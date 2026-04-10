import torch
import numpy as np
from torch.utils.data import random_split
import os
import yaml
from datetime import date
from pathlib import Path
import copy


def _get_labels_for_stratification(dataset):
    """Retourne les labels (np.ndarray) si présents, sinon None."""
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if hasattr(dataset, "df") and "label" in dataset.df.columns:
        return dataset.df["label"].to_numpy()
    return None


def _stratified_split(indices, labels, val_size, test_size, seed):
    """Split stratifié simple en train/val/test à partir d'indices et labels globaux."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        cls_mask = labels == cls
        cls_indices = np.asarray(indices)[cls_mask]
        rng.shuffle(cls_indices)
        n = len(cls_indices)
        n_val = int(round(n * val_size))
        n_test = int(round(n * test_size))
        # Evite de dépasser n (peut arriver avec les arrondis)
        n_val = min(n_val, n)
        n_test = min(n_test, n - n_val)
        n_train = n - n_val - n_test
        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_indices[n_train + n_val:n_train + n_val + n_test].tolist())
    # Mélange pour éviter l'ordre par classe
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _build_folds(indices, labels, k, seed):
    """Construit k folds stratifiés pour CV."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    folds = [[] for _ in range(k)]
    for cls in np.unique(labels):
        cls_idx = np.asarray(indices)[labels == cls]
        rng.shuffle(cls_idx)
        chunks = np.array_split(cls_idx, k)
        for i, chunk in enumerate(chunks):
            folds[i].extend(chunk.tolist())
    # Mélange chaque fold pour éviter l'ordre par classe
    for i in range(k):
        rng.shuffle(folds[i])
    return [np.array(f, dtype=np.int64) for f in folds]


class SequenceSubset(torch.utils.data.Dataset):
    """
    Vue sur un dataset séquentiel pour contrôler l'aléa de sous-séquence par split
    tout en partageant les buffers du dataset de base.
    """
    def __init__(self, base_dataset, indices, random_subseq=None, va_jitter_override=None):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if random_subseq is None and hasattr(base_dataset, "random_subseq"):
            random_subseq = base_dataset.random_subseq
        self.random_subseq = bool(random_subseq)
        self.va_jitter_override = va_jitter_override

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = int(self.indices[idx])
        if hasattr(self.base_dataset, "_get_item"):
            return self.base_dataset._get_item(
                base_idx,
                random_subseq=self.random_subseq,
                va_jitter_override=self.va_jitter_override
            )
        return self.base_dataset[base_idx]


def train_val_split(
    dataset,
    generalize=False,
    val_size=0.2,
    test_size=None,
    seed=8,
    rng=None,
    test=True,
    stratify=False,
    cv_folds=0,
    cv_fold_index=0,
    time_disjoint=False,
    no_overlap=False
):
    """
    Split amélioré:
    - Stratifié optionnel (pour MSTAR) avec tailles val/test configurables.
    - CV optionnel: choisit un fold comme test, split stratifié du reste en train/val.
    - generalize=True conserve le comportement historique (par MMSI) pour RP.
    Retourne: (train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, test_mmsis)
    """
    n = len(dataset)
    all_idx = np.arange(n, dtype=np.int64)
    test_size = (val_size / 2 if test_size is None else test_size)
    subset_random = getattr(dataset, "random_subseq", True)
    def _make_subset(ds, idx_list, jitter_override=None, rnd=None):
        if hasattr(ds, "_get_item"):
            return SequenceSubset(ds, idx_list, random_subseq=rnd if rnd is not None else subset_random,
                                  va_jitter_override=jitter_override)
        return torch.utils.data.Subset(ds, idx_list)

    # Option: split sans chevauchement en repartant des RPs bruts (reconstruction de séquences par split)
    if no_overlap and hasattr(dataset, "df"):
        rng_np = np.random.default_rng(seed)
        raw_indices = np.arange(len(dataset.df), dtype=np.int64)
        labels_raw = getattr(dataset, "raw_labels", None)
        if stratify and labels_raw is not None:
            train_idx, val_idx, test_idx = _stratified_split(
                raw_indices, labels_raw, val_size, test_size, seed
            )
        else:
            rng_np.shuffle(raw_indices)
            n_val = int(round(len(raw_indices) * val_size))
            n_test = int(round(len(raw_indices) * test_size))
            n_val = min(n_val, len(raw_indices))
            n_test = min(n_test, len(raw_indices) - n_val)
            n_train = len(raw_indices) - n_val - n_test
            train_idx = raw_indices[:n_train].tolist()
            val_idx = raw_indices[n_train:n_train + n_val].tolist()
            test_idx = raw_indices[n_train + n_val:n_train + n_val + n_test].tolist()

        def _build_ds(raw_idx):
            cfg = copy.deepcopy(dataset.config)
            return dataset.__class__(cfg, raw_indices=raw_idx)

        train_ds = _build_ds(train_idx)
        val_ds   = _build_ds(val_idx)
        test_ds  = _build_ds(test_idx)
        return train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, []

    # Option: split temporel disjoint par MMSI (pas de fenêtres qui se chevauchent entre splits)
    # Mode blocs contigus : pour chaque MMSI, on garde des segments temporels compacts.
    # On choisit un bloc val puis un bloc test juste après (ou avant) de taille cible,
    # le reste va en train. Cela limite les discontinuités tout en évitant la concentration
    # systématique en début/fin de trajectoire (random start).
    if time_disjoint:
        # indices des séquences (niveau dataset.sequences) : déjà échantillons unitaires
        if not hasattr(dataset, "sequences"):
            raise ValueError("time_disjoint=True mais dataset ne fournit pas .sequences")
        df = getattr(dataset, "df_samples", None) if hasattr(dataset, "df_samples") else getattr(dataset, "df", None)
        if df is None or "mmsi" not in df.columns:
            raise ValueError("time_disjoint=True mais df_samples/df sans colonne mmsi")
        # récupère timestamp associé à chaque séquence (colonnes df_samples)
        if "unix_seconds" in df.columns:
            ts_col = df["unix_seconds"].to_numpy()
        else:
            ts_col = np.zeros(len(dataset.sequences), dtype=np.float64)
        mmsi_col = df["mmsi"].to_numpy()
        rng_np = np.random.default_rng(seed)
        train_idx = []
        val_idx = []
        test_idx = []
        buffer = max(0, getattr(dataset, "sequence_length", 1) - 1)  # pour limiter le chevauchement entre splits
        angle_bins = 12
        for mmsi in np.unique(mmsi_col):
            mask = mmsi_col == mmsi
            seq_indices = np.nonzero(mask)[0]
            # trier par temps
            order = np.argsort(ts_col[seq_indices])
            seq_indices = seq_indices[order]
            # bins angulaires pour stratification légère
            if "viewing_angle" in df.columns:
                ang = df["viewing_angle"].to_numpy()[seq_indices]
            else:
                ang = np.zeros(len(seq_indices))
            bin_ids = np.clip((ang % (2 * np.pi)) / (2 * np.pi) * angle_bins, 0, angle_bins - 1e-6).astype(int)
            # cumul par bin pour calcul rapide d'histo de segments contigus
            cum = np.zeros((angle_bins, len(seq_indices)+1), dtype=int)
            for i, b in enumerate(bin_ids, start=1):
                cum[:, i] = cum[:, i-1]
                cum[b, i] += 1
            n = len(seq_indices)
            if n == 0:
                continue
            n_val = int(round(n * val_size))
            n_test = int(round(n * test_size))
            n_val = min(n_val, n)
            n_test = min(n_test, n - n_val)
            n_train = n - n_val - n_test
            if n_val == 0 or n_test == 0 or n_train <= 0:
                # fallback: tout en train
                train_idx.extend(seq_indices.tolist())
                continue
            start_max = n - (n_val + n_test)
            def hist_block(start, length):
                end = start + length
                end = min(end, len(seq_indices))
                if start >= end:
                    return np.zeros(angle_bins, dtype=int)
                return cum[:, end] - cum[:, start]
            total_hist = hist_block(0, n)
            best_start = 0
            best_score = 1e9
            for s in range(max(1, start_max+1)):
                val_h = hist_block(s, n_val)
                test_h = hist_block(s + n_val, n_test)
                # score = divergence angulaire vs distribution globale
                score = 0.0
                for h in (val_h, test_h):
                    # chi-square-like with smoothing
                    expected = total_hist * (h.sum() / max(total_hist.sum(), 1))
                    num = (h - expected) ** 2
                    den = expected + 1e-6
                    score += float((num / den).sum())
                if score < best_score:
                    best_score = score
                    best_start = s
            val_start = best_start
            val_block = seq_indices[val_start:val_start + n_val]
            test_block = seq_indices[val_start + n_val:val_start + n_val + n_test]
            train_block = np.concatenate([seq_indices[:val_start], seq_indices[val_start + n_val + n_test:]])
            # Retire un petit buffer pour limiter le recouvrement (stride=1)
            if buffer > 0:
                if len(train_block) > buffer:
                    train_block = train_block[:-buffer]
                if len(val_block) > buffer:
                    val_block = val_block[buffer:]  # retire le début pour séparer de train
                if len(test_block) > buffer:
                    test_block = test_block[buffer:]  # sépare val/test
            train_idx.extend(train_block.tolist())
            val_idx.extend(val_block.tolist())
            test_idx.extend(test_block.tolist())
        # Mélange pour casser l'ordre par MMSI
        rng_np.shuffle(train_idx); rng_np.shuffle(val_idx); rng_np.shuffle(test_idx)
        train_ds = _make_subset(dataset, train_idx, jitter_override=None, rnd=subset_random)
        val_ds   = _make_subset(dataset, val_idx, jitter_override=0.0, rnd=False)
        test_ds  = _make_subset(dataset, test_idx, jitter_override=0.0, rnd=False)
        return train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, []

    if generalize:
        rng_np = np.random.default_rng(seed)
        # ----- Tirage par MMSI (standard) -----
        if rng is None or rng == "None":
            mmsi_col = dataset.df["mmsi"].to_numpy()  # (N,)
            unique_mmsis = np.unique(mmsi_col)
            if len(unique_mmsis) == 0:
                train_idx = all_idx
                val_idx = np.array([], dtype=np.int64)
                test_idx = np.array([], dtype=np.int64)
                test_mmsis = np.array([], dtype=mmsi_col.dtype)
            else:
                nb_mmsis = max(1, int(round(len(unique_mmsis) * val_size)))
                nb_mmsis = min(nb_mmsis, len(unique_mmsis))
                if nb_mmsis == 0:
                    train_idx = all_idx
                    val_idx = np.array([], dtype=np.int64)
                    test_idx = np.array([], dtype=np.int64)
                    test_mmsis = np.array([], dtype=unique_mmsis.dtype)
                else:
                    val_mmsis = rng_np.choice(unique_mmsis, nb_mmsis, replace=False)
                    test_mmsis = np.array([], dtype=unique_mmsis.dtype)
                    if test:
                        nb_test = max(nb_mmsis // 2, 1) if nb_mmsis > 1 else 0
                        if nb_test > 0:
                            test_mmsis = rng_np.choice(val_mmsis, nb_test, replace=False)
                            val_mmsis = np.setdiff1d(val_mmsis, test_mmsis, assume_unique=False)

                    mask_val  = np.isin(mmsi_col, val_mmsis)
                    mask_test = np.isin(mmsi_col, test_mmsis) if test_mmsis.size > 0 else np.zeros(n, dtype=bool)
                    mask_train = ~(mask_val | mask_test)

                    val_idx   = all_idx[mask_val]
                    test_idx  = all_idx[mask_test]
                    train_idx = all_idx[mask_train]

            train_ds = _make_subset(dataset, train_idx.tolist(), jitter_override=None, rnd=subset_random)
            val_ds   = _make_subset(dataset, val_idx.tolist(), jitter_override=0.0, rnd=False)
            test_ds  = _make_subset(dataset, test_idx.tolist(), jitter_override=0.0, rnd=False)
            return train_ds, val_ds, test_ds, \
                   train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), test_mmsis.tolist()

        # ----- rng numérique: fenêtre contiguë d'indices (vectorisée) -----
        r = float(rng)
        r = 0.0 if r < 0 else (1.0 if r > 1.0 else r)
        if r == 1.0:
            r = r - val_size  # pour avoir au moins un échantillon en val

        start = int(r * n)
        n_val = int(val_size * n)
        stop  = min(start + n_val, n)
        val_idx = np.arange(start, stop, dtype=np.int64)

        if test and val_idx.size > 0:
            test_idx = rng_np.choice(val_idx, val_idx.size // 2, replace=False)
            keep_mask = np.ones(val_idx.shape[0], dtype=bool)
            test_set = set(test_idx.tolist())
            for j, v in enumerate(val_idx):
                if v in test_set:
                    keep_mask[j] = False
            val_idx = val_idx[keep_mask]
        else:
            test_idx = np.array([], dtype=np.int64)

        keep = np.ones(n, dtype=bool)
        keep[val_idx] = False
        keep[test_idx] = False
        train_idx = all_idx[keep]

        train_ds = _make_subset(dataset, train_idx.tolist(), jitter_override=None, rnd=subset_random)
        val_ds   = _make_subset(dataset, val_idx.tolist(), jitter_override=0.0, rnd=False)
        test_ds  = _make_subset(dataset, test_idx.tolist(), jitter_override=0.0, rnd=False)
        return train_ds, val_ds, test_ds, \
               train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), []

    # ----- generalize=False -----
    labels = _get_labels_for_stratification(dataset) if stratify or cv_folds else None

    if cv_folds and cv_folds >= 2 and labels is not None:
        folds = _build_folds(all_idx, labels, cv_folds, seed)
        fold_idx = int(cv_fold_index) % cv_folds
        test_idx = folds[fold_idx]
        remaining = np.concatenate([folds[i] for i in range(cv_folds) if i != fold_idx])
        # Split val/train à l'intérieur du reste
        train_idx, val_idx, _ = _stratified_split(remaining, labels[remaining], val_size, 0.0, seed)
        train_ds = _make_subset(dataset, train_idx, jitter_override=None, rnd=subset_random)
        val_ds = _make_subset(dataset, val_idx, jitter_override=0.0, rnd=False)
        test_ds = _make_subset(dataset, test_idx.tolist(), jitter_override=0.0, rnd=False)
        return train_ds, val_ds, test_ds, \
               train_idx, val_idx, test_idx.tolist(), []

    if stratify and labels is not None:
        train_idx, val_idx, test_idx = _stratified_split(all_idx, labels, val_size, test_size if test else 0.0, seed)
        train_ds = _make_subset(dataset, train_idx, jitter_override=None, rnd=subset_random)
        val_ds   = _make_subset(dataset, val_idx, jitter_override=0.0, rnd=False)
        test_ds  = _make_subset(dataset, test_idx, jitter_override=0.0, rnd=False)
        return train_ds, val_ds, test_ds, \
               train_idx, val_idx, test_idx, []

    # Fallback: random_split comme avant
    g = torch.Generator().manual_seed(seed)
    n_val = int(round(n * val_size))
    train_len = n - n_val
    train_ds, val_ds = random_split(dataset, [train_len, n_val], generator=g)
    train_idx = np.array([all_idx[i] for i in train_ds.indices], dtype=np.int64)
    n_test = int(round(n * test_size)) if test else 0
    if n_test > 0:
        n_test = min(n_test, n_val)
        val_len = n_val - n_test
        val_ds, test_ds = random_split(val_ds, [val_len, n_test], generator=g)
        val_idx = np.array([all_idx[i] for i in val_ds.indices], dtype=np.int64)
        test_idx = np.array([all_idx[i] for i in test_ds.indices], dtype=np.int64)
    else:
        test_ds = torch.utils.data.Subset(dataset, [])
        test_idx = np.array([], dtype=np.int64)

    # Optionnel : enlever les chevauchements de RP entre splits
    if no_overlap and hasattr(dataset, "sequences"):
        seqs = dataset.sequences
        def raw_set(idxs):
            s = set()
            for i in idxs:
                s.update(seqs[int(i)])
            return s
        test_raw = raw_set(test_idx)
        # retire des val ceux qui recouvrent test
        val_idx = [i for i in val_idx.tolist() if test_raw.isdisjoint(seqs[int(i)])]
        val_raw = raw_set(val_idx)
        occupied = test_raw | val_raw
        train_idx = [i for i in train_idx.tolist() if occupied.isdisjoint(seqs[int(i)])]
    train_ds = _make_subset(dataset, train_idx.tolist(), jitter_override=None, rnd=subset_random)
    val_ds   = _make_subset(dataset, val_idx.tolist(), jitter_override=0.0, rnd=False)
    test_ds  = _make_subset(dataset, test_idx.tolist(), jitter_override=0.0, rnd=False) if test_idx.size > 0 else test_ds

    return train_ds, val_ds, test_ds, \
           train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), []

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)

    return config

def get_save_path_create_folder(config, seed):
    # figure_path = "results RPclf" -> base_dir="results", run_name="RPclf"
    parts = str(config.get("figure_path", "results run")).split()
    base_dir = Path(parts[0]) / "Samples" / str(date.today())
    run_name = parts[1] if len(parts) > 1 else "run"
    seed_dir = base_dir / f"{run_name}_seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    # Versioning: sous-dossiers numériques
    existing = [int(p.name) for p in seed_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    z = max(existing) + 1 if existing else 0
    run_dir = seed_dir / str(z)
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)

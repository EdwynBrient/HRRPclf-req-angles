import torch
import numpy as np
from torch.utils.data import random_split
import os
import yaml
from datetime import date
from pathlib import Path


def _get_labels_for_stratification(dataset):
    """Return labels (np.ndarray) when available, otherwise None."""
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    if hasattr(dataset, "df") and "label" in dataset.df.columns:
        return dataset.df["label"].to_numpy()
    return None


def _stratified_split(indices, labels, val_size, test_size, seed):
    """Simple stratified train/val/test split from global indices and labels."""
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
        # Avoid exceeding n (can happen due to rounding)
        n_val = min(n_val, n)
        n_test = min(n_test, n - n_val)
        n_train = n - n_val - n_test
        train_idx.extend(cls_indices[:n_train].tolist())
        val_idx.extend(cls_indices[n_train:n_train + n_val].tolist())
        test_idx.extend(cls_indices[n_train + n_val:n_train + n_val + n_test].tolist())
    # Shuffle to avoid class-ordered indices
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _build_folds(indices, labels, k, seed):
    """Build k stratified folds for cross-validation."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    folds = [[] for _ in range(k)]
    for cls in np.unique(labels):
        cls_idx = np.asarray(indices)[labels == cls]
        rng.shuffle(cls_idx)
        chunks = np.array_split(cls_idx, k)
        for i, chunk in enumerate(chunks):
            folds[i].extend(chunk.tolist())
    # Shuffle each fold to avoid class ordering
    for i in range(k):
        rng.shuffle(folds[i])
    return [np.array(f, dtype=np.int64) for f in folds]


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
    cv_fold_index=0
):
    """
    Enhanced split:
    - Optional stratification (for MSTAR) with configurable val/test sizes.
    - Optional CV: select one fold as test, stratified split of the remainder into train/val.
    - generalize=True preserves historical RP behavior (MMSI-based split).
    Returns: (train_ds, val_ds, test_ds, train_idx, val_idx, test_idx, test_mmsis)
    """
    n = len(dataset)
    all_idx = np.arange(n, dtype=np.int64)
    test_size = (val_size / 2 if test_size is None else test_size)

    if generalize:
        # ----- generalize=True -----
        rng_np = np.random.default_rng(seed)

        if rng is None or rng == "None":
            # Sampling by unique MMSI, vectorized implementation
            mmsi_col = dataset.df["mmsi"].to_numpy()  # (N,)
            unique_mmsis = np.unique(mmsi_col)
            if len(unique_mmsis) == 0:
                train_idx = all_idx
                val_idx = np.array([], dtype=np.int64)
                test_idx = np.array([], dtype=np.int64)
                test_mmsis = np.array([], dtype=mmsi_col.dtype)
                train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
                val_ds   = torch.utils.data.Subset(dataset, val_idx.tolist())
                test_ds  = torch.utils.data.Subset(dataset, test_idx.tolist())
                return train_ds, val_ds, test_ds, \
                       train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), []

            # Ensure at least 1 MMSI in validation when possible
            nb_mmsis = max(1, int(round(len(unique_mmsis) * val_size)))
            nb_mmsis = min(nb_mmsis, len(unique_mmsis))

            if nb_mmsis == 0:
                # fallback: rien en val, tout en train
                train_idx = all_idx
                val_idx = np.array([], dtype=np.int64)
                test_idx = np.array([], dtype=np.int64)
                test_mmsis = np.array([], dtype=unique_mmsis.dtype)
            else:
                val_mmsis = rng_np.choice(unique_mmsis, nb_mmsis, replace=False)
                test_mmsis = np.array([], dtype=unique_mmsis.dtype)
                if test:
                    # Move half of validation MMSIs to test
                    nb_test = max(nb_mmsis // 2, 1) if nb_mmsis > 1 else 0
                    if nb_test > 0:
                        test_mmsis = rng_np.choice(val_mmsis, nb_test, replace=False)
                        # MMSI de validation = val_mmsis \ test_mmsis
                        val_mmsis = np.setdiff1d(val_mmsis, test_mmsis, assume_unique=False)

                mask_val  = np.isin(mmsi_col, val_mmsis)
                mask_test = np.isin(mmsi_col, test_mmsis) if test_mmsis.size > 0 else np.zeros(n, dtype=bool)
                mask_train = ~(mask_val | mask_test)

                val_idx   = all_idx[mask_val]
                test_idx  = all_idx[mask_test]
                train_idx = all_idx[mask_train]

            # Build subsets
            train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
            val_ds   = torch.utils.data.Subset(dataset, val_idx.tolist())
            test_ds  = torch.utils.data.Subset(dataset, test_idx.tolist())

            return train_ds, val_ds, test_ds, \
                   train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), test_mmsis.tolist()

        else:
            # Numeric rng: contiguous index window (vectorized)
            r = float(rng)
            r = 0.0 if r < 0 else (1.0 if r > 1.0 else r)
            if r == 1.0:
                r = r - val_size  # ensure at least one validation sample

            start = int(r * n)
            n_val = int(val_size * n)
            stop  = min(start + n_val, n)
            val_idx = np.arange(start, stop, dtype=np.int64)

            if test and val_idx.size > 0:
                # Move half of validation indices to test (stable sampling)
                test_idx = rng_np.choice(val_idx, val_idx.size // 2, replace=False)
                # Remove test indices from validation
                keep_mask = np.ones(val_idx.shape[0], dtype=bool)
                # Mark positions belonging to test_idx as False
                # Fast method: create a set of test values for O(1) lookup
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

            train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
            val_ds   = torch.utils.data.Subset(dataset, val_idx.tolist())
            test_ds  = torch.utils.data.Subset(dataset, test_idx.tolist())
            return train_ds, val_ds, test_ds, \
                   train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), []

    # ----- generalize=False -----
    labels = _get_labels_for_stratification(dataset) if stratify or cv_folds else None

    if cv_folds and cv_folds >= 2 and labels is not None:
        folds = _build_folds(all_idx, labels, cv_folds, seed)
        fold_idx = int(cv_fold_index) % cv_folds
        test_idx = folds[fold_idx]
        remaining = np.concatenate([folds[i] for i in range(cv_folds) if i != fold_idx])
        # Split val/train inside the remaining indices
        train_idx, val_idx, _ = _stratified_split(remaining, labels[remaining], val_size, 0.0, seed)
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds = torch.utils.data.Subset(dataset, val_idx)
        test_ds = torch.utils.data.Subset(dataset, test_idx.tolist())
        return train_ds, val_ds, test_ds, \
               train_idx, val_idx, test_idx.tolist(), []

    if stratify and labels is not None:
        train_idx, val_idx, test_idx = _stratified_split(all_idx, labels, val_size, test_size if test else 0.0, seed)
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        val_ds   = torch.utils.data.Subset(dataset, val_idx)
        test_ds  = torch.utils.data.Subset(dataset, test_idx)
        return train_ds, val_ds, test_ds, \
               train_idx, val_idx, test_idx, []

    # Fallback: random_split as before
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
    return train_ds, val_ds, test_ds, \
           train_idx.tolist(), val_idx.tolist(), test_idx.tolist(), []

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)

    return config

def get_save_path_create_folder(config, seed):
    """
        Create a versioned save directory.
        `figure_path` can be either:
            - a path with two space-separated tokens (base + name)
            - a simple path (e.g., results/Samples/ships_film_angles)
    """
    fp = str(config.get("figure_path", "results/Samples/"))
    tokens = fp.split()
    if len(tokens) >= 2:
        base_dir, exp_name = tokens[0], tokens[1]
    else:
        p = Path(fp)
        # If `figure_path` is only a name, place it under results/Samples
        if p.suffix:  # unlikely case: file path
            base_dir = str(p.parent) if str(p.parent) not in ("", ".") else "results/Samples"
            exp_name = p.stem
        else:
            base_dir = str(p.parent) if str(p.parent) not in ("", ".") else "results/Samples"
            exp_name = p.name if p.name else "exp"

    date_dir = os.path.join(base_dir, str(date.today()))
    save_root = os.path.join(date_dir, f"{exp_name}_seed{seed}")
    os.makedirs(save_root, exist_ok=True)
    dirs = [int(d) for d in os.listdir(save_root) if d.isdigit()]
    z = (np.max(dirs) + 1) if len(dirs) > 0 else 0
    run_dir = os.path.join(save_root, str(z))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

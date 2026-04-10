from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
try:
    from .dataset import *
except ImportError:
    from dataset import *
import torch
try:
    from .models import *
except ImportError:
    from models import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from .utils import *
except ImportError:
    from utils import *
from torchvision import transforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
from datetime import date, datetime
from lightning.pytorch.profilers import PyTorchProfiler
import argparse
import shutil
import numpy as np
import math
import csv
from pathlib import Path
import logging
from typing import Optional

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
# Silence noisy symbolic-shape warnings from torch.compile
logging.getLogger("torch.fx.experimental.symbolic_shapes").setLevel(logging.ERROR)
parser = argparse.ArgumentParser(description='Train a DDPM model.')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch-name', type=str, default=None, help="Identifiant du batch pour ranger metrics.csv dans un dossier dédié")
parser.add_argument('--metrics-path', type=str, default=None, help="Chemin explicite du metrics.csv de batch (un fichier unique par multiple_jobs)")
args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

n_cpu = os.getenv("SLURM_CPUS_PER_TASK")
partition = os.getenv("SLURM_JOB_PARTITION")

# Resolve configuration path relative to the repo root (location of this file) to avoid CWD issues on SLURM.
def resolve_config_path(cfg_arg: str):
    base_dir = Path(__file__).resolve().parent  # folder containing train.py
    cfg_path = Path(cfg_arg)
    # Absolute path: use as-is
    if cfg_path.is_absolute():
        return cfg_path
    # 1) try relative to current working directory
    cwd_candidate = (Path.cwd() / cfg_path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    # 2) try relative to script location
    script_candidate = (base_dir / cfg_path).resolve()
    if script_candidate.exists():
        return script_candidate
    # 3) try base_dir/config and base_dir/config/configs_v100
    for sub in ["config", "config/configs_v100"]:
        candidate = (base_dir / sub / cfg_path).resolve()
        if candidate.exists():
            return candidate
    # Fallback: return what would be under config/configs_v100 (will error later if missing)
    return (base_dir / "config" / "configs_v100" / cfg_path).resolve()


def infer_experiment_name(config: dict, config_path: Path) -> str:
    """Derive a readable experiment name for logging/metrics."""
    if config.get("experiment_name"):
        return str(config["experiment_name"])
    if config.get("model_name"):
        return str(config["model_name"])
    figure_path = str(config.get("figure_path", "") or "")
    if figure_path:
        tokens = [tok.strip("/").strip() for tok in figure_path.split() if tok.strip()]
        if len(tokens) >= 2:
            return "_".join(tokens[1:])
        if tokens:
            return tokens[-1]
    return Path(config_path).stem


# folder to load config file
CONFIG_PATH = resolve_config_path(args.config)
config = load_config(str(CONFIG_PATH))
xp_name = infer_experiment_name(config, CONFIG_PATH)
config.setdefault("experiment_name", xp_name)
config.setdefault("model_name", xp_name)
config["clf"]["cond_op"] = config["conditionned"]["cond_op"]
config["clf"]["cond"] = config["conditionned"]["bool"]
bs = config["batch_size"]
device_cfg = config.get("device", "auto")
split_cfg = config.get("split", {})
val_size = split_cfg.get("val_size", 0.10)
test_size = split_cfg.get("test_size", val_size / 2)
stratified = bool(split_cfg.get("stratified", config["dataset"] == "MSTAR"))
cv_folds = int(split_cfg.get("cv_folds", 0))
cv_fold_index = int(split_cfg.get("fold_index", 0))
# Par défaut en classification: pas de split par MMSI (generalize) sauf override explicite
generalize_default = 0
generalize = split_cfg.get("generalize", generalize_default)
time_disjoint = bool(split_cfg.get("time_disjoint", False))
no_overlap = bool(split_cfg.get("no_overlap", False))
max_epochs_cfg = int(config.get("epochs", 0))
min_epochs_cfg = int(config.get("min_epochs", 0))
grad_clip_val = config.get("gradient_clip_val", None)


def resolve_metrics_path(custom_path: Optional[str] = None) -> Path:
    """
    Return the path to the batch metrics file.
    Default: results/Samples/YYYY-MM-DD/metricsHH.csv (HH = hour of launch).
    """
    if custom_path:
        return Path(custom_path)
    now = datetime.now()
    date_dir = Path("results") / "Samples" / now.date().isoformat()
    date_dir.mkdir(parents=True, exist_ok=True)
    filename = f"metrics{now.strftime('%H')}.csv"
    return date_dir / filename


# Folder dedicated to this batch of experiments (per-run metrics live here)
batch_metrics_path = resolve_metrics_path(args.metrics_path)

# Single experiment save root (config saved once per experiment, not per fold)
experiment_save_root = get_save_path_create_folder(config, args.seed)
shutil.copyfile(CONFIG_PATH, os.path.join(experiment_save_root, "config.yaml"))

def choose_accelerator(dev_cfg):
    if dev_cfg == "cpu":
        return "cpu", 1
    if dev_cfg == "gpu":
        if torch.cuda.is_available():
            return "gpu", torch.cuda.device_count()
        else:
            print("Warning: device=gpu demandé mais CUDA indisponible, bascule sur CPU.")
            return "cpu", 1
    # auto
    if torch.cuda.is_available():
        return "gpu", torch.cuda.device_count()
    return "cpu", 1

accelerator, n_devices = choose_accelerator(device_cfg)

def compute_class_weights(dataset, num_classes=None):
    """
    Calcule des poids inverses des fréquences de classe à partir du dataset (ou SequenceSubset).
    """
    labels = None
    if hasattr(dataset, "labels"):
        labels = torch.as_tensor(dataset.labels)
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "labels") and hasattr(dataset, "indices"):
        labels_full = torch.as_tensor(dataset.dataset.labels)
        labels = labels_full[torch.as_tensor(dataset.indices, dtype=torch.long)]
    elif hasattr(dataset, "base_dataset") and hasattr(dataset.base_dataset, "labels"):
        labels_full = torch.as_tensor(dataset.base_dataset.labels)
        labels = labels_full[torch.as_tensor(dataset.indices, dtype=torch.long)]
    if labels is None:
        return None
    labels = labels.view(-1).long()
    if labels.numel() == 0:
        return None
    inferred_num_classes = int(labels.max().item()) + 1
    if num_classes is None:
        target_num_classes = inferred_num_classes
    else:
        target_num_classes = max(int(num_classes), inferred_num_classes)
    counts = torch.bincount(labels, minlength=target_num_classes)
    weights = counts.sum() / (counts.float().clamp(min=1) * counts.numel())
    present_classes = int((counts > 0).sum().item())
    print(
        f"[INFO] weighted CE: classes présentes dans train={present_classes}/{target_num_classes}, "
        f"samples_train={int(counts.sum().item())}"
    )
    missing = (counts == 0).sum().item()
    if missing > 0:
        print(f"[WARN] {missing} classe(s) absente(s) du split train. Poids étendus à {counts.numel()} classes.")
    return weights

# Setting dataset
if config["dataset"] == "MSTAR":
    # Par défaut on utilise le même fichier que le notebook d'énergie (pkl préprocessé)
    default_mstar_path = "data/MSTAR_hrrp.pkl"
    data_path = config.get("data_path", default_mstar_path)
    dataset = MSTAR_dataset(config, path=data_path)
else:
    dataset = RP_ImageDataset(config)
print("Dataset length : ", len(dataset))
if config["dataset"] != "MSTAR" and hasattr(dataset, "mmsi_values"):
    detected_num_classes = int(len(dataset.mmsi_values))
    if detected_num_classes > 0:
        cfg_num_classes = int(config.get("num_classes", detected_num_classes))
        if cfg_num_classes != detected_num_classes:
            print(
                f"[WARN] num_classes={cfg_num_classes} mais {detected_num_classes} MMSI sélectionnés. "
                f"num_classes ajusté à {detected_num_classes}."
            )
        config["num_classes"] = detected_num_classes
if config["dataset"] != "MSTAR":
    requested_mmsi = int(getattr(dataset, "requested_mmsi_n", 0) or 0)
    selected_mmsi = int(getattr(dataset, "selected_mmsi_n", len(getattr(dataset, "mmsi_values", []))))
    selection_mode = getattr(dataset, "mmsi_selection_mode", "default")
    if requested_mmsi > 0:
        print(
            f"[INFO] MMSI retenus pour CE: {selected_mmsi}/{requested_mmsi} "
            f"(mode={selection_mode}, min_samples={getattr(dataset, 'min_mmsi_samples', 'n/a')})"
        )
    else:
        print(
            f"[INFO] MMSI retenus pour CE: {selected_mmsi} "
            f"(mode={selection_mode}, min_samples={getattr(dataset, 'min_mmsi_samples', 'n/a')})"
        )
min_rp, max_rp = dataset.min_rp, dataset.max_rp


def build_dataloaders(train_dataset, val_dataset, test_dataset):
    if n_cpu:
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=int(n_cpu)-1, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2
        )
        val_dl = torch.utils.data.DataLoader(
            val_dataset, batch_size=bs, shuffle=False, num_workers=int(n_cpu)-1, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        ) if len(val_dataset) > 0 else None
        test_dl = torch.utils.data.DataLoader(
            test_dataset, batch_size=bs, shuffle=False, num_workers=int(n_cpu)-1, pin_memory=True,
            persistent_workers=True, prefetch_factor=2
        ) if len(test_dataset) > 0 else None
    else:
        cpu_ct = os.cpu_count() or 2
        workers = max(1, cpu_ct - 1)
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=workers,
            persistent_workers=True, prefetch_factor=2, pin_memory=True
        )
        val_dl = torch.utils.data.DataLoader(
            val_dataset, batch_size=bs, shuffle=True, num_workers=workers,
            persistent_workers=True, prefetch_factor=2, pin_memory=True
        ) if len(val_dataset) > 0 else None
        test_dl = torch.utils.data.DataLoader(
            test_dataset, batch_size=bs, shuffle=False, num_workers=workers,
            persistent_workers=True, prefetch_factor=2, pin_memory=True
        ) if len(test_dataset) > 0 else None
    return train_dl, val_dl, test_dl


def append_results_csv(row, csv_path="results/metrics.csv"):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def next_version_dir(base_dir: str = "results/Samples") -> str:
    """Create/return a new versioned subfolder results/Samples/YYYY-MM-DD/{n}."""
    base_dir = str(base_dir)
    date_dir = os.path.join(base_dir, date.today().isoformat())
    os.makedirs(date_dir, exist_ok=True)
    existing = [
        int(d) for d in os.listdir(date_dir)
        if os.path.isdir(os.path.join(date_dir, d)) and d.isdigit()
    ]
    next_idx = (max(existing) + 1) if existing else 0
    version_dir = os.path.join(date_dir, str(next_idx))
    os.makedirs(version_dir, exist_ok=True)
    return version_dir

early_stop_cfg = config.get("early_stopping")
use_early_stopping = bool(early_stop_cfg)
monitor = (early_stop_cfg or {}).get("monitor", "val_loss")
mode = (early_stop_cfg or {}).get("mode", "min")
patience_checks = patience_epochs = None

if use_early_stopping:
    patience_checks = early_stop_cfg.get("patience")  # patience expressed in validation checks
    patience_epochs = int(early_stop_cfg.get("patience_epochs", 50))
    # Enforce a minimum number of epochs to avoid stopping before convergence kicks in
    min_epochs = max(
        1,
        int(early_stop_cfg.get("min_epochs", patience_epochs)),
        min_epochs_cfg,
        patience_epochs,
    )
    max_epochs = max(max_epochs_cfg, min_epochs) if max_epochs_cfg else min_epochs
    print(
        f"Early stopping -> monitor={monitor}, mode={mode}, patience_epochs={patience_epochs}, "
        f"min_epochs={min_epochs}, max_epochs={max_epochs} (config_epochs={max_epochs_cfg})"
    )
else:
    # No patience configured: run for the configured number of epochs.
    min_epochs = max(1, min_epochs_cfg)
    max_epochs = max(max_epochs_cfg, min_epochs) if max_epochs_cfg else min_epochs
    print(f"Early stopping disabled -> min_epochs={min_epochs}, max_epochs={max_epochs} (config_epochs={max_epochs_cfg})")

fold_indices = range(cv_folds) if cv_folds and cv_folds > 1 else [cv_fold_index]
all_test_metrics = []
run_rows = []

for fold in fold_indices:
    train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx, test_mmsis = train_val_split(
        dataset,
        generalize=generalize,
        val_size=val_size,
        test_size=test_size,
        seed=args.seed,
        rng=None,
        stratify=stratified,
        cv_folds=cv_folds,
        cv_fold_index=fold,
        time_disjoint=time_disjoint,
        no_overlap=no_overlap
    )
    # Aléas par split : random subseq pour train/val, pas pour test ; pas de jitter en val/test
    for ds, rnd, jitter in [
        # train : jitter normal du config (None => valeur dataset)
        (train_dataset, True, None),
        # val/test : pas de jitter
        (val_dataset, True, 0.0),
        (test_dataset, False, 0.0),
    ]:
        if ds is None:
            continue
        if hasattr(ds, "random_subseq"):
            ds.random_subseq = rnd
        if hasattr(ds, "va_jitter_override"):
            ds.va_jitter_override = jitter

    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(train_dataset, val_dataset, test_dataset)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="results/lightning_logs/{}".format(date.today()),
        name=config["figure_path"].split(" ")[-1],
        version=f"fold{fold}" if cv_folds and cv_folds > 1 else None
    )

    if use_early_stopping:
        # Convert patience in epochs to patience in validation checks to handle datasets with different steps/epoch
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / bs))
        # We now validate once per epoch (check_val_every_n_epoch=1)
        checks_per_epoch = 1
        effective_patience = patience_checks if patience_checks is not None else patience_epochs * checks_per_epoch
        effective_patience = int(effective_patience)
        print(f"[fold {fold}] Patience (val checks): {effective_patience} | steps/epoch: {steps_per_epoch}")
        early_stop_callback = EarlyStopping(monitor=monitor, patience=effective_patience, mode=mode, verbose=True)
    else:
        early_stop_callback = None

    ckpt_dir = os.path.join(experiment_save_root, f"fold{fold}") if cv_folds and cv_folds > 1 else experiment_save_root
    os.makedirs(ckpt_dir, exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        dirpath=ckpt_dir,
        filename="best",
        auto_insert_metric_name=False,
    )
    class_weights = compute_class_weights(train_dataset, num_classes=config.get("num_classes"))
    model = ClassifierPL(config, class_weights=class_weights)
    # Compile uniquement le modèle (pas Lightning) pour accélérer la passe avant
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        try:
            model.model = torch.compile(model.model, dynamic=False)
            print("[INFO] torch.compile activé sur le backbone/classifier.")
        except Exception as e:
            print(f"[WARN] torch.compile désactivé (raison: {e})")

    if partition is not None:
        if len(os.getenv("SLURM_JOB_GPUS")) > 1:
            trainer_args = {
                'accelerator': 'auto',
                'strategy':'ddp',
                'devices': int(torch.cuda.device_count()),
                'num_nodes': int(os.environ['SLURM_NNODES']),
                'max_epochs': max_epochs,
                'check_val_every_n_epoch': 1,
                'log_every_n_steps': 1,
                'num_sanity_val_steps': 0,  # Disable sanity check
                'logger': tb_logger,
                # 'profiler': profiler,
            }
        else:
            trainer_args = {
                'accelerator': 'gpu',
                'devices': int(os.environ['SLURM_GPUS_ON_NODE']),
                'num_nodes': int(os.environ['SLURM_NNODES']),
                'max_epochs': max_epochs,
                'check_val_every_n_epoch': 1,
                'log_every_n_steps': 1,
                'num_sanity_val_steps': 1,
                'logger': tb_logger,
                # 'profiler': profiler,
            }
        
    else:
        trainer_args = {
            'accelerator': accelerator,
            **({'devices': n_devices} if accelerator != "cpu" else {}),
            'max_epochs': max_epochs,
            'check_val_every_n_epoch': 1,
            'log_every_n_steps': 1,
            'num_sanity_val_steps': 1,
            'logger': tb_logger,
            # 'profiler': profiler,
        }

    # Mixed precision to speed up training (safe on GPU; bypassed on CPU)
    if trainer_args.get('accelerator') == 'gpu':
        trainer_args['precision'] = '16-mixed'

    trainer_args['min_epochs'] = min_epochs
    trainer_args['callbacks'] = [cb for cb in [model_checkpoint, early_stop_callback] if cb is not None]
    trainer_args['enable_checkpointing'] = True
    if grad_clip_val is not None:
        trainer_args['gradient_clip_val'] = float(grad_clip_val)

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader if val_dataloader is not None else None)
    stopped_epoch = getattr(early_stop_callback, "stopped_epoch", 0)
    best_score = getattr(early_stop_callback, "best_score", None)
    if hasattr(best_score, "item"):
        try:
            best_score = best_score.item()
        except Exception:
            pass
    if stopped_epoch:
        print(f"EarlyStopping stopped at epoch {stopped_epoch} (best {monitor}={best_score})")
    # Log test metrics on the same TensorBoard run using the best checkpoint from training
    if test_dataloader is not None:
        test_metrics = trainer.test(model=model, dataloaders=test_dataloader, ckpt_path="best")
        if test_metrics:
            metrics_raw = test_metrics[0]
            metrics = {}
            for k, v in metrics_raw.items():
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                metrics[k] = v
            all_test_metrics.append(metrics)
            test_macro_f1 = metrics.get("test_macro_f1", None)
            if test_macro_f1 is None:
                for k, v in metrics.items():
                    if "test_macro_f1" in k:
                        test_macro_f1 = v
                        break
            if test_macro_f1 is None:
                print("[WARN] test_macro_f1 absent des métriques de test: le CSV contiendra une valeur vide.")
            cond_levels = config["clf"].get("cond_levels", None)
            cond_levels_str = ",".join(str(x) for x in cond_levels) if cond_levels else ""
            row = {
                "date": date.today().isoformat(),
                "model_name": config.get("model_name", ""),
                "dataset": config.get("dataset", ""),
                "cond": config["conditionned"]["bool"],
                "cond_op": config["conditionned"]["cond_op"],
                "cond_levels": cond_levels_str,
                "num_classes_used": int(config.get("num_classes", 0)),
                "selected_mmsi_n": int(getattr(dataset, "selected_mmsi_n", 0) or 0),
                "requested_mmsi_n": int(getattr(dataset, "requested_mmsi_n", 0) or 0),
                "mmsi_min_samples": int(getattr(dataset, "min_mmsi_samples", 0) or 0),
                "test_macro_f1": test_macro_f1,
                "seed": args.seed,
                "fold": fold,
                "cv_folds": cv_folds,
                "val_size": val_size,
                "test_size": test_size,
            }
            for k, v in metrics.items():
                row[k] = v
            append_results_csv(row)
            run_rows.append(row)

if cv_folds and cv_folds > 1 and all_test_metrics:
    # Moyenne simple des métriques de test sur les folds
    mean_metrics = {}
    for k in all_test_metrics[0].keys():
        mean_metrics[k] = float(np.mean([m[k] for m in all_test_metrics]))
    print("CV summary (mean over folds):", mean_metrics)

if run_rows:
    batch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not batch_metrics_path.exists()
    with open(batch_metrics_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=run_rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(run_rows)
    print(f"Batch metrics appended to {batch_metrics_path}")

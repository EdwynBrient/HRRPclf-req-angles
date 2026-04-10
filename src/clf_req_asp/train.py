from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
try:
    from .dataset import *
except ImportError:  # pragma: no cover - fallback for direct script execution
    from dataset import *
import torch
try:
    from .models import *
except ImportError:  # pragma: no cover - fallback for direct script execution
    from models import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from .utils import *
except ImportError:  # pragma: no cover - fallback for direct script execution
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
from typing import Optional, Union

torch.set_float32_matmul_precision('medium')
parser = argparse.ArgumentParser(description='Train a DDPM model.')
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch-name', type=str, default=None, help="Batch identifier used to group metrics.csv in a dedicated folder")
parser.add_argument('--metrics-path', type=str, default=None, help="Explicit path to the batch metrics.csv (single file for multiple_jobs)")
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

# folder to load config file
CONFIG_PATH = resolve_config_path(args.config)
config = load_config(str(CONFIG_PATH))
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
# Default for classification: no MMSI-based split (generalize) unless explicitly overridden
generalize_default = 0
generalize = split_cfg.get("generalize", generalize_default)
max_epochs_cfg = int(config.get("epochs", 0))
min_epochs_cfg = int(config.get("min_epochs", 0))
base_class_weights = config.get("class_weights", None)


def compute_class_weights(labels, indices, num_classes=None):
    """
    Compute inverse-frequency class weights from training indices.
    weight_c = N / (C * count_c), zeros are clipped to 1 to avoid inf.
    """
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    idx_t = torch.as_tensor(indices, dtype=torch.long)
    if num_classes is None:
        num_classes = int(labels_t.max().item() + 1)
    counts = torch.bincount(labels_t[idx_t], minlength=num_classes)
    counts_f = counts.float()
    weights = counts_f.sum() / (counts_f.clamp(min=1.0) * float(num_classes))
    # If a class is absent from the split, set its weight to 0 to avoid over-weighting noise.
    weights = torch.where(counts == 0, torch.zeros_like(weights), weights)
    return weights


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
            print("Warning: device=gpu requested but CUDA is unavailable, falling back to CPU.")
            return "cpu", 1
    # auto
    if torch.cuda.is_available():
        return "gpu", torch.cuda.device_count()
    return "cpu", 1

accelerator, n_devices = choose_accelerator(device_cfg)

# Setting dataset
if config["dataset"] == "MSTAR":
    # By default, use the same file as the energy notebook (preprocessed pkl)
    default_mstar_path = "../data/MSTAR_data/MSTAR-10-Classes/train_hrrp_dataframe_mean.pkl"
    data_path = config.get("data_path", default_mstar_path)
    dataset = MSTAR_dataset(config, path=data_path)
else:
    dataset = RP_ImageDataset(config)
print("Dataset length : ", len(dataset))
detected_num_classes = 0
if config["dataset"] != "MSTAR" and hasattr(dataset, "mmsi_values"):
    detected_num_classes = int(len(dataset.mmsi_values))
elif hasattr(dataset, "labels"):
    labels_t = torch.as_tensor(dataset.labels, dtype=torch.long).view(-1)
    if labels_t.numel() > 0:
        detected_num_classes = int(labels_t.max().item() + 1)
if detected_num_classes > 0:
    cfg_num_classes = int(config.get("num_classes", detected_num_classes))
    if cfg_num_classes != detected_num_classes:
        print(
            f"[WARN] num_classes={cfg_num_classes} but {detected_num_classes} classes were detected. "
            f"num_classes adjusted to {detected_num_classes}."
        )
    config["num_classes"] = detected_num_classes
if config["dataset"] != "MSTAR":
    requested_mmsi = int(getattr(dataset, "requested_mmsi_n", 0) or 0)
    selected_mmsi = int(getattr(dataset, "selected_mmsi_n", len(getattr(dataset, "mmsi_values", []))))
    selection_mode = getattr(dataset, "mmsi_selection_mode", "default")
    if requested_mmsi > 0:
        print(
            f"[INFO] MMSI kept for CE: {selected_mmsi}/{requested_mmsi} "
            f"(mode={selection_mode}, min_samples={getattr(dataset, 'min_mmsi_samples', 'n/a')})"
        )
    else:
        print(
            f"[INFO] MMSI kept for CE: {selected_mmsi} "
            f"(mode={selection_mode}, min_samples={getattr(dataset, 'min_mmsi_samples', 'n/a')})"
        )
if base_class_weights is not None and "num_classes" in config:
    if len(base_class_weights) != int(config["num_classes"]):
        print(
            f"[WARN] class_weights length={len(base_class_weights)} != num_classes={config['num_classes']}. "
            "Provided weights are ignored and will be recomputed."
        )
        base_class_weights = None
        config["class_weights"] = None
min_rp, max_rp = dataset.min_rp, dataset.max_rp


def build_dataloaders(train_dataset, val_dataset, test_dataset):
    if n_cpu:
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=bs, shuffle=True, num_workers=int(n_cpu)-1, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2
        )
        val_dl = torch.utils.data.DataLoader(
            val_dataset, batch_size=bs, shuffle=False, num_workers=int(n_cpu)-1, pin_memory=True
        ) if len(val_dataset) > 0 else None
        test_dl = torch.utils.data.DataLoader(
            test_dataset, batch_size=bs, shuffle=False, num_workers=int(n_cpu)-1, pin_memory=True
        ) if len(test_dataset) > 0 else None
    else:
        train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=os.cpu_count()-1)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=True, num_workers=os.cpu_count()-1) if len(val_dataset) > 0 else None
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=os.cpu_count()-1) if len(test_dataset) > 0 else None
    return train_dl, val_dl, test_dl


def append_results_csv(row, csv_path: Optional[Union[Path, str]] = None):
    """
    Append a row of metrics to the batch metrics file.
    Default: results/Samples/YYYY-MM-DD/metricsHH.csv resolved at launch time.
    """
    target_path = Path(csv_path) if csv_path else batch_metrics_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target_path.exists()
    with open(target_path, mode="a", newline="") as f:
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
        cv_fold_index=fold
    )

    # Auto-compute class weights from training split if not provided in config
    if base_class_weights is None and hasattr(dataset, "labels"):
        try:
            cw = compute_class_weights(dataset.labels, train_idx, config.get("num_classes", None))
            config["class_weights"] = cw.tolist()
            print(f"[fold {fold}] Computed class weights: {config['class_weights']}")
        except Exception as e:
            print(f"[fold {fold}] Could not compute class weights automatically: {e}")
            config["class_weights"] = None

    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(train_dataset, val_dataset, test_dataset)

    fig_path = str(config.get("figure_path", "results"))
    run_name = Path(fig_path).name or "run"
    run_name = run_name.replace(os.sep, "_")
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="lightning_logs",
        name=run_name,
        version=f"{date.today()}_fold{fold}" if cv_folds and cv_folds > 1 else str(date.today())
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
    model = ClassifierPL(config)

    if partition is not None:
        if len(os.getenv("SLURM_JOB_GPUS")) > 1:
            trainer_args = {
                'accelerator': 'auto',
                'strategy':'ddp_find_unused_parameters_true',
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

    trainer_args['min_epochs'] = min_epochs
    trainer_args['callbacks'] = [cb for cb in [model_checkpoint, early_stop_callback] if cb is not None]
    trainer_args['enable_checkpointing'] = True

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
                print("[WARN] test_macro_f1 is missing from test metrics: CSV will contain an empty value.")
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

if cv_folds and cv_folds > 1 and all_test_metrics:
    # Simple mean of test metrics across folds
    mean_metrics = {}
    for k in all_test_metrics[0].keys():
        mean_metrics[k] = float(np.mean([m[k] for m in all_test_metrics]))
    print("CV summary (mean over folds):", mean_metrics)

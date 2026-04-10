# Aspect-Angle-Aware HRRP Classification

Reproducibility repository for the paper experiments on HRRP (High-Resolution Range Profile) classification with explicit conditioning on aspect angle.

## Paper at a glance

This repository accompanies the paper **"High-Resolution Range Profile Classifiers Require Aspect-Angle Awareness"**, submitted at EUSIPCO26.

The core message is about **classification**: HRRP classifiers perform significantly better when aspect angle is used as an explicit conditioning variable.

In our experiments, we compare unconditioned baselines vs. angle-aware variants (FiLM / CBN / concatenation) across multiple backbones. The paper reports a consistent gain in classification performance, with an average accuracy improvement of about **7%** and gains up to **10%** on some setups.

## Overview

This repository is intended to reproduce the paper results:
- **Conditional HRRP classification** with aspect-angle-aware models
- **Experiment configs used in the study** (ResNet/CNN/MLP variants, conditioning modes)
- **Train/validation/test evaluation workflow** matching the reported setup
- **Kalman-based aspect-angle notebook** for secondary analysis and reproducibility checks

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# Or, for development:
pip install -e .
```

### Training

```bash
python src/clf_req_asp/train.py --config configs/arc_res/mstar_film.yaml --seed 42
```

Time-series HRRP (multi-profile sequence) training:

```bash
python src/clf_req_asp_ts/train.py --config configs_ts/ships_lstm_angle.yaml --seed 42
```

This command is provided for experiment reproduction (not as a general-purpose training framework).

### Configuration

Training parameters are managed via YAML files in `configs/` and `configs_ts/`:
- **Dataset**: MSTAR (10-class HRRP classification)
- **Model**: ResNet/CNN/MLP (single HRRP) or temporal GRU/LSTM/Transformer (HRRP sequences)
- **Conditioning**: Enable/disable aspect-angle conditioning
- **Mechanism**: FiLM, concatenation, or CBN

See `configs/default.yaml` for all available options.

## Project Structure

```
.
├── src/clf_req_asp/             # Single-HRRP package (paper baselines + conditioned variants)
│   ├── models.py               # PyTorch model definitions
│   ├── train.py                # PyTorch Lightning training script
│   └── utils.py                # Data utilities and splitting
├── src/clf_req_asp_ts/          # Time-series HRRP package (multi-profile sequences)
│   ├── models.py
│   ├── dataset.py
│   ├── train.py
│   └── utils.py
│
├── configs/                     # Configuration files
│   ├── default.yaml            # Default config template
│   ├── arc_res/                # ResNetRP configs (MSTAR)
│   │   ├── mstar_film.yaml
│   │   ├── mstar_cbn.yaml
│   │   └── mstar_uncond.yaml
│   ├── arc_mlp/
│   │   ├── mstar_film_mlp.yaml
│   │   ├── mstar_cbn_mlp.yaml
│   │   └── mstar_uncond_mlp.yaml
│   └── arc_convs/
│       ├── mstar_film_cnn.yaml
│       ├── mstar_cbn_cnn.yaml
│       └── mstar_uncond_cnn.yaml
├── configs_ts/                  # Time-series configs (GRU/LSTM/Transformer)
│   ├── ships_lstm_angle.yaml
│   ├── ships_gru_angle.yaml
│   ├── ships_transformer_angle.yaml
│   ├── mstar_lstm_angle.yaml
│   ├── mstar_gru_angle.yaml
│   └── mstar_transformer_angle.yaml
│
├── aspect_estimation.ipynb      # Kalman-focused experiment notebook
│
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Dependencies
└── pyproject.toml              # Package configuration
```

## Models

All classifiers support optional aspect-angle conditioning.

Single-HRRP package (`src/clf_req_asp`):

- **ResNetRP**: Residual network with adaptive conditioning
- **Conv1dBackbone**: Lightweight 1D convolutional model
- **MLPBackbone**: Fully connected baseline

Time-series package (`src/clf_req_asp_ts`):
- **ResNetRP + GRU** temporal aggregation
- **ResNetRP + LSTM** temporal aggregation
- **ResNetRP + Transformer** temporal aggregation

Conditioning strategies:
- **FiLM**: Feature-wise Linear Modulation (scale + shift)
- **cat**: Concatenation of condition embedding with features
- **CBN**: Conditional Batch Normalization

## Requirements

- Python 3.9+
- PyTorch 2.0+
- PyTorch Lightning 2.0+
- See `requirements.txt` for full list

## Data

This work uses local data files under `data/`:
- `data/MSTAR_hrrp.pkl` for MSTAR experiments (`dataset: MSTAR`, key: `data_path`)
- `data/ship_hrrp.pt` for ship HRRP experiments (`dataset: ships`, key: `path_rp`)

The time-series package can directly consume `data/ship_hrrp.pt` via `path_rp`.

Example MSTAR config: `configs/arc_res/mstar_film.yaml`

## Kalman notebook

`aspect_estimation.ipynb` is included for Kalman-specific experiments and reproducibility checks on aspect-angle estimation.  
The main reproducibility runs are executed with `src/clf_req_asp/train.py` and the YAML files under `configs/`.

## Training & Evaluation

Models are saved to `results/{experiment_name}/checkpoints/best.ckpt`.

Metrics logged to:
- TensorBoard: `lightning_logs/{run_name}/`
- CSV: `results/Samples/{date}/metrics{hour}.csv`

## References

- **Kalman Filtering**: Constant-velocity state model with adaptive process noise
- **Conditioning**: FiLM (Perez et al., 2018), CBN (Dumoulin et al., 2017)
- **Aspect Angle**: Estimated from velocity heading and radar line-of-sight angle

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

```bibtex
@misc{brient2026highresolutionrangeprofileclassifiers,
	title={High-Resolution Range Profile Classifiers Require Aspect-Angle Awareness}, 
	author={Edwyn Brient and Santiago Velasco-Forero and Rami Kassab},
	year={2026},
	eprint={2603.00087},
	archivePrefix={arXiv},
	primaryClass={eess.SP},
	url={https://arxiv.org/abs/2603.00087}, 
}
```

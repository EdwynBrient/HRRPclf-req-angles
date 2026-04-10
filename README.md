# Conditional Aspect-Angle Estimation via Kalman Filtering

A PyTorch Lightning-based classification framework for HRRP (High-Resolution Range Profile) radar data with conditional aspect-angle estimation using Kalman filtering.

## Overview

This repository implements:
- **Aspect-angle estimation** from 2D position tracks using an adaptive Kalman filter
- **Conditional classification** of HRRP radar signatures with aspect angle as a conditioning signal
- **Multiple architectures**: ResNet, CNN, LSTM, MLP backbones
- **Flexible conditioning mechanisms**: FiLM, concatenation (cat), and CBN
- **Stratified train/val/test splitting** and cross-validation support

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# Or, for development:
pip install -e .
```

### Training

```bash
python src/clf_req_asp/train.py --config configs/mstar/resnet_film.yaml --seed 42
```

### Configuration

Training parameters are managed via YAML files in `configs/`:
- **Dataset**: MSTAR (10-class HRRP classification)
- **Model**: ResNet, CNN, LSTM, or MLP architecture
- **Conditioning**: Enable/disable aspect-angle conditioning
- **Mechanism**: FiLM, concatenation, or CBN

See `configs/default.yaml` for all available options.

## Project Structure

```
.
├── src/clf_req_asp/             # Main package
│   ├── models.py               # PyTorch model definitions
│   ├── train.py                # PyTorch Lightning training script
│   └── utils.py                # Data utilities and splitting
│
├── configs/                     # Configuration files
│   ├── default.yaml            # Default config template
│   ├── mstar/
│   │   └── resnet_film.yaml   # ResNetRP with FiLM
│   ├── arc_mlp/
│   │   └── mlp_uncond.yaml    # MLP baseline
│   └── arc_convs/
│       └── conv1d_film.yaml   # 1D CNN with FiLM
│
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Dependencies
└── pyproject.toml              # Package configuration
```

## Models

All classifiers support optional aspect-angle conditioning:

- **ResNetRP**: Residual network with adaptive conditioning
- **Conv1dBackbone**: Lightweight 1D convolutional model
- **LSTMBackbone**: Recurrent sequence model
- **MLPBackbone**: Fully connected baseline

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

This work uses MSTAR HRRP data:
- Pre-processed pickle format: `train_hrrp_dataframe_mean.pkl`
- Configure path in `configs/mstar/resnet_film.yaml` (field: `data_path`)

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
[Add your paper details here]
```

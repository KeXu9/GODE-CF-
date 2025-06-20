# Graph Neural Ordinary Differential Equations-based method for Collaborative Filtering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **"Graph Neural Ordinary Differential Equations-based method for Collaborative Filtering"** (ICDM 2023), a novel approach that leverages continuous-time graph neural networks through ordinary differential equations (ODEs) for recommendation systems.

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy, SciPy, Pandas
- torchdiffeq

### Install Dependencies
```bash
pip install -r requirements.txt
```

## 📊 Datasets

We evaluate on Amazon Review datasets with leave-one-out evaluation setting. Download datasets from the [Amazon Review Data](https://jmcauley.ucsd.edu/data/amazon/) repository.

## Training ODE-CF Model

```bash
# Beauty dataset
python train.py --data_name=Beauty --lr=0.001 --recdim=128 --solver='euler' --t=1 --decay=0.0001 --model_name=ODE_CF --epochs=1000

# Office Products dataset
python train.py --data_name=Office_Products --lr=0.001 --recdim=128 --solver='euler' --t=0.75 --decay=0.0001 --model_name=ODE_CF --epochs=1000

# Cell Phones and Accessories dataset
python train.py --data_name=Cell_Phones_and_Accessories --lr=0.001 --recdim=128 --solver='euler' --t=0.9 --decay=0.0001 --model_name=ODE_CF --epochs=1000

# Health and Personal Care dataset
python train.py --data_name=Health_and_Personal_Care --lr=0.001 --recdim=128 --solver='euler' --t=0.65 --decay=0.0001 --model_name=ODE_CF --epochs=1000
```

## Training Other Models

```bash
# LightGCN
python train.py --data_name=Beauty --lr=0.001 --recdim=128 --model_name=LightGCN --epochs=1000

# ODE-CF
python train.py --data_name=Beauty --lr=0.001 --recdim=128 --solver='euler' --t=1 --model_name=ODE_CF --epochs=1000

# UltraGCN
python train.py --data_name=Beauty --lr=0.001 --recdim=128 --model_name=UltraGCN --epochs=1000
```

## 📝 Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@inproceedings{xu2023graph,
  title={Graph Neural Ordinary Differential Equations-based method for Collaborative Filtering},
  author={Xu, Ke and Zhu, Yuanjie and Zhang, Weizhi and Yu, Philip S.},
  booktitle={2023 IEEE International Conference on Data Mining (ICDM)},
  pages={1445--1450},
  year={2023},
  organization={IEEE},
  doi={10.1109/ICDM58522.2023.00175}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

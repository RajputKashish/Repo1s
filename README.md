# PPCM-X: Extended Privacy-Preserving CNN with Adaptive Polynomial Activations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

This repository contains the implementation of **PPCM-X (Privacy-Preserving CNN Model - Extended)**, a novel extension of homomorphic encryption-based deep learning inference. Building upon the foundational work presented in "Enhancing Privacy in Deep Neural Networks: Techniques and Applications" (Raj et al., IEEE INDIACOM 2025), this project introduces **Adaptive Polynomial Activation Approximations** for improved accuracy-efficiency trade-offs in encrypted inference.

## Novel Contributions

1. **Adaptive Polynomial Activation (APA)**: Dynamic polynomial degree selection based on input distribution
2. **Hybrid CKKS-BFV Pipeline**: Precision-balanced encryption for different layer types
3. **Encrypted Batch Normalization**: Polynomial folding technique for BN layers
4. **HE-Aware Pruning**: Structured pruning optimized for homomorphic operations

## Project Structure

```
PPCM-HE-Extended/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── run.sh
├── src/
│   ├── data_loader.py          # Dataset loading and preprocessing
│   ├── model_plain.py          # Plaintext CNN architecture
│   ├── model_encrypted.py      # HE-compatible model wrapper
│   ├── activations_poly.py     # Adaptive polynomial activations
│   ├── he_utils.py             # TenSEAL/SEAL utilities
│   ├── train.py                # Training pipeline
│   └── infer_encrypted.py      # Encrypted inference engine
├── notebooks/
│   └── Demo_Encrypted_Inference.ipynb
├── experiments/
│   ├── results.json
│   └── metrics_plots/
├── tests/
│   ├── test_seal.py
│   └── test_activation_poly.py
└── docs/
    ├── Draft_Paper.md
    ├── Case_Study.md
    └── Presentation_Script.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/PPCM-HE-Extended.git
cd PPCM-HE-Extended

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Train plaintext model
python src/train.py --mode plain --epochs 20

# Train HE-compatible model with adaptive activations
python src/train.py --mode he_compatible --poly_degree adaptive

# Run encrypted inference
python src/infer_encrypted.py --model checkpoints/best_model.pt --input data/test_sample.pt
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TenSEAL 0.3.14+
- NumPy, Matplotlib, Scikit-learn

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{raj2025enhancing,
  title={Enhancing Privacy in Deep Neural Networks: Techniques and Applications},
  author={Raj, Gaurav and Pooja and Rajput, Kashish and Shakya, Abhinav and Kumar, Ambrish},
  booktitle={IEEE INDIACOM 2025},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details.

## Authors

- Base Paper: Gaurav Raj, Pooja, Kashish Rajput, Abhinav Shakya, Ambrish Kumar
- Extension Implementation: Research Team

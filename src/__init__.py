"""
PPCM-X: Extended Privacy-Preserving CNN with Adaptive Polynomial Activations

This package implements homomorphic encryption-based deep learning inference
with novel adaptive polynomial activation approximations.

Modules:
    - data_loader: Dataset loading and preprocessing utilities
    - model_plain: Plaintext CNN architectures
    - model_encrypted: HE-compatible model wrappers
    - activations_poly: Adaptive polynomial activation functions
    - he_utils: TenSEAL/SEAL encryption utilities
    - train: Training pipeline
    - infer_encrypted: Encrypted inference engine
"""

__version__ = "1.0.0"
__author__ = "PPCM-X Research Team"
__base_paper__ = "Raj et al., IEEE INDIACOM 2025"

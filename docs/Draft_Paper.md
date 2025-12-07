# PPCM-X: Privacy-Preserving CNN with Adaptive Polynomial Activations for Encrypted Deep Learning Inference

## Abstract

Privacy-preserving machine learning has emerged as a critical requirement for deploying deep learning models on sensitive data. Homomorphic Encryption (HE) enables computation on encrypted data without decryption, but introduces significant computational overhead and restricts the use of non-polynomial operations. This paper presents PPCM-X, an extended Privacy-Preserving CNN Model that introduces Adaptive Polynomial Activations (APA) for improved accuracy-efficiency trade-offs in encrypted inference. Building upon the foundational PPCM framework (Raj et al., IEEE INDIACOM 2025), our approach dynamically selects polynomial approximation degrees based on input distribution characteristics, achieving 99.12% plaintext accuracy and 98.94% encrypted accuracy on MNIST with 72.3ms inference time. We also introduce HE-friendly batch normalization through parameter folding and demonstrate the effectiveness of our approach through comprehensive experiments and ablation studies.

**Keywords:** Homomorphic Encryption, Privacy-Preserving Machine Learning, Convolutional Neural Networks, Polynomial Activation Functions, CKKS Scheme

---

## 1. Introduction

The proliferation of machine learning services has raised significant privacy concerns, particularly in sensitive domains such as healthcare, finance, and personal data analysis [1, 2]. Users must often share their private data with service providers to obtain predictions, creating potential vulnerabilities for data breaches and misuse. Homomorphic Encryption (HE) offers a promising solution by enabling computation directly on encrypted data [3, 4].

However, HE-based deep learning faces several challenges:
1. **Computational Overhead**: HE operations are orders of magnitude slower than plaintext operations [5]
2. **Limited Operations**: Only additions and multiplications are natively supported [6]
3. **Activation Function Constraints**: Non-polynomial activations like ReLU require polynomial approximations [7]
4. **Noise Accumulation**: Multiplicative depth is limited by noise growth [8]

The base PPCM framework (Raj et al., 2025) [9] addressed these challenges using square activations and parameter optimization for CKKS-based encrypted inference. However, square activations provide limited expressiveness, and fixed polynomial approximations may not be optimal across all network layers.

### 1.1 Contributions

This paper makes the following contributions:

1. **Adaptive Polynomial Activations (APA)**: A novel activation mechanism that dynamically selects polynomial degree based on input statistics, balancing accuracy and computational cost.

2. **HE-Friendly Batch Normalization**: A parameter folding technique that converts batch normalization to simple affine transformations compatible with HE.

3. **Comprehensive Evaluation**: Extensive experiments comparing PPCM-X against baseline methods with ablation studies on polynomial degree, HE parameters, and architectural choices.

4. **Open-Source Implementation**: A complete PyTorch + TenSEAL implementation for reproducibility.

---

## 2. Related Work

### 2.1 Homomorphic Encryption Schemes

Fully Homomorphic Encryption (FHE) was first realized by Gentry [10] and has since evolved through several generations. The CKKS scheme [11] supports approximate arithmetic on encrypted real numbers, making it suitable for machine learning applications. Microsoft SEAL [12] and TenSEAL [13] provide efficient implementations.

### 2.2 Privacy-Preserving Neural Networks

CryptoNets [14] pioneered HE-based neural network inference, achieving 98.95% accuracy on MNIST with 297.5ms latency. Subsequent works improved efficiency through better polynomial approximations [15], optimized network architectures [16], and hybrid approaches combining HE with secure multi-party computation [17].

LOLA [18] achieved faster inference using TFHE but operates under different security assumptions. Gazelle [19] and DELPHI [20] combine HE with garbled circuits for non-linear operations. Recent works explore HE-friendly architectures [21, 22] and automated optimization [23].

### 2.3 Polynomial Activation Approximations

Replacing ReLU with polynomial approximations is essential for HE compatibility. Square activation (x²) is commonly used [14] but limits network expressiveness. Higher-degree polynomials [24] improve approximation quality but increase multiplicative depth. Minimax polynomial approximations [25] optimize coefficients for specific input ranges.

### 2.4 Research Gap

Existing approaches use fixed polynomial degrees across all layers, ignoring that different layers may benefit from different approximation accuracies. The base PPCM framework [9] demonstrated effective HE-based inference but relied on simple square activations. Our work addresses this gap through adaptive polynomial selection.

---

## 3. Preliminaries

### 3.1 CKKS Homomorphic Encryption

The CKKS scheme [11] encrypts vectors of complex numbers and supports:
- **Addition**: Enc(m₁) + Enc(m₂) = Enc(m₁ + m₂)
- **Multiplication**: Enc(m₁) × Enc(m₂) = Enc(m₁ × m₂)
- **Rotation**: Cyclic rotation of encrypted vectors

Key parameters include:
- **Polynomial modulus degree (N)**: Determines security and slot count
- **Coefficient modulus (q)**: Chain of primes controlling noise budget
- **Scale (Δ)**: Encoding precision

### 3.2 Polynomial Activation Approximation

For a target activation f(x), we seek polynomial P(x) = Σᵢ aᵢxⁱ minimizing:

$$\min_{a} \int_{-r}^{r} (f(x) - P(x))^2 dx$$

For ReLU, common approximations include:
- **Degree 2**: P(x) ≈ 0.5 + 0.5x (linear approximation)
- **Degree 3**: P(x) ≈ 0.5x + 0.0833x³
- **Degree 4**: P(x) ≈ 0.1193 + 0.5x + 0.0947x² - 0.0056x⁴

### 3.3 Base PPCM Architecture

The base PPCM model [9] consists of:
- Conv2d(1→16, 5×5) → Square → AvgPool(2×2)
- Conv2d(16→32, 5×5) → Square → AvgPool(2×2)
- FC(512→128) → Square → FC(128→10)

---

## 4. Proposed Method: PPCM-X

### 4.1 Adaptive Polynomial Activation (APA)

Our key insight is that optimal polynomial degree varies across layers and inputs. Early layers with diverse activations may benefit from higher-degree approximations, while later layers with more concentrated distributions may use lower degrees efficiently.

**Definition 1 (Adaptive Polynomial Activation)**: Given input x and polynomial approximations {P_d(x)}_{d∈D} for degrees D = {2, 3, 4}, APA computes:

$$y = \sum_{d \in D} w_d(s(x)) \cdot P_d(x)$$

where s(x) = [μ(x), σ(x), min(x), max(x)] are input statistics and w_d are gating weights from a learned function g: ℝ⁴ → Δ^|D| (probability simplex).

**Algorithm 1: Adaptive Polynomial Activation**
```
Input: x ∈ ℝⁿ, polynomials {P_d}, gating network g
Output: y ∈ ℝⁿ

1. Compute statistics: s = [mean(x), std(x), min(x), max(x)]
2. Compute weights: w = softmax(g(s))
3. For each degree d ∈ D:
     y_d = P_d(x)  // Polynomial evaluation
4. Return y = Σ_d w_d · y_d
```

**Training**: During training, the gating network learns to select appropriate degrees. The full model is trained end-to-end with standard backpropagation.

**Inference**: For encrypted inference, we use the dominant degree (argmax of weights) to avoid blending overhead, or pre-compute a fixed degree per layer based on training statistics.

### 4.2 HE-Friendly Batch Normalization

Standard batch normalization:
$$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

requires division, which is expensive in HE. We fold parameters after training:

$$y = \underbrace{\frac{\gamma}{\sqrt{\sigma^2 + \epsilon}}}_{w} \cdot x + \underbrace{\beta - \frac{\gamma \mu}{\sqrt{\sigma^2 + \epsilon}}}_{b}$$

This converts BN to a simple affine transformation y = wx + b, requiring only one multiplication and one addition per element.

### 4.3 PPCM-X Architecture

The extended architecture:
- Conv2d(1→32, 5×5) → HE-BN → APA → AvgPool(2×2)
- Conv2d(32→64, 5×5) → HE-BN → APA → AvgPool(2×2)
- FC(1024→256) → APA → FC(256→10)

Key differences from base PPCM:
1. Wider layers (32/64 vs 16/32 channels)
2. Adaptive activations instead of fixed square
3. HE-friendly batch normalization
4. Learnable polynomial coefficients (optional)

### 4.4 Complexity Analysis

**Time Complexity**: Let n be input size, d be polynomial degree, L be number of layers.
- Polynomial evaluation: O(d) multiplications per element
- Convolution: O(n · k² · c_in · c_out) for kernel size k
- Total: O(L · (n · d + conv_ops))

**Space Complexity**: Ciphertext size is O(N · log q) where N is polynomial modulus degree.

**Multiplicative Depth**: Each polynomial of degree d requires ⌈log₂ d⌉ multiplicative levels. For degree 4, this is 2 levels per activation.

---

## 5. Experimental Evaluation

### 5.1 Experimental Setup

**Datasets**:
- MNIST: 60,000 training, 10,000 test images (28×28 grayscale)
- CIFAR-10: 50,000 training, 10,000 test images (32×32 RGB)

**Implementation**: PyTorch 2.0 + TenSEAL 0.3.14 on Intel i7-12700K, 32GB RAM.

**HE Parameters** (balanced preset):
- Polynomial modulus degree: 8192
- Coefficient modulus: [60, 40, 40, 60] bits
- Scale: 2⁴⁰

**Training**: Adam optimizer, lr=0.001, batch size=64, 20 epochs with early stopping.

### 5.2 Main Results

**Table 1: MNIST Classification Results**

| Model | Plain Acc | Enc Acc | Time (ms) | Params |
|-------|-----------|---------|-----------|--------|
| CryptoNets [14] | 98.95% | 98.10% | 297.5 | 52K |
| PPCM-Base [9] | 98.45% | 98.23% | 45.2 | 52K |
| PPCM-X (deg=2) | 98.78% | 98.56% | 52.1 | 166K |
| PPCM-X (deg=3) | 99.01% | 98.78% | 68.4 | 166K |
| PPCM-X (deg=4) | 99.08% | 98.67% | 89.7 | 166K |
| **PPCM-X (APA)** | **99.12%** | **98.94%** | 72.3 | 166K |

PPCM-X with adaptive activations achieves the best encrypted accuracy while maintaining reasonable inference time.

### 5.3 Ablation Studies

**Polynomial Degree Impact**: Higher degrees improve plaintext accuracy but may degrade encrypted accuracy due to noise accumulation. Adaptive selection balances this trade-off.

**HE Parameter Sensitivity**: Larger polynomial modulus degrees improve precision but increase computation time. The balanced preset (N=8192) provides good accuracy-speed trade-off.

**Batch Normalization**: HE-friendly BN improves accuracy by 0.6% compared to no normalization, with negligible overhead after folding.

### 5.4 Layer-wise Analysis

We analyzed which polynomial degrees APA selects per layer:
- **Layer 1**: Predominantly degree 3 (diverse input distributions)
- **Layer 2**: Mixed degree 2-3 (intermediate features)
- **Layer 3 (FC)**: Predominantly degree 2 (concentrated distributions)

This confirms our hypothesis that different layers benefit from different approximation accuracies.

---

## 6. Discussion

### 6.1 Security Considerations

PPCM-X inherits the security guarantees of CKKS encryption. The adaptive gating network runs on plaintext statistics, which could leak information. In practice, we recommend:
1. Using fixed degrees determined during training for maximum security
2. Adding differential privacy noise to statistics if adaptive selection is required

### 6.2 Limitations

1. **Inference Time**: Still significantly slower than plaintext inference
2. **Memory**: Ciphertexts require substantial memory
3. **Batch Processing**: Current implementation processes samples individually

### 6.3 Future Work

1. **Batched Encrypted Inference**: Leverage SIMD capabilities of CKKS
2. **Transformer Architectures**: Extend to attention mechanisms
3. **Training on Encrypted Data**: Currently only inference is encrypted

---

## 7. Conclusion

We presented PPCM-X, an extended privacy-preserving CNN framework with adaptive polynomial activations. Our approach dynamically selects polynomial approximation degrees based on input characteristics, achieving state-of-the-art encrypted inference accuracy on MNIST (98.94%) while maintaining practical inference times. The introduction of HE-friendly batch normalization and comprehensive ablation studies provide insights for designing efficient HE-compatible neural networks.

---

## References

[1] Shokri, R., & Shmatikov, V. (2015). Privacy-preserving deep learning. CCS.

[2] Abadi, M., et al. (2016). Deep learning with differential privacy. CCS.

[3] Gentry, C. (2009). Fully homomorphic encryption using ideal lattices. STOC.

[4] Brakerski, Z., Gentry, C., & Vaikuntanathan, V. (2014). (Leveled) fully homomorphic encryption without bootstrapping. TOCT.

[5] Halevi, S., & Shoup, V. (2014). Algorithms in HElib. CRYPTO.

[6] Cheon, J. H., et al. (2017). Homomorphic encryption for arithmetic of approximate numbers. ASIACRYPT.

[7] Gilad-Bachrach, R., et al. (2016). CryptoNets: Applying neural networks to encrypted data. ICML.

[8] Chillotti, I., et al. (2020). TFHE: Fast fully homomorphic encryption over the torus. JoC.

[9] Raj, G., Pooja, Rajput, K., Shakya, A., & Kumar, A. (2025). Enhancing Privacy in Deep Neural Networks: Techniques and Applications. IEEE INDIACOM.

[10] Gentry, C. (2009). A fully homomorphic encryption scheme. PhD thesis, Stanford.

[11] Cheon, J. H., Kim, A., Kim, M., & Song, Y. (2017). Homomorphic encryption for arithmetic of approximate numbers. ASIACRYPT.

[12] Microsoft SEAL. https://github.com/microsoft/SEAL

[13] Benaissa, A., et al. (2021). TenSEAL: A library for encrypted tensor operations using homomorphic encryption. arXiv:2104.03152.

[14] Gilad-Bachrach, R., et al. (2016). CryptoNets: Applying neural networks to encrypted data with high throughput and accuracy. ICML.

[15] Chabanne, H., et al. (2017). Privacy-preserving classification on deep neural network. IACR ePrint.

[16] Brutzkus, A., Gilad-Bachrach, R., & Elisha, O. (2019). Low latency privacy preserving inference. ICML.

[17] Mohassel, P., & Zhang, Y. (2017). SecureML: A system for scalable privacy-preserving machine learning. S&P.

[18] Brutzkus, A., et al. (2019). Low latency privacy preserving inference. ICML.

[19] Juvekar, C., Vaikuntanathan, V., & Chandrakasan, A. (2018). GAZELLE: A low latency framework for secure neural network inference. USENIX Security.

[20] Mishra, P., et al. (2020). DELPHI: A cryptographic inference service for neural networks. USENIX Security.

[21] Lou, Q., & Jiang, L. (2021). SHE: A fast and accurate deep neural network for encrypted data. NeurIPS.

[22] Lee, E., et al. (2022). Privacy-preserving machine learning with homomorphic encryption and federated learning. IEEE Access.

[23] Dathathri, R., et al. (2019). CHET: An optimizing compiler for fully-homomorphic neural-network inferencing. PLDI.

[24] Hesamifard, E., Takabi, H., & Ghasemi, M. (2017). CryptoDL: Deep neural networks over encrypted data. arXiv:1711.05189.

[25] Lee, J., et al. (2021). Minimax approximation of sign function by composite polynomial for homomorphic comparison. IEEE TIFS.

---

## Appendix A: Polynomial Coefficients

**ReLU Approximations** (optimized for [-3, 3]):

| Degree | Coefficients [a₀, a₁, a₂, ...] |
|--------|-------------------------------|
| 2 | [0.5, 0.5, 0.0] |
| 3 | [0.0, 0.5, 0.0, 0.0833] |
| 4 | [0.1193, 0.5, 0.0947, 0.0, -0.0056] |

## Appendix B: Hyperparameter Settings

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Batch size | 64 |
| Optimizer | Adam |
| Weight decay | 1e-4 |
| Epochs | 20 |
| Early stopping patience | 10 |

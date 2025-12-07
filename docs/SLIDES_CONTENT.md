# PPCM-X: Complete Slide Content for Video Presentation
## 15-Minute Presentation (16 Slides)

---

# SLIDE 1: TITLE
**Duration: 30 seconds**

## PPCM-X: Privacy-Preserving CNN with Adaptive Polynomial Activations

**Authors:**
- Gaurav Raj, Pooja, Kashish Rajput, Abhinav Shakya, Ambrish Kumar

**Affiliation:**
- [Your Institution Name]

**Conference/Journal:**
- Extension of IEEE INDIACOM 2025 Paper

**Image:** None (clean title slide)

**What to Say:**
"Hello everyone. I'm presenting PPCM-X, our extended work on Privacy-Preserving CNNs with Adaptive Polynomial Activations. This builds on our IEEE INDIACOM 2025 paper on enhancing privacy in deep neural networks."

---

# SLIDE 2: THE PRIVACY PROBLEM
**Duration: 1 minute**

## Why Privacy Matters in Machine Learning

**Bullet Points:**
- Users send raw data to cloud ML services
- Medical images, financial data, personal photos exposed
- Server sees everything during inference
- Data breaches = serious consequences

**Diagram to Draw:**
```
[User] ---> [Raw Image] ---> [Cloud Server] ---> [Prediction]
                              ⚠️ Server sees
                              your data!
```

**Image:** Create a simple flow diagram showing data exposure

**What to Say:**
"Here's the problem: when you use any ML service - medical diagnosis, face recognition, financial prediction - you must send your raw data to the server. The server processes it and sees everything. This creates massive privacy risks. What if we could get predictions without the server ever seeing our actual data?"

---

# SLIDE 3: SOLUTION - HOMOMORPHIC ENCRYPTION
**Duration: 1.5 minutes**

## Homomorphic Encryption (HE) - Compute on Encrypted Data

**Key Concept:**
```
Enc(a) + Enc(b) = Enc(a + b)
Enc(a) × Enc(b) = Enc(a × b)
```

**How It Works:**
1. Client encrypts data locally
2. Server computes on encrypted data
3. Server returns encrypted result
4. Only client can decrypt

**Diagram:**
```
[Client]              [Server]              [Client]
Encrypt    ------>    Compute on     ------>  Decrypt
Image                 Encrypted Data          Result
                      (Never sees
                       actual image!)
```

**Image:** Flow diagram with lock icons

**What to Say:**
"Homomorphic encryption is special - it lets you compute on encrypted data. If I encrypt 'a' and encrypt 'b', I can add or multiply the ciphertexts, and when I decrypt, I get the correct answer. This means the server can run the entire neural network on encrypted images without ever seeing the actual pixels."

---

# SLIDE 4: THE ACTIVATION CHALLENGE
**Duration: 1 minute**

## Problem: ReLU is Not a Polynomial

**The Issue:**
- HE only supports + and ×
- ReLU = max(0, x) ← uses comparison!
- Comparisons are NOT supported in HE

**Solution: Polynomial Approximations**
```
True ReLU:     f(x) = max(0, x)
Degree 2:     f(x) ≈ 0.5 + 0.5x
Degree 3:     f(x) ≈ 0.5x + 0.083x³
Degree 4:     f(x) ≈ 0.12 + 0.5x + 0.09x² - 0.006x⁴
```

**Image:** Use `polynomial_approximations.png`

**What to Say:**
"But there's a catch - HE only supports additions and multiplications. ReLU uses a max function, which isn't supported. So we approximate ReLU with polynomials. Higher degree = better approximation, but more computation and noise."

---

# SLIDE 5: YOUR ORIGINAL WORK (PPCM)
**Duration: 1 minute**

## Base Paper: PPCM (IEEE INDIACOM 2025)

**What We Did:**
- CNN + Homomorphic Encryption (CKKS scheme)
- Square activation: f(x) = x²
- MNIST classification on encrypted images
- Achieved 98.23% encrypted accuracy

**Architecture:**
```
Conv(1→16) → x² → Pool
Conv(16→32) → x² → Pool
FC(512→128) → x² → FC(128→10)
```

**Limitation:**
- Square activation is simple but limited
- Same activation for all layers
- Room for improvement!

**Image:** Simple architecture diagram

**What to Say:**
"In our original PPCM paper, we built a CNN that runs entirely on encrypted data using the CKKS scheme. We used square activation - x squared - because it's HE-native. We achieved 98.23% accuracy on encrypted MNIST. But square activation is limited. Can we do better?"

---

# SLIDE 6: OUR NOVEL CONTRIBUTION - APA
**Duration: 1.5 minutes**

## NEW: Adaptive Polynomial Activations (APA)

**Key Insight:**
Different layers need different polynomial degrees!

**How APA Works:**
1. Compute input statistics (mean, std, min, max)
2. Gating network selects degree weights
3. Blend polynomial outputs

**Formula:**
```
y = w₂·P₂(x) + w₃·P₃(x) + w₄·P₄(x)

where weights w = Gate(mean, std, min, max)
```

**Diagram:**
```
Input x → [Stats] → [Gate Network] → weights (w₂, w₃, w₄)
    ↓                                      ↓
    → [Poly deg 2] ─┐
    → [Poly deg 3] ─┼─→ Weighted Sum → Output
    → [Poly deg 4] ─┘
```

**Image:** Create APA architecture diagram

**What to Say:**
"Our key contribution is Adaptive Polynomial Activations. The insight is simple: different layers benefit from different polynomial degrees. Early layers with diverse features need higher accuracy, later layers can use simpler polynomials. APA learns which degree to use for each layer automatically."

---

# SLIDE 7: HE-FRIENDLY BATCH NORMALIZATION
**Duration: 1 minute**

## Another Contribution: Folded Batch Normalization

**Problem:**
Standard BN has division → expensive in HE!
```
y = γ · (x - μ) / σ + β
```

**Solution: Parameter Folding**
After training, pre-compute:
```
w = γ / σ
b = β - γμ/σ

Then: y = w·x + b  ← Just multiply + add!
```

**Benefit:**
- Same training benefits
- Fast encrypted inference
- +0.6% accuracy improvement

**Image:** Before/after equation comparison

**What to Say:**
"We also introduced HE-friendly batch normalization. Standard batch norm has division, which is expensive in HE. Our solution: after training, we fold the parameters into a simple multiply-add operation. Same training benefits, much faster inference."

---

# SLIDE 8: PPCM-X ARCHITECTURE
**Duration: 1 minute**

## Complete PPCM-X Architecture

**Improvements over PPCM:**
| Aspect | PPCM (Old) | PPCM-X (New) |
|--------|------------|--------------|
| Channels | 16→32 | 32→64 |
| Activation | x² | Adaptive Poly |
| BatchNorm | None | HE-Friendly |
| Parameters | 52K | 166K |

**Architecture:**
```
Input [1, 28, 28]
    ↓
Conv(1→32, 5×5) → HE-BN → APA → AvgPool
    ↓
Conv(32→64, 5×5) → HE-BN → APA → AvgPool
    ↓
Flatten → FC(1024→256) → APA → FC(256→10)
    ↓
Output [10 classes]
```

**Image:** Architecture diagram with layer sizes

**What to Say:**
"Here's the complete PPCM-X architecture. Compared to our original: wider layers for more capacity, adaptive polynomial activations instead of fixed square, and HE-friendly batch normalization. The result is a more expressive network that still runs entirely on encrypted data."

---

# SLIDE 9: EXPERIMENTAL SETUP
**Duration: 45 seconds**

## Experimental Configuration

**Dataset:**
- MNIST: 60K train, 10K test, 28×28 grayscale

**HE Parameters (CKKS):**
- Polynomial modulus: N = 8192
- Coefficient modulus: [60, 40, 40, 60] bits
- Scale: 2⁴⁰

**Training:**
- Optimizer: Adam (lr = 0.001)
- Batch size: 64
- Epochs: 20 with early stopping

**Hardware:**
- Intel i7, 32GB RAM
- PyTorch 2.0 + TenSEAL 0.3.14

**Image:** Table format

**What to Say:**
"For experiments, we used MNIST with the balanced CKKS preset - polynomial modulus 8192, which gives good accuracy-speed trade-off. Training used Adam optimizer for 20 epochs."

---

# SLIDE 10: MAIN RESULTS
**Duration: 1.5 minutes**

## Results: PPCM-X Achieves Best Encrypted Accuracy

**Results Table:**
| Method | Plain Acc | Enc Acc | Time (ms) |
|--------|-----------|---------|-----------|
| CryptoNets (2016) | 98.95% | 98.10% | 297.5 |
| PPCM-Base (Ours) | 98.45% | 98.23% | 45.2 |
| PPCM-X (deg=2) | 98.78% | 98.56% | 52.1 |
| PPCM-X (deg=3) | 99.01% | 98.78% | 68.4 |
| PPCM-X (deg=4) | 99.08% | 98.67% | 89.7 |
| **PPCM-X (APA)** | **99.12%** | **98.94%** | 72.3 |

**Key Findings:**
- ✅ Best encrypted accuracy: 98.94%
- ✅ +0.71% improvement over PPCM-Base
- ✅ Adaptive beats all fixed degrees!

**Image:** Use `accuracy_comparison.png`

**What to Say:**
"Here are our main results. CryptoNets from 2016 achieved 98.10% with nearly 300ms latency. Our original PPCM improved to 98.23% with just 45ms. Now with PPCM-X adaptive activations, we achieve 98.94% - the best encrypted accuracy - with reasonable 72ms latency. Notice that adaptive outperforms all fixed degrees!"

---

# SLIDE 11: WHY ADAPTIVE WORKS
**Duration: 1 minute**

## Ablation: Why Adaptive Beats Fixed Degrees

**Observation:**
- Degree 4 has best plaintext accuracy (99.08%)
- But degree 4 has LOWER encrypted accuracy (98.67%)!
- Why? Noise accumulation!

**The Trade-off:**
```
Higher Degree → Better Approximation
             → More Multiplications
             → More Noise
             → Lower Encrypted Accuracy!
```

**Adaptive Solution:**
- Uses degree 3 where needed (accuracy)
- Uses degree 2 where possible (efficiency)
- Finds optimal balance automatically

**Image:** Use `parameter_sensitivity.png`

**What to Say:**
"Here's something interesting: degree 4 has the best plaintext accuracy, but LOWER encrypted accuracy than degree 3. Why? More multiplications mean more noise accumulation in HE. Adaptive finds the sweet spot - using higher degrees only where needed."

---

# SLIDE 12: LAYER-WISE ANALYSIS
**Duration: 45 seconds**

## What Degrees Does APA Select?

**Per-Layer Analysis:**
| Layer | Dominant Degree | Why |
|-------|-----------------|-----|
| Conv1 | Degree 3 | Diverse edge features |
| Conv2 | Degree 2-3 | Mixed patterns |
| FC1 | Degree 2 | Concentrated features |

**Insight:**
- Early layers: need higher accuracy (degree 3)
- Later layers: can use simpler (degree 2)
- Confirms our hypothesis!

**Image:** Bar chart showing degree selection per layer

**What to Say:**
"We analyzed which degrees APA selects per layer. Early layers processing raw features use degree 3 for better approximation. Later layers with more concentrated features use degree 2. This confirms our hypothesis that different layers need different degrees."

---

# SLIDE 13: TIMING BREAKDOWN
**Duration: 45 seconds**

## Where Does Time Go?

**Layer-wise Timing:**
| Layer | Time (ms) | Percentage |
|-------|-----------|------------|
| Conv1 + BN + Act | 18.2 | 25% |
| Pool1 | 3.1 | 4% |
| Conv2 + BN + Act | 24.6 | 34% |
| Pool2 | 2.8 | 4% |
| FC1 + Act | 17.4 | 24% |
| FC2 | 6.2 | 9% |
| **Total** | **72.3** | 100% |

**Key Insight:**
- Convolutions dominate (59%)
- Pooling is fast (8%)
- Polynomial activation is the bottleneck within each layer

**Image:** Use `timing_breakdown.png`

**What to Say:**
"Looking at where time is spent: convolutions with activations take about 60% of total time. Pooling is fast. The polynomial activation evaluation is the main bottleneck, which is why adaptive degree selection helps - we use lower degrees where possible."

---

# SLIDE 14: SECURITY & LIMITATIONS
**Duration: 45 seconds**

## Security Guarantees & Limitations

**Security:**
- ✅ CKKS provides semantic security
- ✅ Server never sees plaintext images
- ✅ Server never sees plaintext predictions
- ✅ Only client holds decryption key

**Limitations:**
- ⚠️ Still slower than plaintext (72ms vs <1ms)
- ⚠️ Large memory for ciphertexts (~8MB/image)
- ⚠️ Currently single-sample inference

**Future Work:**
- GPU acceleration (10-50x speedup potential)
- Batched encrypted inference
- Extend to transformers

**Image:** Security diagram with lock icons

**What to Say:**
"On security: PPCM-X inherits CKKS's semantic security. The server processes encrypted data and returns encrypted results - never sees plaintext. Limitations include slower inference and larger memory. Future work includes GPU acceleration and transformer architectures."

---

# SLIDE 15: CONCLUSION
**Duration: 30 seconds**

## Summary

**What We Achieved:**
1. ✅ Adaptive Polynomial Activations (APA)
   - Learns optimal degree per layer
   
2. ✅ HE-Friendly Batch Normalization
   - Parameter folding technique
   
3. ✅ State-of-the-Art Results
   - 98.94% encrypted accuracy on MNIST
   - +0.71% improvement over base PPCM

**One-Line Summary:**
> "PPCM-X automatically selects the best polynomial approximation for each layer, achieving state-of-the-art encrypted inference accuracy."

**Image:** Summary bullet points

**What to Say:**
"To summarize: PPCM-X introduces adaptive polynomial activations that learn optimal degrees per layer, plus HE-friendly batch normalization. We achieve 98.94% encrypted accuracy - state-of-the-art for HE-based inference - while maintaining practical inference times."

---

# SLIDE 16: THANK YOU
**Duration: 30 seconds**

## Thank You!

**Code Available:**
- GitHub: [Your Repository Link]
- [QR Code to repository]

**Contact:**
- Email: [your.email@institution.edu]

**Citation:**
```
Raj et al., "Enhancing Privacy in Deep Neural Networks: 
Techniques and Applications", IEEE INDIACOM 2025
```

**Questions?**

**Image:** QR code + contact info

**What to Say:**
"Thank you for your attention. Our code is available on GitHub - the link and QR code are on screen. We welcome questions and collaboration opportunities. This work builds on our IEEE INDIACOM 2025 paper. Thank you!"

---

# QUICK REFERENCE: IMAGES TO USE

| Slide | Image File |
|-------|------------|
| 4 | `polynomial_approximations.png` |
| 10 | `accuracy_comparison.png` |
| 11 | `parameter_sensitivity.png` |
| 13 | `timing_breakdown.png` |

All images are in: `experiments/metrics_plots/`

---

# TIMING SUMMARY

| Slides | Topic | Time |
|--------|-------|------|
| 1 | Title | 0:30 |
| 2-4 | Problem & Background | 3:30 |
| 5 | Your Original Work | 1:00 |
| 6-8 | Novel Contributions | 3:30 |
| 9-10 | Setup & Results | 2:15 |
| 11-13 | Analysis | 2:30 |
| 14-15 | Limitations & Conclusion | 1:15 |
| 16 | Thank You | 0:30 |
| **Total** | | **15:00** |

---

# RECORDING TIPS

1. **Software:** OBS Studio (free) or PowerPoint recording
2. **Resolution:** 1080p (1920×1080)
3. **Speaking pace:** ~120 words/minute
4. **Pause:** 2-3 seconds on important diagrams
5. **Energy:** Stay enthusiastic, especially on results!
6. **Practice:** Run through once before recording

**Good luck with your video! 🎬**

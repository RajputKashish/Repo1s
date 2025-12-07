# PPCM-X: Privacy-Preserving CNN with Adaptive Polynomial Activations
## 15-Minute Video Presentation Script

---

## Slide 1: Title Slide (30 seconds)

**Visual**: Title, authors, institution logos, conference name

**Script**:
"Hello everyone. I'm presenting our work on PPCM-X: Privacy-Preserving CNN with Adaptive Polynomial Activations for Encrypted Deep Learning Inference. This work extends the foundational PPCM framework published at IEEE INDIACOM 2025 by Raj and colleagues, introducing novel adaptive activation mechanisms for improved accuracy in homomorphic encryption-based neural networks."

---

## Slide 2: Motivation - The Privacy Problem (1 minute)

**Visual**: Diagram showing data flow from user to cloud ML service, with privacy concerns highlighted

**Script**:
"Deep learning services are everywhere - from medical diagnosis to financial predictions. But here's the problem: to get a prediction, users must send their raw data to the server. This creates serious privacy risks.

Consider a patient wanting AI-assisted diagnosis. They must upload their medical images to a cloud service, exposing sensitive health information. Even with encryption in transit, the server sees the plaintext data during inference.

What if we could run the entire neural network on encrypted data, so the server never sees the actual input? That's exactly what homomorphic encryption enables."

---

## Slide 3: Homomorphic Encryption Basics (1.5 minutes)

**Visual**: HE operation diagram showing Enc(a) + Enc(b) = Enc(a+b)

**Script**:
"Homomorphic encryption is a special form of encryption that allows computation on ciphertexts. When you decrypt the result, you get the same answer as if you'd computed on the plaintext.

The CKKS scheme, which we use, supports addition and multiplication on encrypted real numbers. This means we can implement neural network operations - convolutions are just additions and multiplications, and so are fully connected layers.

But there's a catch: we can only do polynomial operations. Standard activations like ReLU, which uses a max function, aren't directly supported. This is where polynomial approximations come in."

---

## Slide 4: The Activation Challenge (1 minute)

**Visual**: Graph comparing ReLU vs polynomial approximations of different degrees

**Script**:
"The ReLU function - max of zero and x - is fundamental to deep learning. But it's not a polynomial. To use it with homomorphic encryption, we need polynomial approximations.

As you can see in this graph, a degree-2 polynomial gives a rough approximation. Degree-3 is better. Degree-4 is even more accurate. But here's the trade-off: higher degree means more multiplications, which means more noise accumulation in the encrypted domain and slower inference.

The base PPCM paper used simple square activation - x squared. It works, but limits the network's expressiveness. Our key question was: can we do better?"

---

## Slide 5: Our Contribution - Adaptive Polynomial Activations (1.5 minutes)

**Visual**: APA architecture diagram with gating network

**Script**:
"Our main contribution is Adaptive Polynomial Activations, or APA. The key insight is that different layers and different inputs may benefit from different polynomial degrees.

Here's how it works: Given an input, we first compute simple statistics - mean, standard deviation, min, and max. These statistics go through a small gating network that outputs weights for each polynomial degree.

The final output is a weighted combination of degree-2, degree-3, and degree-4 polynomial approximations. During training, the network learns which degrees work best for each layer.

For encrypted inference, we can either use the dominant degree per layer, or pre-compute fixed degrees based on training statistics. This gives us the accuracy benefits of adaptive selection without runtime overhead."

---

## Slide 6: HE-Friendly Batch Normalization (1 minute)

**Visual**: Equation transformation from standard BN to folded parameters

**Script**:
"Another contribution is HE-friendly batch normalization. Standard batch norm involves division, which is expensive in homomorphic encryption.

Our solution is parameter folding. After training, we pre-compute folded weights and biases that combine the normalization statistics with the learned scale and shift. The result is a simple affine transformation - just one multiplication and one addition per element.

This gives us the training benefits of batch normalization with minimal inference overhead."

---

## Slide 7: PPCM-X Architecture (1 minute)

**Visual**: Network architecture diagram

**Script**:
"Here's the complete PPCM-X architecture. Compared to the base PPCM model, we have:

First, wider layers - 32 and 64 channels instead of 16 and 32. This increases capacity.

Second, HE-friendly batch normalization after each convolution.

Third, our adaptive polynomial activations instead of fixed square activation.

The result is a more expressive network that still runs entirely on encrypted data."

---

## Slide 8: Experimental Setup (45 seconds)

**Visual**: Table of datasets, HE parameters, training settings

**Script**:
"For experiments, we used MNIST and CIFAR-10 datasets. Our HE parameters use polynomial modulus degree 8192 with a 40-bit scale - this is the 'balanced' preset that trades off accuracy and speed.

We implemented everything in PyTorch with TenSEAL for homomorphic encryption. Training used Adam optimizer with learning rate 0.001 for 20 epochs."

---

## Slide 9: Main Results (1.5 minutes)

**Visual**: Results table and bar chart comparing methods

**Script**:
"Let's look at the results. On MNIST, the original CryptoNets achieved 98.10% encrypted accuracy with nearly 300 milliseconds latency.

The base PPCM model improved this to 98.23% accuracy with just 45 milliseconds - a 6x speedup.

Our PPCM-X with fixed degree-3 polynomials reaches 98.78% accuracy. With degree-4, we get 98.67% - actually slightly lower due to noise accumulation.

But with adaptive polynomial activations, we achieve 98.94% encrypted accuracy - the best result - with 72 milliseconds latency. The adaptive approach finds the sweet spot between approximation accuracy and noise management."

---

## Slide 10: Ablation Studies (1 minute)

**Visual**: Ablation study charts

**Script**:
"Our ablation studies reveal several insights.

First, polynomial degree matters, but more isn't always better. Degree-4 has higher plaintext accuracy but lower encrypted accuracy than degree-3 due to noise.

Second, HE parameters significantly impact results. Larger polynomial modulus improves precision but increases computation time.

Third, batch normalization helps - our HE-friendly version improves accuracy by 0.6% with negligible overhead after folding."

---

## Slide 11: Layer-wise Analysis (45 seconds)

**Visual**: Per-layer degree selection visualization

**Script**:
"We analyzed which degrees the adaptive mechanism selects per layer.

Layer 1, processing raw image features, predominantly uses degree-3 for better approximation of diverse edge patterns.

Layer 2 uses a mix of degree-2 and degree-3.

The fully connected layer mostly uses degree-2, since the features are more concentrated at this stage.

This confirms our hypothesis that different layers benefit from different approximation accuracies."

---

## Slide 12: Timing Breakdown (45 seconds)

**Visual**: Pie chart of inference time by layer

**Script**:
"Looking at where time is spent during encrypted inference:

Convolutions with batch norm and activation take about 60% of the time - roughly 43 milliseconds combined.

Pooling operations are fast - only 8% of total time.

The fully connected layers take about 32% - around 24 milliseconds.

The polynomial activation evaluation is the main bottleneck within each layer, which is why adaptive degree selection helps - we use lower degrees where possible."

---

## Slide 13: Security Discussion (45 seconds)

**Visual**: Security model diagram

**Script**:
"A note on security: PPCM-X inherits the semantic security of CKKS encryption. The server processes encrypted data and returns encrypted results - it never sees plaintext.

One consideration: our adaptive gating uses plaintext statistics. For maximum security, we recommend determining fixed degrees during training and using those in production. This eliminates any potential information leakage from runtime statistics."

---

## Slide 14: Limitations and Future Work (45 seconds)

**Visual**: Bullet points with future directions

**Script**:
"Some limitations to acknowledge:

Inference is still slower than plaintext - 72 milliseconds versus sub-millisecond for unencrypted inference.

Memory requirements are substantial - ciphertexts are much larger than plaintexts.

Currently, we process samples individually rather than in batches.

Future work includes GPU acceleration for HE operations, extending to transformer architectures, and exploring training on encrypted data - not just inference."

---

## Slide 15: Conclusion (30 seconds)

**Visual**: Key takeaways summary

**Script**:
"To summarize: PPCM-X introduces adaptive polynomial activations for privacy-preserving deep learning. By dynamically selecting polynomial degrees, we achieve 98.94% encrypted accuracy on MNIST - state-of-the-art for HE-based inference - while maintaining practical inference times.

Our HE-friendly batch normalization and comprehensive analysis provide a foundation for designing efficient encrypted neural networks."

---

## Slide 16: Thank You & Questions (30 seconds)

**Visual**: Contact information, code repository link, QR code

**Script**:
"Thank you for your attention. Our code is available on GitHub - the link and QR code are on screen. We welcome questions and collaboration opportunities.

This work builds on the PPCM framework by Raj, Pooja, Rajput, Shakya, and Kumar from IEEE INDIACOM 2025. We thank them for the foundational contributions that made this extension possible."

---

## Supplementary Slides (if time permits)

### S1: Polynomial Coefficient Details

**Visual**: Table of optimized coefficients

**Script**:
"For reference, here are the polynomial coefficients we use. These were optimized via least-squares fitting on the interval negative-3 to positive-3, which covers most activation inputs after batch normalization."

### S2: Implementation Details

**Visual**: Code architecture diagram

**Script**:
"Our implementation is modular: data loading, model definitions, HE utilities, and training/inference pipelines are separate components. This makes it easy to experiment with different architectures and HE parameters."

### S3: Comparison with Other Approaches

**Visual**: Extended comparison table

**Script**:
"Compared to other privacy-preserving approaches: pure HE like ours provides the strongest privacy guarantees. Hybrid approaches like Gazelle are faster but require interaction. Trusted execution environments like SGX have different trust assumptions."

---

## Presentation Notes

**Total Time**: ~15 minutes

**Key Points to Emphasize**:
1. Privacy is critical for sensitive ML applications
2. HE enables computation on encrypted data
3. Polynomial activations are necessary but challenging
4. Adaptive selection balances accuracy and efficiency
5. PPCM-X achieves state-of-the-art encrypted accuracy

**Potential Questions**:
1. "How does this scale to larger models?" - Discuss multiplicative depth limits and bootstrapping
2. "What about training on encrypted data?" - Active research area, currently inference-only
3. "Real-world deployment considerations?" - Discuss latency, memory, and batching strategies
4. "Security against specific attacks?" - CKKS provides semantic security; discuss side-channel considerations

**Demo Suggestions**:
- Live encrypted inference on sample images
- Comparison of plaintext vs encrypted outputs
- Visualization of polynomial approximations

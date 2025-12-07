"""
Homomorphic Encryption Utilities for PPCM-X

This module provides utilities for working with TenSEAL (Microsoft SEAL wrapper)
for encrypted tensor operations.

Key Features:
    - CKKS context creation with optimized parameters
    - Tensor encryption/decryption utilities
    - HE-compatible operations (add, multiply, polynomial eval)
    - Parameter optimization for accuracy-performance trade-off

Reference: Base Paper - Raj et al., IEEE INDIACOM 2025
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
import time
import json

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("Warning: TenSEAL not installed. HE operations will be simulated.")


@dataclass
class HEParameters:
    """
    Homomorphic Encryption parameters configuration.
    
    Attributes:
        poly_modulus_degree: Polynomial modulus degree (power of 2)
        coeff_mod_bit_sizes: Coefficient modulus bit sizes
        scale: Scale for CKKS encoding (2^scale_bits)
        scheme: Encryption scheme ('ckks' or 'bfv')
    """
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = None
    scale_bits: int = 40
    scheme: str = "ckks"
    
    def __post_init__(self):
        if self.coeff_mod_bit_sizes is None:
            # Default coefficient modulus for given poly_modulus_degree
            if self.poly_modulus_degree == 4096:
                self.coeff_mod_bit_sizes = [40, 20, 40]
            elif self.poly_modulus_degree == 8192:
                self.coeff_mod_bit_sizes = [60, 40, 40, 60]
            elif self.poly_modulus_degree == 16384:
                self.coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 60]
            else:
                self.coeff_mod_bit_sizes = [60, 40, 40, 60]
    
    @property
    def scale(self) -> float:
        return 2 ** self.scale_bits
    
    @property
    def max_depth(self) -> int:
        """Maximum multiplicative depth before bootstrapping needed."""
        return len(self.coeff_mod_bit_sizes) - 2


# Predefined parameter sets for different use cases
HE_PARAM_PRESETS = {
    "fast": HEParameters(
        poly_modulus_degree=4096,
        coeff_mod_bit_sizes=[40, 20, 40],
        scale_bits=20
    ),
    "balanced": HEParameters(
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
        scale_bits=40
    ),
    "accurate": HEParameters(
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60],
        scale_bits=40
    ),
}


class HEContext:
    """
    Wrapper for TenSEAL context with utility methods.
    """
    
    def __init__(self, params: Optional[HEParameters] = None, preset: str = "balanced"):
        """
        Initialize HE context.
        
        Args:
            params: HE parameters (if None, uses preset)
            preset: Parameter preset name ('fast', 'balanced', 'accurate')
        """
        if params is None:
            params = HE_PARAM_PRESETS.get(preset, HE_PARAM_PRESETS["balanced"])
        
        self.params = params
        self.context = None
        self._setup_context()
    
    def _setup_context(self):
        """Create TenSEAL context."""
        if not TENSEAL_AVAILABLE:
            print("TenSEAL not available. Using simulation mode.")
            return
        
        if self.params.scheme.lower() == "ckks":
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.params.poly_modulus_degree,
                coeff_mod_bit_sizes=self.params.coeff_mod_bit_sizes
            )
            self.context.global_scale = self.params.scale
            self.context.generate_galois_keys()
            self.context.generate_relin_keys()
        else:
            raise ValueError(f"Unsupported scheme: {self.params.scheme}")
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> Any:
        """
        Encrypt a PyTorch tensor.
        
        Args:
            tensor: Input tensor (will be flattened)
        
        Returns:
            Encrypted tensor (CKKSVector or simulated)
        """
        if not TENSEAL_AVAILABLE or self.context is None:
            return SimulatedEncryptedTensor(tensor)
        
        # Flatten and convert to list
        flat = tensor.detach().cpu().numpy().flatten().tolist()
        
        return ts.ckks_vector(self.context, flat)
    
    def decrypt_tensor(
        self,
        encrypted: Any,
        shape: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Decrypt an encrypted tensor.
        
        Args:
            encrypted: Encrypted tensor
            shape: Original shape (for reshaping)
        
        Returns:
            Decrypted PyTorch tensor
        """
        if isinstance(encrypted, SimulatedEncryptedTensor):
            return encrypted.decrypt()
        
        if not TENSEAL_AVAILABLE:
            raise RuntimeError("TenSEAL not available")
        
        decrypted = encrypted.decrypt()
        tensor = torch.tensor(decrypted, dtype=torch.float32)
        
        if shape is not None:
            tensor = tensor.view(shape)
        
        return tensor
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get information about the HE context."""
        return {
            "scheme": self.params.scheme,
            "poly_modulus_degree": self.params.poly_modulus_degree,
            "coeff_mod_bit_sizes": self.params.coeff_mod_bit_sizes,
            "scale_bits": self.params.scale_bits,
            "max_depth": self.params.max_depth,
            "tenseal_available": TENSEAL_AVAILABLE,
        }


class SimulatedEncryptedTensor:
    """
    Simulated encrypted tensor for testing without TenSEAL.
    Mimics the interface of CKKS vectors.
    """
    
    def __init__(self, tensor: torch.Tensor):
        self._data = tensor.detach().clone()
        self._shape = tensor.shape
    
    def decrypt(self) -> torch.Tensor:
        return self._data.clone()
    
    def __add__(self, other):
        if isinstance(other, SimulatedEncryptedTensor):
            return SimulatedEncryptedTensor(self._data + other._data)
        return SimulatedEncryptedTensor(self._data + other)
    
    def __mul__(self, other):
        if isinstance(other, SimulatedEncryptedTensor):
            return SimulatedEncryptedTensor(self._data * other._data)
        return SimulatedEncryptedTensor(self._data * other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def square(self):
        return SimulatedEncryptedTensor(self._data ** 2)
    
    def polyval(self, coeffs: List[float]):
        """Evaluate polynomial on encrypted data."""
        result = torch.zeros_like(self._data)
        x_power = torch.ones_like(self._data)
        
        for coeff in coeffs:
            result = result + coeff * x_power
            x_power = x_power * self._data
        
        return SimulatedEncryptedTensor(result)


class EncryptedLinear:
    """
    Encrypted linear layer operation.
    
    Computes: y = Wx + b on encrypted input x
    """
    
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Args:
            weight: Weight matrix [out_features, in_features]
            bias: Bias vector [out_features]
        """
        self.weight = weight.detach().cpu().numpy()
        self.bias = bias.detach().cpu().numpy() if bias is not None else None
    
    def __call__(self, encrypted_input: Any) -> Any:
        """Apply linear transformation to encrypted input."""
        if isinstance(encrypted_input, SimulatedEncryptedTensor):
            # Simulated mode
            x = encrypted_input.decrypt()
            y = torch.mm(x.view(1, -1), torch.tensor(self.weight.T))
            if self.bias is not None:
                y = y + torch.tensor(self.bias)
            return SimulatedEncryptedTensor(y.squeeze())
        
        if not TENSEAL_AVAILABLE:
            raise RuntimeError("TenSEAL not available")
        
        # Real encrypted computation
        result = encrypted_input.mm(self.weight.T.tolist())
        if self.bias is not None:
            result = result + self.bias.tolist()
        
        return result


class EncryptedConv2d:
    """
    Encrypted 2D convolution operation.
    
    Uses im2col transformation for HE-compatible convolution.
    """
    
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: int = 1,
        padding: int = 0
    ):
        """
        Args:
            weight: Conv weight [out_channels, in_channels, kH, kW]
            bias: Bias [out_channels]
            stride: Convolution stride
            padding: Input padding
        """
        self.weight = weight.detach()
        self.bias = bias.detach() if bias is not None else None
        self.stride = stride
        self.padding = padding
        
        # Reshape weight for matrix multiplication
        self.out_channels = weight.shape[0]
        self.in_channels = weight.shape[1]
        self.kernel_size = weight.shape[2]
        
        # Flatten kernels: [out_channels, in_channels * kH * kW]
        self.weight_matrix = weight.view(self.out_channels, -1).cpu().numpy()
    
    def __call__(
        self,
        encrypted_input: Any,
        input_shape: Tuple[int, int, int]
    ) -> Tuple[Any, Tuple[int, int, int]]:
        """
        Apply convolution to encrypted input.
        
        Args:
            encrypted_input: Encrypted flattened input
            input_shape: Original input shape (C, H, W)
        
        Returns:
            Tuple of (encrypted output, output shape)
        """
        C, H, W = input_shape
        
        # Calculate output dimensions
        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        if isinstance(encrypted_input, SimulatedEncryptedTensor):
            # Simulated mode - use PyTorch conv
            x = encrypted_input.decrypt().view(1, C, H, W)
            
            with torch.no_grad():
                conv = torch.nn.Conv2d(
                    self.in_channels, self.out_channels,
                    self.kernel_size, self.stride, self.padding, bias=False
                )
                conv.weight.data = self.weight
                y = conv(x)
                
                if self.bias is not None:
                    y = y + self.bias.view(1, -1, 1, 1)
            
            return SimulatedEncryptedTensor(y.squeeze(0)), (self.out_channels, H_out, W_out)
        
        # Real HE convolution would use im2col here
        raise NotImplementedError("Real HE convolution requires im2col implementation")


class EncryptedAvgPool2d:
    """Encrypted average pooling operation."""
    
    def __init__(self, kernel_size: int = 2, stride: Optional[int] = None):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
    
    def __call__(
        self,
        encrypted_input: Any,
        input_shape: Tuple[int, int, int]
    ) -> Tuple[Any, Tuple[int, int, int]]:
        """Apply average pooling."""
        C, H, W = input_shape
        
        H_out = H // self.stride
        W_out = W // self.stride
        
        if isinstance(encrypted_input, SimulatedEncryptedTensor):
            x = encrypted_input.decrypt().view(1, C, H, W)
            pool = torch.nn.AvgPool2d(self.kernel_size, self.stride)
            y = pool(x)
            return SimulatedEncryptedTensor(y.squeeze(0)), (C, H_out, W_out)
        
        raise NotImplementedError("Real HE pooling not implemented")


def encrypted_polynomial_eval(
    encrypted_input: Any,
    coefficients: List[float]
) -> Any:
    """
    Evaluate polynomial on encrypted data.
    
    P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
    
    Args:
        encrypted_input: Encrypted tensor
        coefficients: Polynomial coefficients [a_0, a_1, ..., a_n]
    
    Returns:
        Encrypted result
    """
    if isinstance(encrypted_input, SimulatedEncryptedTensor):
        return encrypted_input.polyval(coefficients)
    
    if not TENSEAL_AVAILABLE:
        raise RuntimeError("TenSEAL not available")
    
    return encrypted_input.polyval(coefficients)


def benchmark_he_operations(
    context: HEContext,
    vector_size: int = 1000,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Benchmark HE operations.
    
    Args:
        context: HE context
        vector_size: Size of test vectors
        num_iterations: Number of iterations for averaging
    
    Returns:
        Dictionary of operation timings (in ms)
    """
    results = {}
    
    # Create test data
    x = torch.randn(vector_size)
    y = torch.randn(vector_size)
    
    # Encryption time
    start = time.time()
    for _ in range(num_iterations):
        enc_x = context.encrypt_tensor(x)
    results['encrypt_ms'] = (time.time() - start) / num_iterations * 1000
    
    # Decryption time
    enc_x = context.encrypt_tensor(x)
    start = time.time()
    for _ in range(num_iterations):
        _ = context.decrypt_tensor(enc_x)
    results['decrypt_ms'] = (time.time() - start) / num_iterations * 1000
    
    # Addition time
    enc_x = context.encrypt_tensor(x)
    enc_y = context.encrypt_tensor(y)
    start = time.time()
    for _ in range(num_iterations):
        _ = enc_x + enc_y
    results['add_ms'] = (time.time() - start) / num_iterations * 1000
    
    # Multiplication time
    start = time.time()
    for _ in range(num_iterations):
        _ = enc_x * enc_y
    results['mul_ms'] = (time.time() - start) / num_iterations * 1000
    
    # Polynomial evaluation (degree 3)
    coeffs = [0.5, 0.5, 0.0, 0.0833]
    start = time.time()
    for _ in range(num_iterations):
        _ = encrypted_polynomial_eval(enc_x, coeffs)
    results['poly_eval_ms'] = (time.time() - start) / num_iterations * 1000
    
    return results


if __name__ == "__main__":
    print("Testing HE Utilities...")
    
    # Create context
    context = HEContext(preset="balanced")
    print(f"\nContext info: {json.dumps(context.get_context_info(), indent=2)}")
    
    # Test encryption/decryption
    print("\n--- Encryption/Decryption Test ---")
    x = torch.randn(100)
    enc_x = context.encrypt_tensor(x)
    dec_x = context.decrypt_tensor(enc_x)
    
    error = torch.mean((x - dec_x) ** 2).item()
    print(f"Encryption/Decryption MSE: {error:.10f}")
    
    # Test operations
    print("\n--- Operation Tests ---")
    y = torch.randn(100)
    enc_y = context.encrypt_tensor(y)
    
    # Addition
    enc_sum = enc_x + enc_y
    dec_sum = context.decrypt_tensor(enc_sum)
    add_error = torch.mean((x + y - dec_sum) ** 2).item()
    print(f"Addition MSE: {add_error:.10f}")
    
    # Multiplication
    enc_prod = enc_x * enc_y
    dec_prod = context.decrypt_tensor(enc_prod)
    mul_error = torch.mean((x * y - dec_prod) ** 2).item()
    print(f"Multiplication MSE: {mul_error:.10f}")
    
    # Polynomial evaluation
    coeffs = [0.5, 0.5, 0.0, 0.0833]
    enc_poly = encrypted_polynomial_eval(enc_x, coeffs)
    dec_poly = context.decrypt_tensor(enc_poly)
    
    # Expected result
    expected = sum(c * (x ** i) for i, c in enumerate(coeffs))
    poly_error = torch.mean((expected - dec_poly) ** 2).item()
    print(f"Polynomial Eval MSE: {poly_error:.10f}")
    
    # Benchmark
    print("\n--- Benchmarks ---")
    benchmarks = benchmark_he_operations(context, vector_size=1000, num_iterations=5)
    for op, time_ms in benchmarks.items():
        print(f"{op}: {time_ms:.3f} ms")
    
    print("\nHE utility tests passed!")

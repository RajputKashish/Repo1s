"""
Tests for Homomorphic Encryption utilities.

Tests cover:
    - HE context creation
    - Encryption/decryption accuracy
    - HE operations (add, multiply, polynomial)
    - Parameter presets
"""

import pytest
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.he_utils import (
    HEContext, HEParameters, HE_PARAM_PRESETS,
    SimulatedEncryptedTensor, encrypted_polynomial_eval,
    EncryptedLinear, benchmark_he_operations
)


class TestHEParameters:
    """Tests for HE parameter configuration."""
    
    def test_default_parameters(self):
        """Test default parameter initialization."""
        params = HEParameters()
        assert params.poly_modulus_degree == 8192
        assert params.scheme == "ckks"
        assert params.scale_bits == 40
        assert params.coeff_mod_bit_sizes is not None
    
    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        params = HEParameters(
            poly_modulus_degree=4096,
            coeff_mod_bit_sizes=[40, 20, 40],
            scale_bits=20
        )
        assert params.poly_modulus_degree == 4096
        assert params.coeff_mod_bit_sizes == [40, 20, 40]
        assert params.scale == 2 ** 20
    
    def test_max_depth_calculation(self):
        """Test multiplicative depth calculation."""
        params = HEParameters(coeff_mod_bit_sizes=[60, 40, 40, 60])
        assert params.max_depth == 2  # len - 2
    
    def test_presets_exist(self):
        """Test that all presets are defined."""
        assert "fast" in HE_PARAM_PRESETS
        assert "balanced" in HE_PARAM_PRESETS
        assert "accurate" in HE_PARAM_PRESETS


class TestHEContext:
    """Tests for HE context creation and operations."""
    
    def test_context_creation_balanced(self):
        """Test balanced preset context creation."""
        context = HEContext(preset="balanced")
        assert context.params is not None
        info = context.get_context_info()
        assert info['scheme'] == 'ckks'
    
    def test_context_creation_fast(self):
        """Test fast preset context creation."""
        context = HEContext(preset="fast")
        assert context.params.poly_modulus_degree == 4096
    
    def test_context_creation_accurate(self):
        """Test accurate preset context creation."""
        context = HEContext(preset="accurate")
        assert context.params.poly_modulus_degree == 16384
    
    def test_encrypt_decrypt_vector(self):
        """Test encryption and decryption of vectors."""
        context = HEContext(preset="fast")
        
        x = torch.randn(100)
        enc_x = context.encrypt_tensor(x)
        dec_x = context.decrypt_tensor(enc_x)
        
        # Check shape preserved
        assert dec_x.shape == x.shape
        
        # Check values approximately equal (simulated mode is exact)
        mse = torch.mean((x - dec_x) ** 2).item()
        assert mse < 1e-6  # Very small error in simulation
    
    def test_encrypt_decrypt_2d(self):
        """Test encryption of 2D tensors."""
        context = HEContext(preset="fast")
        
        x = torch.randn(10, 10)
        enc_x = context.encrypt_tensor(x)
        dec_x = context.decrypt_tensor(enc_x)
        
        # Flattened comparison
        assert dec_x.numel() == x.numel()


class TestSimulatedEncryptedTensor:
    """Tests for simulated encrypted tensor operations."""
    
    def test_creation(self):
        """Test tensor creation."""
        x = torch.randn(50)
        enc = SimulatedEncryptedTensor(x)
        dec = enc.decrypt()
        
        assert torch.allclose(x, dec)
    
    def test_addition(self):
        """Test encrypted addition."""
        x = torch.randn(50)
        y = torch.randn(50)
        
        enc_x = SimulatedEncryptedTensor(x)
        enc_y = SimulatedEncryptedTensor(y)
        
        enc_sum = enc_x + enc_y
        dec_sum = enc_sum.decrypt()
        
        assert torch.allclose(x + y, dec_sum)
    
    def test_multiplication(self):
        """Test encrypted multiplication."""
        x = torch.randn(50)
        y = torch.randn(50)
        
        enc_x = SimulatedEncryptedTensor(x)
        enc_y = SimulatedEncryptedTensor(y)
        
        enc_prod = enc_x * enc_y
        dec_prod = enc_prod.decrypt()
        
        assert torch.allclose(x * y, dec_prod)
    
    def test_scalar_operations(self):
        """Test operations with scalars."""
        x = torch.randn(50)
        scalar = 2.5
        
        enc_x = SimulatedEncryptedTensor(x)
        
        # Scalar addition
        enc_add = enc_x + scalar
        assert torch.allclose(x + scalar, enc_add.decrypt())
        
        # Scalar multiplication
        enc_mul = enc_x * scalar
        assert torch.allclose(x * scalar, enc_mul.decrypt())
    
    def test_square(self):
        """Test square operation."""
        x = torch.randn(50)
        enc_x = SimulatedEncryptedTensor(x)
        
        enc_sq = enc_x.square()
        dec_sq = enc_sq.decrypt()
        
        assert torch.allclose(x ** 2, dec_sq)
    
    def test_polynomial_evaluation(self):
        """Test polynomial evaluation."""
        x = torch.randn(50)
        coeffs = [1.0, 2.0, 3.0]  # 1 + 2x + 3x^2
        
        enc_x = SimulatedEncryptedTensor(x)
        enc_poly = enc_x.polyval(coeffs)
        dec_poly = enc_poly.decrypt()
        
        expected = 1.0 + 2.0 * x + 3.0 * x ** 2
        assert torch.allclose(expected, dec_poly)


class TestEncryptedPolynomialEval:
    """Tests for encrypted polynomial evaluation function."""
    
    def test_constant_polynomial(self):
        """Test constant polynomial (degree 0)."""
        x = torch.randn(20)
        enc_x = SimulatedEncryptedTensor(x)
        
        result = encrypted_polynomial_eval(enc_x, [5.0])
        dec = result.decrypt()
        
        expected = torch.full_like(x, 5.0)
        assert torch.allclose(expected, dec)
    
    def test_linear_polynomial(self):
        """Test linear polynomial (degree 1)."""
        x = torch.randn(20)
        enc_x = SimulatedEncryptedTensor(x)
        
        # 2 + 3x
        result = encrypted_polynomial_eval(enc_x, [2.0, 3.0])
        dec = result.decrypt()
        
        expected = 2.0 + 3.0 * x
        assert torch.allclose(expected, dec)
    
    def test_quadratic_polynomial(self):
        """Test quadratic polynomial (degree 2)."""
        x = torch.randn(20)
        enc_x = SimulatedEncryptedTensor(x)
        
        # 1 + 2x + 3x^2
        result = encrypted_polynomial_eval(enc_x, [1.0, 2.0, 3.0])
        dec = result.decrypt()
        
        expected = 1.0 + 2.0 * x + 3.0 * x ** 2
        assert torch.allclose(expected, dec)
    
    def test_relu_approximation(self):
        """Test ReLU polynomial approximation."""
        x = torch.linspace(-2, 2, 50)
        enc_x = SimulatedEncryptedTensor(x)
        
        # Degree 3 ReLU approximation coefficients
        coeffs = [0.0, 0.5, 0.0, 0.0833]
        result = encrypted_polynomial_eval(enc_x, coeffs)
        dec = result.decrypt()
        
        # Should be close to ReLU for small x
        true_relu = torch.relu(x)
        # Allow larger error since it's an approximation
        mse = torch.mean((true_relu - dec) ** 2).item()
        assert mse < 1.0  # Reasonable approximation


class TestEncryptedLinear:
    """Tests for encrypted linear layer."""
    
    def test_linear_no_bias(self):
        """Test linear layer without bias."""
        in_features = 10
        out_features = 5
        
        weight = torch.randn(out_features, in_features)
        enc_linear = EncryptedLinear(weight, bias=None)
        
        x = torch.randn(in_features)
        enc_x = SimulatedEncryptedTensor(x)
        
        enc_out = enc_linear(enc_x)
        dec_out = enc_out.decrypt()
        
        expected = torch.mm(x.view(1, -1), weight.T).squeeze()
        assert torch.allclose(expected, dec_out, atol=1e-5)
    
    def test_linear_with_bias(self):
        """Test linear layer with bias."""
        in_features = 10
        out_features = 5
        
        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)
        enc_linear = EncryptedLinear(weight, bias)
        
        x = torch.randn(in_features)
        enc_x = SimulatedEncryptedTensor(x)
        
        enc_out = enc_linear(enc_x)
        dec_out = enc_out.decrypt()
        
        expected = torch.mm(x.view(1, -1), weight.T).squeeze() + bias
        assert torch.allclose(expected, dec_out, atol=1e-5)


class TestBenchmarks:
    """Tests for benchmarking functions."""
    
    def test_benchmark_runs(self):
        """Test that benchmarks complete without error."""
        context = HEContext(preset="fast")
        results = benchmark_he_operations(
            context,
            vector_size=100,
            num_iterations=2
        )
        
        assert 'encrypt_ms' in results
        assert 'decrypt_ms' in results
        assert 'add_ms' in results
        assert 'mul_ms' in results
        assert 'poly_eval_ms' in results
        
        # All times should be positive
        for key, value in results.items():
            assert value >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

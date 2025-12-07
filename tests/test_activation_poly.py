"""
Tests for Adaptive Polynomial Activation functions.

Tests cover:
    - Polynomial coefficient correctness
    - Activation approximation accuracy
    - Adaptive activation behavior
    - HE-friendly batch normalization
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.activations_poly import (
    PolynomialCoefficients, polynomial_eval, polynomial_eval_horner,
    PolyReLU, PolySigmoid, PolyTanh, SquareActivation,
    AdaptivePolyActivation, HEFriendlyBatchNorm, get_activation
)


class TestPolynomialEvaluation:
    """Tests for polynomial evaluation functions."""
    
    def test_polynomial_eval_constant(self):
        """Test constant polynomial."""
        x = torch.randn(10)
        coeffs = torch.tensor([5.0])
        result = polynomial_eval(x, coeffs)
        
        expected = torch.full_like(x, 5.0)
        assert torch.allclose(result, expected)
    
    def test_polynomial_eval_linear(self):
        """Test linear polynomial."""
        x = torch.tensor([1.0, 2.0, 3.0])
        coeffs = torch.tensor([1.0, 2.0])  # 1 + 2x
        result = polynomial_eval(x, coeffs)
        
        expected = torch.tensor([3.0, 5.0, 7.0])
        assert torch.allclose(result, expected)
    
    def test_polynomial_eval_quadratic(self):
        """Test quadratic polynomial."""
        x = torch.tensor([0.0, 1.0, 2.0])
        coeffs = torch.tensor([1.0, 0.0, 1.0])  # 1 + x^2
        result = polynomial_eval(x, coeffs)
        
        expected = torch.tensor([1.0, 2.0, 5.0])
        assert torch.allclose(result, expected)
    
    def test_horner_matches_standard(self):
        """Test that Horner's method matches standard evaluation."""
        x = torch.randn(100)
        coeffs = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        result_standard = polynomial_eval(x, coeffs)
        result_horner = polynomial_eval_horner(x, coeffs)
        
        assert torch.allclose(result_standard, result_horner, atol=1e-5)


class TestPolyReLU:
    """Tests for polynomial ReLU approximation."""
    
    def test_degree_2(self):
        """Test degree 2 approximation."""
        poly_relu = PolyReLU(degree=2)
        x = torch.linspace(-2, 2, 50)
        y = poly_relu(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any()
    
    def test_degree_3(self):
        """Test degree 3 approximation."""
        poly_relu = PolyReLU(degree=3)
        x = torch.linspace(-2, 2, 50)
        y = poly_relu(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any()
    
    def test_degree_4(self):
        """Test degree 4 approximation."""
        poly_relu = PolyReLU(degree=4)
        x = torch.linspace(-2, 2, 50)
        y = poly_relu(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any()
    
    def test_approximation_quality(self):
        """Test that approximation is reasonable."""
        poly_relu = PolyReLU(degree=3)
        x = torch.linspace(-1, 1, 100)
        
        error = poly_relu.approximation_error(x)
        # Error should be bounded
        assert error < 1.0
    
    def test_learnable_coefficients(self):
        """Test learnable coefficient mode."""
        poly_relu = PolyReLU(degree=3, learnable=True)
        
        # Coefficients should be parameters
        assert isinstance(poly_relu.coeffs, nn.Parameter)
        assert poly_relu.coeffs.requires_grad
    
    def test_fixed_coefficients(self):
        """Test fixed coefficient mode."""
        poly_relu = PolyReLU(degree=3, learnable=False)
        
        # Coefficients should be buffers
        assert not isinstance(poly_relu.coeffs, nn.Parameter)
    
    def test_gradient_flow(self):
        """Test gradient flow through learnable activation."""
        poly_relu = PolyReLU(degree=3, learnable=True)
        x = torch.randn(10, requires_grad=True)
        
        y = poly_relu(x)
        loss = y.sum()
        loss.backward()
        
        # Gradients should exist
        assert x.grad is not None
        assert poly_relu.coeffs.grad is not None


class TestPolySigmoid:
    """Tests for polynomial sigmoid approximation."""
    
    def test_output_range(self):
        """Test that output is approximately in [0, 1]."""
        poly_sigmoid = PolySigmoid(degree=3)
        x = torch.linspace(-2, 2, 50)
        y = poly_sigmoid(x)
        
        # Should be approximately bounded
        assert y.min() > -0.5
        assert y.max() < 1.5
    
    def test_center_value(self):
        """Test value at x=0."""
        poly_sigmoid = PolySigmoid(degree=3)
        x = torch.tensor([0.0])
        y = poly_sigmoid(x)
        
        # Should be close to 0.5
        assert abs(y.item() - 0.5) < 0.1


class TestPolyTanh:
    """Tests for polynomial tanh approximation."""
    
    def test_output_range(self):
        """Test that output is approximately in [-1, 1]."""
        poly_tanh = PolyTanh(degree=3)
        x = torch.linspace(-1, 1, 50)
        y = poly_tanh(x)
        
        # Should be approximately bounded
        assert y.min() > -1.5
        assert y.max() < 1.5
    
    def test_center_value(self):
        """Test value at x=0."""
        poly_tanh = PolyTanh(degree=3)
        x = torch.tensor([0.0])
        y = poly_tanh(x)
        
        # Should be close to 0
        assert abs(y.item()) < 0.1
    
    def test_odd_function(self):
        """Test that approximation is approximately odd."""
        poly_tanh = PolyTanh(degree=3)
        x = torch.tensor([0.5])
        
        y_pos = poly_tanh(x)
        y_neg = poly_tanh(-x)
        
        # f(-x) ≈ -f(x)
        assert torch.allclose(y_neg, -y_pos, atol=0.1)


class TestSquareActivation:
    """Tests for square activation."""
    
    def test_square_computation(self):
        """Test that square is computed correctly."""
        square = SquareActivation()
        x = torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0])
        y = square(x)
        
        expected = torch.tensor([1.0, 4.0, 9.0, 1.0, 4.0])
        assert torch.allclose(y, expected)
    
    def test_gradient(self):
        """Test gradient computation."""
        square = SquareActivation()
        x = torch.tensor([2.0], requires_grad=True)
        
        y = square(x)
        y.backward()
        
        # d/dx(x^2) = 2x
        assert torch.allclose(x.grad, torch.tensor([4.0]))


class TestAdaptivePolyActivation:
    """Tests for adaptive polynomial activation."""
    
    def test_creation(self):
        """Test module creation."""
        apa = AdaptivePolyActivation(degrees=[2, 3, 4])
        assert len(apa.degrees) == 3
    
    def test_forward_pass(self):
        """Test forward pass."""
        apa = AdaptivePolyActivation(degrees=[2, 3, 4])
        x = torch.randn(8, 64)
        y = apa(x)
        
        assert y.shape == x.shape
        assert not torch.isnan(y).any()
    
    def test_input_stats_computation(self):
        """Test input statistics computation."""
        apa = AdaptivePolyActivation(degrees=[2, 3, 4])
        x = torch.randn(4, 100)
        
        stats = apa.compute_input_stats(x)
        
        # Should have 4 statistics per sample
        assert stats.shape == (4, 4)
    
    def test_effective_degree(self):
        """Test effective degree selection."""
        apa = AdaptivePolyActivation(degrees=[2, 3, 4])
        x = torch.randn(8, 64)
        
        degree = apa.get_effective_degree(x)
        
        # Should be one of the available degrees
        assert degree in [2, 3, 4]
    
    def test_gradient_flow(self):
        """Test gradient flow through adaptive activation."""
        apa = AdaptivePolyActivation(
            degrees=[2, 3, 4],
            learnable_coeffs=True,
            learnable_gate=True
        )
        x = torch.randn(4, 32, requires_grad=True)
        
        y = apa(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_different_activation_types(self):
        """Test different base activation types."""
        for act_type in ["relu", "sigmoid", "tanh"]:
            apa = AdaptivePolyActivation(
                degrees=[2, 3],
                activation_type=act_type
            )
            x = torch.randn(4, 32)
            y = apa(x)
            
            assert y.shape == x.shape


class TestHEFriendlyBatchNorm:
    """Tests for HE-friendly batch normalization."""
    
    def test_training_mode(self):
        """Test batch norm in training mode."""
        bn = HEFriendlyBatchNorm(64)
        bn.train()
        
        x = torch.randn(32, 64)
        y = bn(x)
        
        assert y.shape == x.shape
    
    def test_eval_mode_standard(self):
        """Test batch norm in eval mode (standard)."""
        bn = HEFriendlyBatchNorm(64)
        
        # Train to update running stats
        bn.train()
        for _ in range(10):
            x = torch.randn(32, 64)
            _ = bn(x)
        
        # Eval mode
        bn.eval()
        x = torch.randn(4, 64)
        y = bn(x)
        
        assert y.shape == x.shape
    
    def test_parameter_folding(self):
        """Test parameter folding for HE."""
        bn = HEFriendlyBatchNorm(64)
        
        # Train
        bn.train()
        for _ in range(10):
            x = torch.randn(32, 64)
            _ = bn(x)
        
        # Fold parameters
        bn.eval()
        bn.fold_parameters()
        
        assert bn.folded
        assert bn.folded_weight is not None
        assert bn.folded_bias is not None
    
    def test_folded_output_matches(self):
        """Test that folded output matches standard output."""
        bn = HEFriendlyBatchNorm(64)
        
        # Train
        bn.train()
        for _ in range(10):
            x = torch.randn(32, 64)
            _ = bn(x)
        
        # Get standard eval output
        bn.eval()
        x = torch.randn(4, 64)
        y_standard = bn(x).clone()
        
        # Fold and get folded output
        bn.fold_parameters()
        y_folded = bn(x)
        
        # Should be very close
        assert torch.allclose(y_standard, y_folded, atol=1e-5)
    
    def test_conv_output_shape(self):
        """Test with conv-like output (4D tensor)."""
        bn = HEFriendlyBatchNorm(32)
        
        # Train
        bn.train()
        for _ in range(10):
            x = torch.randn(8, 32, 14, 14)
            _ = bn(x)
        
        # Fold and test
        bn.eval()
        bn.fold_parameters()
        
        x = torch.randn(4, 32, 14, 14)
        y = bn(x)
        
        assert y.shape == x.shape


class TestGetActivation:
    """Tests for activation factory function."""
    
    def test_get_relu(self):
        """Test getting standard ReLU."""
        act = get_activation("relu")
        assert isinstance(act, nn.ReLU)
    
    def test_get_poly_relu(self):
        """Test getting polynomial ReLU."""
        act = get_activation("poly_relu", degree=3)
        assert isinstance(act, PolyReLU)
    
    def test_get_square(self):
        """Test getting square activation."""
        act = get_activation("square")
        assert isinstance(act, SquareActivation)
    
    def test_get_adaptive(self):
        """Test getting adaptive activation."""
        act = get_activation("poly_relu", adaptive=True)
        assert isinstance(act, AdaptivePolyActivation)
    
    def test_invalid_activation(self):
        """Test error on invalid activation."""
        with pytest.raises(ValueError):
            get_activation("invalid_activation")


class TestPolynomialCoefficients:
    """Tests for pre-defined polynomial coefficients."""
    
    def test_relu_coefficients_exist(self):
        """Test ReLU coefficients are defined."""
        assert 2 in PolynomialCoefficients.RELU
        assert 3 in PolynomialCoefficients.RELU
        assert 4 in PolynomialCoefficients.RELU
    
    def test_sigmoid_coefficients_exist(self):
        """Test sigmoid coefficients are defined."""
        assert 2 in PolynomialCoefficients.SIGMOID
        assert 3 in PolynomialCoefficients.SIGMOID
    
    def test_tanh_coefficients_exist(self):
        """Test tanh coefficients are defined."""
        assert 2 in PolynomialCoefficients.TANH
        assert 3 in PolynomialCoefficients.TANH
    
    def test_coefficient_lengths(self):
        """Test coefficient array lengths match degrees."""
        for degree, coeffs in PolynomialCoefficients.RELU.items():
            assert len(coeffs) == degree + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

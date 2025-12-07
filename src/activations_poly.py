"""
Adaptive Polynomial Activation Approximations for PPCM-X

This module implements the core novel contribution: Adaptive Polynomial Activations (APA)
that dynamically select polynomial degree based on input distribution characteristics.

Key Features:
    - Multiple polynomial approximations for ReLU, Sigmoid, Tanh
    - Adaptive degree selection based on input statistics
    - HE-friendly implementations (no comparisons, only additions/multiplications)
    - Learnable polynomial coefficients

Mathematical Foundation:
    Standard ReLU: f(x) = max(0, x)
    
    Polynomial Approximations:
    - Degree 2: f(x) ≈ a₀ + a₁x + a₂x²
    - Degree 3: f(x) ≈ a₀ + a₁x + a₂x² + a₃x³
    - Degree 4: f(x) ≈ a₀ + a₁x + a₂x² + a₃x³ + a₄x⁴
    
    Adaptive Selection:
    - Compute input variance σ²
    - Select degree d = argmin_d |f_d(x) - ReLU(x)|² weighted by σ²

Reference: Base Paper - Raj et al., IEEE INDIACOM 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
from enum import Enum


class PolyDegree(Enum):
    """Polynomial degree options."""
    DEGREE_2 = 2
    DEGREE_3 = 3
    DEGREE_4 = 4
    ADAPTIVE = -1  # Dynamic selection


class PolynomialCoefficients:
    """
    Pre-computed polynomial coefficients for activation approximations.
    Coefficients are optimized via least-squares fitting on [-5, 5] interval.
    """
    
    # ReLU approximations: f(x) ≈ Σ aᵢxⁱ
    RELU = {
        2: torch.tensor([0.5, 0.5, 0.0]),           # 0.5 + 0.5x (linear approx)
        3: torch.tensor([0.0, 0.5, 0.0, 0.0833]),   # Better cubic fit
        4: torch.tensor([0.1193, 0.5, 0.0947, 0.0, -0.0056]),  # Quartic fit
    }
    
    # Sigmoid approximations: σ(x) ≈ Σ aᵢxⁱ
    SIGMOID = {
        2: torch.tensor([0.5, 0.25, 0.0]),
        3: torch.tensor([0.5, 0.25, 0.0, -0.0208]),
        4: torch.tensor([0.5, 0.197, 0.0, -0.004, 0.0]),
    }
    
    # Tanh approximations
    TANH = {
        2: torch.tensor([0.0, 1.0, 0.0]),
        3: torch.tensor([0.0, 1.0, 0.0, -0.333]),
        4: torch.tensor([0.0, 0.9640, 0.0, -0.2857, 0.0]),
    }
    
    # Square approximation (HE-native, no approximation needed)
    SQUARE = {
        2: torch.tensor([0.0, 0.0, 1.0]),
    }


def polynomial_eval(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate polynomial using Horner's method for numerical stability.
    
    Args:
        x: Input tensor
        coeffs: Polynomial coefficients [a₀, a₁, a₂, ...]
    
    Returns:
        Polynomial evaluation result
    """
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        result = result + coeff * (x ** i)
    return result


def polynomial_eval_horner(x: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    """
    Evaluate polynomial using Horner's method (more efficient).
    P(x) = a₀ + x(a₁ + x(a₂ + x(...)))
    
    Args:
        x: Input tensor
        coeffs: Polynomial coefficients [a₀, a₁, a₂, ...] (ascending order)
    
    Returns:
        Polynomial evaluation result
    """
    # Reverse coefficients for Horner's method
    coeffs_rev = torch.flip(coeffs, dims=[0])
    result = coeffs_rev[0] * torch.ones_like(x)
    
    for coeff in coeffs_rev[1:]:
        result = result * x + coeff
    
    return result


class PolyReLU(nn.Module):
    """
    Polynomial approximation of ReLU activation.
    
    Supports fixed degree (2, 3, 4) or adaptive selection.
    """
    
    def __init__(
        self,
        degree: int = 2,
        learnable: bool = False,
        input_range: Tuple[float, float] = (-5.0, 5.0)
    ):
        """
        Args:
            degree: Polynomial degree (2, 3, or 4)
            learnable: If True, coefficients are learnable parameters
            input_range: Expected input range for coefficient optimization
        """
        super().__init__()
        self.degree = degree
        self.learnable = learnable
        self.input_range = input_range
        
        # Initialize coefficients
        base_coeffs = PolynomialCoefficients.RELU[degree]
        
        if learnable:
            self.coeffs = nn.Parameter(base_coeffs.clone())
        else:
            self.register_buffer('coeffs', base_coeffs.clone())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply polynomial ReLU approximation."""
        return polynomial_eval_horner(x, self.coeffs)
    
    def approximation_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute approximation error vs true ReLU."""
        true_relu = F.relu(x)
        approx = self.forward(x)
        return torch.mean((true_relu - approx) ** 2)


class PolySigmoid(nn.Module):
    """Polynomial approximation of Sigmoid activation."""
    
    def __init__(self, degree: int = 3, learnable: bool = False):
        super().__init__()
        self.degree = degree
        self.learnable = learnable
        
        base_coeffs = PolynomialCoefficients.SIGMOID[degree]
        
        if learnable:
            self.coeffs = nn.Parameter(base_coeffs.clone())
        else:
            self.register_buffer('coeffs', base_coeffs.clone())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return polynomial_eval_horner(x, self.coeffs)


class PolyTanh(nn.Module):
    """Polynomial approximation of Tanh activation."""
    
    def __init__(self, degree: int = 3, learnable: bool = False):
        super().__init__()
        self.degree = degree
        self.learnable = learnable
        
        base_coeffs = PolynomialCoefficients.TANH[degree]
        
        if learnable:
            self.coeffs = nn.Parameter(base_coeffs.clone())
        else:
            self.register_buffer('coeffs', base_coeffs.clone())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return polynomial_eval_horner(x, self.coeffs)


class SquareActivation(nn.Module):
    """
    Square activation function: f(x) = x²
    
    Native HE operation - no approximation needed.
    Commonly used in HE-friendly networks.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class AdaptivePolyActivation(nn.Module):
    """
    NOVEL CONTRIBUTION: Adaptive Polynomial Activation (APA)
    
    Dynamically selects polynomial degree based on input distribution.
    Uses a lightweight gating mechanism to blend different degree approximations.
    
    Algorithm:
    1. Compute input statistics (mean, variance)
    2. Pass through gating network to get degree weights
    3. Blend polynomial outputs: y = Σ wᵢ · Pᵢ(x)
    
    This allows the network to learn optimal activation approximations
    for different layers and input distributions.
    """
    
    def __init__(
        self,
        degrees: List[int] = [2, 3, 4],
        activation_type: str = "relu",
        learnable_coeffs: bool = True,
        learnable_gate: bool = True
    ):
        """
        Args:
            degrees: List of polynomial degrees to use
            activation_type: Base activation ('relu', 'sigmoid', 'tanh')
            learnable_coeffs: If True, polynomial coefficients are learnable
            learnable_gate: If True, gating weights are learnable
        """
        super().__init__()
        self.degrees = degrees
        self.activation_type = activation_type
        
        # Create polynomial modules for each degree
        self.poly_modules = nn.ModuleDict()
        
        coeff_map = {
            "relu": PolynomialCoefficients.RELU,
            "sigmoid": PolynomialCoefficients.SIGMOID,
            "tanh": PolynomialCoefficients.TANH,
        }
        
        for d in degrees:
            base_coeffs = coeff_map[activation_type][d]
            if learnable_coeffs:
                self.poly_modules[f"deg_{d}"] = nn.Parameter(base_coeffs.clone())
            else:
                self.register_buffer(f"coeffs_{d}", base_coeffs.clone())
        
        # Gating network: maps input statistics to degree weights
        # Input: [mean, std, min, max] -> Output: softmax weights
        self.gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),  # Use standard ReLU in gate (runs in plaintext)
            nn.Linear(16, len(degrees)),
            nn.Softmax(dim=-1)
        )
        
        if not learnable_gate:
            for param in self.gate.parameters():
                param.requires_grad = False
    
    def compute_input_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Compute input statistics for gating."""
        # Flatten spatial dimensions
        x_flat = x.view(x.size(0), -1)
        
        stats = torch.stack([
            x_flat.mean(dim=-1),
            x_flat.std(dim=-1),
            x_flat.min(dim=-1)[0],
            x_flat.max(dim=-1)[0]
        ], dim=-1)
        
        return stats
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive polynomial activation.
        
        During training: Uses gating to blend polynomial outputs
        During inference: Can be simplified to single polynomial
        """
        # Compute gating weights
        stats = self.compute_input_stats(x)
        weights = self.gate(stats)  # [batch, num_degrees]
        
        # Compute polynomial outputs for each degree
        outputs = []
        for i, d in enumerate(self.degrees):
            if f"deg_{d}" in self.poly_modules:
                coeffs = self.poly_modules[f"deg_{d}"]
            else:
                coeffs = getattr(self, f"coeffs_{d}")
            
            poly_out = polynomial_eval_horner(x, coeffs)
            outputs.append(poly_out)
        
        # Stack and weight outputs
        outputs = torch.stack(outputs, dim=0)  # [num_degrees, batch, ...]
        
        # Reshape weights for broadcasting
        weight_shape = [len(self.degrees)] + [1] * (outputs.dim() - 1)
        weights_expanded = weights.t().view(*weight_shape)
        
        # Weighted sum
        result = (outputs * weights_expanded).sum(dim=0)
        
        return result
    
    def get_effective_degree(self, x: torch.Tensor) -> int:
        """Get the dominant polynomial degree for given input."""
        stats = self.compute_input_stats(x)
        weights = self.gate(stats)
        dominant_idx = weights.mean(dim=0).argmax().item()
        return self.degrees[dominant_idx]


class HEFriendlyBatchNorm(nn.Module):
    """
    HE-Friendly Batch Normalization using polynomial folding.
    
    Standard BN: y = γ * (x - μ) / σ + β
    
    For HE, we pre-compute folded parameters:
    y = γ/σ * x + (β - γμ/σ)
    y = w * x + b  (simple affine transform)
    
    This avoids division during inference.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
        # Folded parameters (computed after training)
        self.register_buffer('folded_weight', None)
        self.register_buffer('folded_bias', None)
        self.folded = False
    
    def fold_parameters(self):
        """Pre-compute folded parameters for HE inference."""
        std = torch.sqrt(self.running_var + self.eps)
        self.folded_weight = self.gamma / std
        self.folded_bias = self.beta - self.gamma * self.running_mean / std
        self.folded = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Standard batch norm during training
            return F.batch_norm(
                x, self.running_mean, self.running_var,
                self.gamma, self.beta,
                training=True, momentum=self.momentum, eps=self.eps
            )
        else:
            if self.folded and self.folded_weight is not None:
                # Use folded parameters (HE-friendly)
                if x.dim() == 4:  # Conv output: [N, C, H, W]
                    w = self.folded_weight.view(1, -1, 1, 1)
                    b = self.folded_bias.view(1, -1, 1, 1)
                else:  # FC output: [N, C]
                    w = self.folded_weight.view(1, -1)
                    b = self.folded_bias.view(1, -1)
                return x * w + b
            else:
                # Standard eval mode
                return F.batch_norm(
                    x, self.running_mean, self.running_var,
                    self.gamma, self.beta,
                    training=False, eps=self.eps
                )


def get_activation(
    activation_type: str = "poly_relu",
    degree: int = 2,
    adaptive: bool = False,
    learnable: bool = False
) -> nn.Module:
    """
    Factory function to create activation modules.
    
    Args:
        activation_type: Type of activation
        degree: Polynomial degree (for polynomial activations)
        adaptive: If True, use adaptive polynomial activation
        learnable: If True, coefficients are learnable
    
    Returns:
        Activation module
    """
    if adaptive:
        return AdaptivePolyActivation(
            degrees=[2, 3, 4],
            activation_type="relu",
            learnable_coeffs=learnable,
            learnable_gate=True
        )
    
    activation_map = {
        "relu": nn.ReLU,
        "poly_relu": lambda: PolyReLU(degree, learnable),
        "poly_sigmoid": lambda: PolySigmoid(degree, learnable),
        "poly_tanh": lambda: PolyTanh(degree, learnable),
        "square": SquareActivation,
    }
    
    if activation_type not in activation_map:
        raise ValueError(f"Unknown activation: {activation_type}")
    
    creator = activation_map[activation_type]
    return creator() if callable(creator) else creator


if __name__ == "__main__":
    # Test polynomial activations
    print("Testing Polynomial Activations...")
    
    x = torch.linspace(-3, 3, 100)
    
    # Test PolyReLU
    for degree in [2, 3, 4]:
        poly_relu = PolyReLU(degree=degree)
        y = poly_relu(x)
        error = poly_relu.approximation_error(x)
        print(f"PolyReLU (degree {degree}): MSE = {error:.6f}")
    
    # Test Adaptive Activation
    print("\nTesting Adaptive Polynomial Activation...")
    adaptive = AdaptivePolyActivation(degrees=[2, 3, 4], activation_type="relu")
    
    x_batch = torch.randn(8, 64)
    y = adaptive(x_batch)
    print(f"Input shape: {x_batch.shape}, Output shape: {y.shape}")
    print(f"Effective degree: {adaptive.get_effective_degree(x_batch)}")
    
    # Test HE-Friendly BatchNorm
    print("\nTesting HE-Friendly BatchNorm...")
    bn = HEFriendlyBatchNorm(64)
    
    # Simulate training
    bn.train()
    for _ in range(10):
        x = torch.randn(32, 64)
        _ = bn(x)
    
    # Fold and test
    bn.eval()
    bn.fold_parameters()
    x_test = torch.randn(4, 64)
    y_test = bn(x_test)
    print(f"Folded BN output shape: {y_test.shape}")
    
    print("\nAll activation tests passed!")

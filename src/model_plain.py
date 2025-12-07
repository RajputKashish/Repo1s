"""
Plaintext CNN Architectures for PPCM-X

This module defines CNN architectures that can be trained in plaintext
and later converted to HE-compatible versions.

Architectures:
    - PPCM_CNN: Base CNN from the original paper (Raj et al., 2025)
    - PPCM_X_CNN: Extended CNN with adaptive activations
    - PPCM_X_Deep: Deeper variant for CIFAR-10

Reference: Base Paper - Raj et al., IEEE INDIACOM 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
from .activations_poly import (
    PolyReLU, SquareActivation, AdaptivePolyActivation,
    HEFriendlyBatchNorm, get_activation
)


class PPCM_CNN(nn.Module):
    """
    Base Privacy-Preserving CNN Model from the original paper.
    
    Architecture:
        Conv2d(1, 16, 5) -> Square -> AvgPool(2)
        Conv2d(16, 32, 5) -> Square -> AvgPool(2)
        Flatten -> FC(512, 128) -> Square -> FC(128, 10)
    
    Uses square activation for HE compatibility.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=0)
        
        # Pooling
        self.pool = nn.AvgPool2d(2, 2)
        
        # Activation (square for HE)
        self.activation = SquareActivation()
        
        # Calculate FC input size
        self._fc_input_size = self._calculate_fc_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self._fc_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def _calculate_fc_size(self) -> int:
        """Calculate the flattened size after conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            x = self.pool(self.activation(self.conv1(x)))
            x = self.pool(self.activation(self.conv2(x)))
            return x.view(1, -1).size(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x
    
    def get_layer_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate layer outputs for analysis."""
        outputs = {}
        
        outputs['input'] = x.clone()
        
        x = self.conv1(x)
        outputs['conv1'] = x.clone()
        x = self.activation(x)
        outputs['act1'] = x.clone()
        x = self.pool(x)
        outputs['pool1'] = x.clone()
        
        x = self.conv2(x)
        outputs['conv2'] = x.clone()
        x = self.activation(x)
        outputs['act2'] = x.clone()
        x = self.pool(x)
        outputs['pool2'] = x.clone()
        
        x = x.view(x.size(0), -1)
        outputs['flatten'] = x.clone()
        
        x = self.fc1(x)
        outputs['fc1'] = x.clone()
        x = self.activation(x)
        outputs['act3'] = x.clone()
        
        x = self.fc2(x)
        outputs['output'] = x.clone()
        
        return outputs


class PPCM_X_CNN(nn.Module):
    """
    NOVEL: Extended Privacy-Preserving CNN with Adaptive Polynomial Activations.
    
    Key improvements over base PPCM_CNN:
    1. Adaptive polynomial activations (APA) instead of fixed square
    2. HE-friendly batch normalization
    3. Configurable polynomial degrees per layer
    4. Optional learnable activation coefficients
    
    Architecture:
        Conv2d(1, 32, 5) -> HE-BN -> APA -> AvgPool(2)
        Conv2d(32, 64, 5) -> HE-BN -> APA -> AvgPool(2)
        Flatten -> FC(1024, 256) -> APA -> FC(256, 10)
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        input_size: int = 28,
        poly_degree: int = 3,
        adaptive_activation: bool = True,
        use_bn: bool = True,
        learnable_coeffs: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.input_size = input_size
        self.poly_degree = poly_degree
        self.adaptive_activation = adaptive_activation
        self.use_bn = use_bn
        
        # Convolutional layers (wider than base)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        
        # Batch normalization (HE-friendly)
        if use_bn:
            self.bn1 = HEFriendlyBatchNorm(32)
            self.bn2 = HEFriendlyBatchNorm(64)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        
        # Pooling
        self.pool = nn.AvgPool2d(2, 2)
        
        # Activations (adaptive or fixed polynomial)
        if adaptive_activation:
            self.act1 = AdaptivePolyActivation(
                degrees=[2, 3, 4], activation_type="relu",
                learnable_coeffs=learnable_coeffs
            )
            self.act2 = AdaptivePolyActivation(
                degrees=[2, 3, 4], activation_type="relu",
                learnable_coeffs=learnable_coeffs
            )
            self.act3 = AdaptivePolyActivation(
                degrees=[2, 3, 4], activation_type="relu",
                learnable_coeffs=learnable_coeffs
            )
        else:
            self.act1 = PolyReLU(degree=poly_degree, learnable=learnable_coeffs)
            self.act2 = PolyReLU(degree=poly_degree, learnable=learnable_coeffs)
            self.act3 = PolyReLU(degree=poly_degree, learnable=learnable_coeffs)
        
        # Calculate FC input size
        self._fc_input_size = self._calculate_fc_size()
        
        # Fully connected layers (wider than base)
        self.fc1 = nn.Linear(self._fc_input_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _calculate_fc_size(self) -> int:
        """Calculate the flattened size after conv layers."""
        with torch.no_grad():
            x = torch.zeros(1, self.input_channels, self.input_size, self.input_size)
            x = self.conv1(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            return x.view(1, -1).size(1)
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        
        return x
    
    def prepare_for_he(self):
        """Prepare model for HE inference by folding BN parameters."""
        self.eval()
        if self.use_bn:
            self.bn1.fold_parameters()
            self.bn2.fold_parameters()
    
    def get_activation_stats(self, x: torch.Tensor) -> Dict[str, Any]:
        """Get statistics about adaptive activation selections."""
        stats = {}
        
        if self.adaptive_activation:
            # Forward pass to populate stats
            _ = self.forward(x)
            
            stats['layer1_degree'] = self.act1.get_effective_degree(x)
            
            # Get intermediate for layer 2
            with torch.no_grad():
                h = self.pool(self.act1(self.bn1(self.conv1(x))))
                stats['layer2_degree'] = self.act2.get_effective_degree(h)
                
                h = self.pool(self.act2(self.bn2(self.conv2(h))))
                h = h.view(h.size(0), -1)
                h = self.fc1(h)
                stats['layer3_degree'] = self.act3.get_effective_degree(h)
        
        return stats


class PPCM_X_Deep(nn.Module):
    """
    Deeper variant of PPCM-X for more complex datasets (CIFAR-10).
    
    Architecture:
        Conv2d(3, 64, 3) -> HE-BN -> APA
        Conv2d(64, 64, 3) -> HE-BN -> APA -> AvgPool(2)
        Conv2d(64, 128, 3) -> HE-BN -> APA
        Conv2d(128, 128, 3) -> HE-BN -> APA -> AvgPool(2)
        Flatten -> FC(2048, 512) -> APA -> FC(512, 10)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        input_size: int = 32,
        poly_degree: int = 3,
        adaptive_activation: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Block 1
        self.conv1a = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.bn1a = HEFriendlyBatchNorm(64)
        self.conv1b = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1b = HEFriendlyBatchNorm(64)
        
        # Block 2
        self.conv2a = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2a = HEFriendlyBatchNorm(128)
        self.conv2b = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2b = HEFriendlyBatchNorm(128)
        
        self.pool = nn.AvgPool2d(2, 2)
        
        # Activations
        if adaptive_activation:
            self.activations = nn.ModuleList([
                AdaptivePolyActivation(degrees=[2, 3, 4]) for _ in range(5)
            ])
        else:
            self.activations = nn.ModuleList([
                PolyReLU(degree=poly_degree) for _ in range(5)
            ])
        
        # FC layers
        self._fc_size = 128 * (input_size // 4) ** 2
        self.fc1 = nn.Linear(self._fc_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.activations[0](self.bn1a(self.conv1a(x)))
        x = self.activations[1](self.bn1b(self.conv1b(x)))
        x = self.pool(x)
        
        # Block 2
        x = self.activations[2](self.bn2a(self.conv2a(x)))
        x = self.activations[3](self.bn2b(self.conv2b(x)))
        x = self.pool(x)
        
        # FC
        x = x.view(x.size(0), -1)
        x = self.activations[4](self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def prepare_for_he(self):
        """Prepare for HE inference."""
        self.eval()
        for bn in [self.bn1a, self.bn1b, self.bn2a, self.bn2b]:
            bn.fold_parameters()


def get_model(
    model_name: str = "ppcm_x",
    dataset: str = "mnist",
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Model architecture name
        dataset: Dataset name (determines input size/channels)
        **kwargs: Additional model arguments
    
    Returns:
        Model instance
    """
    # Dataset configurations
    dataset_config = {
        "mnist": {"input_channels": 1, "input_size": 28, "num_classes": 10},
        "fashion_mnist": {"input_channels": 1, "input_size": 28, "num_classes": 10},
        "cifar10": {"input_channels": 3, "input_size": 32, "num_classes": 10},
    }
    
    if dataset.lower() not in dataset_config:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    config = dataset_config[dataset.lower()]
    config.update(kwargs)
    
    model_map = {
        "ppcm": PPCM_CNN,
        "ppcm_x": PPCM_X_CNN,
        "ppcm_x_deep": PPCM_X_Deep,
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_map[model_name.lower()](**config)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """Generate model summary string."""
    lines = []
    lines.append(f"Model: {model.__class__.__name__}")
    lines.append(f"Total parameters: {count_parameters(model):,}")
    lines.append(f"Input size: {input_size}")
    lines.append("\nLayers:")
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            params = sum(p.numel() for p in module.parameters())
            lines.append(f"  {name}: {module.__class__.__name__} ({params:,} params)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test models
    print("Testing PPCM-X Models...")
    
    # Test base model
    print("\n--- PPCM (Base) ---")
    model = get_model("ppcm", "mnist")
    x = torch.randn(4, 1, 28, 28)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test extended model
    print("\n--- PPCM-X (Extended) ---")
    model = get_model("ppcm_x", "mnist", adaptive_activation=True)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Get activation stats
    stats = model.get_activation_stats(x)
    print(f"Activation degrees: {stats}")
    
    # Test deep model for CIFAR
    print("\n--- PPCM-X Deep (CIFAR-10) ---")
    model = get_model("ppcm_x_deep", "cifar10")
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {count_parameters(model):,}")
    
    print("\nAll model tests passed!")

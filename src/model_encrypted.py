"""
Encrypted Model Wrapper for PPCM-X

This module provides wrappers to convert trained plaintext models
into HE-compatible encrypted inference engines.

Key Features:
    - Automatic layer conversion to HE operations
    - Support for PPCM and PPCM-X architectures
    - Batched encrypted inference
    - Performance profiling

Reference: Base Paper - Raj et al., IEEE INDIACOM 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import time
import numpy as np

from .he_utils import (
    HEContext, HEParameters, HE_PARAM_PRESETS,
    EncryptedLinear, EncryptedConv2d, EncryptedAvgPool2d,
    encrypted_polynomial_eval, SimulatedEncryptedTensor
)
from .activations_poly import PolynomialCoefficients
from .model_plain import PPCM_CNN, PPCM_X_CNN, PPCM_X_Deep


class EncryptedPPCM:
    """
    Encrypted inference wrapper for PPCM models.
    
    Converts a trained PPCM_CNN or PPCM_X_CNN to encrypted operations.
    """
    
    def __init__(
        self,
        model: nn.Module,
        he_context: Optional[HEContext] = None,
        he_preset: str = "balanced"
    ):
        """
        Args:
            model: Trained plaintext model
            he_context: HE context (creates new if None)
            he_preset: HE parameter preset if creating new context
        """
        self.model = model
        self.model.eval()
        
        # Prepare model for HE (fold BN if applicable)
        if hasattr(model, 'prepare_for_he'):
            model.prepare_for_he()
        
        # Create or use HE context
        self.he_context = he_context or HEContext(preset=he_preset)
        
        # Extract and convert layers
        self._setup_encrypted_layers()
        
        # Profiling data
        self.layer_times = {}
    
    def _setup_encrypted_layers(self):
        """Convert model layers to encrypted operations."""
        self.enc_layers = []
        
        # Determine model type and extract layers
        if isinstance(self.model, PPCM_CNN):
            self._setup_ppcm_layers()
        elif isinstance(self.model, (PPCM_X_CNN, PPCM_X_Deep)):
            self._setup_ppcm_x_layers()
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
    
    def _setup_ppcm_layers(self):
        """Setup layers for base PPCM model."""
        model = self.model
        
        # Conv1
        self.enc_conv1 = EncryptedConv2d(
            model.conv1.weight, model.conv1.bias, stride=1, padding=0
        )
        
        # Conv2
        self.enc_conv2 = EncryptedConv2d(
            model.conv2.weight, model.conv2.bias, stride=1, padding=0
        )
        
        # Pooling
        self.enc_pool = EncryptedAvgPool2d(kernel_size=2)
        
        # FC layers
        self.enc_fc1 = EncryptedLinear(model.fc1.weight, model.fc1.bias)
        self.enc_fc2 = EncryptedLinear(model.fc2.weight, model.fc2.bias)
        
        # Activation coefficients (square: x^2)
        self.activation_coeffs = [0.0, 0.0, 1.0]
    
    def _setup_ppcm_x_layers(self):
        """Setup layers for PPCM-X model."""
        model = self.model
        
        # Conv layers
        self.enc_conv1 = EncryptedConv2d(
            model.conv1.weight, model.conv1.bias, stride=1, padding=0
        )
        self.enc_conv2 = EncryptedConv2d(
            model.conv2.weight, model.conv2.bias, stride=1, padding=0
        )
        
        # Pooling
        self.enc_pool = EncryptedAvgPool2d(kernel_size=2)
        
        # FC layers
        self.enc_fc1 = EncryptedLinear(model.fc1.weight, model.fc1.bias)
        self.enc_fc2 = EncryptedLinear(model.fc2.weight, model.fc2.bias)
        
        # Get activation coefficients
        self._extract_activation_coeffs()
    
    def _extract_activation_coeffs(self):
        """Extract polynomial coefficients from adaptive activations."""
        model = self.model
        
        self.activation_coeffs_list = []
        
        for act in [model.act1, model.act2, model.act3]:
            if hasattr(act, 'coeffs'):
                # Fixed polynomial
                coeffs = act.coeffs.detach().cpu().tolist()
            elif hasattr(act, 'poly_modules'):
                # Adaptive - use degree 3 as default for HE
                if 'deg_3' in act.poly_modules:
                    coeffs = act.poly_modules['deg_3'].detach().cpu().tolist()
                else:
                    coeffs = PolynomialCoefficients.RELU[3].tolist()
            else:
                # Default to degree 3 ReLU approximation
                coeffs = PolynomialCoefficients.RELU[3].tolist()
            
            self.activation_coeffs_list.append(coeffs)
    
    def encrypt_input(self, x: torch.Tensor) -> Any:
        """Encrypt input tensor."""
        return self.he_context.encrypt_tensor(x)
    
    def decrypt_output(self, enc_output: Any, shape: Tuple = None) -> torch.Tensor:
        """Decrypt output tensor."""
        return self.he_context.decrypt_tensor(enc_output, shape)
    
    def forward_encrypted(
        self,
        encrypted_input: Any,
        input_shape: Tuple[int, int, int]
    ) -> Any:
        """
        Run encrypted forward pass.
        
        Args:
            encrypted_input: Encrypted input tensor
            input_shape: Original input shape (C, H, W)
        
        Returns:
            Encrypted output (logits)
        """
        self.layer_times = {}
        
        # Conv1
        start = time.time()
        x, shape = self.enc_conv1(encrypted_input, input_shape)
        self.layer_times['conv1'] = time.time() - start
        
        # Activation 1
        start = time.time()
        coeffs = self.activation_coeffs_list[0] if hasattr(self, 'activation_coeffs_list') else self.activation_coeffs
        x = encrypted_polynomial_eval(x, coeffs)
        self.layer_times['act1'] = time.time() - start
        
        # Pool 1
        start = time.time()
        x, shape = self.enc_pool(x, shape)
        self.layer_times['pool1'] = time.time() - start
        
        # Conv2
        start = time.time()
        x, shape = self.enc_conv2(x, shape)
        self.layer_times['conv2'] = time.time() - start
        
        # Activation 2
        start = time.time()
        coeffs = self.activation_coeffs_list[1] if hasattr(self, 'activation_coeffs_list') else self.activation_coeffs
        x = encrypted_polynomial_eval(x, coeffs)
        self.layer_times['act2'] = time.time() - start
        
        # Pool 2
        start = time.time()
        x, shape = self.enc_pool(x, shape)
        self.layer_times['pool2'] = time.time() - start
        
        # Flatten (implicit in encrypted domain)
        
        # FC1
        start = time.time()
        x = self.enc_fc1(x)
        self.layer_times['fc1'] = time.time() - start
        
        # Activation 3
        start = time.time()
        coeffs = self.activation_coeffs_list[2] if hasattr(self, 'activation_coeffs_list') else self.activation_coeffs
        x = encrypted_polynomial_eval(x, coeffs)
        self.layer_times['act3'] = time.time() - start
        
        # FC2
        start = time.time()
        x = self.enc_fc2(x)
        self.layer_times['fc2'] = time.time() - start
        
        return x
    
    def __call__(
        self,
        x: torch.Tensor,
        return_encrypted: bool = False
    ) -> torch.Tensor:
        """
        Run inference on input tensor.
        
        Args:
            x: Input tensor [N, C, H, W] or [C, H, W]
        
        Returns:
            Output logits
        """
        # Handle batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        results = []
        
        for i in range(batch_size):
            # Get single sample
            sample = x[i]
            input_shape = tuple(sample.shape)
            
            # Encrypt
            enc_input = self.encrypt_input(sample)
            
            # Forward pass
            enc_output = self.forward_encrypted(enc_input, input_shape)
            
            if return_encrypted:
                results.append(enc_output)
            else:
                # Decrypt
                dec_output = self.decrypt_output(enc_output)
                results.append(dec_output)
        
        if return_encrypted:
            return results
        
        return torch.stack(results)
    
    def get_timing_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown of last forward pass."""
        total = sum(self.layer_times.values())
        breakdown = {k: v for k, v in self.layer_times.items()}
        breakdown['total'] = total
        return breakdown
    
    def profile(self, x: torch.Tensor, num_runs: int = 5) -> Dict[str, Any]:
        """
        Profile encrypted inference.
        
        Args:
            x: Sample input
            num_runs: Number of profiling runs
        
        Returns:
            Profiling results
        """
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            _ = self(x)
            times.append(time.time() - start)
        
        return {
            'mean_time_ms': np.mean(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'layer_breakdown': self.get_timing_breakdown(),
        }


class HybridEncryptedModel:
    """
    Hybrid model that can switch between plaintext and encrypted inference.
    
    Useful for:
    - Debugging and validation
    - Performance comparison
    - Gradual migration to encrypted inference
    """
    
    def __init__(
        self,
        model: nn.Module,
        he_context: Optional[HEContext] = None
    ):
        self.plain_model = model
        self.plain_model.eval()
        
        self.encrypted_model = EncryptedPPCM(model, he_context)
        self.mode = "plain"
    
    def set_mode(self, mode: str):
        """Set inference mode ('plain' or 'encrypted')."""
        if mode not in ["plain", "encrypted"]:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "plain":
            with torch.no_grad():
                return self.plain_model(x)
        else:
            return self.encrypted_model(x)
    
    def compare_outputs(self, x: torch.Tensor) -> Dict[str, Any]:
        """Compare plaintext and encrypted outputs."""
        with torch.no_grad():
            plain_out = self.plain_model(x)
        
        enc_out = self.encrypted_model(x)
        
        # Compute differences
        diff = plain_out - enc_out
        
        return {
            'plain_output': plain_out,
            'encrypted_output': enc_out,
            'mse': torch.mean(diff ** 2).item(),
            'max_abs_diff': torch.max(torch.abs(diff)).item(),
            'plain_pred': plain_out.argmax(dim=-1).tolist(),
            'encrypted_pred': enc_out.argmax(dim=-1).tolist(),
            'predictions_match': (plain_out.argmax(dim=-1) == enc_out.argmax(dim=-1)).all().item(),
        }


def convert_to_encrypted(
    model: nn.Module,
    he_preset: str = "balanced"
) -> EncryptedPPCM:
    """
    Convert a trained model to encrypted inference.
    
    Args:
        model: Trained PyTorch model
        he_preset: HE parameter preset
    
    Returns:
        EncryptedPPCM wrapper
    """
    return EncryptedPPCM(model, he_preset=he_preset)


def validate_encrypted_model(
    plain_model: nn.Module,
    test_loader,
    num_samples: int = 100,
    he_preset: str = "balanced"
) -> Dict[str, Any]:
    """
    Validate encrypted model against plaintext.
    
    Args:
        plain_model: Trained plaintext model
        test_loader: Test data loader
        num_samples: Number of samples to validate
        he_preset: HE parameter preset
    
    Returns:
        Validation results
    """
    encrypted_model = EncryptedPPCM(plain_model, he_preset=he_preset)
    hybrid = HybridEncryptedModel(plain_model)
    
    results = {
        'total_samples': 0,
        'matching_predictions': 0,
        'mse_values': [],
        'plain_correct': 0,
        'encrypted_correct': 0,
    }
    
    sample_count = 0
    for inputs, labels in test_loader:
        for i in range(inputs.size(0)):
            if sample_count >= num_samples:
                break
            
            x = inputs[i:i+1]
            y = labels[i].item()
            
            comparison = hybrid.compare_outputs(x)
            
            results['total_samples'] += 1
            results['mse_values'].append(comparison['mse'])
            
            if comparison['predictions_match']:
                results['matching_predictions'] += 1
            
            if comparison['plain_pred'][0] == y:
                results['plain_correct'] += 1
            
            if comparison['encrypted_pred'][0] == y:
                results['encrypted_correct'] += 1
            
            sample_count += 1
        
        if sample_count >= num_samples:
            break
    
    # Compute summary statistics
    results['prediction_match_rate'] = results['matching_predictions'] / results['total_samples']
    results['mean_mse'] = np.mean(results['mse_values'])
    results['plain_accuracy'] = results['plain_correct'] / results['total_samples']
    results['encrypted_accuracy'] = results['encrypted_correct'] / results['total_samples']
    
    return results


if __name__ == "__main__":
    from .model_plain import get_model
    from .data_loader import get_sample_batch
    
    print("Testing Encrypted Model Wrapper...")
    
    # Create and test model
    model = get_model("ppcm_x", "mnist", adaptive_activation=True)
    
    # Get sample input
    x, y = get_sample_batch("mnist", batch_size=2)
    
    # Test plaintext forward
    print("\n--- Plaintext Forward ---")
    with torch.no_grad():
        plain_out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {plain_out.shape}")
    print(f"Predictions: {plain_out.argmax(dim=-1).tolist()}")
    
    # Test encrypted forward
    print("\n--- Encrypted Forward ---")
    encrypted_model = EncryptedPPCM(model, he_preset="fast")
    enc_out = encrypted_model(x)
    print(f"Output shape: {enc_out.shape}")
    print(f"Predictions: {enc_out.argmax(dim=-1).tolist()}")
    
    # Compare outputs
    print("\n--- Comparison ---")
    hybrid = HybridEncryptedModel(model)
    comparison = hybrid.compare_outputs(x[:1])
    print(f"MSE: {comparison['mse']:.6f}")
    print(f"Max diff: {comparison['max_abs_diff']:.6f}")
    print(f"Predictions match: {comparison['predictions_match']}")
    
    # Timing
    print("\n--- Timing ---")
    timing = encrypted_model.get_timing_breakdown()
    for layer, t in timing.items():
        print(f"  {layer}: {t*1000:.2f} ms")
    
    print("\nEncrypted model tests passed!")

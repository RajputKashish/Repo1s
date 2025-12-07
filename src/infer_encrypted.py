"""
Encrypted Inference Engine for PPCM-X

This module provides the complete encrypted inference pipeline including:
    - Model loading and conversion
    - Encrypted inference execution
    - Performance benchmarking
    - Accuracy validation

Reference: Base Paper - Raj et al., IEEE INDIACOM 2025
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
import os
import json
import time
import argparse
from datetime import datetime
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_data_loaders, get_sample_batch, get_dataset_info
from src.model_plain import get_model, count_parameters
from src.model_encrypted import EncryptedPPCM, HybridEncryptedModel, validate_encrypted_model
from src.he_utils import HEContext, HE_PARAM_PRESETS, benchmark_he_operations


class EncryptedInferenceEngine:
    """
    Complete encrypted inference engine with benchmarking capabilities.
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = "ppcm_x",
        dataset: str = "mnist",
        he_preset: str = "balanced",
        device: str = "cpu"
    ):
        """
        Args:
            model_path: Path to trained model weights
            model_type: Model architecture type
            dataset: Dataset name (for model configuration)
            he_preset: HE parameter preset
            device: Device for plaintext operations
        """
        self.model_path = model_path
        self.model_type = model_type
        self.dataset = dataset
        self.he_preset = he_preset
        self.device = torch.device(device)
        
        # Load model
        self._load_model()
        
        # Setup HE context
        self.he_context = HEContext(preset=he_preset)
        
        # Create encrypted model wrapper
        self.encrypted_model = EncryptedPPCM(
            self.plain_model,
            he_context=self.he_context
        )
        
        # Statistics
        self.inference_times = []
        self.accuracy_results = {}
    
    def _load_model(self):
        """Load the trained plaintext model."""
        # Create model architecture
        adaptive = self.model_type in ["ppcm_x", "ppcm_x_deep"]
        self.plain_model = get_model(
            self.model_type, self.dataset,
            adaptive_activation=adaptive
        )
        
        # Load weights
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Handle checkpoint format
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            self.plain_model.load_state_dict(state_dict)
            print(f"Loaded model from: {self.model_path}")
        else:
            print(f"Warning: Model file not found at {self.model_path}")
            print("Using randomly initialized model for demonstration.")
        
        self.plain_model.eval()
        self.plain_model.to(self.device)
    
    def infer_single(
        self,
        x: torch.Tensor,
        return_timing: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        Run encrypted inference on a single sample.
        
        Args:
            x: Input tensor [C, H, W] or [1, C, H, W]
            return_timing: Whether to return timing breakdown
        
        Returns:
            Tuple of (predictions, timing_dict)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        start_time = time.time()
        output = self.encrypted_model(x)
        total_time = time.time() - start_time
        
        self.inference_times.append(total_time)
        
        predictions = output.argmax(dim=-1)
        
        if return_timing:
            timing = self.encrypted_model.get_timing_breakdown()
            timing['total'] = total_time
            return predictions, timing
        
        return predictions, None
    
    def infer_batch(
        self,
        loader,
        num_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run encrypted inference on a batch of samples.
        
        Args:
            loader: Data loader
            num_samples: Maximum number of samples (None for all)
            verbose: Print progress
        
        Returns:
            Results dictionary
        """
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_times = []
        
        iterator = tqdm(loader, desc="Encrypted Inference") if verbose else loader
        
        for inputs, labels in iterator:
            for i in range(inputs.size(0)):
                if num_samples is not None and total >= num_samples:
                    break
                
                x = inputs[i]
                y = labels[i].item()
                
                pred, timing = self.infer_single(x, return_timing=True)
                pred = pred.item()
                
                all_predictions.append(pred)
                all_labels.append(y)
                all_times.append(timing['total'])
                
                if pred == y:
                    correct += 1
                total += 1
            
            if num_samples is not None and total >= num_samples:
                break
        
        accuracy = correct / total if total > 0 else 0
        
        results = {
            'total_samples': total,
            'correct': correct,
            'accuracy': accuracy,
            'mean_inference_time_ms': np.mean(all_times) * 1000,
            'std_inference_time_ms': np.std(all_times) * 1000,
            'min_inference_time_ms': np.min(all_times) * 1000,
            'max_inference_time_ms': np.max(all_times) * 1000,
            'predictions': all_predictions,
            'labels': all_labels,
        }
        
        return results
    
    def compare_with_plaintext(
        self,
        loader,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compare encrypted inference with plaintext.
        
        Args:
            loader: Data loader
            num_samples: Number of samples to compare
        
        Returns:
            Comparison results
        """
        hybrid = HybridEncryptedModel(self.plain_model, self.he_context)
        
        matching = 0
        plain_correct = 0
        enc_correct = 0
        total = 0
        mse_values = []
        
        for inputs, labels in tqdm(loader, desc="Comparing"):
            for i in range(inputs.size(0)):
                if total >= num_samples:
                    break
                
                x = inputs[i:i+1]
                y = labels[i].item()
                
                comparison = hybrid.compare_outputs(x)
                
                if comparison['predictions_match']:
                    matching += 1
                
                if comparison['plain_pred'][0] == y:
                    plain_correct += 1
                
                if comparison['encrypted_pred'][0] == y:
                    enc_correct += 1
                
                mse_values.append(comparison['mse'])
                total += 1
            
            if total >= num_samples:
                break
        
        return {
            'total_samples': total,
            'prediction_match_rate': matching / total,
            'plaintext_accuracy': plain_correct / total,
            'encrypted_accuracy': enc_correct / total,
            'mean_output_mse': np.mean(mse_values),
            'max_output_mse': np.max(mse_values),
        }
    
    def benchmark(self, num_iterations: int = 10) -> Dict[str, Any]:
        """
        Comprehensive benchmark of encrypted inference.
        
        Args:
            num_iterations: Number of benchmark iterations
        
        Returns:
            Benchmark results
        """
        # Get sample input
        x, _ = get_sample_batch(self.dataset, batch_size=1)
        
        # Warmup
        for _ in range(3):
            _ = self.encrypted_model(x)
        
        # Benchmark inference
        inference_times = []
        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            start = time.time()
            _ = self.encrypted_model(x)
            inference_times.append(time.time() - start)
        
        # Benchmark HE operations
        he_benchmarks = benchmark_he_operations(
            self.he_context,
            vector_size=784,  # MNIST flattened size
            num_iterations=num_iterations
        )
        
        # Get layer breakdown
        _ = self.encrypted_model(x)
        layer_breakdown = self.encrypted_model.get_timing_breakdown()
        
        return {
            'inference': {
                'mean_ms': np.mean(inference_times) * 1000,
                'std_ms': np.std(inference_times) * 1000,
                'min_ms': np.min(inference_times) * 1000,
                'max_ms': np.max(inference_times) * 1000,
            },
            'he_operations': he_benchmarks,
            'layer_breakdown_ms': {k: v * 1000 for k, v in layer_breakdown.items()},
            'he_context': self.he_context.get_context_info(),
        }
    
    def generate_report(
        self,
        test_loader,
        num_test_samples: int = 100,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive inference report.
        
        Args:
            test_loader: Test data loader
            num_test_samples: Number of test samples
            output_path: Path to save report (optional)
        
        Returns:
            Complete report dictionary
        """
        print("\n" + "="*60)
        print("PPCM-X Encrypted Inference Report")
        print("="*60)
        
        # Model info
        print("\n[1/4] Model Information...")
        model_info = {
            'model_type': self.model_type,
            'dataset': self.dataset,
            'num_parameters': count_parameters(self.plain_model),
            'he_preset': self.he_preset,
        }
        
        # Accuracy evaluation
        print("\n[2/4] Evaluating Accuracy...")
        accuracy_results = self.infer_batch(
            test_loader,
            num_samples=num_test_samples,
            verbose=True
        )
        
        # Comparison with plaintext
        print("\n[3/4] Comparing with Plaintext...")
        comparison_results = self.compare_with_plaintext(
            test_loader,
            num_samples=min(50, num_test_samples)
        )
        
        # Benchmarking
        print("\n[4/4] Running Benchmarks...")
        benchmark_results = self.benchmark(num_iterations=10)
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': model_info,
            'accuracy': {
                'encrypted_accuracy': accuracy_results['accuracy'],
                'total_samples': accuracy_results['total_samples'],
            },
            'comparison': comparison_results,
            'performance': {
                'mean_inference_time_ms': accuracy_results['mean_inference_time_ms'],
                'std_inference_time_ms': accuracy_results['std_inference_time_ms'],
            },
            'benchmarks': benchmark_results,
        }
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Model: {self.model_type} ({model_info['num_parameters']:,} params)")
        print(f"Dataset: {self.dataset}")
        print(f"HE Preset: {self.he_preset}")
        print(f"\nEncrypted Accuracy: {accuracy_results['accuracy']*100:.2f}%")
        print(f"Plaintext Accuracy: {comparison_results['plaintext_accuracy']*100:.2f}%")
        print(f"Prediction Match Rate: {comparison_results['prediction_match_rate']*100:.2f}%")
        print(f"\nMean Inference Time: {accuracy_results['mean_inference_time_ms']:.2f} ms")
        print(f"Output MSE (vs plaintext): {comparison_results['mean_output_mse']:.6f}")
        print("="*60)
        
        # Save report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nReport saved to: {output_path}")
        
        return report


def run_encrypted_inference(
    model_path: str,
    dataset: str = "mnist",
    model_type: str = "ppcm_x",
    he_preset: str = "balanced",
    num_samples: int = 100,
    output_dir: str = "./experiments"
) -> Dict[str, Any]:
    """
    High-level function to run encrypted inference.
    
    Args:
        model_path: Path to trained model
        dataset: Dataset name
        model_type: Model architecture
        he_preset: HE parameter preset
        num_samples: Number of test samples
        output_dir: Output directory
    
    Returns:
        Inference results
    """
    # Create engine
    engine = EncryptedInferenceEngine(
        model_path=model_path,
        model_type=model_type,
        dataset=dataset,
        he_preset=he_preset
    )
    
    # Get test loader
    _, _, test_loader = get_data_loaders(
        dataset_name=dataset,
        batch_size=1,
        num_workers=0
    )
    
    # Generate report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(
        output_dir,
        f"encrypted_inference_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    report = engine.generate_report(
        test_loader,
        num_test_samples=num_samples,
        output_path=report_path
    )
    
    return report


def main():
    """Main entry point for encrypted inference."""
    parser = argparse.ArgumentParser(description="PPCM-X Encrypted Inference")
    
    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model weights")
    parser.add_argument("--model_type", type=str, default="ppcm_x",
                       choices=["ppcm", "ppcm_x", "ppcm_x_deep"],
                       help="Model architecture type")
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar10", "fashion_mnist"],
                       help="Dataset name")
    parser.add_argument("--he_preset", type=str, default="balanced",
                       choices=["fast", "balanced", "accurate"],
                       help="HE parameter preset")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="Output directory")
    
    args = parser.parse_args()
    
    results = run_encrypted_inference(
        model_path=args.model,
        dataset=args.dataset,
        model_type=args.model_type,
        he_preset=args.he_preset,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    return results


if __name__ == "__main__":
    main()

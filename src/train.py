"""
Training Pipeline for PPCM-X Models

This module provides comprehensive training functionality including:
    - Standard training loop with validation
    - Learning rate scheduling
    - Early stopping
    - Checkpoint management
    - Training visualization

Reference: Base Paper - Raj et al., IEEE INDIACOM 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple, Callable
import os
import json
import time
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np

# Import local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_data_loaders, get_dataset_info
from src.model_plain import get_model, count_parameters, PPCM_CNN, PPCM_X_CNN


class Trainer:
    """
    Comprehensive trainer for PPCM models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "auto",
        output_dir: str = "./experiments",
        experiment_name: Optional[str] = None
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to use ('auto', 'cuda', 'cpu')
            output_dir: Directory for outputs
            experiment_name: Name for this experiment
        """
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup output directory
        self.output_dir = output_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
        }
        
        # Default training components (can be overridden)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
    
    def setup_training(
        self,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adam",
        scheduler_type: str = "cosine",
        epochs: int = 20
    ):
        """
        Setup optimizer and scheduler.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay for regularization
            optimizer_type: Optimizer type ('adam', 'sgd', 'adamw')
            scheduler_type: Scheduler type ('cosine', 'step', 'none')
            epochs: Total epochs (for scheduler)
        """
        # Setup optimizer
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Setup scheduler
        if scheduler_type.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        elif scheduler_type.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        elif scheduler_type.lower() == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "Eval") -> Tuple[float, float]:
        """
        Evaluate model on a data loader.
        
        Args:
            loader: Data loader to evaluate on
            desc: Description for progress bar
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(loader, desc=desc):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        epochs: int = 20,
        lr: float = 0.001,
        early_stopping_patience: int = 5,
        save_best: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            epochs: Number of epochs
            lr: Learning rate
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            **kwargs: Additional arguments for setup_training
        
        Returns:
            Training results dictionary
        """
        # Setup training
        self.setup_training(lr=lr, epochs=epochs, **kwargs)
        
        # Training loop
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.evaluate(self.val_loader, "Validation")
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            print(f"  LR: {current_lr:.6f}")
            
            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                patience_counter = 0
                
                if save_best:
                    self.save_checkpoint("best_model.pt")
                    print(f"  New best model saved! (Val Acc: {val_acc*100:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Final evaluation on test set
        test_loss, test_acc = self.evaluate(self.test_loader, "Test")
        
        training_time = time.time() - start_time
        
        # Compile results
        results = {
            'model_name': self.model.__class__.__name__,
            'num_parameters': count_parameters(self.model),
            'epochs_trained': self.epoch + 1,
            'best_val_acc': self.best_val_acc,
            'final_test_acc': test_acc,
            'final_test_loss': test_loss,
            'training_time_seconds': training_time,
            'history': self.history,
            'device': str(self.device),
        }
        
        # Save results
        self.save_results(results)
        
        print(f"\n{'='*50}")
        print(f"Training Complete!")
        print(f"Best Val Acc: {self.best_val_acc*100:.2f}%")
        print(f"Test Acc: {test_acc*100:.2f}%")
        print(f"Training Time: {training_time:.1f}s")
        print(f"{'='*50}")
        
        return results
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.exp_dir, "checkpoints", filename)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.exp_dir, "checkpoints", filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results = convert(results)
        
        path = os.path.join(self.exp_dir, "results.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)


def train_model(
    model_name: str = "ppcm_x",
    dataset: str = "mnist",
    mode: str = "he_compatible",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
    poly_degree: int = 3,
    adaptive: bool = True,
    output_dir: str = "./experiments",
    **kwargs
) -> Dict[str, Any]:
    """
    High-level training function.
    
    Args:
        model_name: Model architecture name
        dataset: Dataset name
        mode: Training mode ('plain', 'he_compatible')
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        poly_degree: Polynomial degree for activations
        adaptive: Use adaptive activations
        output_dir: Output directory
    
    Returns:
        Training results
    """
    print(f"\n{'='*60}")
    print(f"PPCM-X Training Pipeline")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Mode: {mode}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    if mode == "he_compatible":
        print(f"Polynomial Degree: {'adaptive' if adaptive else poly_degree}")
    print(f"{'='*60}\n")
    
    # Get data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=dataset,
        batch_size=batch_size,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    # Create model
    if mode == "plain":
        model = get_model("ppcm", dataset)
    else:
        model = get_model(
            model_name, dataset,
            poly_degree=poly_degree,
            adaptive_activation=adaptive
        )
    
    print(f"Model Parameters: {count_parameters(model):,}")
    
    # Create trainer
    exp_name = f"{model_name}_{dataset}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        output_dir=output_dir,
        experiment_name=exp_name
    )
    
    # Train
    results = trainer.train(
        epochs=epochs,
        lr=lr,
        early_stopping_patience=10,
        **kwargs
    )
    
    # Save final model
    final_path = os.path.join(output_dir, f"{mode}_best.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")
    
    return results


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="PPCM-X Training Pipeline")
    
    parser.add_argument("--mode", type=str, default="he_compatible",
                       choices=["plain", "he_compatible"],
                       help="Training mode")
    parser.add_argument("--model", type=str, default="ppcm_x",
                       choices=["ppcm", "ppcm_x", "ppcm_x_deep"],
                       help="Model architecture")
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "cifar10", "fashion_mnist"],
                       help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--poly_degree", type=str, default="adaptive",
                       help="Polynomial degree (2, 3, 4, or 'adaptive')")
    parser.add_argument("--output_dir", type=str, default="./experiments",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Parse polynomial degree
    adaptive = args.poly_degree.lower() == "adaptive"
    poly_degree = 3 if adaptive else int(args.poly_degree)
    
    # Run training
    results = train_model(
        model_name=args.model,
        dataset=args.dataset,
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        poly_degree=poly_degree,
        adaptive=adaptive,
        output_dir=args.output_dir
    )
    
    return results


if __name__ == "__main__":
    main()

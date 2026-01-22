"""Training utilities for face recognition models."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.device import get_device
from ..utils.config import DictConfig
from .metrics import evaluate_one_shot_accuracy, evaluate_siamese_accuracy


class Trainer:
    """Trainer class for face recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.device = device or get_device()
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_criterion()
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        # Create save directory
        self.save_dir = Path(config.logging.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.epochs,
            )
        elif self.config.training.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.epochs // 3,
                gamma=0.1,
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        if hasattr(self.model, "criterion"):
            return self.model.criterion
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if hasattr(self.model, "forward_one"):
                    # For Siamese networks
                    embeddings = self.model.forward_one(images)
                    loss = self.criterion(embeddings, labels)
                else:
                    # For standard models
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
            elif isinstance(batch, dict):
                # For one-shot learning
                support_images = batch["support_images"].to(self.device)
                support_labels = batch["support_labels"].to(self.device)
                query_images = batch["query_images"].to(self.device)
                query_labels = batch["query_labels"].to(self.device)
                
                # Forward pass
                if hasattr(self.model, "forward_one"):
                    support_embeddings = self.model.forward_one(support_images)
                    query_embeddings = self.model.forward_one(query_images)
                    
                    # Compute distances and loss
                    distances = torch.cdist(query_embeddings, support_embeddings)
                    loss = self.criterion(distances, query_labels)
                else:
                    outputs = self.model(support_images, query_images)
                    loss = self.criterion(outputs, query_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.
        
        Args:
            val_loader: Validation data loader.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        
        if hasattr(self.model, "forward_one"):
            # One-shot learning evaluation
            return self._validate_one_shot(val_loader)
        else:
            # Standard classification evaluation
            return self._validate_classification(val_loader)
    
    def _validate_one_shot(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate one-shot learning model."""
        total_metrics = {}
        num_episodes = 0
        
        with torch.no_grad():
            for batch in val_loader:
                support_images = batch["support_images"].to(self.device)
                support_labels = batch["support_labels"].to(self.device)
                query_images = batch["query_images"].to(self.device)
                query_labels = batch["query_labels"].to(self.device)
                
                metrics = evaluate_one_shot_accuracy(
                    self.model,
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    self.device,
                )
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
                num_episodes += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_episodes
        
        return total_metrics
    
    def _validate_classification(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate classification model."""
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        
        return {"accuracy": accuracy, "loss": avg_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Number of epochs to train.
            
        Returns:
            Training history.
        """
        num_epochs = num_epochs or self.config.training.epochs
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_metrics = self.validate(val_loader)
            val_accuracy = val_metrics.get("accuracy", 0.0)
            self.val_accuracies.append(val_accuracy)
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 50)
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.save_checkpoint("best_model.pth")
            
            # Save checkpoint
            if (epoch + 1) % self.config.logging.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        
        return {
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint.
        
        Args:
            filename: Checkpoint filename.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_accuracy": self.best_accuracy,
            "config": self.config,
        }
        
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]
        
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, best accuracy: {self.best_accuracy:.4f}")

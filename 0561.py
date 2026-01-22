"""
Project 561: Face Recognition with One-Shot Learning - Modern PyTorch Implementation

This is a modernized version of the original Keras/TensorFlow implementation.
The project has been completely refactored to use PyTorch 2.x with modern best practices.

Key improvements:
- PyTorch 2.x implementation with mixed precision training
- Automatic device detection (CUDA → MPS → CPU)
- Type hints and comprehensive documentation
- Modern project structure with proper testing
- Interactive Streamlit demo
- Production-ready configuration management

For the full implementation, see the src/ directory and run:
    python scripts/train.py --data_dir data/raw
    streamlit run demo/app.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ModernSiameseNetwork(nn.Module):
    """Modern Siamese network implementation using PyTorch."""
    
    def __init__(self, embedding_dim: int = 512):
        """Initialize Siamese network.
        
        Args:
            embedding_dim: Dimension of embedding vectors.
        """
        super().__init__()
        
        # Base network (simplified CNN)
        self.base_network = nn.Sequential(
            nn.Conv2d(3, 64, 10, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
        )
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single input.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Normalized embedding tensor.
        """
        embedding = self.base_network(x)
        return F.normalize(embedding, p=2, dim=1)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for pair of inputs.
        
        Args:
            x1: First input tensor.
            x2: Second input tensor.
            
        Returns:
            Tuple of embeddings for both inputs.
        """
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        return embedding1, embedding2


def euclidean_distance(embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance between embeddings.
    
    Args:
        embedding1: First embedding batch.
        embedding2: Second embedding batch.
        
    Returns:
        Distance tensor.
    """
    return torch.norm(embedding1 - embedding2, dim=1)


def main():
    """Example usage of the modern Siamese network."""
    print("Modern Face Recognition with One-Shot Learning")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = ModernSiameseNetwork(embedding_dim=512).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data (replace with actual face images)
    batch_size = 2
    input_shape = (3, 105, 105)
    
    image_a = torch.randn(batch_size, *input_shape).to(device)
    image_b = torch.randn(batch_size, *input_shape).to(device)
    
    print(f"Input shape: {input_shape}")
    print(f"Batch size: {batch_size}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        embedding_a, embedding_b = model(image_a, image_b)
        
        # Compute similarity (lower distance = higher similarity)
        distances = euclidean_distance(embedding_a, embedding_b)
        
        print(f"\nEmbedding dimensions: {embedding_a.shape}")
        print(f"Distances between face pairs:")
        for i, dist in enumerate(distances):
            similarity = max(0, 1 - dist.item())  # Convert distance to similarity
            print(f"  Pair {i+1}: Distance={dist.item():.4f}, Similarity={similarity:.4f}")
    
    print("\nFor full training and evaluation, run:")
    print("  python scripts/train.py --data_dir data/raw")
    print("  streamlit run demo/app.py")


if __name__ == "__main__":
    main()

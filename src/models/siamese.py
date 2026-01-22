"""Siamese network architectures for one-shot face recognition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    import timm
except ImportError:
    timm = None


class SiameseNetwork(nn.Module):
    """Siamese network for face recognition."""
    
    def __init__(
        self, 
        backbone: str = "resnet18", 
        embedding_dim: int = 512,
        dropout: float = 0.3,
        pretrained: bool = True
    ):
        """Initialize Siamese network.
        
        Args:
            backbone: Backbone architecture name.
            embedding_dim: Dimension of embedding vectors.
            dropout: Dropout rate.
            pretrained: Whether to use pretrained weights.
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Create backbone
        if timm and backbone in timm.list_models():
            self.backbone = timm.create_model(
                backbone, 
                pretrained=pretrained, 
                num_classes=0  # Remove classifier
            )
            backbone_dim = self.backbone.num_features
        else:
            # Fallback to custom CNN
            self.backbone = self._create_custom_backbone()
            backbone_dim = 512
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )
        
        # L2 normalization
        self.normalize = True
    
    def _create_custom_backbone(self) -> nn.Module:
        """Create custom CNN backbone."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single input.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        features = self.backbone(x)
        embedding = self.projection(features)
        
        if self.normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding
    
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


class ContrastiveLoss(nn.Module):
    """Contrastive loss for Siamese networks."""
    
    def __init__(self, margin: float = 1.0):
        """Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs.
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            embedding1: First embedding batch.
            embedding2: Second embedding batch.
            labels: Binary labels (1 for same, 0 for different).
            
        Returns:
            Contrastive loss.
        """
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        
        positive_loss = labels * torch.pow(euclidean_distance, 2)
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(positive_loss + negative_loss)
        return loss


class TripletLoss(nn.Module):
    """Triplet loss for face recognition."""
    
    def __init__(self, margin: float = 0.5):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss.
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings.
            positive: Positive embeddings.
            negative: Negative embeddings.
            
        Returns:
            Triplet loss.
        """
        pos_dist = F.pairwise_distance(anchor, positive)
        neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return torch.mean(loss)


class ProtoNet(nn.Module):
    """Prototypical network for few-shot learning."""
    
    def __init__(self, backbone: str = "resnet18", embedding_dim: int = 512):
        """Initialize ProtoNet.
        
        Args:
            backbone: Backbone architecture.
            embedding_dim: Embedding dimension.
        """
        super().__init__()
        self.siamese = SiameseNetwork(backbone, embedding_dim)
    
    def forward(self, support: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Forward pass for prototypical network.
        
        Args:
            support: Support set embeddings.
            query: Query embeddings.
            
        Returns:
            Distance matrix between query and prototypes.
        """
        # Compute prototypes (mean of support embeddings)
        prototypes = torch.mean(support, dim=1)  # (n_way, embedding_dim)
        
        # Compute distances between query and prototypes
        distances = torch.cdist(query, prototypes)  # (n_query, n_way)
        
        return distances

"""CLIP-based face recognition models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


class CLIPFaceRecognition(nn.Module):
    """CLIP-based face recognition model."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP face recognition model.
        
        Args:
            model_name: CLIP model name.
        """
        super().__init__()
        
        if CLIPModel is None:
            raise ImportError("transformers library is required for CLIP models")
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP vision encoder.
        
        Args:
            images: Input images.
            
        Returns:
            Image embeddings.
        """
        return self.clip_model.get_image_features(images)
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text using CLIP text encoder.
        
        Args:
            text: Input text.
            
        Returns:
            Text embeddings.
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        return self.clip_model.get_text_features(**inputs)
    
    def forward(self, images: torch.Tensor, text: Optional[str] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: Input images.
            text: Optional text description.
            
        Returns:
            Image embeddings.
        """
        return self.encode_image(images)


class CLIPSiameseNetwork(nn.Module):
    """Siamese network using CLIP embeddings."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """Initialize CLIP Siamese network.
        
        Args:
            model_name: CLIP model name.
        """
        super().__init__()
        
        self.clip_model = CLIPFaceRecognition(model_name)
        
        # Additional projection layers
        clip_dim = self.clip_model.clip_model.config.vision_config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
        )
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for single input.
        
        Args:
            x: Input tensor.
            
        Returns:
            Embedding tensor.
        """
        clip_embedding = self.clip_model.encode_image(x)
        embedding = self.projection(clip_embedding)
        return F.normalize(embedding, p=2, dim=1)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for pair of inputs.
        
        Args:
            x1: First input tensor.
            x2: Second input tensor.
            
        Returns:
            Tuple of embeddings.
        """
        embedding1 = self.forward_one(x1)
        embedding2 = self.forward_one(x2)
        return embedding1, embedding2


class ArcFaceLoss(nn.Module):
    """ArcFace loss for face recognition."""
    
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 64.0):
        """Initialize ArcFace loss.
        
        Args:
            embedding_dim: Embedding dimension.
            num_classes: Number of classes.
            margin: Angular margin.
            scale: Feature scale.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss.
        
        Args:
            embeddings: Input embeddings.
            labels: Ground truth labels.
            
        Returns:
            ArcFace loss.
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Apply margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_theta = theta[torch.arange(0, embeddings.size(0)), labels].view(-1, 1)
        theta_margin = target_theta + self.margin
        
        # Compute logits
        logits = self.scale * torch.cos(theta_margin)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss


class CosFaceLoss(nn.Module):
    """CosFace loss for face recognition."""
    
    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.35, scale: float = 64.0):
        """Initialize CosFace loss.
        
        Args:
            embedding_dim: Embedding dimension.
            num_classes: Number of classes.
            margin: Cosine margin.
            scale: Feature scale.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CosFace loss.
        
        Args:
            embeddings: Input embeddings.
            labels: Ground truth labels.
            
        Returns:
            CosFace loss.
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weight)
        
        # Apply margin
        target_cosine = cosine[torch.arange(0, embeddings.size(0)), labels].view(-1, 1)
        cosine_margin = target_cosine - self.margin
        
        # Replace target cosine with margin cosine
        logits = cosine.scatter(1, labels.view(-1, 1), cosine_margin)
        logits = self.scale * logits
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        return loss

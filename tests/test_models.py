"""Test suite for face recognition with one-shot learning."""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.siamese import SiameseNetwork, ContrastiveLoss, TripletLoss
from src.models.clip_models import ArcFaceLoss, CosFaceLoss
from src.utils.device import get_device, set_seed
from src.utils.face_utils import FaceDetector
from src.eval.metrics import compute_cmc_curve, compute_tpr_fpr


class TestSiameseNetwork:
    """Test Siamese network functionality."""
    
    def test_siamese_network_creation(self):
        """Test Siamese network creation."""
        model = SiameseNetwork(
            backbone="resnet18",
            embedding_dim=512,
            dropout=0.3,
            pretrained=False
        )
        
        assert model is not None
        assert model.embedding_dim == 512
    
    def test_siamese_forward(self):
        """Test Siamese network forward pass."""
        model = SiameseNetwork(
            backbone="resnet18",
            embedding_dim=512,
            dropout=0.3,
            pretrained=False
        )
        
        # Create dummy input
        x1 = torch.randn(2, 3, 112, 112)
        x2 = torch.randn(2, 3, 112, 112)
        
        # Forward pass
        embedding1, embedding2 = model(x1, x2)
        
        assert embedding1.shape == (2, 512)
        assert embedding2.shape == (2, 512)
        assert torch.allclose(torch.norm(embedding1, dim=1), torch.ones(2), atol=1e-6)
        assert torch.allclose(torch.norm(embedding2, dim=1), torch.ones(2), atol=1e-6)
    
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(margin=1.0)
        
        # Create dummy embeddings and labels
        embedding1 = torch.randn(4, 512)
        embedding2 = torch.randn(4, 512)
        labels = torch.tensor([1, 0, 1, 0])
        
        loss = loss_fn(embedding1, embedding2, labels)
        
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])
    
    def test_triplet_loss(self):
        """Test triplet loss computation."""
        loss_fn = TripletLoss(margin=0.5)
        
        # Create dummy embeddings
        anchor = torch.randn(4, 512)
        positive = torch.randn(4, 512)
        negative = torch.randn(4, 512)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert loss.item() >= 0
        assert loss.shape == torch.Size([])


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_cmc_curve(self):
        """Test CMC curve computation."""
        # Create dummy distances and labels
        distances = torch.tensor([
            [0.1, 0.5, 0.3, 0.8],
            [0.6, 0.2, 0.9, 0.4],
        ])
        labels = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        
        cmc_values = compute_cmc_curve(distances, labels)
        
        assert len(cmc_values) == 4
        assert all(0 <= val <= 1 for val in cmc_values)
        assert cmc_values[0] == 1.0  # Rank-1 accuracy should be 100%
    
    def test_tpr_fpr(self):
        """Test TPR/FPR computation."""
        distances = torch.tensor([0.1, 0.3, 0.7, 0.9])
        labels = torch.tensor([1, 0, 1, 0])
        thresholds = [0.2, 0.5, 0.8]
        
        tpr_values, fpr_values = compute_tpr_fpr(distances, labels, thresholds)
        
        assert len(tpr_values) == len(thresholds)
        assert len(fpr_values) == len(thresholds)
        assert all(0 <= val <= 1 for val in tpr_values)
        assert all(0 <= val <= 1 for val in fpr_values)


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(1).item()
        np_rand = np.random.rand()
        
        # Set seed again and generate again
        set_seed(42)
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        
        assert torch_rand == torch_rand2
        assert np_rand == np_rand2


class TestFaceDetector:
    """Test face detection functionality."""
    
    def test_face_detector_creation(self):
        """Test face detector creation."""
        detector = FaceDetector(method="opencv")
        assert detector is not None
        assert detector.method == "opencv"
    
    def test_face_detection_dummy(self):
        """Test face detection with dummy image."""
        detector = FaceDetector(method="opencv")
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # This might not detect faces in random noise, but should not crash
        faces = detector.detect_faces(dummy_image)
        assert isinstance(faces, list)


if __name__ == "__main__":
    pytest.main([__file__])

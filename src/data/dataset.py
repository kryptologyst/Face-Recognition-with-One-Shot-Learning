"""Dataset classes for face recognition with one-shot learning."""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ..utils.face_utils import FaceDetector


class FaceDataset(Dataset):
    """Dataset for face recognition with one-shot learning."""
    
    def __init__(
        self,
        data_dir: str,
        face_detector: Optional[FaceDetector] = None,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (112, 112),
        cache_embeddings: bool = False,
    ):
        """Initialize face dataset.
        
        Args:
            data_dir: Directory containing face images organized by identity.
            face_detector: Face detector for preprocessing.
            transform: Image transformations.
            target_size: Target size for face images.
            cache_embeddings: Whether to cache preprocessed faces.
        """
        self.data_dir = Path(data_dir)
        self.face_detector = face_detector or FaceDetector()
        self.transform = transform
        self.target_size = target_size
        self.cache_embeddings = cache_embeddings
        
        # Load dataset structure
        self.identities = self._load_identities()
        self.samples = self._load_samples()
        
        # Cache for preprocessed images
        self.cache = {} if cache_embeddings else None
    
    def _load_identities(self) -> List[str]:
        """Load list of identity directories."""
        identities = []
        for item in self.data_dir.iterdir():
            if item.is_dir():
                identities.append(item.name)
        return sorted(identities)
    
    def _load_samples(self) -> List[Tuple[str, str, int]]:
        """Load all samples with their identity labels."""
        samples = []
        for identity_idx, identity in enumerate(self.identities):
            identity_dir = self.data_dir / identity
            for image_path in identity_dir.glob("*.jpg"):
                samples.append((str(image_path), identity, identity_idx))
            for image_path in identity_dir.glob("*.png"):
                samples.append((str(image_path), identity, identity_idx))
        return samples
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image_tensor, identity_label).
        """
        image_path, identity, label = self.samples[idx]
        
        # Check cache first
        if self.cache is not None and image_path in self.cache:
            image = self.cache[image_path]
        else:
            # Load and preprocess image
            image = self._load_image(image_path)
            if self.cache is not None:
                self.cache[image_path] = image
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Detect and extract face
            face = self.face_detector.preprocess_image(image_path, self.target_size)
            if face is None:
                # Fallback: resize original image
                image = image.resize(self.target_size)
                face = np.array(image).astype(np.float32) / 255.0
            
            # Convert to tensor
            face_tensor = torch.from_numpy(face).permute(2, 0, 1)
            return face_tensor
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return dummy image
            return torch.zeros(3, *self.target_size)
    
    def get_identity_samples(self, identity: str) -> List[int]:
        """Get all sample indices for a given identity."""
        indices = []
        for idx, (_, identity_name, _) in enumerate(self.samples):
            if identity_name == identity:
                indices.append(idx)
        return indices


class OneShotDataset(Dataset):
    """Dataset for one-shot learning evaluation."""
    
    def __init__(
        self,
        support_set: FaceDataset,
        query_set: FaceDataset,
        n_way: int = 5,
        k_shot: int = 1,
    ):
        """Initialize one-shot dataset.
        
        Args:
            support_set: Support set dataset.
            query_set: Query set dataset.
            n_way: Number of classes per episode.
            k_shot: Number of support samples per class.
        """
        self.support_set = support_set
        self.query_set = query_set
        self.n_way = n_way
        self.k_shot = k_shot
        
        # Create episodes
        self.episodes = self._create_episodes()
    
    def _create_episodes(self) -> List[Dict]:
        """Create evaluation episodes."""
        episodes = []
        
        # Get all identities
        all_identities = list(set(self.support_set.identities) & set(self.query_set.identities))
        
        for _ in range(1000):  # Create 1000 episodes
            # Sample n_way identities
            selected_identities = random.sample(all_identities, min(self.n_way, len(all_identities)))
            
            episode = {
                "support": [],
                "query": [],
                "labels": selected_identities,
            }
            
            # Sample support and query sets
            for identity in selected_identities:
                # Support samples
                support_indices = self.support_set.get_identity_samples(identity)
                if len(support_indices) >= self.k_shot:
                    episode["support"].extend(random.sample(support_indices, self.k_shot))
                
                # Query samples
                query_indices = self.query_set.get_identity_samples(identity)
                if query_indices:
                    episode["query"].extend(random.sample(query_indices, min(5, len(query_indices))))
            
            if len(episode["support"]) == self.n_way * self.k_shot and len(episode["query"]) > 0:
                episodes.append(episode)
        
        return episodes
    
    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get episode by index."""
        episode = self.episodes[idx]
        
        # Load support samples
        support_images = []
        support_labels = []
        for sample_idx in episode["support"]:
            image, label = self.support_set[sample_idx]
            support_images.append(image)
            support_labels.append(label)
        
        # Load query samples
        query_images = []
        query_labels = []
        for sample_idx in episode["query"]:
            image, label = self.query_set[sample_idx]
            query_images.append(image)
            query_labels.append(label)
        
        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels),
            "class_labels": episode["labels"],
        }


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    target_size: Tuple[int, int] = (112, 112),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory containing face images.
        batch_size: Batch size for data loaders.
        num_workers: Number of worker processes.
        train_split: Fraction of data for training.
        val_split: Fraction of data for validation.
        target_size: Target size for face images.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Create datasets
    full_dataset = FaceDataset(data_dir, transform=None, target_size=target_size)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

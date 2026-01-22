"""Evaluation metrics for face recognition."""

import numpy as np
import torch
from typing import List, Tuple, Dict


def compute_cmc_curve(distances: torch.Tensor, labels: torch.Tensor) -> List[float]:
    """Compute Cumulative Matching Characteristic (CMC) curve.
    
    Args:
        distances: Distance matrix of shape (n_query, n_gallery).
        labels: Binary labels indicating matches.
        
    Returns:
        List of CMC values for ranks 1 to n_gallery.
    """
    n_query, n_gallery = distances.shape
    cmc_values = []
    
    for rank in range(1, n_gallery + 1):
        # Get top-k predictions
        _, top_k_indices = torch.topk(distances, rank, dim=1, largest=False)
        
        # Check if correct match is in top-k
        correct_matches = 0
        for i in range(n_query):
            if labels[i] in top_k_indices[i]:
                correct_matches += 1
        
        cmc_values.append(correct_matches / n_query)
    
    return cmc_values


def compute_tpr_fpr(distances: torch.Tensor, labels: torch.Tensor, thresholds: List[float]) -> Tuple[List[float], List[float]]:
    """Compute True Positive Rate and False Positive Rate for different thresholds.
    
    Args:
        distances: Distance matrix.
        labels: Binary labels (1 for match, 0 for non-match).
        thresholds: List of distance thresholds.
        
    Returns:
        Tuple of (TPR values, FPR values).
    """
    tpr_values = []
    fpr_values = []
    
    for threshold in thresholds:
        predictions = (distances < threshold).float()
        
        # True Positives: correctly predicted matches
        tp = torch.sum(predictions * labels).item()
        
        # False Positives: incorrectly predicted matches
        fp = torch.sum(predictions * (1 - labels)).item()
        
        # True Negatives: correctly predicted non-matches
        tn = torch.sum((1 - predictions) * (1 - labels)).item()
        
        # False Negatives: incorrectly predicted non-matches
        fn = torch.sum((1 - predictions) * labels).item()
        
        # Compute rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    
    return tpr_values, fpr_values


def compute_auc(tpr_values: List[float], fpr_values: List[float]) -> float:
    """Compute Area Under Curve (AUC) for ROC curve.
    
    Args:
        tpr_values: True Positive Rate values.
        fpr_values: False Positive Rate values.
        
    Returns:
        AUC value.
    """
    # Sort by FPR
    sorted_indices = np.argsort(fpr_values)
    sorted_fpr = np.array(fpr_values)[sorted_indices]
    sorted_tpr = np.array(tpr_values)[sorted_indices]
    
    # Compute AUC using trapezoidal rule
    auc = np.trapz(sorted_tpr, sorted_fpr)
    return auc


def compute_eer(tpr_values: List[float], fpr_values: List[float]) -> float:
    """Compute Equal Error Rate (EER).
    
    Args:
        tpr_values: True Positive Rate values.
        fpr_values: False Positive Rate values.
        
    Returns:
        EER value.
    """
    # Find point where TPR = 1 - FPR
    for i in range(len(tpr_values)):
        if tpr_values[i] <= 1 - fpr_values[i]:
            return fpr_values[i]
    
    return 0.0


def evaluate_one_shot_accuracy(
    model: torch.nn.Module,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate one-shot learning accuracy.
    
    Args:
        model: Trained model.
        support_images: Support set images.
        support_labels: Support set labels.
        query_images: Query set images.
        query_labels: Query set labels.
        device: Device to run evaluation on.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    
    with torch.no_grad():
        # Move to device
        support_images = support_images.to(device)
        query_images = query_images.to(device)
        
        # Get embeddings
        support_embeddings = model.forward_one(support_images)
        query_embeddings = model.forward_one(query_images)
        
        # Compute distances
        distances = torch.cdist(query_embeddings, support_embeddings)
        
        # Get predictions
        _, predicted_indices = torch.min(distances, dim=1)
        predicted_labels = support_labels[predicted_indices]
        
        # Compute accuracy
        correct = (predicted_labels == query_labels).sum().item()
        accuracy = correct / len(query_labels)
        
        # Compute CMC curve
        binary_labels = (support_labels.unsqueeze(0) == query_labels.unsqueeze(1)).float()
        cmc_values = compute_cmc_curve(distances, binary_labels)
        
        # Compute TPR/FPR
        thresholds = torch.linspace(0, 2, 100).tolist()
        tpr_values, fpr_values = compute_tpr_fpr(distances.flatten(), binary_labels.flatten(), thresholds)
        
        # Compute additional metrics
        auc = compute_auc(tpr_values, fpr_values)
        eer = compute_eer(tpr_values, fpr_values)
        
        return {
            "accuracy": accuracy,
            "cmc_rank1": cmc_values[0] if len(cmc_values) > 0 else 0.0,
            "cmc_rank5": cmc_values[4] if len(cmc_values) > 4 else 0.0,
            "cmc_rank10": cmc_values[9] if len(cmc_values) > 9 else 0.0,
            "auc": auc,
            "eer": eer,
        }


def evaluate_siamese_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate Siamese network accuracy.
    
    Args:
        model: Trained Siamese model.
        data_loader: Data loader for evaluation.
        device: Device to run evaluation on.
        threshold: Distance threshold for classification.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images1, images2, labels = batch
            images1 = images1.to(device)
            images2 = images2.to(device)
            labels = labels.to(device)
            
            # Get embeddings
            embedding1, embedding2 = model(images1, images2)
            
            # Compute distances
            distances = torch.norm(embedding1 - embedding2, dim=1)
            
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)
    
    # Compute predictions
    predictions = (all_distances < threshold).astype(int)
    
    # Compute metrics
    accuracy = np.mean(predictions == all_labels)
    
    # Compute TPR/FPR
    thresholds = np.linspace(0, 2, 100)
    tpr_values, fpr_values = compute_tpr_fpr(
        torch.tensor(all_distances), torch.tensor(all_labels), thresholds.tolist()
    )
    
    auc = compute_auc(tpr_values, fpr_values)
    eer = compute_eer(tpr_values, fpr_values)
    
    return {
        "accuracy": accuracy,
        "auc": auc,
        "eer": eer,
        "threshold": threshold,
    }

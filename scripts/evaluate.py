"""Evaluation script for face recognition with one-shot learning."""

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_default_config, load_config
from src.utils.device import get_device, set_seed
from src.models.siamese import SiameseNetwork
from src.models.clip_models import CLIPSiameseNetwork
from src.data.dataset import create_data_loaders, OneShotDataset
from src.eval.metrics import evaluate_one_shot_accuracy, evaluate_siamese_accuracy


def create_model(config: DictConfig) -> torch.nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration.
        
    Returns:
        Created model.
    """
    model_name = config.model.name.lower()
    
    if model_name == "siamese_resnet":
        model = SiameseNetwork(
            backbone=config.model.backbone,
            embedding_dim=config.model.embedding_dim,
            dropout=config.model.dropout,
            pretrained=config.model.pretrained,
        )
        
    elif model_name == "clip_siamese":
        model = CLIPSiameseNetwork()
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: DictConfig,
) -> dict:
    """Evaluate model on test set.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device to run evaluation on.
        config: Configuration.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    
    if hasattr(model, "forward_one"):
        # One-shot learning evaluation
        return evaluate_one_shot_learning(model, test_loader, device, config)
    else:
        # Standard classification evaluation
        return evaluate_siamese_accuracy(model, test_loader, device)


def evaluate_one_shot_learning(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: DictConfig,
) -> dict:
    """Evaluate one-shot learning performance.
    
    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device to run evaluation on.
        config: Configuration.
        
    Returns:
        Dictionary of evaluation metrics.
    """
    total_metrics = {}
    num_episodes = 0
    
    print("Evaluating one-shot learning performance...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            support_images = batch["support_images"].to(device)
            support_labels = batch["support_labels"].to(device)
            query_images = batch["query_images"].to(device)
            query_labels = batch["query_labels"].to(device)
            
            metrics = evaluate_one_shot_accuracy(
                model,
                support_images,
                support_labels,
                query_images,
                query_labels,
                device,
            )
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            num_episodes += 1
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1} episodes...")
    
    # Average metrics
    for key in total_metrics:
        total_metrics[key] /= num_episodes
    
    return total_metrics


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate face recognition model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory")
    parser.add_argument("--device", type=str, help="Device to use")
    parser.add_argument("--output", type=str, help="Path to save results")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.device:
        config.device = args.device
    
    # Set device
    if config.device == "auto":
        device = get_device()
    else:
        device = torch.device(config.device)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    print(f"Evaluation Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data directory: {config.data.data_dir}")
    print(f"  Device: {device}")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Best accuracy: {checkpoint.get('best_accuracy', 'unknown')}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        target_size=tuple(config.data.input_size),
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate model
    print("Starting evaluation...")
    results = evaluate_model(model, test_loader, device, config)
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: {value}")
    
    # Save results
    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = {k: float(v) if isinstance(v, (int, float)) else str(v) 
                       for k, v in results.items()}
        
        with open(output_path, "w") as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()

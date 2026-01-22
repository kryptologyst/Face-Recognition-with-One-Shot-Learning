#!/usr/bin/env python3
"""Complete workflow demonstration for face recognition with one-shot learning."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.device import get_device, set_seed
from src.utils.config import get_default_config


def main():
    """Demonstrate the complete workflow."""
    print("Face Recognition with One-Shot Learning - Complete Workflow")
    print("=" * 60)
    
    # Set up
    device = get_device()
    set_seed(42)
    config = get_default_config()
    
    print(f"Device: {device}")
    print(f"Configuration loaded: {config.model.name}")
    
    # Check if data exists
    data_dir = Path("data/raw")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("\nNo data found. Generating sample dataset...")
        os.system("python scripts/generate_sample_data.py --num_identities 5 --images_per_identity 10 --realistic")
        print("Sample dataset generated!")
    else:
        print(f"\nData directory found: {data_dir}")
        identities = [d.name for d in data_dir.iterdir() if d.is_dir()]
        print(f"Found {len(identities)} identities: {identities[:3]}{'...' if len(identities) > 3 else ''}")
    
    # Check if model exists
    checkpoint_path = Path("checkpoints/best_model.pth")
    if not checkpoint_path.exists():
        print("\nNo trained model found. Starting training...")
        print("Run the following command to train:")
        print("  python scripts/train.py --data_dir data/raw --epochs 10")
        print("\nFor now, let's run the demo with the original example...")
        
        # Run the original example
        print("\nRunning original example:")
        os.system("python 0561.py")
        
    else:
        print(f"\nTrained model found: {checkpoint_path}")
        print("Running evaluation...")
        os.system(f"python scripts/evaluate.py --checkpoint {checkpoint_path} --data_dir data/raw")
    
    # Demo instructions
    print("\n" + "=" * 60)
    print("DEMO INSTRUCTIONS:")
    print("=" * 60)
    print("1. Generate sample data:")
    print("   python scripts/generate_sample_data.py --realistic")
    print()
    print("2. Train the model:")
    print("   python scripts/train.py --data_dir data/raw --epochs 50")
    print()
    print("3. Evaluate the model:")
    print("   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth")
    print()
    print("4. Launch interactive demo:")
    print("   streamlit run demo/app.py")
    print()
    print("5. Run tests:")
    print("   python -m pytest tests/ -v")
    print()
    print("6. Format code:")
    print("   black src/ scripts/ tests/")
    print("   ruff check src/ scripts/ tests/")
    
    print("\nProject structure:")
    print("src/")
    print("├── models/          # Siamese networks, CLIP models")
    print("├── data/            # Dataset classes and loaders")
    print("├── train/           # Training utilities")
    print("├── eval/            # Evaluation metrics")
    print("└── utils/           # Device, config, face detection")
    print()
    print("scripts/")
    print("├── train.py         # Training script")
    print("├── evaluate.py      # Evaluation script")
    print("└── generate_sample_data.py  # Sample data generation")
    print()
    print("demo/")
    print("└── app.py           # Streamlit demo")
    print()
    print("configs/")
    print("└── default.yaml     # Default configuration")
    
    print("\nKey Features:")
    print("✓ Modern PyTorch 2.x implementation")
    print("✓ Automatic device detection (CUDA → MPS → CPU)")
    print("✓ Type hints and comprehensive documentation")
    print("✓ Multiple model architectures (Siamese, ProtoNet, CLIP)")
    print("✓ Comprehensive evaluation metrics (CMC, TPR/FPR, AUC)")
    print("✓ Interactive Streamlit demo")
    print("✓ Production-ready structure and testing")
    print("✓ Deterministic seeding for reproducibility")
    
    print("\nReady to use! 🚀")


if __name__ == "__main__":
    main()

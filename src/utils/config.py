"""Configuration management using OmegaConf."""

from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        DictConfig: Loaded configuration.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        save_path: Path to save configuration.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """Merge two configurations.
    
    Args:
        base_config: Base configuration.
        override_config: Configuration to override with.
        
    Returns:
        DictConfig: Merged configuration.
    """
    return OmegaConf.merge(base_config, override_config)


def get_default_config() -> DictConfig:
    """Get default configuration for face recognition with one-shot learning.
    
    Returns:
        DictConfig: Default configuration.
    """
    return OmegaConf.create({
        "model": {
            "name": "siamese_resnet",
            "backbone": "resnet18",
            "embedding_dim": 512,
            "dropout": 0.3,
            "pretrained": True,
        },
        "data": {
            "input_size": [112, 112],
            "batch_size": 32,
            "num_workers": 4,
            "data_dir": "data/raw",
            "processed_dir": "data/processed",
            "augmentation": {
                "horizontal_flip": 0.5,
                "rotation": 15,
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
            },
        },
        "training": {
            "epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "scheduler": "cosine",
            "warmup_epochs": 5,
            "gradient_clip": 1.0,
            "accumulation_steps": 1,
            "mixed_precision": True,
        },
        "evaluation": {
            "metrics": ["accuracy", "cmc", "tpr_fpr"],
            "k_values": [1, 5, 10],
            "threshold": 0.5,
        },
        "logging": {
            "log_dir": "logs",
            "save_dir": "checkpoints",
            "log_interval": 10,
            "save_interval": 10,
            "use_wandb": False,
            "project_name": "face-recognition-one-shot",
        },
        "device": "auto",
        "seed": 42,
    })

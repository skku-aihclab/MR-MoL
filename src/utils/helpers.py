"""Utility functions."""

import random
import yaml
import torch
import numpy as np
from typing import Dict, Any


# use seeds: 42, 43, 44
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    else:
        return str(n)


def print_model_info(model, title: str = "Model Information"):
    """Print model information."""
    params = count_parameters(model)

    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Total parameters:     {format_number(params['total'])}")
    print(f"Trainable parameters: {format_number(params['trainable'])}")
    print(f"Frozen parameters:    {format_number(params['frozen'])}")
    print(f"{'='*50}\n")

"""Reproducibility helpers for PyTorch experiments."""

import random

import torch


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch random number generators."""
    random.seed(seed)
    torch.manual_seed(seed)


def torch_generator(seed: int) -> torch.Generator:
    """Return an independently seeded CPU generator."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator

"""Shared utilities: seeding, device, data loading."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_SHAPES = PROJECT_ROOT / "data" / "shapes"
DATA_DIR_SETS = PROJECT_ROOT / "data" / "sets"
FIGURES_DIR = PROJECT_ROOT / "figures"
CKPT_DIR = PROJECT_ROOT / "checkpoints"
FIGURES_DIR.mkdir(exist_ok=True)
CKPT_DIR.mkdir(exist_ok=True)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_shape(name: str) -> np.ndarray:
    """name in {'circle', 'square', 'dinohead'} -> (N, 2) float32."""
    path = DATA_DIR_SHAPES / f"{name}_pts.npy"
    return np.load(path).astype(np.float32)


def load_stars() -> np.ndarray:
    """(1024, 1024, 2) float32: 1024 star samples, each with 1024 points."""
    return np.load(DATA_DIR_SETS / "stars_data.npy").astype(np.float32)

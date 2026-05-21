"""
spectra/utils.py
Shared utilities: config loading, seeding, logging helpers.
"""
import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class Config:
    """Dot-access wrapper around a nested YAML dict."""

    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __repr__(self) -> str:
        return str(self.__dict__)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, "spectra_best.pth")
        torch.save(state, best_path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors in a dict/list/tuple to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    return batch


def normalise_tensor(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalise a tensor to [0, 1]."""
    mn = x.flatten(1).min(dim=1).values.view(-1, 1, 1, 1)
    mx = x.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
    return (x - mn) / (mx - mn + eps)

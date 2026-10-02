"""A small CNN baseline for Part B, sized to train on a laptop CPU.

Four 3x3 convolution blocks (16, 32, 64 and 128 channels; batch norm and ReLU
in each; 2x2 max pooling after the first three), global average pooling,
dropout 0.2 and one linear layer. Input: one grayscale channel, already
resized by ``mri_pipeline.io.load_image``; pixel values are scaled to 0-1
here. Training: Adam, cross-entropy, random horizontal flips, a fixed number
of epochs, and one seed for the weights, the shuffling and the flips.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn


def build_model(n_classes: int) -> nn.Sequential:
    def block(c_in: int, c_out: int, pool: bool = True) -> list[nn.Module]:
        layers: list[nn.Module] = [
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        ]
        return [*layers, nn.MaxPool2d(2)] if pool else layers

    return nn.Sequential(
        *block(1, 16),
        *block(16, 32),
        *block(32, 64),
        *block(64, 128, pool=False),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(128, n_classes),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def to_uint8_tensor(images: Sequence[np.ndarray]) -> torch.Tensor:
    """Stack equal-sized uint8 grayscale images into an (N, 1, H, W) uint8 tensor."""
    return torch.from_numpy(np.ascontiguousarray(np.stack(images)[:, None]))


def _scale(x_u8: torch.Tensor) -> torch.Tensor:
    return x_u8.float().div_(255.0)


def train(
    x_u8: torch.Tensor,
    y: Sequence[int] | np.ndarray,
    n_classes: int,
    epochs: int,
    seed: int,
    batch_size: int = 64,
    lr: float = 1e-3,
    flip: bool = True,
    log: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[nn.Sequential, list[dict[str, Any]]]:
    """Train from scratch; returns the model after the last epoch and a per-epoch history.

    The training accuracy in the history is the running accuracy on the
    (flipped) batches during that epoch, not a separate evaluation.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = build_model(n_classes)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    labels = torch.as_tensor(np.asarray(y), dtype=torch.long)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        order = torch.randperm(len(labels), generator=generator)
        loss_sum, correct = 0.0, 0
        for begin in range(0, len(order), batch_size):
            idx = order[begin : begin + batch_size]
            xb, yb = _scale(x_u8[idx]), labels[idx]
            if flip:
                mask = torch.rand(len(idx), generator=generator) < 0.5
                xb[mask] = xb[mask].flip(-1)
            optimiser.zero_grad()
            logits = model(xb)
            loss = nn.functional.cross_entropy(logits, yb)
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * len(idx)
            correct += int((logits.argmax(1) == yb).sum())
        row = {
            "epoch": epoch,
            "train_loss": round(loss_sum / len(labels), 4),
            "train_accuracy_running": round(correct / len(labels), 4),
            "seconds": round(time.perf_counter() - start, 2),
        }
        history.append(row)
        if log is not None:
            log(row)
    return model, history


def predict(model: nn.Module, x_u8: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        parts = [model(_scale(xb)).argmax(1) for xb in torch.split(x_u8, batch_size)]
    return torch.cat(parts).numpy()

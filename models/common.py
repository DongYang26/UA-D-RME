"""Slim helper utilities for DUPRME models.

Only the three symbols required by ``models.decentralized.local_uncertainty_rme``
are retained: ``build_mlp``, ``safe_precision``, ``safe_variance``. Heavier
scene/episode adapters, frequency-feature encoders, and dense-query helpers
that exist in the parent project have been removed to keep this package
self-contained.
"""
from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 3,
    activation: type[nn.Module] = nn.SiLU,
    dropout: float = 0.0,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(max(1, num_layers - 1)):
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def safe_precision(raw_precision: torch.Tensor, min_precision: float = 1e-4) -> torch.Tensor:
    return torch.nn.functional.softplus(raw_precision) + min_precision


def safe_variance(raw_variance: torch.Tensor, min_variance: float = 1e-6) -> torch.Tensor:
    return torch.nn.functional.softplus(raw_variance) + min_variance

from __future__ import annotations

import torch


def local_observation_nll(
    pred_mean: torch.Tensor,
    pred_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if pred_mean.numel() == 0 or pred_variance.numel() == 0 or target.numel() == 0:
        return torch.zeros((), device=pred_mean.device, dtype=pred_mean.dtype)
    variance = torch.clamp(pred_variance, min=1e-6)
    sq_error = (target - pred_mean).pow(2)
    return 0.5 * torch.mean(torch.log(variance) + sq_error / variance)


def compute_edge_agreement_loss(
    local_mean: torch.Tensor,
    recv_mean: torch.Tensor,
    local_precision: torch.Tensor,
    recv_precision: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        local_mean.shape != recv_mean.shape
        or local_mean.shape != local_precision.shape
        or local_mean.shape != recv_precision.shape
    ):
        raise ValueError(
            "local_mean, recv_mean, local_precision, and recv_precision must have matching shapes."
        )
    if (
        local_mean.numel() == 0
        or recv_mean.numel() == 0
        or local_precision.numel() == 0
        or recv_precision.numel() == 0
    ):
        zero = torch.zeros((), device=local_mean.device, dtype=local_mean.dtype)
        return {"overlap": zero, "uncertainty": zero}

    local_precision = torch.clamp(local_precision, min=1e-6)
    recv_precision = torch.clamp(recv_precision, min=1e-6)
    overlap = torch.mean((local_precision + recv_precision) * (local_mean - recv_mean).pow(2))
    uncertainty = torch.mean(torch.abs(torch.log(local_precision) - torch.log(recv_precision)))
    return {"overlap": overlap, "uncertainty": uncertainty}

from __future__ import annotations

from typing import Any

import numpy as np
import torch

_MIN_PRECISION = 1e-6


def _to_numpy_array(value: Any, *, dtype: np.dtype[Any] | type[np.generic] | None = None) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.astype(dtype, copy=False) if dtype is not None else array
    return np.asarray(value, dtype=dtype)


def compute_global_nmse(
    pred_map: Any,
    target_map: Any,
    mask: Any,
) -> float:
    pred_map_arr = _to_numpy_array(pred_map, dtype=np.float32)
    target_map_arr = _to_numpy_array(target_map, dtype=np.float32)
    mask_arr = _to_numpy_array(mask, dtype=bool)

    if pred_map_arr.shape != target_map_arr.shape or mask_arr.shape != target_map_arr.shape:
        raise ValueError("pred_map, target_map, and mask must have matching shapes.")

    if not np.any(mask_arr):
        return float("nan")

    squared_error = (pred_map_arr - target_map_arr) ** 2
    target_energy = target_map_arr**2
    numerator = float(np.sum(squared_error[mask_arr]))
    denominator = float(np.sum(target_energy[mask_arr]))
    return float(numerator / denominator) if denominator > 0.0 else float("nan")


def compute_overlap_disagreement(
    local_mean: Any,
    recv_mean: Any,
    local_precision: Any,
    recv_precision: Any,
) -> dict[str, float]:
    local_mean_arr = _to_numpy_array(local_mean, dtype=np.float32)
    recv_mean_arr = _to_numpy_array(recv_mean, dtype=np.float32)
    local_precision_arr = np.clip(_to_numpy_array(local_precision, dtype=np.float32), a_min=_MIN_PRECISION, a_max=None)
    recv_precision_arr = np.clip(_to_numpy_array(recv_precision, dtype=np.float32), a_min=_MIN_PRECISION, a_max=None)

    if (
        local_mean_arr.shape != recv_mean_arr.shape
        or local_mean_arr.shape != local_precision_arr.shape
        or local_mean_arr.shape != recv_precision_arr.shape
    ):
        raise ValueError(
            "local_mean, recv_mean, local_precision, and recv_precision must have matching shapes."
        )
    if local_mean_arr.size == 0:
        return {"mean_abs": 0.0, "log_precision_abs": 0.0}

    mean_abs = float(np.mean(np.abs(local_mean_arr - recv_mean_arr)))
    log_precision_abs = float(
        np.mean(np.abs(np.log(local_precision_arr) - np.log(recv_precision_arr)))
    )
    return {"mean_abs": mean_abs, "log_precision_abs": log_precision_abs}

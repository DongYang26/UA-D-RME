from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class DecentralizedMessage:
    decoded: torch.Tensor
    num_bytes: float


class IdentityCodec(nn.Module):
    def encode(self, tensor: torch.Tensor) -> DecentralizedMessage:
        return DecentralizedMessage(
            decoded=tensor.detach().clone(),
            num_bytes=float(tensor.element_size() * tensor.numel()),
        )


class ScalarQuantizationCodec(nn.Module):
    METADATA_OVERHEAD_BYTES = 8.0

    def __init__(self, bits: int) -> None:
        super().__init__()
        if bits <= 0:
            raise ValueError("bits must be positive for scalar quantization.")
        self.bits = bits
        self.levels = float(2**bits - 1)

    def encode(self, tensor: torch.Tensor) -> DecentralizedMessage:
        if tensor.numel() == 0:
            return DecentralizedMessage(decoded=tensor.detach().clone(), num_bytes=0.0)

        min_v = tensor.min()
        max_v = tensor.max()
        scale = torch.clamp(max_v - min_v, min=1e-6)
        quantized = torch.round(((tensor - min_v) / scale) * self.levels) / self.levels
        decoded = (min_v + quantized * scale).detach().clone()
        payload_bytes = tensor.numel() * self.bits / 8.0
        metadata_bytes = self.METADATA_OVERHEAD_BYTES
        num_bytes = float(payload_bytes + metadata_bytes)
        return DecentralizedMessage(decoded=decoded, num_bytes=num_bytes)


def build_decentralized_codec(config: dict[str, Any]) -> nn.Module:
    mode = str(config.get("mode", "none"))
    if mode == "none":
        return IdentityCodec()
    if mode == "scalar_quantization":
        return ScalarQuantizationCodec(bits=int(config.get("bits", 8)))
    raise ValueError(f"Unsupported codec mode '{mode}'.")

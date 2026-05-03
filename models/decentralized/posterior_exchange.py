from __future__ import annotations

from typing import Any

import torch


def exchange_edge_statistics(
    local_stats: dict[tuple[int, int], dict[int, dict[str, torch.Tensor]]],
    codec: Any,
) -> dict[tuple[int, int], dict[int, dict[str, torch.Tensor | float]]]:
    exchanged: dict[tuple[int, int], dict[int, dict[str, torch.Tensor | float]]] = {}
    for edge, endpoint_stats in local_stats.items():
        src, dst = edge

        src_mean_msg = codec.encode(endpoint_stats[src]["mean"])
        src_precision_msg = codec.encode(endpoint_stats[src]["precision"])
        dst_mean_msg = codec.encode(endpoint_stats[dst]["mean"])
        dst_precision_msg = codec.encode(endpoint_stats[dst]["precision"])

        exchanged[edge] = {
            src: {
                "received_mean": dst_mean_msg.decoded.detach(),
                "received_precision": dst_precision_msg.decoded.detach(),
                "sent_bytes": src_mean_msg.num_bytes + src_precision_msg.num_bytes,
            },
            dst: {
                "received_mean": src_mean_msg.decoded.detach(),
                "received_precision": src_precision_msg.decoded.detach(),
                "sent_bytes": dst_mean_msg.num_bytes + dst_precision_msg.num_bytes,
            },
        }
    return exchanged

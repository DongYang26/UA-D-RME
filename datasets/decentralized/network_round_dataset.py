from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from datasets.decentralized.manifest import EdgeOverlapSample, LocalNodeSample, NetworkRoundSample


class NetworkRoundDataset:
    """Iterates :class:`NetworkRoundSample` objects from per-split manifests.

    Both the node and edge manifests are expected to live in the same directory
    (or share a common ``data_root``). Each manifest's ``artifact_path`` column
    may be absolute or relative; relative paths are resolved against
    ``data_root`` (default = directory of ``node_manifest_path``). The manifest
    file format is auto-detected from the file extension and may be either
    ``.parquet`` or ``.csv``.
    """

    def __init__(
        self,
        node_manifest_path: str,
        edge_manifest_path: str,
        split: str,
        node_loader: Callable[[str], dict[str, Any]],
        edge_loader: Callable[[str], dict[str, Any]],
        num_agents: int,
        graph_edges: list[list[int]] | np.ndarray,
        data_root: str | Path | None = None,
    ) -> None:
        self._node_loader = node_loader
        self._edge_loader = edge_loader
        self._num_agents = int(num_agents)
        self._graph_edges = np.asarray(graph_edges, dtype=np.int64)

        self._node_manifest_path = Path(node_manifest_path)
        self._edge_manifest_path = Path(edge_manifest_path)
        self._data_root = (
            Path(data_root) if data_root is not None else self._node_manifest_path.parent
        )

        self._node_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._edge_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        node_rows = self._read_manifest(self._node_manifest_path)
        node_rows = node_rows[node_rows["split"] == split]
        for record in node_rows.to_dict(orient="records"):
            self._node_groups[str(record["scene_id"])].append(record)

        edge_rows = self._read_manifest(self._edge_manifest_path)
        edge_rows = edge_rows[edge_rows["split"] == split]
        for record in edge_rows.to_dict(orient="records"):
            self._edge_groups[str(record["scene_id"])].append(record)

        node_scene_ids = set(self._node_groups.keys())
        edge_scene_ids = set(self._edge_groups.keys())
        edge_only_scene_ids = sorted(edge_scene_ids - node_scene_ids)
        if edge_only_scene_ids:
            raise ValueError(
                "Found scene_id values in edge manifest without node rows for the selected split: "
                f"{edge_only_scene_ids}"
            )

        self._scene_ids = sorted(node_scene_ids)

    @staticmethod
    def _read_manifest(path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _resolve(self, artifact_path: str) -> Path:
        candidate = Path(artifact_path)
        if candidate.is_absolute():
            return candidate
        return self._data_root / candidate

    def __len__(self) -> int:
        return len(self._scene_ids)

    def __getitem__(self, index: int) -> NetworkRoundSample:
        scene_id = self._scene_ids[index]
        node_records = sorted(self._node_groups[scene_id], key=lambda item: int(item["agent_id"]))
        edge_records = sorted(
            self._edge_groups.get(scene_id, []),
            key=lambda item: (int(item["src_agent_id"]), int(item["dst_agent_id"])),
        )

        nodes: list[LocalNodeSample] = []
        for record in node_records:
            payload = self._node_loader(str(self._resolve(record["artifact_path"])))
            grid_z = payload.get("grid_z")
            grid_z_array = np.asarray(grid_z, dtype=np.float32) if grid_z is not None else None
            if grid_z_array is not None and grid_z_array.size == 0:
                grid_z_array = None
            nodes.append(
                LocalNodeSample(
                    scene_id=scene_id,
                    agent_id=int(record["agent_id"]),
                    obs_coords=np.asarray(payload["obs_coords"], dtype=np.float32),
                    obs_values=np.asarray(payload["obs_values"], dtype=np.float32),
                    region_mask=np.asarray(payload["region_mask"], dtype=np.uint8),
                    target_map=np.asarray(payload["target_map"], dtype=np.float32),
                    target_mask=np.asarray(payload["target_mask"], dtype=np.uint8),
                    local_building_mask=np.asarray(
                        payload.get("local_building_mask", np.zeros_like(payload["target_map"])),
                        dtype=np.uint8,
                    ),
                    grid_x=np.asarray(
                        payload.get("grid_x", np.arange(np.asarray(payload["target_map"]).shape[-1])),
                        dtype=np.float32,
                    ),
                    grid_y=np.asarray(
                        payload.get("grid_y", np.arange(np.asarray(payload["target_map"]).shape[-2])),
                        dtype=np.float32,
                    ),
                    grid_z=grid_z_array,
                )
            )

        if self._num_agents > 0 and len(nodes) != self._num_agents:
            raise ValueError("node count does not match configured num_agents.")

        edges: list[EdgeOverlapSample] = []
        for record in edge_records:
            payload = self._edge_loader(str(self._resolve(record["artifact_path"])))
            edges.append(
                EdgeOverlapSample(
                    scene_id=scene_id,
                    src_agent_id=int(record["src_agent_id"]),
                    dst_agent_id=int(record["dst_agent_id"]),
                    anchor_coords=np.asarray(payload["anchor_coords"], dtype=np.float32),
                    overlap_mask=np.asarray(payload["overlap_mask"], dtype=np.uint8),
                )
            )

        sample = NetworkRoundSample(
            scene_id=scene_id,
            nodes=nodes,
            edges=edges,
            graph_edges=self._graph_edges,
        )
        sample.validate()
        return sample

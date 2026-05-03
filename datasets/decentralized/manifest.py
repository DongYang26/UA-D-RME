from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def validate_undirected_connected_graph(num_agents: int, edges: np.ndarray) -> None:
    if num_agents <= 0:
        raise ValueError("num_agents must be positive.")

    edge_array = np.asarray(edges)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape [E, 2].")
    if not np.issubdtype(edge_array.dtype, np.integer):
        raise ValueError("edges must contain integer node ids.")
    if np.any(edge_array < 0) or np.any(edge_array >= num_agents):
        raise ValueError("edges must reference node ids in [0, num_agents).")
    if np.any(edge_array[:, 0] == edge_array[:, 1]):
        raise ValueError("Self-loops are not allowed in undirected graphs.")

    adjacency: list[list[int]] = [[] for _ in range(num_agents)]
    for src, dst in edge_array:
        src_id = int(src)
        dst_id = int(dst)
        adjacency[src_id].append(dst_id)
        adjacency[dst_id].append(src_id)

    visited = np.zeros(num_agents, dtype=bool)
    stack = [0]
    while stack:
        node = stack.pop()
        if visited[node]:
            continue
        visited[node] = True
        for neighbor in adjacency[node]:
            if not visited[neighbor]:
                stack.append(neighbor)

    if not np.all(visited):
        raise ValueError("Undirected communication graph must be connected.")


@dataclass
class LocalNodeSample:
    scene_id: str
    agent_id: int
    obs_coords: np.ndarray
    obs_values: np.ndarray
    region_mask: np.ndarray
    target_map: np.ndarray
    target_mask: np.ndarray
    local_building_mask: np.ndarray | None = None
    grid_x: np.ndarray | None = None
    grid_y: np.ndarray | None = None
    grid_z: np.ndarray | None = None

    def validate(self) -> None:
        if self.agent_id < 0:
            raise ValueError("agent_id must be non-negative.")
        if self.obs_coords.ndim != 2:
            raise ValueError("obs_coords must have shape [N, D].")
        if self.obs_values.ndim != 1:
            raise ValueError("obs_values must have shape [N].")
        if self.obs_coords.shape[0] != self.obs_values.shape[0]:
            raise ValueError("obs_coords and obs_values must align on N.")
        if self.region_mask.shape != self.target_map.shape:
            raise ValueError("region_mask and target_map must have the same shape.")
        if self.target_mask.shape != self.target_map.shape:
            raise ValueError("target_mask and target_map must have the same shape.")
        if self.obs_coords.shape[1] != self.target_map.ndim:
            raise ValueError("obs_coords dimensionality must match target_map.ndim.")
        if self.local_building_mask is not None and self.local_building_mask.shape != self.target_map.shape:
            raise ValueError("local_building_mask and target_map must have the same shape.")
        if self.grid_x is not None and self.grid_x.ndim != 1:
            raise ValueError("grid_x must be a 1D vector.")
        if self.grid_y is not None and self.grid_y.ndim != 1:
            raise ValueError("grid_y must be a 1D vector.")
        if self.target_map.ndim == 2:
            if self.grid_x is not None and self.grid_x.shape[0] != self.target_map.shape[1]:
                raise ValueError("grid_x length must match target_map width.")
            if self.grid_y is not None and self.grid_y.shape[0] != self.target_map.shape[0]:
                raise ValueError("grid_y length must match target_map height.")
            if self.grid_z is not None and self.grid_z.size != 0:
                raise ValueError("2D samples must not provide grid_z.")
        else:
            if self.grid_x is not None and self.grid_x.shape[0] != self.target_map.shape[2]:
                raise ValueError("grid_x length must match target_map width.")
            if self.grid_y is not None and self.grid_y.shape[0] != self.target_map.shape[1]:
                raise ValueError("grid_y length must match target_map height.")
            if self.grid_z is None:
                raise ValueError("3D samples must provide grid_z.")
            if self.grid_z.ndim != 1 or self.grid_z.shape[0] != self.target_map.shape[0]:
                raise ValueError("grid_z length must match target_map depth.")


@dataclass
class EdgeOverlapSample:
    scene_id: str
    src_agent_id: int
    dst_agent_id: int
    anchor_coords: np.ndarray
    overlap_mask: np.ndarray

    def validate(self) -> None:
        if self.src_agent_id < 0 or self.dst_agent_id < 0:
            raise ValueError("edge endpoints must be non-negative.")
        if self.src_agent_id == self.dst_agent_id:
            raise ValueError("edge endpoints must be different nodes.")
        if self.anchor_coords.ndim != 2:
            raise ValueError("anchor_coords must have shape [Q, D].")
        if self.overlap_mask.ndim < 2:
            raise ValueError("overlap_mask must have at least 2 dimensions.")
        if self.anchor_coords.shape[1] != self.overlap_mask.ndim:
            raise ValueError("anchor_coords dimensionality must match overlap_mask.ndim.")


@dataclass
class NetworkRoundSample:
    scene_id: str
    nodes: list[LocalNodeSample]
    edges: list[EdgeOverlapSample]
    graph_edges: np.ndarray

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("nodes must be non-empty.")

        agent_ids: list[int] = []
        for node in self.nodes:
            node.validate()
            if node.scene_id != self.scene_id:
                raise ValueError("All nodes must share the network scene_id.")
            agent_ids.append(node.agent_id)
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("nodes must have unique agent_id values.")

        agent_id_set = set(agent_ids)

        graph_edges = np.asarray(self.graph_edges)
        if graph_edges.ndim != 2 or graph_edges.shape[1] != 2:
            raise ValueError("graph_edges must have shape [E, 2].")
        if not np.issubdtype(graph_edges.dtype, np.integer):
            raise ValueError("graph_edges must contain integer agent ids.")

        graph_edge_pairs: set[tuple[int, int]] = set()
        for src, dst in graph_edges:
            src_id = int(src)
            dst_id = int(dst)
            if src_id not in agent_id_set or dst_id not in agent_id_set:
                raise ValueError("graph_edges endpoints must exist in nodes.")
            graph_edge_pairs.add(tuple(sorted((src_id, dst_id))))

        overlap_edge_pairs: set[tuple[int, int]] = set()
        for edge in self.edges:
            edge.validate()
            if edge.scene_id != self.scene_id:
                raise ValueError("All overlap edges must share the network scene_id.")
            if edge.src_agent_id not in agent_id_set or edge.dst_agent_id not in agent_id_set:
                raise ValueError("Edge endpoints must exist in nodes.")
            edge_pair = tuple(sorted((edge.src_agent_id, edge.dst_agent_id)))
            if edge_pair in overlap_edge_pairs:
                raise ValueError(
                    "Duplicate overlap artifacts are not allowed for the same undirected edge."
                )
            overlap_edge_pairs.add(edge_pair)
            if edge_pair not in graph_edge_pairs:
                raise ValueError("Each overlap edge must correspond to a graph edge.")
        if missing_overlap_pairs := graph_edge_pairs - overlap_edge_pairs:
            raise ValueError(
                "Each graph edge must have a corresponding overlap edge artifact; "
                f"missing overlap edges for {sorted(missing_overlap_pairs)}."
            )

        node_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        indexed_edges = np.asarray(
            [[node_id_to_idx[int(src)], node_id_to_idx[int(dst)]] for src, dst in graph_edges],
            dtype=np.int64,
        ).reshape(-1, 2)
        validate_undirected_connected_graph(num_agents=len(self.nodes), edges=indexed_edges)

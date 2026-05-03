from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from datasets.decentralized.manifest import EdgeOverlapSample, LocalNodeSample, NetworkRoundSample
from evaluation.decentralized_metrics import compute_overlap_disagreement
from losses.decentralized_network_loss import compute_edge_agreement_loss, local_observation_nll
from models.decentralized.local_uncertainty_rme import LocalUncertaintyRME
from models.decentralized.message_codec import build_decentralized_codec
from models.decentralized.posterior_exchange import exchange_edge_statistics


class LightningDUPRME(pl.LightningModule):
    automatic_optimization = False

    def __init__(
        self,
        num_agents: int,
        model_config: dict[str, Any],
        loss_weights: dict[str, float],
        optimizer_config: dict[str, Any],
        codec_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.model_config = dict(model_config)
        self.loss_weights = dict(loss_weights)
        self.optimizer_config = dict(optimizer_config)
        self.agent_models = nn.ModuleDict(
            {
                str(agent_id): LocalUncertaintyRME(**self.model_config)
                for agent_id in range(self.num_agents)
            }
        )
        self.codec = build_decentralized_codec(codec_config)

    def configure_optimizers(self) -> list[torch.optim.Optimizer]:
        lr = float(self.optimizer_config["lr"])
        weight_decay = float(self.optimizer_config.get("weight_decay", 0.0))
        return [
            torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            for model in self.agent_models.values()
        ]

    def _to_tensor(self, array: Any, *, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(array, dtype=dtype, device=self.device)

    def _empty_anchor_coords(self) -> torch.Tensor:
        coord_dim = int(self.model_config.get("coord_dim", 2))
        return torch.empty((0, coord_dim), dtype=torch.float32, device=self.device)

    def _empty_values(self) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.float32, device=self.device)

    def _grid_vectors_for_node(
        self,
        node: LocalNodeSample,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        grid_x = None if node.grid_x is None else self._to_tensor(node.grid_x, dtype=torch.float32)
        grid_y = None if node.grid_y is None else self._to_tensor(node.grid_y, dtype=torch.float32)
        grid_z = None if node.grid_z is None else self._to_tensor(node.grid_z, dtype=torch.float32)
        return grid_x, grid_y, grid_z

    def _build_masked_query_coords(
        self,
        support_mask: torch.Tensor,
        node: LocalNodeSample,
    ) -> torch.Tensor:
        support_indices = torch.nonzero(support_mask, as_tuple=False)
        if support_indices.numel() == 0:
            return self._empty_anchor_coords()
        grid_x, grid_y, grid_z = self._grid_vectors_for_node(node)
        if grid_x is None or grid_y is None:
            coord_order = list(range(support_indices.shape[1] - 1, -1, -1))
            return support_indices[:, coord_order].to(dtype=torch.float32, device=self.device)
        if support_mask.ndim == 2:
            ys = support_indices[:, 0]
            xs = support_indices[:, 1]
            return torch.stack([grid_x[xs], grid_y[ys]], dim=-1)
        if grid_z is None:
            raise ValueError("grid_z is required for 3D masked query coordinates.")
        zs = support_indices[:, 0]
        ys = support_indices[:, 1]
        xs = support_indices[:, 2]
        return torch.stack([grid_x[xs], grid_y[ys], grid_z[zs]], dim=-1)

    def _extract_local_supervision(
        self,
        node: LocalNodeSample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_mask = torch.as_tensor(node.target_mask > 0, dtype=torch.bool, device=self.device)
        region_mask = torch.as_tensor(node.region_mask > 0, dtype=torch.bool, device=self.device)
        support_mask = target_mask & region_mask
        if not torch.any(support_mask):
            return self._empty_anchor_coords(), self._empty_values()
        target_map = self._to_tensor(node.target_map, dtype=torch.float32)
        query_coords = self._build_masked_query_coords(support_mask, node)
        target_values = target_map[support_mask]
        return query_coords, target_values

    def _collect_anchor_coords(
        self,
        sample: NetworkRoundSample,
    ) -> tuple[dict[int, torch.Tensor], dict[tuple[int, int], dict[int, slice]]]:
        per_agent_chunks: dict[int, list[torch.Tensor]] = {
            node.agent_id: [] for node in sample.nodes
        }
        per_edge_slices: dict[tuple[int, int], dict[int, slice]] = {}
        per_agent_sizes = {node.agent_id: 0 for node in sample.nodes}

        for edge in sample.edges:
            edge_key = (edge.src_agent_id, edge.dst_agent_id)
            anchor_coords = self._to_tensor(edge.anchor_coords, dtype=torch.float32)
            per_edge_slices[edge_key] = {}

            for agent_id in (edge.src_agent_id, edge.dst_agent_id):
                start = per_agent_sizes[agent_id]
                stop = start + anchor_coords.shape[0]
                per_agent_chunks[agent_id].append(anchor_coords)
                per_edge_slices[edge_key][agent_id] = slice(start, stop)
                per_agent_sizes[agent_id] = stop

        per_agent_anchor_coords: dict[int, torch.Tensor] = {}
        for agent_id, chunks in per_agent_chunks.items():
            if chunks:
                per_agent_anchor_coords[agent_id] = torch.cat(chunks, dim=0)
            else:
                per_agent_anchor_coords[agent_id] = self._empty_anchor_coords()

        return per_agent_anchor_coords, per_edge_slices

    def _build_agent_slots(self, sample: NetworkRoundSample) -> dict[int, int]:
        agent_ids = sorted({node.agent_id for node in sample.nodes})
        if len(agent_ids) > self.num_agents:
            raise ValueError(
                f"Sample contains {len(agent_ids)} unique agent IDs but LightningDUPRME "
                f"was initialized with num_agents={self.num_agents}."
            )
        return {agent_id: slot for slot, agent_id in enumerate(agent_ids)}

    def _forward_node(
        self,
        node: LocalNodeSample,
        query_coords: torch.Tensor,
        anchor_coords: torch.Tensor,
        agent_slots: dict[int, int],
    ) -> dict[str, torch.Tensor]:
        model = self.agent_models[str(agent_slots[node.agent_id])]
        obs_coords = self._to_tensor(node.obs_coords, dtype=torch.float32)
        obs_values = self._to_tensor(node.obs_values, dtype=torch.float32)
        region_mask = self._to_tensor(node.region_mask, dtype=torch.float32)
        local_building_mask = None
        if node.local_building_mask is not None:
            local_building_mask = self._to_tensor(node.local_building_mask, dtype=torch.float32)
        grid_x, grid_y, grid_z = self._grid_vectors_for_node(node)
        return model(
            obs_coords=obs_coords,
            obs_values=obs_values,
            query_coords=query_coords,
            anchor_coords=anchor_coords,
            region_mask=region_mask,
            local_building_mask=local_building_mask,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
        )

    def _edge_local_stats(
        self,
        edge: EdgeOverlapSample,
        node_outputs: dict[int, dict[str, torch.Tensor]],
        per_edge_slices: dict[tuple[int, int], dict[int, slice]],
    ) -> dict[int, dict[str, torch.Tensor]]:
        edge_key = (edge.src_agent_id, edge.dst_agent_id)
        return {
            edge.src_agent_id: {
                "mean": node_outputs[edge.src_agent_id]["anchor_mean"][
                    per_edge_slices[edge_key][edge.src_agent_id]
                ],
                "precision": node_outputs[edge.src_agent_id]["anchor_precision"][
                    per_edge_slices[edge_key][edge.src_agent_id]
                ],
            },
            edge.dst_agent_id: {
                "mean": node_outputs[edge.dst_agent_id]["anchor_mean"][
                    per_edge_slices[edge_key][edge.dst_agent_id]
                ],
                "precision": node_outputs[edge.dst_agent_id]["anchor_precision"][
                    per_edge_slices[edge_key][edge.dst_agent_id]
                ],
            },
        }

    def _require_single_sample(
        self,
        batch: list[NetworkRoundSample],
        *,
        step_name: str,
    ) -> NetworkRoundSample:
        if len(batch) != 1:
            raise ValueError(
                f"LightningDUPRME currently expects one NetworkRoundSample per {step_name} step."
            )
        return batch[0]

    def training_step(
        self,
        batch: list[NetworkRoundSample],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        sample = self._require_single_sample(batch, step_name="training")
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        agent_slots = self._build_agent_slots(sample)
        per_agent_anchor_coords, per_edge_slices = self._collect_anchor_coords(sample)

        node_outputs: dict[int, dict[str, torch.Tensor]] = {}
        local_reconstruction_outputs: dict[int, dict[str, torch.Tensor]] = {}
        local_reconstruction_targets: dict[int, torch.Tensor] = {}
        for node in sample.nodes:
            obs_query_coords = self._to_tensor(node.obs_coords, dtype=torch.float32)
            node_outputs[node.agent_id] = self._forward_node(
                node,
                obs_query_coords,
                per_agent_anchor_coords[node.agent_id],
                agent_slots,
            )
            local_query_coords, local_targets = self._extract_local_supervision(node)
            local_reconstruction_outputs[node.agent_id] = self._forward_node(
                node,
                local_query_coords,
                self._empty_anchor_coords(),
                agent_slots,
            )
            local_reconstruction_targets[node.agent_id] = local_targets

        local_stats = {
            (edge.src_agent_id, edge.dst_agent_id): self._edge_local_stats(
                edge=edge,
                node_outputs=node_outputs,
                per_edge_slices=per_edge_slices,
            )
            for edge in sample.edges
        }
        exchanged = exchange_edge_statistics(local_stats=local_stats, codec=self.codec)

        total_losses = {
            node.agent_id: torch.zeros((), dtype=torch.float32, device=self.device)
            for node in sample.nodes
        }

        for node in sample.nodes:
            output = node_outputs[node.agent_id]
            target = self._to_tensor(node.obs_values, dtype=torch.float32)
            reconstruction_output = local_reconstruction_outputs[node.agent_id]
            reconstruction_target = local_reconstruction_targets[node.agent_id]
            observation_loss = local_observation_nll(
                pred_mean=output["map_mean"],
                pred_variance=output["map_variance"],
                target=target,
            )
            reconstruction_loss = local_observation_nll(
                pred_mean=reconstruction_output["map_mean"],
                pred_variance=reconstruction_output["map_variance"],
                target=reconstruction_target,
            )
            total_losses[node.agent_id] = total_losses[node.agent_id] + (
                float(self.loss_weights["local"])
                * (observation_loss + reconstruction_loss)
            )

        for edge in sample.edges:
            edge_out = exchanged[(edge.src_agent_id, edge.dst_agent_id)]
            for agent_id in (edge.src_agent_id, edge.dst_agent_id):
                loss_terms = compute_edge_agreement_loss(
                    local_mean=node_outputs[agent_id]["anchor_mean"][
                        per_edge_slices[(edge.src_agent_id, edge.dst_agent_id)][agent_id]
                    ],
                    recv_mean=edge_out[agent_id]["received_mean"],
                    local_precision=node_outputs[agent_id]["anchor_precision"][
                        per_edge_slices[(edge.src_agent_id, edge.dst_agent_id)][agent_id]
                    ],
                    recv_precision=edge_out[agent_id]["received_precision"],
                )
                total_losses[agent_id] = total_losses[agent_id] + (
                    float(self.loss_weights["overlap"]) * loss_terms["overlap"]
                    + float(self.loss_weights["uncertainty"]) * loss_terms["uncertainty"]
                )

        for agent_id, loss in total_losses.items():
            self.manual_backward(loss)
            optimizers[agent_slots[agent_id]].step()

        return torch.stack(list(total_losses.values())).sum()

    def validation_step(
        self,
        batch: list[NetworkRoundSample],
        batch_idx: int,
    ) -> torch.Tensor:
        del batch_idx
        sample = self._require_single_sample(batch, step_name="validation")
        agent_slots = self._build_agent_slots(sample)
        per_agent_anchor_coords, per_edge_slices = self._collect_anchor_coords(sample)

        local_losses: dict[int, torch.Tensor] = {}
        node_outputs: dict[int, dict[str, torch.Tensor]] = {}
        for node in sample.nodes:
            obs_query_coords = self._to_tensor(node.obs_coords, dtype=torch.float32)
            output = self._forward_node(
                node,
                obs_query_coords,
                per_agent_anchor_coords[node.agent_id],
                agent_slots,
            )
            node_outputs[node.agent_id] = output
            local_query_coords, local_target = self._extract_local_supervision(node)
            local_reconstruction_output = self._forward_node(
                node,
                local_query_coords,
                self._empty_anchor_coords(),
                agent_slots,
            )
            target = self._to_tensor(node.obs_values, dtype=torch.float32)
            local_loss = local_observation_nll(
                pred_mean=output["map_mean"],
                pred_variance=output["map_variance"],
                target=target,
            )
            reconstruction_loss = local_observation_nll(
                pred_mean=local_reconstruction_output["map_mean"],
                pred_variance=local_reconstruction_output["map_variance"],
                target=local_target,
            )
            local_losses[node.agent_id] = local_loss + reconstruction_loss
            self.log(
                f"val/local_observation_nll_agent_{node.agent_id}",
                local_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )
            self.log(
                f"val/local_reconstruction_nll_agent_{node.agent_id}",
                reconstruction_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

        local_stats = {
            (edge.src_agent_id, edge.dst_agent_id): self._edge_local_stats(
                edge=edge,
                node_outputs=node_outputs,
                per_edge_slices=per_edge_slices,
            )
            for edge in sample.edges
        }
        exchanged = exchange_edge_statistics(local_stats=local_stats, codec=self.codec)
        total_validation_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        for edge in sample.edges:
            edge_key = (edge.src_agent_id, edge.dst_agent_id)
            edge_mean_abs: list[float] = []
            edge_log_precision_abs: list[float] = []
            for agent_id in (edge.src_agent_id, edge.dst_agent_id):
                loss_terms = compute_edge_agreement_loss(
                    local_mean=node_outputs[agent_id]["anchor_mean"][
                        per_edge_slices[edge_key][agent_id]
                    ],
                    recv_mean=exchanged[edge_key][agent_id]["received_mean"],
                    local_precision=node_outputs[agent_id]["anchor_precision"][
                        per_edge_slices[edge_key][agent_id]
                    ],
                    recv_precision=exchanged[edge_key][agent_id]["received_precision"],
                )
                total_validation_loss = total_validation_loss + (
                    float(self.loss_weights["overlap"]) * loss_terms["overlap"]
                    + float(self.loss_weights["uncertainty"]) * loss_terms["uncertainty"]
                )
                edge_metrics = compute_overlap_disagreement(
                    local_mean=node_outputs[agent_id]["anchor_mean"][
                        per_edge_slices[edge_key][agent_id]
                    ],
                    recv_mean=exchanged[edge_key][agent_id]["received_mean"],
                    local_precision=node_outputs[agent_id]["anchor_precision"][
                        per_edge_slices[edge_key][agent_id]
                    ],
                    recv_precision=exchanged[edge_key][agent_id]["received_precision"],
                )
                edge_mean_abs.append(edge_metrics["mean_abs"])
                edge_log_precision_abs.append(edge_metrics["log_precision_abs"])

            edge_label = f"{edge.src_agent_id}_{edge.dst_agent_id}"
            edge_metric_value = local_losses[edge.src_agent_id].new_tensor(
                sum(edge_mean_abs) / len(edge_mean_abs)
            )
            self.log(
                f"val/overlap_mean_abs_{edge_label}",
                edge_metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )
            edge_precision_value = local_losses[edge.src_agent_id].new_tensor(
                sum(edge_log_precision_abs) / len(edge_log_precision_abs)
            )
            self.log(
                f"val/overlap_log_precision_abs_{edge_label}",
                edge_precision_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

        total_validation_loss = total_validation_loss + torch.stack(list(local_losses.values())).sum()
        self.log(
            "val/total_loss",
            total_validation_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )
        return total_validation_loss

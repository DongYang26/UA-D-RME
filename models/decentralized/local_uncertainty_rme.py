from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.common import build_mlp, safe_precision, safe_variance


class LocalUncertaintyRME(nn.Module):
    """Local uncertainty-aware predictor for query and anchor coordinates."""

    def __init__(self, coord_dim: int = 2, hidden_dim: int = 64, anchor_dim: int = 64) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.obs_encoder = build_mlp(
            input_dim=coord_dim + 1 + hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )
        self.env_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.summary_fuser = build_mlp(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
        )
        self.query_head = build_mlp(
            input_dim=coord_dim + hidden_dim + hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=2,
            num_layers=2,
        )
        self.anchor_head = build_mlp(
            input_dim=coord_dim + hidden_dim + hidden_dim,
            hidden_dim=anchor_dim,
            output_dim=2,
            num_layers=2,
        )
        self.empty_obs_summary = nn.Parameter(torch.zeros(hidden_dim))
        self.empty_env_summary = nn.Parameter(torch.zeros(hidden_dim))

    def _prepare_environment_image(
        self,
        local_building_mask: torch.Tensor,
    ) -> torch.Tensor:
        if local_building_mask.ndim == 2:
            return local_building_mask.to(dtype=self.empty_env_summary.dtype)
        if local_building_mask.ndim == 3:
            # 2.5D environment: collapse broadcast Z slices back to an XY height map.
            return torch.amax(local_building_mask, dim=0).to(dtype=self.empty_env_summary.dtype)
        raise ValueError("local_building_mask must be [H, W] or [D, H, W].")

    def _collapse_region_mask_xy(
        self,
        region_mask: torch.Tensor,
    ) -> torch.Tensor:
        if region_mask.ndim == 2:
            return region_mask
        if region_mask.ndim == 3:
            return torch.amax(region_mask, dim=0)
        raise ValueError("region_mask must be [H, W] or [D, H, W].")

    def _coords_to_nearest_xy_indices(
        self,
        coords: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
    ) -> torch.Tensor:
        if coords.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long, device=coords.device)
        x_deltas = torch.abs(grid_x.unsqueeze(0) - coords[:, 0].unsqueeze(1))
        y_deltas = torch.abs(grid_y.unsqueeze(0) - coords[:, 1].unsqueeze(1))
        xs = torch.argmin(x_deltas, dim=1)
        ys = torch.argmin(y_deltas, dim=1)
        return torch.stack([ys, xs], dim=-1)

    def _encode_environment_feature_map(
        self,
        local_building_mask: torch.Tensor,
    ) -> torch.Tensor:
        env_image = self._prepare_environment_image(local_building_mask)
        return self.env_encoder(env_image.unsqueeze(0).unsqueeze(0)).squeeze(0)

    def _sample_environment_features(
        self,
        env_feature_map: torch.Tensor | None,
        coords: torch.Tensor,
        grid_x: torch.Tensor | None,
        grid_y: torch.Tensor | None,
        grid_z: torch.Tensor | None,
    ) -> torch.Tensor:
        del grid_z
        channel_dim = self.empty_env_summary.shape[0]
        if env_feature_map is None or grid_x is None or grid_y is None or coords.numel() == 0:
            return torch.zeros((coords.shape[0], channel_dim), dtype=self.empty_env_summary.dtype, device=coords.device)
        indices = self._coords_to_nearest_xy_indices(coords, grid_x=grid_x, grid_y=grid_y)
        ys = indices[:, 0]
        xs = indices[:, 1]
        return env_feature_map[:, ys, xs].transpose(0, 1)

    def _summarize_observations(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_env_features: torch.Tensor,
    ) -> torch.Tensor:
        if obs_coords.numel() == 0:
            return self.empty_obs_summary
        obs_tokens = torch.cat([obs_coords, obs_values.unsqueeze(-1), obs_env_features], dim=-1)
        encoded = self.obs_encoder(obs_tokens)
        return encoded.mean(dim=0)

    def _summarize_environment(
        self,
        region_mask: torch.Tensor | None,
        env_feature_map: torch.Tensor | None,
    ) -> torch.Tensor:
        if region_mask is None or env_feature_map is None:
            return self.empty_env_summary
        region_mask_xy = self._collapse_region_mask_xy(region_mask).to(dtype=torch.bool)
        if not torch.any(region_mask_xy):
            return self.empty_env_summary
        masked = env_feature_map[:, region_mask_xy]
        return masked.mean(dim=1)

    def _condition_with_summary(
        self,
        coords: torch.Tensor,
        sampled_env_features: torch.Tensor,
        summary: torch.Tensor,
    ) -> torch.Tensor:
        expanded_summary = summary.unsqueeze(0).expand(coords.shape[0], -1)
        return torch.cat([coords, sampled_env_features, expanded_summary], dim=-1)

    def forward(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        query_coords: torch.Tensor,
        anchor_coords: torch.Tensor,
        region_mask: torch.Tensor | None = None,
        local_building_mask: torch.Tensor | None = None,
        grid_x: torch.Tensor | None = None,
        grid_y: torch.Tensor | None = None,
        grid_z: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        env_feature_map = (
            None
            if local_building_mask is None
            else self._encode_environment_feature_map(local_building_mask)
        )
        obs_env_features = self._sample_environment_features(
            env_feature_map=env_feature_map,
            coords=obs_coords,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
        )
        query_env_features = self._sample_environment_features(
            env_feature_map=env_feature_map,
            coords=query_coords,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
        )
        anchor_env_features = self._sample_environment_features(
            env_feature_map=env_feature_map,
            coords=anchor_coords,
            grid_x=grid_x,
            grid_y=grid_y,
            grid_z=grid_z,
        )
        obs_summary = self._summarize_observations(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_env_features=obs_env_features,
        )
        env_summary = self._summarize_environment(
            region_mask=region_mask,
            env_feature_map=env_feature_map,
        )
        summary = self.summary_fuser(torch.cat([obs_summary, env_summary], dim=-1))
        query_out = self.query_head(self._condition_with_summary(query_coords, query_env_features, summary))
        anchor_out = self.anchor_head(self._condition_with_summary(anchor_coords, anchor_env_features, summary))

        map_mean = query_out[:, 0]
        map_variance = safe_variance(query_out[:, 1])
        anchor_mean = anchor_out[:, 0]
        anchor_precision = safe_precision(anchor_out[:, 1])
        return {
            "map_mean": map_mean,
            "map_variance": map_variance,
            "anchor_mean": anchor_mean,
            "anchor_precision": anchor_precision,
        }

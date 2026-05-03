from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from scripts._bootstrap import bootstrap_project_root  # noqa: E402

bootstrap_project_root()

from datasets.decentralized.network_round_dataset import NetworkRoundDataset
from trainers.lightning_duprme_data import LightningDecentralizedRMEDataModule
from trainers.lightning_duprme_module import LightningDUPRME
from utils.config import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train DUPRME with PyTorch Lightning.")
    parser.add_argument("--config", default="configs/train/duprme.yaml")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data.data_root in the loaded config (resolves relative manifest paths).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override trainer.epochs in the loaded config.",
    )
    parser.add_argument(
        "--accelerator",
        default=None,
        help="Override trainer.accelerator (e.g. 'cpu', 'gpu', 'auto').",
    )
    return parser


def _configure_runtime_env(trainer_config: dict[str, Any]) -> None:
    env_config = dict(trainer_config.get("env", {}))
    for key, value in env_config.items():
        os.environ.setdefault(str(key), str(value))


def _load_npz_payload(path: str, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in keys}


def _load_local_node_artifact(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        data = {
            "obs_coords": payload["obs_coords"],
            "obs_values": payload["obs_values"],
            "region_mask": payload["region_mask"],
            "target_map": payload["target_map"],
            "target_mask": payload["target_mask"],
        }
        if "local_building_mask" in payload.files:
            data["local_building_mask"] = payload["local_building_mask"]
        if "grid_x" in payload.files:
            data["grid_x"] = payload["grid_x"]
        if "grid_y" in payload.files:
            data["grid_y"] = payload["grid_y"]
        if "grid_z" in payload.files:
            data["grid_z"] = payload["grid_z"]
        return data


def _load_edge_overlap_artifact(path: str) -> dict[str, np.ndarray]:
    return _load_npz_payload(path, ("anchor_coords", "overlap_mask"))


def _build_dataset(
    *,
    split: str | None,
    data_config: dict[str, Any],
    graph_config: dict[str, Any],
) -> NetworkRoundDataset | None:
    if not split:
        return None
    return NetworkRoundDataset(
        node_manifest_path=str(data_config["node_manifest_path"]),
        edge_manifest_path=str(data_config["edge_manifest_path"]),
        split=split,
        node_loader=_load_local_node_artifact,
        edge_loader=_load_edge_overlap_artifact,
        num_agents=int(graph_config["num_agents"]),
        graph_edges=graph_config["graph_edges"],
        data_root=data_config.get("data_root"),
    )


def _build_callbacks(
    *,
    trainer_config: dict[str, Any],
    has_validation: bool,
    output_dir: Path,
) -> list[Any]:
    if not has_validation:
        return []

    callbacks: list[Any] = []
    checkpoint_config = dict(trainer_config.get("checkpoint", {}))
    if checkpoint_config.get("enabled", True):
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir / "checkpoints",
                monitor=str(checkpoint_config.get("monitor", "val/total_loss")),
                mode=str(checkpoint_config.get("mode", "min")),
                save_top_k=int(checkpoint_config.get("save_top_k", 1)),
                save_last=bool(checkpoint_config.get("save_last", True)),
                filename="{epoch:03d}-{val_total_loss:.6f}",
                auto_insert_metric_name=False,
            )
        )

    early_stopping_config = dict(trainer_config.get("early_stopping", {}))
    if early_stopping_config.get("enabled", True):
        callbacks.append(
            EarlyStopping(
                monitor=str(early_stopping_config.get("monitor", "val/total_loss")),
                mode=str(early_stopping_config.get("mode", "min")),
                patience=int(early_stopping_config.get("patience", 10)),
                min_delta=float(early_stopping_config.get("min_delta", 0.0)),
            )
        )
    return callbacks


def train_with_config(config: dict[str, Any]) -> dict[str, str]:
    trainer_config = dict(config["trainer"])
    graph_config = dict(config["graph"])
    data_config = dict(config["data"])

    _configure_runtime_env(trainer_config)
    seed_everything(int(trainer_config.get("seed", 0)), workers=True)

    output_dir = Path(trainer_config.get("output_dir", "artifacts/train/duprme"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _build_dataset(
        split=data_config.get("train_split", "train"),
        data_config=data_config,
        graph_config=graph_config,
    )
    val_dataset = _build_dataset(
        split=data_config.get("val_split"),
        data_config=data_config,
        graph_config=graph_config,
    )
    if val_dataset is not None and len(val_dataset) == 0:
        val_dataset = None

    datamodule = LightningDecentralizedRMEDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=int(trainer_config.get("batch_size", 1)),
        num_workers=int(trainer_config.get("num_workers", 0)),
        pin_memory=bool(trainer_config.get("pin_memory", False)),
        persistent_workers=bool(trainer_config.get("persistent_workers", False)),
        prefetch_factor=trainer_config.get("prefetch_factor"),
    )

    model = LightningDUPRME(
        num_agents=int(graph_config["num_agents"]),
        model_config=dict(config["model"]),
        loss_weights=dict(config["losses"]),
        optimizer_config={
            "lr": trainer_config["lr"],
            "weight_decay": trainer_config.get("weight_decay", 0.0),
        },
        codec_config=dict(config.get("codec", {"mode": "none"})),
    )

    trainer_kwargs: dict[str, Any] = {
        "default_root_dir": str(output_dir),
        "max_epochs": int(trainer_config.get("epochs", 1)),
    }
    if val_dataset is None:
        trainer_kwargs["limit_val_batches"] = 0
        trainer_kwargs["num_sanity_val_steps"] = 0
    else:
        callbacks = _build_callbacks(
            trainer_config=trainer_config,
            has_validation=True,
            output_dir=output_dir,
        )
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks
    for key in ("accelerator", "devices", "strategy", "precision"):
        if key in trainer_config:
            trainer_kwargs[key] = trainer_config[key]

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, datamodule=datamodule)

    return {"output_dir": str(output_dir)}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.data_root is not None:
        config.setdefault("data", {})["data_root"] = args.data_root
    if args.max_epochs is not None:
        config.setdefault("trainer", {})["epochs"] = args.max_epochs
    if args.accelerator is not None:
        trainer_section = config.setdefault("trainer", {})
        trainer_section["accelerator"] = args.accelerator
        if args.accelerator == "cpu":
            trainer_section["devices"] = 1
            trainer_section["strategy"] = "auto"
            trainer_section["precision"] = "32-true"
    print(train_with_config(config))


if __name__ == "__main__":
    main()

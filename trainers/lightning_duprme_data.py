from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


def collate_network_rounds(batch: list[Any]) -> list[Any]:
    return batch


class LightningDecentralizedRMEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset[Any] | None,
        val_dataset: Dataset[Any] | None = None,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int | None = None,
    ) -> None:
        super().__init__()
        if int(batch_size) != 1:
            raise ValueError(
                "LightningDecentralizedRMEDataModule supports only batch_size=1 "
                "because LightningDUPRME consumes one NetworkRoundSample per step."
            )
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.persistent_workers = bool(persistent_workers) and self.num_workers > 0
        self.prefetch_factor = prefetch_factor

    def _loader_kwargs(self, shuffle: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": collate_network_rounds,
        }
        if self.num_workers > 0 and self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)
        return kwargs

    def train_dataloader(self) -> DataLoader[Any]:
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized.")
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader[Any]:
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized.")
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

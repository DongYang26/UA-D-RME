"""Convert raw absolute-path CSV manifests into relative-path Parquet bundles.

Usage:
    python scripts/package_manifest.py \
        --node-manifest /abs/path/local_node_manifest.csv \
        --edge-manifest /abs/path/edge_overlap_manifest.csv \
        --source-root  /abs/path/                            \
        --dest-root    data/sample                           \
        [--scenes-per-split 30]

The script does the following:
1. Reads the source CSV manifests.
2. Optionally retains only the first ``--scenes-per-split N`` scene_ids per
   ``split`` value (deterministic alphabetical order). When ``--scenes-per-split``
   is omitted, the full manifest is exported.
3. Rewrites every ``artifact_path`` to be relative to ``--source-root`` (paths
   are required to start with ``local_nodes/`` or ``edge_overlaps/``).
4. Writes ``local_node_manifest.parquet`` and ``edge_overlap_manifest.parquet``
   to ``--dest-root`` and copies the referenced ``.npz`` shards into
   ``<dest-root>/local_nodes/`` and ``<dest-root>/edge_overlaps/``.
5. Re-reads the parquet outputs and asserts every relative path resolves to an
   existing file under ``--dest-root``.

After running, ``<dest-root>`` is a self-contained directory that can be used
as ``data.data_root`` (or, equivalently, as the manifest directory under the
default behaviour where ``data_root`` is unset).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REQUIRED_NODE_COLS = ("scene_id", "agent_id", "split", "artifact_path")
REQUIRED_EDGE_COLS = ("scene_id", "src_agent_id", "dst_agent_id", "split", "artifact_path")
ALLOWED_SUBDIRS = ("local_nodes", "edge_overlaps")


def _read_csv(path: Path, required_cols: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [col for col in required_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Manifest {path} is missing required columns: {missing}")
    return frame


def _stratified_scene_filter(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    scenes_per_split: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_scene_ids: list[str] = []
    for split_value, group in nodes.groupby("split", sort=True):
        scene_ids = sorted(group["scene_id"].astype(str).unique())
        kept_scene_ids.extend(scene_ids[:scenes_per_split])
    kept_set = set(kept_scene_ids)
    filtered_nodes = nodes[nodes["scene_id"].astype(str).isin(kept_set)].reset_index(drop=True)
    filtered_edges = edges[edges["scene_id"].astype(str).isin(kept_set)].reset_index(drop=True)
    return filtered_nodes, filtered_edges


def _relativize(absolute_path: str, source_root: Path) -> str:
    abs_p = Path(absolute_path)
    if not abs_p.is_absolute():
        rel = abs_p
    else:
        try:
            rel = abs_p.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(
                f"Artifact path {abs_p} is not under source_root {source_root}."
            ) from exc
    parts = rel.parts
    if not parts or parts[0] not in ALLOWED_SUBDIRS:
        raise ValueError(
            f"Artifact path must start with one of {ALLOWED_SUBDIRS}: got {rel}"
        )
    return rel.as_posix()


def _copy_shards(
    relative_paths: list[str],
    source_root: Path,
    dest_root: Path,
) -> int:
    bytes_copied = 0
    for rel in relative_paths:
        src = source_root / rel
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            raise FileNotFoundError(f"Missing source shard {src}")
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        bytes_copied += dst.stat().st_size
    return bytes_copied


def _verify_dest(parquet_path: Path, dest_root: Path) -> None:
    frame = pd.read_parquet(parquet_path)
    for rel in frame["artifact_path"].tolist():
        full = dest_root / rel
        if not full.exists():
            raise FileNotFoundError(f"Verified path missing: {full}")


def package(
    *,
    node_manifest: Path,
    edge_manifest: Path,
    source_root: Path,
    dest_root: Path,
    scenes_per_split: int | None,
) -> dict[str, int]:
    nodes = _read_csv(node_manifest, REQUIRED_NODE_COLS)
    edges = _read_csv(edge_manifest, REQUIRED_EDGE_COLS)

    if scenes_per_split is not None:
        nodes, edges = _stratified_scene_filter(nodes, edges, scenes_per_split)

    nodes = nodes.copy()
    edges = edges.copy()
    nodes["artifact_path"] = nodes["artifact_path"].astype(str).map(
        lambda p: _relativize(p, source_root)
    )
    edges["artifact_path"] = edges["artifact_path"].astype(str).map(
        lambda p: _relativize(p, source_root)
    )

    dest_root.mkdir(parents=True, exist_ok=True)
    node_parquet = dest_root / "local_node_manifest.parquet"
    edge_parquet = dest_root / "edge_overlap_manifest.parquet"
    nodes.to_parquet(node_parquet, engine="pyarrow", index=False)
    edges.to_parquet(edge_parquet, engine="pyarrow", index=False)

    relative_paths = sorted(set(nodes["artifact_path"].tolist() + edges["artifact_path"].tolist()))
    bytes_copied = _copy_shards(relative_paths, source_root, dest_root)

    _verify_dest(node_parquet, dest_root)
    _verify_dest(edge_parquet, dest_root)

    return {
        "node_rows": int(len(nodes)),
        "edge_rows": int(len(edges)),
        "scenes": int(nodes["scene_id"].nunique()),
        "shard_files": int(len(relative_paths)),
        "bytes_copied": int(bytes_copied),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--node-manifest", required=True, type=Path)
    parser.add_argument("--edge-manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--dest-root", required=True, type=Path)
    parser.add_argument(
        "--scenes-per-split",
        type=int,
        default=None,
        help="Per-split deterministic top-N scene_id selection. Omit for full export.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = package(
        node_manifest=args.node_manifest,
        edge_manifest=args.edge_manifest,
        source_root=args.source_root,
        dest_root=args.dest_root,
        scenes_per_split=args.scenes_per_split,
    )
    print(summary)


if __name__ == "__main__":
    main()

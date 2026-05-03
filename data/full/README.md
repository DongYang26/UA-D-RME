# Full DUPRME dataset

This directory is a placeholder for the full ~15 GB processed DUPRME dataset.
The bundled `data/sample/` directory at the repository root contains a small
stratified subsample (~22 MB) suitable for smoke-testing the pipeline.

## Download

Download the full dataset archive from the link below:

> **Google Drive:** `It will be released.`

The archive contains:

```
local_node_manifest.parquet      # ~448,640 rows
edge_overlap_manifest.parquet    # ~560,800 rows
local_nodes/*.npz                # ~448,640 per-agent local artifacts
edge_overlaps/*.npz              # ~560,800 per-edge overlap artifacts
```

Total uncompressed footprint is approximately 15 GB.

## Layout

After download, extract the archive into this directory so the resulting
layout matches:

```
data/full/
├── local_node_manifest.parquet
├── edge_overlap_manifest.parquet
├── local_nodes/
│   └── *.npz
└── edge_overlaps/
    └── *.npz
```

The bundled `configs/train/duprme.yaml` already points at this layout. Both
manifests are expected to live in the same directory (`NetworkRoundDataset`
defaults `data_root` to the directory containing the node manifest). If you
prefer to keep the manifests and the shards apart, set `data.data_root`
explicitly in the YAML or pass `--data-root PATH` to
`scripts/train_duprme_lightning.py`.

## Verification

After extraction, sanity-check the row counts and split coverage:

```bash
python -c "
import pandas as pd
n = pd.read_parquet('data/full/local_node_manifest.parquet')
print('node rows:', len(n))
print('per-split scenes:', n.groupby('split')['scene_id'].nunique().to_dict())
"
```

You should see a roughly 12:1:1 train/val/test split across approximately
56,000 unique scenes.

## Provenance

The dataset is derived from the RadioMapSeer simulator and partitioned into
8-agent decentralized scenes with a fixed grid communication topology. The
generation pipeline lives in the parent multi-agent research codebase and is
out of scope for this release; only the consumer-side artifacts are shipped.

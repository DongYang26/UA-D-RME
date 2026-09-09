# UA-D-RME

## Outline

```
UA-D-RME/
├── README.md                              this file
├── LICENSE
├── requirements.txt                       runtime + test dependencies
├── configs/train/
│   ├── duprme.yaml                        full 8-GPU DDP configuration
│   └── duprme_singlegpu.yaml              single-GPU toy configuration
├── data/
│   ├── sample/                            bundled 90-scene sample (~22 MB for toy test)
│   │   ├── local_node_manifest.parquet
│   │   ├── edge_overlap_manifest.parquet
│   │   ├── local_nodes/*.npz              720 per-agent local artifacts
│   │   └── edge_overlaps/*.npz            900 per-edge overlap artifacts
│   └── full/
│       └── README.md                      Data download instructions
├── datasets/decentralized/                schemas + parquet/CSV-aware loader
├── models/                                slim model utilities + decentralized predictor
│   ├── common.py                          build_mlp / safe_precision / safe_variance
│   └── decentralized/                     LocalUncertaintyRME, codecs, exchange
├── trainers/                              Lightning module + Lightning datamodule
├── losses/                                local NLL + edge agreement losses
├── evaluation/                            decentralized metrics
├── utils/                                 small YAML config helper
├── scripts/
│   ├── _bootstrap.py                      sys.path shim for direct script invocation
│   ├── train_duprme_lightning.py          training entry point
│   ├── package_manifest.py                CSV → parquet relativizer
│   └── verify.sh                          end-to-end verification gate
└── Supplementary.pdf                      The detailed proofs for the theorems in manuscript.
```

## Requirements

Install dependencies into a fresh virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> The verification script `scripts/verify.sh` deliberately runs the smoke
> on CPU to remain portable across torch wheels and CUDA versions.

## Start

```bash
git clone git@github.com:DongYang26/UA-D-RME.git
cd UA-D-RME
pip install -r requirements.txt
python scripts/train_duprme_lightning.py --config configs/train/duprme.yaml
```

>≥10 GB GPU memory for `duprme_singlegpu.yaml`; ≥24 GB GPU memory per
> device for the full `duprme.yaml` 8-GPU DDP configuration

> **Sample data purpose.** The bundled `data/sample/` is a toy test that
> proves the code runs end-to-end and produces a finite `val/total_loss`. It
> is **not** a paper-reproduction sample. To reproduce paper metrics,
> download the full dataset (see below).

## Full dataset

The full processed dataset is ~15 GB across two manifests and roughly one
million `.npz` shards. It is hosted on Google Drive:

> **Full dataset download:** `[To be updated]`

After downloading, extract into `data/full/` so that the layout becomes:

```
data/full/
├── local_node_manifest.parquet
├── edge_overlap_manifest.parquet
├── local_nodes/*.npz
└── edge_overlaps/*.npz
```

Then launch full training:

```bash
python scripts/train_duprme_lightning.py --config configs/train/duprme.yaml
```

The full configuration uses 8-GPU DDP, 16-mixed precision, and 1000 epochs
with early stopping on `val/total_loss`. Adjust `trainer.devices`,
`trainer.strategy`, and `trainer.precision` to match your hardware.

## Repackaging your own data

If you produce UA-D-RME-format `.npz` shards from a different source dataset
and want to ship them with relative paths and a parquet manifest, run:

```bash
python scripts/package_manifest.py \
    --node-manifest /path/to/local_node_manifest.csv \
    --edge-manifest /path/to/edge_overlap_manifest.csv \
    --source-root  /path/to/dataset_root             \
    --dest-root    data/full
```

Add `--scenes-per-split N` to produce a stratified subsample.

## License

See [`LICENSE`](LICENSE) for the full license text.


## Contact

For reproduction questions, please contact: dnyang26@gmail.com

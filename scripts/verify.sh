#!/usr/bin/env bash
# Self-contained end-to-end verification for the DUPRME release tarball.
#
# Runs every check that gates "open-source-ready":
#   1. Fresh-venv pip install of declared requirements.
#   2. Manifest filename consistency between package_manifest.py outputs and YAML configs.
#   3. Offline unit tests (no GPU, no sample data needed).
#   4. Slim-closure greps: forbidden upstream symbols are absent from the slim package.
#   5. Sample manifest val/train/test rows are all non-empty.
#   6. End-to-end 1-epoch smoke run on the bundled sample with --data-root override.
#   7. Checkpoint file existence after the smoke run.
#   8. External-data relocation: copy sample to /tmp/external_data, run with --data-root.
#
# Exits 0 only if every step passes.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

PY_BIN="${PY_BIN:-python}"
VENV_DIR="${VENV_DIR:-/tmp/duprme_venv}"

echo "[1/8] fresh-venv pip install"
if [ ! -d "$VENV_DIR" ]; then
  "$PY_BIN" -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet -r requirements.txt
VENV_PY="$VENV_DIR/bin/python"

echo "[2/8] manifest filename consistency (acceptance A11)"
grep -q "local_node_manifest.parquet" configs/train/duprme.yaml
grep -q "edge_overlap_manifest.parquet" configs/train/duprme.yaml
grep -q "local_node_manifest.parquet" configs/train/duprme_singlegpu.yaml
grep -q "edge_overlap_manifest.parquet" configs/train/duprme_singlegpu.yaml

echo "[3/8] unit tests"
"$VENV_PY" -m pytest -q tests/

echo "[4/8] slim-closure forbidden-symbol grep"
forbidden=$(grep -rE '(AgentEpisode|CanonicalScene|to_torch_scene|to_torch_episode|FourierFeatureEncoding|build_dense_query_coords|normalize_coords|relative_tx_features|coords_to_nearest_indices|sample_context|gather_spatial_values|precision_to_variance|masked_mean|ensure_list|TorchScene|TorchEpisode)' \
  --include='*.py' . || true)
if [ -n "$forbidden" ]; then
  echo "FAIL: forbidden upstream symbol referenced inside ./duprme/:"
  echo "$forbidden"
  exit 1
fi

echo "[5/8] sample manifest val-non-empty assertion"
"$VENV_PY" -c "
import pandas as pd
n = pd.read_parquet('data/sample/local_node_manifest.parquet')
for split in ('train', 'val', 'test'):
    count = (n['split'] == split).sum()
    assert count > 0, f'sample manifest has zero {split} rows'
    print(f'{split}: {count} rows')
"

echo "[6/8] 1-epoch CPU smoke run"
# CPU smoke is intentional: makes verify.sh portable across torch wheels, CUDA versions,
# and reviewer hardware. --data-root is also passed deliberately to exercise the CLI
# override path even though the YAML has data_root: null. Reviewers with a GPU can run
# the same config without --accelerator cpu to get a GPU smoke.
"$VENV_PY" scripts/train_duprme_lightning.py \
  --config configs/train/duprme_singlegpu.yaml \
  --data-root "$HERE/data/sample" \
  --max-epochs 1 \
  --accelerator cpu

echo "[7/8] checkpoint existence"
shopt -s nullglob
CKPTS=( artifacts/train/duprme*/checkpoints/*.ckpt )
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "FAIL: no checkpoint produced under artifacts/train/duprme*/checkpoints/"
  exit 1
fi
echo "  checkpoint(s): ${CKPTS[*]}"

echo "[8/8] external-data relocation (CPU)"
EXT_DATA="/tmp/duprme_external_data_$$"
rm -rf "$EXT_DATA"
cp -r data/sample "$EXT_DATA"
"$VENV_PY" scripts/train_duprme_lightning.py \
  --config configs/train/duprme_singlegpu.yaml \
  --data-root "$EXT_DATA" \
  --max-epochs 1 \
  --accelerator cpu
rm -rf "$EXT_DATA"

echo "OK: all 8 verification steps passed."

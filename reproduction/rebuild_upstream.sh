#!/usr/bin/env bash
set -euo pipefail

: "${QUANT_DATA_ROOT:?source setup_env.sh first}"
: "${TEAMMATE_WT:?source setup_env.sh first}"
: "${SOTA_WORK:?source setup_env.sh first}"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPECTED_UPSTREAM="0ad2c976d02ff3fc2d1965b5cb0a484800503470"
if [[ -f "$TEAMMATE_WT/UPSTREAM_COMMIT" ]]; then
  ACTUAL_UPSTREAM="$(tr -d '[:space:]' < "$TEAMMATE_WT/UPSTREAM_COMMIT")"
else
  ACTUAL_UPSTREAM="$(git -C "$TEAMMATE_WT" rev-parse HEAD)"
fi
if [[ "$ACTUAL_UPSTREAM" != "$EXPECTED_UPSTREAM" ]]; then
  echo "TEAMMATE_WT commit differs: $ACTUAL_UPSTREAM != $EXPECTED_UPSTREAM" >&2
  exit 1
fi

"$PYTHON_BIN" prepare_parent_official.py

"$PYTHON_BIN" -m repro.baseline02.train_candidate \
  --candidate cs_rank --folds 6 7 8 --production \
  --parent-oof-dir "$TEAMMATE_WT/artifacts/oof" \
  --train-dir "$QUANT_DATA_ROOT/train" --test-dir "$QUANT_DATA_ROOT/test" \
  --out "$SOTA_WORK/cs_rank"

"$PYTHON_BIN" -m repro.baseline05.train_long \
  --folds 6 7 8 --production \
  --parent-oof-dir "$TEAMMATE_WT/artifacts/oof" \
  --train-dir "$QUANT_DATA_ROOT/train" --test-dir "$QUANT_DATA_ROOT/test" \
  --out "$SOTA_WORK/cs_rank_long"

"$PYTHON_BIN" -m repro.baseline03.train_model_c_long \
  --folds 6 7 8 --gap 0 \
  --parent-members "$SOTA_WORK/parent_official" \
  --train-dir "$QUANT_DATA_ROOT/train" \
  --out "$SOTA_WORK/model_c_long"

"$PYTHON_BIN" -m repro.baseline03.train_model_c_long \
  --production --gap 0 \
  --train-dir "$QUANT_DATA_ROOT/train" --test-dir "$QUANT_DATA_ROOT/test" \
  --out "$SOTA_WORK/model_c_long/production"

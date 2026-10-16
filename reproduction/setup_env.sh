#!/usr/bin/env bash
# === Environment setup for reproducing the 0.0039380 submission ===
# Edit these paths to match your machine, then: source setup_env.sh

# Competition data root (contains train/ and test/ subdirs with parquet partitions)
export QUANT_DATA_ROOT="${QUANT_DATA_ROOT:-/absolute/path/to/public_release_20260630/data}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Writable copy of upstream/ containing repro/ plus generated artifacts/.
# The source-only snapshot is bundled at "$PACKAGE_DIR/upstream".
export TEAMMATE_WT="${TEAMMATE_WT:-$PACKAGE_DIR/upstream}"

# SOTA work directory (contains all member fold_*.npz and production/*.parquet)
export SOTA_WORK="${SOTA_WORK:-/absolute/path/to/repro_work/BEST_0.0039380}"

# Python environment (activate your venv first, or override PYTHON_BIN)
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$TEAMMATE_WT:${PYTHONPATH:-}"

mkdir -p "$SOTA_WORK"

echo "Environment configured:"
echo "  QUANT_DATA_ROOT = $QUANT_DATA_ROOT"
echo "  TEAMMATE_WT     = $TEAMMATE_WT"
echo "  SOTA_WORK       = $SOTA_WORK"
echo "  PYTHON_BIN      = $PYTHON_BIN"

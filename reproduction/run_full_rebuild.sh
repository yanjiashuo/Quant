#!/usr/bin/env bash
set -euo pipefail

: "${QUANT_DATA_ROOT:?source setup_env.sh first}"
: "${TEAMMATE_WT:?source setup_env.sh first}"
: "${SOTA_WORK:?source setup_env.sh first}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PACKAGE_DIR"

"$PYTHON_BIN" audit_package.py --stage environment
bash rebuild_upstream.sh

CSRD_LAG=1 "$PYTHON_BIN" scripts/train_cs_rank_delta.py
CSRD_LAG=5 "$PYTHON_BIN" scripts/train_cs_rank_delta.py
"$PYTHON_BIN" scripts/train_cs_rank_vol.py
"$PYTHON_BIN" scripts/train_asset_demean.py
"$PYTHON_BIN" scripts/train_double_demean.py
"$PYTHON_BIN" scripts/train_double_demean_assetaware.py
"$PYTHON_BIN" scripts/train_mlp_member.py
"$PYTHON_BIN" scripts/train_mlp_variants.py
"$PYTHON_BIN" scripts/train_mlp_deep.py

"$PYTHON_BIN" audit_package.py --stage members
"$PYTHON_BIN" build_15m.py
"$PYTHON_BIN" audit_package.py --stage final

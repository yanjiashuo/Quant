"""Shared path contract for every BEST_0.0039380 training entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _required_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(
            f"{name} is not set. Edit and source BEST_0.0039380/setup_env.sh first."
        )
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


DATA_ROOT = _required_path("QUANT_DATA_ROOT")
TEAMMATE_WT = _required_path("TEAMMATE_WT")
SOTA_WORK = Path(os.environ.get("SOTA_WORK", "")).expanduser().resolve()
if not os.environ.get("SOTA_WORK"):
    raise RuntimeError(
        "SOTA_WORK is not set. Edit and source BEST_0.0039380/setup_env.sh first."
    )

TRAIN = DATA_ROOT / "train"
TEST = DATA_ROOT / "test"
if not TRAIN.is_dir() or not TEST.is_dir():
    raise FileNotFoundError(
        f"QUANT_DATA_ROOT must contain train/ and test/: {DATA_ROOT}"
    )
if not (TEAMMATE_WT / "repro").is_dir():
    raise FileNotFoundError(f"TEAMMATE_WT lacks repro/: {TEAMMATE_WT}")

sys.path.insert(0, str(TEAMMATE_WT))


def member_work(name: str) -> Path:
    path = SOTA_WORK / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def cs_rank_fold(fold: int) -> Path:
    path = SOTA_WORK / "cs_rank" / f"fold_{fold}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing cs_rank fold {fold}: {path}; run rebuild_upstream.sh first"
        )
    return path

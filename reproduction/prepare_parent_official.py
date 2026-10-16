"""Re-key the frozen five-member OOF files to official parquet row_id values."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def required_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"{name} is not set; source setup_env.sh")
    return Path(raw).expanduser().resolve()


def main() -> None:
    data = required_path("QUANT_DATA_ROOT")
    wt = required_path("TEAMMATE_WT")
    work = required_path("SOTA_WORK")
    out = work / "parent_official"
    out.mkdir(parents=True, exist_ok=True)

    for fold in (6, 7, 8):
        source_path = wt / "artifacts" / "oof" / f"fold{fold}_five_member_oof.npz"
        frame = pd.read_parquet(
            data / "train" / f"train_partition_{fold:03d}.parquet",
            columns=["row_id", "time_id", "asset_id"],
        )
        with np.load(source_path, allow_pickle=False) as source:
            if not np.array_equal(source["time_id"], frame["time_id"].to_numpy(np.int64)):
                raise RuntimeError(f"fold{fold} time_id differs from official parquet")
            if not np.array_equal(source["asset_id"], frame["asset_id"].to_numpy(np.int64)):
                raise RuntimeError(f"fold{fold} asset_id differs from official parquet")
            payload = {name: np.asarray(source[name]) for name in source.files}
        payload["row_id"] = frame["row_id"].to_numpy(np.int64)
        # Baseline03's frozen loader uses the historical y/w aliases.
        payload["y"] = np.asarray(payload["target"], dtype=np.float64)
        payload["w"] = np.asarray(payload["weight"], dtype=np.float64)
        np.savez(out / f"fold_{fold}.npz", **payload)
        print(f"fold{fold}: {len(frame):,} rows -> {out / f'fold_{fold}.npz'}")


if __name__ == "__main__":
    main()

"""Fail-fast audit for the BEST_0.0039380 reproduction chain."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COMMIT = "0ad2c976d02ff3fc2d1965b5cb0a484800503470"
EXPECTED_ROWS = 3_217_458
EXPECTED_MD5 = "92ef61c93ac86f14f606f0ea0bc66dd0"
EXPECTED_SHA256 = "ae254141a97dafd20ba0b885c2351c5273e51128eede8c808804f996fd44df2f"
MEMBER_KEYS = {
    "cs_rank_delta": "cs_rank_delta",
    "cs_rank_delta5": "cs_rank_delta5",
    "cs_rank_vol": "cs_rank_vol",
    "asset_demean": "asset_demean",
    "double_demean": "double_demean",
    "double_demean_assetaware": "double_demean_assetaware",
    "mlp_ad": "mlp_member",
    "mlp_member": "mlp_member",
    "mlp_deep": "mlp_member",
}
PUBLIC_FILES = {
    "cs_rank_delta": "production/public_cs_rank_delta.parquet",
    "cs_rank_delta5": "production/public_cs_rank_delta5.parquet",
    "cs_rank_vol": "production/public_cs_rank_vol.parquet",
    "asset_demean": "production/public_asset_demean.parquet",
    "double_demean": "production/public_double_demean.parquet",
    "double_demean_assetaware": "production/public_double_demean_assetaware.parquet",
    "mlp_ad": "production/public_mlp_member.parquet",
    "mlp_member": "production/public_mlp_member.parquet",
    "mlp_deep": "production/public_mlp_member.parquet",
}


def env_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"{name} is not set; source setup_env.sh")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_environment() -> tuple[Path, Path, Path]:
    data = env_path("QUANT_DATA_ROOT")
    wt = env_path("TEAMMATE_WT")
    work = env_path("SOTA_WORK")
    if len(list((data / "train").glob("train_partition_*.parquet"))) != 9:
        raise RuntimeError("expected exactly 9 train partitions")
    if len(list((data / "test").glob("test_partition_*.parquet"))) != 3:
        raise RuntimeError("expected exactly 3 test partitions")
    marker = wt / "UPSTREAM_COMMIT"
    if marker.is_file():
        commit = marker.read_text(encoding="utf-8").strip()
    else:
        commit = subprocess.check_output(
            ["git", "-C", str(wt), "rev-parse", "HEAD"], text=True
        ).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"upstream commit differs: {commit} != {EXPECTED_COMMIT}")
    required = [
        wt / "artifacts/oof/fold6_five_member_oof.npz",
        wt / "artifacts/oof/fold7_five_member_oof.npz",
        wt / "artifacts/oof/fold8_five_member_oof.npz",
        wt / "artifacts/member_predictions/test_metadata.csv",
        wt / "artifacts/member_predictions/public_final_fallback.csv",
        wt / "artifacts/member_predictions/public_model_c.csv",
        wt / "artifacts/member_predictions/public_level_pred.csv",
        wt / "artifacts/member_predictions/public_rolling_v1.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen upstream artifacts:\n" + "\n".join(missing))
    print(f"environment PASS: upstream={commit[:12]} train=9 test=3")
    return data, wt, work


def audit_members(work: Path) -> None:
    for member, key in MEMBER_KEYS.items():
        for fold in (6, 7, 8):
            path = work / member / f"fold_{fold}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                for required in ("row_id", "time_id", "asset_id", key):
                    if required not in data.files:
                        raise RuntimeError(f"{path} lacks {required}")
                pred = np.asarray(data[key], dtype=np.float64)
                if len(pred) != len(data["row_id"]) or not np.isfinite(pred).all():
                    raise RuntimeError(f"invalid prediction in {path}")
        public = work / member / PUBLIC_FILES[member]
        if not public.is_file():
            raise FileNotFoundError(public)
        frame = pd.read_parquet(public, columns=["row_id", key])
        if len(frame) != EXPECTED_ROWS or frame["row_id"].duplicated().any():
            raise RuntimeError(f"invalid public rows: {public}")
        if not np.isfinite(frame[key].to_numpy(np.float64)).all():
            raise RuntimeError(f"non-finite public prediction: {public}")
    print(f"members PASS: {len(MEMBER_KEYS)} trained members")


def audit_final(package: Path) -> None:
    path = package / "submission.csv"
    frame = pd.read_csv(path)
    if list(frame.columns) != ["row_id", "target"]:
        raise RuntimeError(f"wrong columns: {list(frame.columns)}")
    if len(frame) != EXPECTED_ROWS or frame["row_id"].duplicated().any():
        raise RuntimeError("submission row contract failed")
    target = frame["target"].to_numpy(np.float64)
    if not np.isfinite(target).all() or abs(target.std(ddof=0) - 0.046) > 1e-12:
        raise RuntimeError("submission target contract failed")
    md5 = digest(path, "md5")
    sha256 = digest(path, "sha256")
    if md5 != EXPECTED_MD5 or sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"submission hash differs: md5={md5} sha256={sha256}")
    print(f"final PASS: rows={len(frame)} std=0.046 md5={md5}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("environment", "members", "final", "all"), default="all")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    _, _, work = audit_environment()
    if args.stage in ("members", "all"):
        audit_members(work)
    if args.stage in ("final", "all"):
        audit_final(package)


if __name__ == "__main__":
    main()

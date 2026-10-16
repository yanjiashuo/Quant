from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from repro.baseline00.build_members import build_fold as build_member_fold
from repro.baseline00.train_base import train_fold as train_base_fold
from repro.baseline00.train_level import train_fold as train_level_fold
from repro.baseline00.train_rolling import train_fold as train_rolling_fold
from repro.baseline03.config import OOF_FOLDS
from repro.config import TRAIN_DIR, ensure_data2_path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def train_parent_folds(
    train_dir: Path,
    output_root: Path,
    folds: Sequence[int] = OOF_FOLDS,
    gap: int = 0,
    random_state: int = 0,
    *,
    resume: bool = True,
) -> dict[str, Any]:
    """Train all five Baseline00 members from source data under the G0 contract."""
    selected = tuple(int(fold) for fold in folds)
    if not selected or any(fold not in OOF_FOLDS for fold in selected):
        raise ValueError(f"folds must be a nonempty subset of {OOF_FOLDS}")
    if gap != 0:
        raise ValueError("Baseline03 G0 requires gap=0")
    if random_state != 0:
        raise ValueError("Baseline03 G0 requires random_state=0")

    root = ensure_data2_path(Path(output_root))
    folds_root = root / "folds"
    members_root = root / "members"
    reports_root = root / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)
    fold_reports: dict[str, Any] = {}

    for fold in selected:
        checkpoint = reports_root / f"fold_{fold}_complete.json"
        archive = members_root / f"fold_{fold}.npz"
        if resume and checkpoint.is_file() and archive.is_file():
            fold_reports[str(fold)] = json.loads(checkpoint.read_text(encoding="utf-8"))
            continue

        report = {
            "fold": fold,
            "gap_time_ids": 0,
            "random_state": 0,
            "base": train_base_fold(
                Path(train_dir), folds_root, fold, 0, None, 0
            ),
            "level": train_level_fold(Path(train_dir), folds_root, fold, 0, None),
            "rolling": train_rolling_fold(Path(train_dir), folds_root, fold, 0, None),
            "members": build_member_fold(folds_root, members_root, fold),
        }
        _write_json(checkpoint, report)
        fold_reports[str(fold)] = report

    summary = {
        "variant": "g0_gap0_parent",
        "folds": list(selected),
        "gap_time_ids": 0,
        "random_state": 0,
        "deterministic": True,
        "force_col_wise": True,
        "fold_reports": fold_reports,
    }
    _write_json(reports_root / "training_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(OOF_FOLDS))
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = train_parent_folds(
        args.train_dir,
        args.out,
        args.folds,
        args.gap,
        args.random_state,
        resume=not args.no_resume,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

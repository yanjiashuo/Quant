from __future__ import annotations

"""Aggregate the three Baseline05 method manifests (E1/E2/E3) into one summary.

Reads each method's ``method_manifest.json`` and writes
``summary/submission_manifest.json``. The summary reasserts the offline-only
discipline: no public label is used for fit or gate, no public leaderboard score
is recorded, and no bundle is private-deployable.
"""

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from repro.baseline04.contract import write_json


def _load_manifest(method_root: Path) -> dict[str, object]:
    path = Path(method_root) / "method_manifest.json"
    if not path.is_file():
        raise ValueError(f"method manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    experiment_root: Path,
    methods: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Aggregate ``{method: member_names}`` bundles under ``experiment_root``."""
    root = Path(experiment_root)
    method_manifests = {
        method: _load_manifest(root / method) for method in methods
    }
    for method, manifest in method_manifests.items():
        if manifest.get("private_deployable") is not False:
            raise ValueError(f"{method} must not be private-deployable")
        if manifest.get("public_labels_used_for_fit_or_gate") is not False:
            raise ValueError(f"{method} public-label discipline flag differs")
        if manifest.get("public_score") is not None:
            raise ValueError(f"{method} must not record a public score")
    summary = {
        "experiment": "baseline05_cs_family",
        "parent": {
            "method": "baseline04_seven_member",
            "public_score": 0.0035828872,
        },
        "methods": {
            method: {
                "member_names": list(methods[method]),
                "offline_verdict": manifest["validation"]["offline_verdict"],
                "rows": manifest["validation"]["rows"],
                "std_pop": manifest["validation"]["std_pop"],
                "submission_sha256": manifest["validation"]["submission_sha256"],
                "private_deployable": False,
            }
            for method, manifest in method_manifests.items()
        },
        "public_score": None,
        "public_labels_used_for_fit_or_gate": False,
        "private_deployable": False,
        "scope": "public_csv_and_order_format_replay_only",
    }
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary_dir / "submission_manifest.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        nargs="+",
        metavar=("METHOD", "MEMBER"),
        required=True,
        help="method name followed by its ordered member names",
    )
    args = parser.parse_args()
    methods = {entry[0]: tuple(entry[1:]) for entry in args.method}
    summary = summarize(args.root, methods)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

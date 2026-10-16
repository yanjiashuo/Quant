from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def validate_frozen_recipe(
    path: Path,
    expected_iterations: Mapping[str, int],
    expected_fit_folds: tuple[int, ...] = (5, 6, 7),
) -> dict:
    recipe_path = Path(path)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if not recipe.get("frozen_before_fold8", False):
        raise ValueError("recipe was not frozen before Fold 8")
    fit_folds = tuple(int(fold) for fold in recipe.get("fit_folds", []))
    if fit_folds != expected_fit_folds:
        raise ValueError(
            f"recipe fit_folds must be {list(expected_fit_folds)}, got {list(fit_folds)}"
        )
    iterations = recipe.get("base_model_iterations", {})
    for model_name, expected in expected_iterations.items():
        actual = iterations.get(model_name)
        if actual != expected:
            raise ValueError(
                f"recipe {model_name} iterations must be {expected}, got {actual}"
            )
    gap = recipe.get("purge_gap_time_ids")
    if not isinstance(gap, int) or gap < 0:
        raise ValueError(f"invalid purge_gap_time_ids: {gap!r}")
    return recipe

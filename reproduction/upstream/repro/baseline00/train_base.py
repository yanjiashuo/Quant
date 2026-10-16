from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from repro.config import FEATS, TRAIN_DIR, WORK, ensure_data2_path
from repro.baseline00.frozen_recipe import validate_frozen_recipe
from repro.baseline00.metrics import prediction_summary, weighted_zero_mean_r2
from repro.baseline00.training_io import (
    load_prediction_frame,
    load_training_arrays,
    load_validation_frame,
)


MODEL_RECIPES: dict[str, dict[str, Any]] = {
    "model_a": {
        "window": "all",
        "n_estimators": 139,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "min_data_in_leaf": 5_000,
        "prediction_scale": 0.8769348741364634,
        "label_mode": "target",
    },
    "model_b": {
        "window": 3,
        "n_estimators": 56,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "min_data_in_leaf": 2_000,
        "prediction_scale": 0.9258936648050373,
        "label_mode": "target",
    },
    "model_c": {
        "window": 4,
        "n_estimators": 201,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "min_data_in_leaf": 2_000,
        "prediction_scale": 1.1921623492997822,
        "label_mode": "target_demean",
    },
}


def model_partitions(fold: int, window: str | int) -> list[int]:
    if window == "all":
        return list(range(fold))
    width = int(window)
    return list(range(max(0, fold - width), fold))


def train_fold(
    train_dir: Path,
    output_root: Path,
    fold: int,
    gap: int = 500,
    frozen_recipe: Path | None = None,
    random_state: int = 0,
) -> dict:
    import lightgbm as lgb

    started = time.perf_counter()
    output_root = ensure_data2_path(output_root)
    fold_dir = output_root / f"fold_{fold}" / "base"
    fold_dir.mkdir(parents=True, exist_ok=True)
    held_out = frozen_recipe is not None and fold == 8
    valid = (
        load_prediction_frame(train_dir, fold, FEATS)
        if held_out
        else load_validation_frame(train_dir, fold, FEATS)
    )
    x_valid = np.nan_to_num(
        valid.loc[:, FEATS].to_numpy(dtype=np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    index_payload = dict(
        row_id=valid["row_id"].to_numpy(dtype=np.int64, copy=False),
        time_id=valid["time_id"].to_numpy(copy=False),
        asset_id=valid["asset_id"].to_numpy(copy=False),
    )
    if not held_out:
        index_payload.update(
            y=valid["target"].to_numpy(dtype=np.float64, copy=False),
            w=np.maximum(
                valid["weight"].to_numpy(dtype=np.float64, copy=False), 0.0
            ),
        )
    np.savez(fold_dir / "valid_index.npz", **index_payload)
    report: dict[str, Any] = {
        "fold": fold,
        "gap_time_ids": gap,
        "valid_rows": int(len(valid)),
        "models": {},
        "frozen_recipe_source": "existing deploy artifacts; no Fold5-7 tuning",
        "frozen_recipe": str(frozen_recipe) if frozen_recipe else None,
        "held_out_labels_read": False if held_out else None,
        "random_state": int(random_state),
        "deterministic": True,
        "force_col_wise": True,
    }
    cutoff = int(valid["time_id"].min()) - gap
    for name, recipe in MODEL_RECIPES.items():
        model_started = time.perf_counter()
        partitions = model_partitions(fold, recipe["window"])
        x_train, y_train, w_train, rows_by_partition = load_training_arrays(
            train_dir,
            partitions,
            FEATS,
            cutoff_time_id=cutoff,
            label_mode=recipe["label_mode"],
        )
        model = lgb.LGBMRegressor(
            n_estimators=recipe["n_estimators"],
            num_leaves=recipe["num_leaves"],
            learning_rate=recipe["learning_rate"],
            min_data_in_leaf=recipe["min_data_in_leaf"],
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            reg_alpha=0.0,
            reg_lambda=1.0,
            n_jobs=64,
            verbosity=-1,
            random_state=random_state,
            deterministic=True,
            force_col_wise=True,
        )
        model.fit(x_train, y_train, sample_weight=w_train)
        prediction = model.predict(x_valid) * float(recipe["prediction_scale"])
        prediction = np.clip(prediction, -3.0, 3.0)
        model_path = fold_dir / f"{name}.txt"
        model.booster_.save_model(str(model_path))
        np.save(fold_dir / f"{name}.npy", prediction.astype(np.float64))
        model_report = {
            "partitions": partitions,
            "rows_by_partition": rows_by_partition,
            "train_rows": int(len(y_train)),
            "label_mode": recipe["label_mode"],
            "n_estimators": recipe["n_estimators"],
            "prediction_scale": recipe["prediction_scale"],
            "seed": int(random_state),
            "deterministic": True,
            "force_col_wise": True,
            "prediction": prediction_summary(prediction),
            "elapsed_seconds": time.perf_counter() - model_started,
        }
        if not held_out:
            model_report["valid_score"] = weighted_zero_mean_r2(
                index_payload["y"], prediction, index_payload["w"]
            )
        report["models"][name] = model_report
        del x_train, y_train, w_train, model, prediction
        gc.collect()
    report["elapsed_seconds"] = time.perf_counter() - started
    (fold_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--gap", type=int, default=500)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, default=WORK / "folds")
    parser.add_argument("--frozen-recipe", type=Path)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    if args.frozen_recipe:
        recipe = validate_frozen_recipe(
            args.frozen_recipe,
            {name: int(config["n_estimators"]) for name, config in MODEL_RECIPES.items()},
        )
        recipe_gap = int(recipe["purge_gap_time_ids"])
        if args.gap != recipe_gap:
            raise ValueError(f"--gap {args.gap} conflicts with frozen recipe gap {recipe_gap}")
    for fold in args.folds:
        report = train_fold(
            args.train_dir,
            args.out,
            fold,
            args.gap,
            args.frozen_recipe,
            args.random_state,
        )
        print(json.dumps({"fold": fold, "models": report["models"]}, indent=2))


if __name__ == "__main__":
    main()

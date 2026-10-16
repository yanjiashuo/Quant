from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from rolling import RollingFeatureState
except ModuleNotFoundError:  # Package import used by local tests.
    from repro.baseline00.rolling import RollingFeatureState


def weighted_level_features(features: np.ndarray, predicted_weight: np.ndarray) -> np.ndarray:
    weight = np.maximum(np.asarray(predicted_weight, dtype=np.float64), 0.0)
    total = float(weight.sum())
    if total <= 0.0:
        return np.zeros((1, features.shape[1]), dtype=np.float32)
    return ((features.T @ weight) / total).astype(np.float32, copy=False).reshape(1, -1)


def select_rolling_columns(matrix: np.ndarray, config: dict) -> np.ndarray:
    values = np.asarray(matrix)
    indices = config.get("rolling_column_indices")
    if indices is None:
        return values
    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= values.shape[1]):
        raise ValueError("rolling_column_indices are outside the generated matrix")
    if np.unique(selected).size != selected.size:
        raise ValueError("rolling_column_indices contain duplicates")
    return np.ascontiguousarray(values[:, selected])


def combine_members(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    pred_c: np.ndarray,
    level: np.ndarray,
    rolling: np.ndarray,
    asset_id: np.ndarray,
    config: dict,
    pred_c_long: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    formula = config["formula"]
    prediction = formula["weight_a"] * pred_a + formula["weight_b"] * pred_b
    prediction = (prediction - formula["post_bias"]) * formula["post_scale"]
    asset_scale = np.asarray(formula["asset_scale"], dtype=np.float64)
    multiplier = 1.0 + formula["asset_scale_shrink"] * (asset_scale - 1.0)
    prediction = prediction * multiplier[np.asarray(asset_id, dtype=np.int64)]
    mean = float(np.mean(prediction))
    cs2 = formula["cs_dev_alpha"] * (prediction - mean) + formula["cs_mean_beta"] * mean
    final = formula["output_scale"] * (
        cs2 + formula["weight_c"] * pred_c + formula["w_level"] * formula["scale_level"] * level
    )
    final_mean = float(np.mean(final))
    l88 = (
        formula["l88_level"] * final_mean
        + formula["l88_deviation"] * (final - final_mean)
    )
    members = {
        "final": final,
        "L88": l88,
        "model_c": pred_c,
        "level": level,
        "rolling_v1": rolling,
    }
    if pred_c_long is not None:
        members["model_c_long"] = np.asarray(pred_c_long, dtype=np.float64)
    missing = [name for name in config["member_names"] if name not in members]
    if missing:
        raise ValueError(f"missing stack member predictions: {missing}")
    matrix = np.column_stack([members[name] for name in config["member_names"]])
    coefficients = np.asarray(config["effective_coefficients"], dtype=np.float64)
    if coefficients.size != matrix.shape[1]:
        raise ValueError("stack coefficients do not match member order")
    prediction = matrix @ coefficients
    prediction *= float(config["amplitude_multiplier"])
    return prediction, members


class Model:
    def __init__(self):
        import lightgbm as lgb

        root = Path(__file__).resolve().parent
        self.config = json.loads((root / "production_config.json").read_text(encoding="utf-8"))
        self.features = tuple(self.config["feature_columns"])
        self.rolling_features = tuple(self.config["rolling_feature_columns"])
        self.num_threads = min(4, int(self.config.get("inference_num_threads", 4)))
        model_names = [
            "model_a",
            "model_b",
            "model_c",
            "weight_model",
            "level_model",
            "rolling_v1",
        ]
        if "model_c_long" in self.config["member_names"]:
            model_names.append("model_c_long")
        missing = [name for name in model_names if not (root / f"{name}.txt").is_file()]
        if missing:
            raise ValueError(f"production package missing models: {missing}")
        self.models = {
            name: lgb.Booster(model_file=str(root / f"{name}.txt"))
            for name in model_names
        }
        self.rolling_state = RollingFeatureState(self.rolling_features)
        with np.load(root / "rolling_history.npz") as history:
            self.rolling_state.restore(
                history["asset_ids"],
                history["values"],
                history["counts"],
                int(history["last_time_id"]),
            )
        self.last_members: dict[str, np.ndarray] = {}

    def _predict(self, name: str, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.models[name].predict(matrix, num_threads=self.num_threads), dtype=np.float64
        )

    def predict(self, test):
        required = {"time_id", "asset_id", *self.features}
        missing = sorted(required - set(test.columns))
        if missing:
            raise ValueError(f"test batch missing columns: {missing[:10]}")
        if test["time_id"].nunique() != 1:
            raise ValueError("Model.predict expects exactly one time_id")
        x = test.loc[:, self.features].to_numpy(dtype=np.float32, copy=True)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        scale = self.config["prediction_scales"]
        pred_a = np.clip(self._predict("model_a", x) * scale["model_a"], -3.0, 3.0)
        pred_b = np.clip(self._predict("model_b", x) * scale["model_b"], -3.0, 3.0)
        pred_c = np.clip(self._predict("model_c", x) * scale["model_c"], -3.0, 3.0)
        pred_c_long = None
        if "model_c_long" in self.models:
            pred_c_long = np.clip(
                self._predict("model_c_long", x) * scale["model_c_long"], -3.0, 3.0
            )
        predicted_weight = np.maximum(self._predict("weight_model", x), 0.0)
        level_features = weighted_level_features(x, predicted_weight)
        group_level = float(self._predict("level_model", level_features)[0])
        level = np.full(len(test), group_level, dtype=np.float64)
        rolling_matrix = select_rolling_columns(
            self.rolling_state.transform_batch(test), self.config
        )
        rolling = self._predict("rolling_v1", rolling_matrix)
        prediction, members = combine_members(
            pred_a,
            pred_b,
            pred_c,
            level,
            rolling,
            test["asset_id"].to_numpy(dtype=np.int64, copy=False),
            self.config,
            pred_c_long,
        )
        self.last_members = members
        if not np.isfinite(prediction).all():
            raise ValueError("production prediction contains non-finite values")
        return prediction

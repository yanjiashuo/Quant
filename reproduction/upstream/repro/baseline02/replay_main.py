from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class Model:
    """Public-test replay model for official runner format/order auditing only."""

    def __init__(self):
        path = Path(__file__).resolve().parent.parent / "submission.csv"
        frame = pd.read_csv(path)
        if list(frame.columns) != ["row_id", "target"]:
            raise ValueError("replay submission columns must be row_id,target")
        self.row_id = frame["row_id"].to_numpy(dtype=np.int64, copy=True)
        self.target = frame["target"].to_numpy(dtype=np.float64, copy=True)
        if np.any(self.row_id[1:] <= self.row_id[:-1]):
            raise ValueError("replay row_id must be strictly increasing")
        if not np.isfinite(self.target).all():
            raise ValueError("replay target must be finite")

    def predict(self, test):
        requested = test["row_id"].to_numpy(dtype=np.int64, copy=False)
        index = np.searchsorted(self.row_id, requested)
        if np.any(index >= self.row_id.size) or not np.array_equal(
            self.row_id[index], requested
        ):
            raise ValueError("runner requested row_id absent from frozen submission")
        return self.target[index]

import numpy as np
import pytest

from quant_competition.metrics import weighted_zero_mean_r2


def test_perfect_prediction_scores_one():
    y = np.array([1.0, -2.0, 0.5])
    w = np.array([1.0, 2.0, 3.0])
    assert weighted_zero_mean_r2(y, y, w) == pytest.approx(1.0)


def test_zero_prediction_scores_zero():
    y = np.array([1.0, -2.0, 0.5])
    w = np.array([1.0, 2.0, 3.0])
    assert weighted_zero_mean_r2(y, np.zeros_like(y), w) == pytest.approx(0.0)


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        weighted_zero_mean_r2(
            np.array([1.0]), np.array([0.0]), np.array([-1.0])
        )

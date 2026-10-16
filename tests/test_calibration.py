import numpy as np
import pytest

from quant_competition.calibration import component_calibration, mean_preserving_std


def test_mean_preserving_std():
    pred = np.array([-2.0, 0.0, 1.0, 5.0])
    calibrated = mean_preserving_std(pred, 0.046)
    assert calibrated.mean() == pytest.approx(pred.mean())
    assert calibrated.std(ddof=0) == pytest.approx(0.046)


def test_identity_component_calibration():
    pred = np.array([0.2, 0.4, -0.1, 0.3])
    time_id = np.array([0, 0, 1, 1])
    asset_id = np.array([0, 1, 0, 1])
    actual = component_calibration(pred, time_id, asset_id)
    np.testing.assert_allclose(actual, pred, atol=1e-12)


def test_time_level_zero_removes_time_mean_differences():
    pred = np.array([1.0, 3.0, 5.0, 7.0])
    time_id = np.array([0, 0, 1, 1])
    asset_id = np.array([0, 1, 0, 1])
    actual = component_calibration(
        pred, time_id, asset_id, time_level=0.0, asset_level=1.0, residual=1.0
    )
    assert actual[:2].mean() == pytest.approx(actual[2:].mean())

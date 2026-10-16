import numpy as np

from quant_competition.stacking import PositiveRidgeStacker


def test_positive_ridge_stacker_learns_non_negative_coefficients():
    rng = np.random.default_rng(7)
    members = rng.normal(size=(500, 3))
    target = 0.7 * members[:, 0] + 0.2 * members[:, 2]
    weight = np.ones(500)
    model = PositiveRidgeStacker(alpha=0.01).fit(members, target, weight)
    assert np.all(model.coef_ >= 0)
    assert np.corrcoef(model.predict(members), target)[0, 1] > 0.99

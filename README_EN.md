# Anonymous Multi-Asset Forecasting System

This repository is the portfolio release of a large-scale forecasting system built for a strictly ordered, anonymous multi-asset time-series competition.

The project covers 13.2 million training rows, 3.2 million public-test rows, 15 assets, 323 anonymous features, and 47 auxiliary responders. The final solution combines static gradient-boosted trees, cross-sectional rank models, per-asset rolling models, structured demeaning targets, and MLP members through a normalized non-negative Ridge stack.

## Highlights

- Built leakage-safe walk-forward OOF validation across ordered partitions.
- Designed a 15-member ensemble spanning static, temporal, cross-sectional, asset-aware, and neural inductive biases.
- Added residualization, leave-one-out analysis, correlation clustering, and duplicate controls before stacking.
- Reproduced a teammate branch in an isolated worktree and verified row alignment, coefficients, and artifact hashes.
- Packaged a byte-identical training-to-submission rebuild for the reproducible public model.
- Audited 3.2 million prediction rows and validated sequential CPU-only inference under a 4-CPU / 12-GB / no-network constraint.
- Documented negative results and the public-to-hidden regime gap rather than presenting leaderboard performance as generalization.

## Architecture

```text
323 anonymous features
  -> walk-forward OOF predictions
  -> LightGBM / cross-sectional rank / rolling history / demeaned targets / MLP
  -> population-std normalization
  -> positive Ridge stack (15 members)
  -> global + time-level + asset-level + residual calibration
  -> audited submission or sequential inference package
```

The reproducible 15-member model achieved `0.002331 / 0.002453` on Fold 7 / Fold 8 and improved over the 8-member E3 baseline by `+0.000143 / +0.000257`. See [Results and limitations](docs/RESULTS_AND_LIMITATIONS.md) for the complete public/private evaluation and its interpretation.

## Quick check

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Competition data, trained weights, caches, and multi-million-row CSV files are intentionally excluded. Full reproduction requirements are documented in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

The code is released under the MIT License. Competition data and official materials remain subject to the organizer's terms.

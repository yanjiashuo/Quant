# BEST_0.0039380 Reproduction Package

This directory freezes the complete training-to-submission chain for the public
leaderboard result reported as **0.0039380**. It includes every additional
member trainer, the exact upstream commit contract, the 15-member Ridge builder,
and fail-fast audits. The 97 MB submitted CSV is intentionally excluded from
Git; a successful rebuild must reproduce its frozen MD5 and SHA256 hashes.

## Frozen result

| Item | Value |
|---|---|
| Public score | `0.0039380` |
| Rows | `3,217,458` |
| Population std | `0.046000` |
| CSV MD5 | `92ef61c93ac86f14f606f0ea0bc66dd0` |
| CSV SHA256 | `ae254141a97dafd20ba0b885c2351c5273e51128eede8c808804f996fd44df2f` |
| Upstream commit | `0ad2c976d02ff3fc2d1965b5cb0a484800503470` |

The earlier draft README incorrectly listed MD5 prefix `7fb562c4146d`. An
isolated rebuild proved that the packaged/submitted artifact is
`92ef61c93ac86f14f606f0ea0bc66dd0`; `build_15m.py` now enforces both the full
MD5 and SHA256 above.

## Model

The final prediction is a positive Ridge stack (`alpha=0.01`, no intercept).
Each member column is divided by its population standard deviation before
fitting. Fold6 coefficients predict Fold7; Fold6+7 coefficients predict Fold8;
Fold6+7+8 coefficients build the public submission. The raw public prediction
is mean-preserving rescaled to population std `0.046`.

| # | Member | How it is learned |
|---:|---|---|
| 1 | `factor` | `(final + L88) / 2` |
| 2 | `contrast` | `final - L88` |
| 3 | `level_pred` | frozen upstream member |
| 4 | `rolling_v1` | frozen upstream member |
| 5 | `cs_rank` | recent-4 LGB on per-time rank features, target minus weighted time mean |
| 6 | `model_c_long` | full-history LGB on time-demeaned target |
| 7 | `csrd_resid` | lag-1 rank-delta LGB, residualized against `cs_rank` |
| 8 | `csrd5_resid` | lag-5 rank-delta LGB, residualized against lag-1 delta |
| 9 | `csrv_resid` | rolling-volatility rank LGB, residualized against rank/delta members |
| 10 | `asset_demean` | full-history LGB on target minus past per-asset mean |
| 11 | `double_demean` | full-history LGB on target minus asset mean and time mean |
| 12 | `double_demean_assetaware` | member 11 plus categorical `asset_id`, min leaf 1000 |
| 13 | `mlp_ad` | MLP 512-256-128 on asset-demeaned target |
| 14 | `mlp_member` | MLP 512-256-128 on double-demeaned target |
| 15 | `mlp_deep` | MLP 1024-512-256 on double-demeaned target |

All MLPs use BatchNorm, ReLU, dropout `0.3`, weighted MSE, Adam, cosine LR,
gradient clipping, seed `0`, and the frozen early-stopping procedure in the
scripts. Production MLP training intentionally preserves the historical
10-row training-slice validation behavior because changing it would change the
submitted artifact.

## Exact environment

The successful rebuild used Linux x86_64, Python `3.13.5`, CUDA `12.4`, and the
versions in `requirements-lock.txt`. GPU is required for the frozen MLP chain.
LightGBM uses up to 64 CPU threads and the complete rebuild requires substantial
RAM, temporary disk, and several hours.

Create a writable upstream runtime from the bundled source snapshot:

```bash
cp -R upstream /absolute/path/to/upstream-runtime
```

Set paths before sourcing the environment file:

```bash
export QUANT_DATA_ROOT=/absolute/path/to/public_release_20260630/data
export TEAMMATE_WT=/absolute/path/to/upstream-runtime
export SOTA_WORK=/absolute/path/to/repro_work/BEST_0.0039380
export PYTHON_BIN=/absolute/path/to/venv/bin/python
source setup_env.sh
```

`QUANT_DATA_ROOT` must contain 9 train parquet partitions and 3 test parquet
partitions. `TEAMMATE_WT` must contain the bundled `repro/` source snapshot and
the generated five-member OOF/public inputs listed in `upstream/README.md`.
Those large intermediate artifacts are retained privately and are not
redistributed in this portfolio repository.

## Rebuild

### One command

```bash
bash run_full_rebuild.sh
```

This performs environment audit, rebuilds upstream `cs_rank`, `cs_rank_long`,
and `model_c_long`, trains every added structural/MLP member, builds the Ridge
submission, and checks the exact CSV hashes.

### Staged commands

```bash
python audit_package.py --stage environment
bash rebuild_upstream.sh

CSRD_LAG=1 python scripts/train_cs_rank_delta.py
CSRD_LAG=5 python scripts/train_cs_rank_delta.py
python scripts/train_cs_rank_vol.py
python scripts/train_asset_demean.py
python scripts/train_double_demean.py
python scripts/train_double_demean_assetaware.py
python scripts/train_mlp_member.py
python scripts/train_mlp_variants.py
python scripts/train_mlp_deep.py

python audit_package.py --stage members
python build_15m.py
python audit_package.py --stage final
```

No source edit is required for the lag-5 member: `CSRD_LAG=5` deterministically
creates `cs_rank_delta5/` and the `cs_rank_delta5` prediction key.

## Rebuild acceptance

`build_15m.py` must print:

```text
E3: f7=0.00218783 f8=0.00219558
15m offline: f7=+0.000143 f8=+0.000257 (vs E3)
Done! rows=3217458 std=0.046000 md5=92ef61c93ac8
```

The exact Fold6+7 Ridge coefficients are recorded in `manifest.json`. The final
audit rejects missing/duplicate rows, non-finite predictions, wrong std, wrong
upstream commit, missing members, or any CSV hash difference.

## Files

```text
reproduction/
├── README.md
├── manifest.json
├── requirements-lock.txt
├── setup_env.sh
├── prepare_parent_official.py
├── rebuild_upstream.sh
├── run_full_rebuild.sh
├── audit_package.py
├── build_15m.py
└── scripts/
    ├── common.py
    ├── train_cs_rank_delta.py
    ├── train_cs_rank_vol.py
    ├── train_asset_demean.py
    ├── train_double_demean.py
    ├── train_double_demean_assetaware.py
    ├── train_mlp_member.py
    ├── train_mlp_variants.py
    └── train_mlp_deep.py
```

The official dataset, submission CSV, caches, trained boosters, and intermediate
OOF/public member predictions are intentionally not committed. The complete
source snapshot is included, while the generated upstream artifacts must be
rebuilt from official data or obtained from the team's private archive.

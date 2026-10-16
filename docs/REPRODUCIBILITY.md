# 复现与制品边界

## 三个复现层级

### 1. 轻量代码验证

不需要竞赛数据，用于验证指标、校准和 stack 实现：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

### 2. 15-member 完整重建

需要：

- 官方 2026-06-30 数据发布中的 9 个 train 和 3 个 test parquet；
- `reproduction/upstream/` 内置的团队上游源码快照；
- Linux x86_64、Python 3.13.5、CUDA 12.4；
- `reproduction/requirements-lock.txt` 中的版本；
- 足够的 RAM、临时磁盘和数小时训练时间。

```bash
cd reproduction
cp -R upstream /absolute/path/to/upstream-runtime
export QUANT_DATA_ROOT=/absolute/path/to/public_release_20260630/data
export TEAMMATE_WT=/absolute/path/to/upstream-runtime
export SOTA_WORK=/absolute/path/to/rebuild-work
export PYTHON_BIN=/absolute/path/to/python
source setup_env.sh
bash run_full_rebuild.sh
```

成功重建的 `submission.csv` 必须满足：

```text
rows    = 3,217,458
std     = 0.046000
MD5     = 92ef61c93ac86f14f606f0ea0bc66dd0
SHA256  = ae254141a97dafd20ba0b885c2351c5273e51128eede8c808804f996fd44df2f
```

### 3. 最终公开校准

在 15-member 重建结果上，使用测试 metadata 的 `time_id` 与 `asset_id`：

```python
from quant_competition.calibration import component_calibration, mean_preserving_std

pred = component_calibration(
    base_prediction,
    time_id,
    asset_id,
    time_level=0.8,
    asset_level=1.0,
    residual=1.0,
)
pred = mean_preserving_std(pred, 0.046)
```

冻结参数和最终文件哈希见 [`configs/final_public_calibration.json`](../configs/final_public_calibration.json)。

## 为什么不提交大文件

当前 GitHub 版本不包含：

- 官方 train/test parquet；
- 训练后的 LightGBM/PyTorch 权重；
- OOF `.npz` 与 production member parquet；
- 97MB 的完整提交 CSV；
- 缓存、日志和 200+ 个失败探针。

这些文件会使仓库难以审阅，也可能违反数据分发边界。它们都属于可生成制品，而非理解方法所需的源代码。

## 可复现性的诚实边界

- `0.0039380` 版本已在隔离目录完成 byte-identical 重建；
- 最终公开 `0.00394663` 是确定性 prediction-level 校准，可由前者重建；
- 完整 15-member 的所有在线 booster 没有在比赛结束前保真封装，因此最终 sequential private package 与公开 CSV 不是完全同一模型；
- 仓库保留这一差异，不声称公开与私榜链路完全一致。

## 团队上游

完整链依赖已内置的团队上游源码，以及由官方数据生成的五成员 OOF/production 中间制品。由于这些制品体积大且包含数据派生预测，公开仓库不分发它们；团队内部私有 bundle 已保留。`reproduction/audit_package.py` 会检查源码快照标记、成员文件、行数、重复行和非有限值。团队贡献与个人工作边界见 [`NOTICE.md`](../NOTICE.md)。

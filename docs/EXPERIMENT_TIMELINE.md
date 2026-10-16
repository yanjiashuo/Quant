# 从 Baseline 到最终版本

## 1. 建立可信 baseline

项目从多个 LightGBM 静态模型和输出校准起步。早期重点是跑通加权零均值 R²、分区训练、提交格式和顺序 runner，并形成 `best_model` 基线。

## 2. 权重与 target level 分解

首先研究评分权重预测和每个 `time_id` 的整体 target level。实验发现：

- 权重模型自身 R² 可达较高水平，但实际权重与预测权重生成的聚合特征相关性已接近 1；
- target 方差的大部分来自 time-level，且相邻时点强持久；
- 这部分持久性依赖测试期不可见的过去真实 target，部署模型只能利用弱得多的 feature-level 信号。

因此停止无效的 weight predictor 强化，保留 acttrain level 作为小幅可部署增强。

## 3. 扩大训练窗口

对核心 `model_a` 做 walk-forward endpoint analog，发现更多历史数据带来的样本量收益超过轻微 drift：长窗口在 Fold 6/7/8 全部提升。该改动把本地 Fold 8 从约 `0.00197` 推到 `0.00211`，是第一项稳定结构突破。

## 4. 寻找低相关成员

Panel NN、multitask NN、自监督编码器、FT-Transformer、XGBoost、CatBoost 与 residual 模型被系统测试。多数模型要么太弱，要么 Fold 7/8 方向不一致。关键教训是：

```text
多样性本身不产生价值；成员必须同时满足“低相关、方向稳定、有残差增量”。
```

## 5. Rolling 历史特征

受官方 sequential example 启发，构造每个资产的历史特征。`rolling_v1` 使用固定特征子集的 lag 与 rolling 变换，standalone 不强，但与静态模型的相关性显著降低，且与其他结构成员互补。

将 rolling 加入正约束 Ridge 后，Fold 8 相对 raw L88 提升约 `+0.000153`，并把公开结果推进到 `0.0034+`。这是项目从静态表格模型走向真正 panel time-series 模型的转折点。

## 6. 系统搜索 feature × temporal 矩阵

对特征选择策略和时间操作做双轴实验：

- 特征轴：模型 gain、跨 fold 稳定性、特征区块、联合策略；
- 时间轴：lag 1/2/3/5、rolling 5/10、更长窗口、trend、delta 和组合。

实验暴露了 rolling_v1 特征列表溯源不足，也验证了简单扩大特征数和窗口并不自动提升。后续将固定 Top50 清单和 rolling 算法写入复现配置。

## 7. 队友分支复现与成员整合

对队友的 cross-sectional family 使用独立 worktree 进行代码审计和 bit-exact 重建：

- 逐时点 rank 变换；
- recent/full-window `cs_rank`；
- 长窗口 `model_c`；
- 行键适配与 OOF 对齐；
- 正约束 Ridge 系数复现。

先完整复现 7-member，再复现 8-member E3，确认方法、数据顺序和系数无误后才扩展最终模型。

## 8. 形成 15-member 集成

在 E3 上新增 rank-delta、lag-5 delta、rank-volatility、asset-demean、double-demean、asset-aware 和三种 MLP。对新增横截面成员先残余化，避免重复解释同一方向。

最终结果：

| 模型 | Fold 7 | Fold 8 |
|---|---:|---:|
| 8-member E3 | 0.002188 | 0.002196 |
| 15-member | 0.002331 | 0.002453 |
| 增量 | +0.000143 | +0.000257 |

15-member 基础版本公开结果 `0.0039380`，并完成隔离环境逐字节重建。

## 9. 校准与最终评测

最终公开最好版本将 time-level 系数调整到 `0.8`，保持 asset-level 和 residual 不变，并把总体标准差恢复为 `0.046`，公开成绩 `0.00394663`。

最终隐藏时段得分 `0.00301583`，显著低于公开时段。这表明公开 regime 上有效的组合和幅度没有完整迁移到更远未来，也暴露了完整 15-member 与最终在线部署包之间的制品保真问题。

项目因此以两条线收束：一条是可逐字节复现的研究模型；另一条是满足最终 API 约束并经过 A/B 的部署 fallback。二者差异被明确记录，而不是被结果包装掩盖。

# 研究与验证方法

## 1. 先验证信息，再扩大模型

项目后期建立了统一门槛：新成员首先检查其与当前 stack 残差的加权相关性，而不是只看 standalone R² 或与现有预测的低相关性。

```text
residual = target - current_stack_prediction
screen = weighted_corr(candidate_prediction, residual)
```

低相关并不等于有增量。一个预测可以和主模型很不相似，但如果它与真实残差无关，只会给 stack 增加噪声。

## 2. Walk-forward 纪律

所有可接受结构增益必须满足：

1. 模型仅使用验证时间之前的分区训练；
2. 超参数与 blend 权重不能在 Fold 8 上选择；
3. Fold 7 与 Fold 8 分别报告相对真实 base 的增量；
4. 任何符号翻转、依赖单 seed、只在单资产成立的增益均降级；
5. production 使用与 OOF 相同的特征定义、窗口和预处理。

## 3. 成员审计

进入最终 stack 前执行：

- **相关性矩阵**：识别几乎重复的成员；
- **leave-one-out**：删除某成员后重拟合，判断其边际贡献；
- **duplicate-control**：复制或微扰成员，检查 Ridge 是否因列数而产生假权重；
- **residualization**：先去掉新成员中可由旧成员线性解释的部分；
- **seed ensemble**：区分真实信号和幸运随机种子；
- **row-key audit**：所有 OOF 和 production 预测按 `row_id` 显式对齐。

## 4. 公榜反馈的边界

公开榜仅用于少量迁移诊断和输出幅度校准，不作为训练标签。项目记录每次提交的假设、文件哈希、与 base 的相关性和实际反馈。

实践中，多个本地增益在公开测试上发生方向翻转，最典型的是 TCN：Fold 7/8 均为正，但公开分数显著下降。这促使后续提高 public-safe 相关性门槛，并最终承认本地验证无法完全代表更远未来。

## 5. 工程质量门

每个提交至少检查：

- 行数和 `row_id` 与 sample submission 完全一致；
- 无重复、无缺失、无 NaN/inf；
- 预测均值、标准差、极值和尾部数量合理；
- 输出文件 MD5/SHA256 可记录；
- runner 无超时、无异常，逐时点调用返回长度正确；
- 大数据、缓存、权重和提交 CSV 不进入 Git。

## 6. 失败实验如何使用

本项目没有把失败实验删除后只保留最好分数。失败结果用于关闭方向并改进方法：

- weight predictor R² 高，但聚合权重已近乎完美，继续强化不提升最终指标；
- target level 有强自相关，但依赖测试期不可见的过去真实 target；
- 多种 residual NN、GRU、TCN、Transformer 和图模型在 fold 间不稳；
- cross-asset 相关性多数来自 own-lag 污染或共同因子；
- public test 与训练分区高度可分，说明 regime gap 是核心风险；
- 更复杂模型不能弥补缺少可迁移信息的问题。

这些结论让研究从“继续堆模型”转向“寻找新的信息轴、验证增量、确保训练与部署一致”。

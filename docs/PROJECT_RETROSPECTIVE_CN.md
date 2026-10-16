# 2026 量化交易研究大赛：项目完整复盘与简历材料

## 一、项目一句话定义

在一个包含 **1322.8 万训练样本、321.7 万公开测试样本、15 个匿名资产、323 个匿名特征和 47 个辅助 responder** 的严格时序预测任务中，构建面向未来风险调整表现 `target` 的多模型集成系统；在显著分布漂移、极弱信号、禁止未来信息和最终无 GPU 在线推理的约束下，将公开榜加权零均值 R² 从早期约 **0.003019** 提升至 **0.00394663**，相对提升约 **30.7%**，取得**公榜第 23 名**；最终私榜得分 **0.00301583**、排名**第 37 名**。项目完整覆盖了研究验证、特征与目标分解、多模型集成、公榜反馈、受限环境部署和可复现交付。

> 结果口径必须区分：
>
> - `0.00394663 / 公榜第 23 名`：最终公开榜最好成绩，也是简历中应使用的公榜结果。
> - `0.0039380`：已经做成逐字节可复现包并推送 GitHub 的 15-member 正约束 Ridge 版本。
> - `0.004574`：8 月 23 日标签回补后的本地 robust calibration 验证结果，不是公开榜或私榜成绩。
> - `0.00301583 / 私榜第 37 名`：最终私榜定榜结果。它低于公榜，必须如实保留，不能用本地回补分数替代。

| 项目 | 信息 |
|---|---|
| 比赛 | 2026 量化交易研究大赛 |
| 联合主办 | 中国科学技术大学、复旦大学、上海交通大学相关学院 |
| 官方网站 | https://race.xhth.cn/ |
| 队伍 | 老老实实队（`sjtu030`） |
| 公榜结果 | `0.00394663`，第 23 名，67 次提交 |
| 最终私榜 | `0.00301583`，第 37 名 |

---

## 二、主办方要求与问题本质

### 2.1 预测任务

每一行表示某个匿名资产 `asset_id` 在匿名时间 `time_id` 的状态。模型需要仅使用当前及历史已知信息，预测该状态之后固定窗口内的风险调整表现 `target`。预测符号表达方向，绝对值表达信号强度。

训练数据包含：

- `row_id`、`time_id`、`asset_id`；
- `feature_000 ... feature_322`，共 323 个匿名数值特征；
- `responder_*`，共 47 个训练期辅助目标；
- `weight`，评分样本权重；
- `target`，最终预测目标。

公开测试和私有评测仅提供索引与 `feature_*`，不提供 `weight`、responder 或 target。

### 2.2 数据规模

| 项目 | 规模 |
|---|---:|
| 原始训练行数 | 13,227,692 |
| 公开测试行数 | 3,217,458 |
| 匿名资产 | 15 |
| 匿名特征 | 323 |
| responder | 47 |
| 原始训练分区 | 9 个连续时序 Parquet |
| 标签回补 | 3,217,458 行，与原公开测试逐 row_id 对齐 |

### 2.3 评分函数

比赛使用加权零均值 R²：

\[
R^2_w = 1 - \frac{\sum_i w_i(y_i-\hat y_i)^2}{\sum_i w_i y_i^2}
\]

全零预测得分为 0。该指标与普通 R² 的关键区别是没有拟合截距后的均值基准，因此预测幅度、方向和高权重样本上的误差都会直接影响分数。

### 2.4 合规和部署约束

- 严禁使用未来 feature、未来分布、未来缺失模式、未来 responder 或 target；
- 不允许使用未授权外部市场数据；
- 私榜按递增 `time_id` 调用 `Model.predict(test)`，每次只给当前时点全部资产；
- 模型可以维护过去状态，但不能一次读取未来测试集；
- 最终环境约 4 CPU、12 GB 内存、无 GPU、无外部网络；
- ZIP 根目录必须包含 `main.py` 和 `requirements.txt`，路径必须相对化；
- 异常、超时、长度不符、NaN/Inf 会被填 0，严重时提交无效；
- 公榜每天原则上最多 5 次成功评分，私榜策略最多提交 10 次；
- 最终入围后综合分为私榜稳健标准分 80% + 答辩评委标准分 20%。

这意味着项目不仅是建模问题，也是严格的时序验证、资源约束推理、数据合规和软件交付问题。

---

## 三、主要技术挑战

1. **信号极弱**：本地加权 R² 仅千分之几，大部分 target 方差不可由当前 feature 解释。
2. **强非平稳性**：训练 fold 与公开测试的 adversarial validation AUC 一度达到约 0.9998，说明 regime 差异巨大。
3. **时间层信号占主导**：约 70% target 方差来自 per-time_id level；其 lag-1 持久性强，但测试时过去真实 target 不可见。
4. **匿名特征不可解释**：不能依赖金融字段语义，只能通过统计稳定性、OOF 表现和结构归纳偏置识别信号。
5. **模型高度相关**：大量 LGB 变体相关系数接近 0.99，简单增加成员只会重复加权或放大噪声。
6. **本地到公榜迁移困难**：多次出现 fold7/fold8 增益与公榜方向相反，必须区分真实结构增益和 regime-specific 偶然性。
7. **计算与工程压力**：千万级行数、数百列和多折训练容易造成 OOM、超时和反复读盘；最终又必须压缩成 CPU 在线推理包。

### 你的核心工作与贡献边界

你的工作不是只训练了一个模型，而是承担了从研究到交付的多条主线：

1. **建立研究基线和验证纪律**：统一加权零均值 R²、walk-forward folds、逐折 delta、held-out 验证和拒绝标准；
2. **推进主要建模路线**：从 LightGBM baseline、acttrain level、long-window model_a，一直推进到 per-asset rolling 和五成员 Ridge stack；
3. **组织大规模探索**：系统测试特征选择 × 时间算子、模型族、神经网络、残差模型、横截面模型、权重代理、cross-asset 与后处理；
4. **使用公榜反馈做受控诊断**：建立提交历史、幅度曲线、风险门和候选审计，而不是只保留最好单点；
5. **整合队友方案**：对队友的 cs-family/15-member 路线进行独立复现、相关性与 LOO 审计、duplicate-control、缺失文件检查和完整链路打包；
6. **负责工程与可复现交付**：缓存、内存优化、后台任务、runner、策略 ZIP、哈希、GitHub release 和队友复现说明；
7. **进行失败复盘与合规审计**：主动关闭未来信息、不可部署 weight、外部数据和 public-only 泄漏方向，并对公私榜落差做最终归因。

团队贡献应准确表述：你可以说自己“主导了研究框架、rolling/stack 路线、实验审计、队友 SOTA 整合和部署复现”；不应声称队友最初提出和编写的所有成员均由你独立完成。

---

## 四、从基线到最终系统的技术演进

### 阶段 1：建立可用的 LightGBM 基线

最初系统以多个 LightGBM 为核心：

- `model_a`：主目标模型，较大叶节点、长历史窗口；
- `model_b`：更短近期窗口，补充近期 regime；
- `model_c`：学习 target 减去 per-time_id 加权均值后的横截面残差；
- `weight_model`：从 feature 预测训练权重，用于可部署的 level 特征聚合；
- `level_model`：预测当前时点整体 target level。

通过 fold6/7 选参、fold8 留作 held-out，逐步加入输出尺度、资产校准、横截面均值与 deviation 校准。`acttrain level` 的关键设计是在训练 level 模型时使用真实训练权重，而测试时使用预测权重，使 level 路径保持可部署。

### 阶段 2：识别长窗口训练的价值

对 model_a 做 endpoint analog 和 walk-forward 比较后发现，全历史训练窗口相对 recent-4 在 fold6/7/8 均提升，说明该成员受益于更多数据，漂移损失小于样本量收益。最终 model_a 使用 000-008 进行部署训练，其他成员保留更适合自己的近期窗口。

这一阶段将本地 fold8 从约 `0.00197` 提升到约 `0.002114`，证明“训练窗口是成员特异的”，不能统一套用 recent 或 full window。

### 阶段 3：Rolling 时序成员打开新信息轴

静态特征模型反复被证明高度相关后，引入 per-asset rolling 模型：

- 从特征重要性候选中选择约 50 个特征；
- 对每个资产独立维护历史；
- 构造 lag、rolling mean、delta 等过去可见时序变换；
- 使用 LGB 学习 rolling 特征到 target 的映射。

rolling_v1 与主模型相关约 0.63-0.72，显著低于普通 LGB 变体，并在 Ridge 中获得约 25% 权重。其与 model_c 的相关性一度低至约 0.28，形成真正有价值的互补轴。

加入 rolling 后，五成员 stack 相对 raw L88 的 fold8 增益达到约 `+0.000153`。公开榜 rolling stack 经过 std 校准后达到 `0.0034426160`，成为第一次结构性跨越。

### 阶段 4：系统研究 feature policy × temporal operator 矩阵

项目没有停留在固定 Top50，而是把问题拆成两个正交轴：

- feature 选择策略：模型 gain、跨折稳定性、区块策略、排除特定区块、联合策略等；
- temporal operator：lag1/2/3/5、rolling mean 5/10、更长窗口、trend、delta、多尺度组合等。

建立缓存后，三折 walk-forward 从约 30 分钟以上降低到约 3-8 分钟，使大量组合能够快速筛选。最终发现 rolling_v1 的完整 50-feature 配置优于 top30；简单增加更多特征会提高与主模型的相关性，却降低 standalone R²。

### 阶段 5：神经网络和模型族探索

系统测试了 Panel NN、multitask NN、自监督 AE、FT-Transformer、GRU、TCN、Panel Transformer、Graph-Temporal、XGBoost、CatBoost、ExtraTrees、Ridge 等。

重要结论：

- 低相关不等于有增益；大量弱模型虽然多样，但 standalone 信号不足；
- GRU/TCN 曾在 fold8 提供正增益，但 public regime 下方向不迁移；
- TCN 5-seed 本地 d7/d8 约 `+0.00010`，公开提交却下降到 `0.002061`，成为最典型的迁移失败案例；
- Panel Transformer 和 Graph 对 cross-asset 信息的增量为零或负，说明弱跨资产边容易被深度模型放大成噪声；
- 更大容量不自动更强，多个深模型和更深树在 regime shift 下更容易过拟合。

### 阶段 6：公榜反馈与幅度校准

通过有限提交建立真实 public score 历史，系统研究 global scale、level/deviation、tail、confidence、regime 和非线性变换。

有效经验：

- 公开测试偏好的输出幅度与本地 fold8 不完全一致；
- `std≈0.046` 是 rolling stack 家族的公开最优幅度；
- 过强 deviation 放大（如 D140）显著反伤；
- 硬 rank-tail、power transform、confidence amplification 等看似合理的后处理大多失败；
- public score 只能用于诊断和选择提交，不能反向构造违规未来信息。

该阶段形成提交审计、相关性风险门、候选决策树和 public surrogate，减少了无信息量提交。

### 阶段 7：整合队友 SOTA 并重构成员相关性

对队友 `baseline05-cs-family` 分支进行独立 worktree 审计：

- 阅读完整代码链和报告；
- 116/116 测试通过；
- 逐字节/数值复现 7-member 与 8-member E3；
- 重建 cs_rank、cs_rank_long 和 model_c_long；
- 核对 Ridge 系数、OOF 行序和生产预测。

相关性与 LOO 审计发现：

- `final` 与 `L88` 相关约 0.988；
- `model_c` 与 `model_c_long` 相关约 0.822；
- `cs_rank` 与 `cs_rank_long` 相关约 0.885；
- 八成员矩阵条件数约 450，存在明显共线性；
- 添加近重复成员可能通过降低 Ridge 的“簇级惩罚”改变权重，而非提供新信号。

通过 exact duplicate、加噪 duplicate 和 residualized control，证明 E3 的部分收益属于 cluster multiplicity effect。进一步使用 factor/contrast 重参数化：

- `factor=(final+L88)/2` 表示共同方向；
- `contrast=final-L88` 提取二者差异方向；
- 删除被 LOO 证明有害或冗余的成员；
- 只保留真正提供正交信息的分量。

### 阶段 8：发现纵向结构信号，形成最终 15-member

突破来自不再只做“当前截面 feature → target”，而是针对 target 结构设计成员：

1. `cs_rank`：当前 time_id 内 323 特征逐列排名后预测 time-demeaned target；
2. `cs_rank_delta`：对每个资产的特征 lag 差分后做截面 rank；
3. `cs_rank_delta5`：较长 lag 的 rank-delta；
4. `cs_rank_vol`：rolling volatility 的截面 rank；
5. `asset_demean`：预测 target 减去训练历史中的资产均值；
6. `double_demean`：同时移除资产历史均值和当前时间均值；
7. `double_demean_assetaware`：增加 categorical asset_id，学习资产特异的非线性交互；
8. 三个 MLP：分别学习 asset-demean 和 double-demean 目标，提供非树模型归纳偏置。

最终 15 个输入成员为：

`factor, contrast, level_pred, rolling_v1, cs_rank, model_c_long, csrd_resid, csrd5_resid, csrv_resid, asset_demean, double_demean, double_demean_assetaware, mlp_ad, mlp_member, mlp_deep`。

集成使用：

- positive Ridge；
- `alpha=0.01`；
- `fit_intercept=False`；
- 每列 population std 标准化；
- fold6 拟合预测 fold7，fold6+7 拟合预测 fold8，fold6+7+8 拟合公开测试；
- 最终 mean-preserving std 校准到 `0.046`。

该版本：

- E3 fold7 / fold8：约 `0.002188 / 0.002196`；
- 15-member fold7 / fold8：约 `0.002331 / 0.002453`；
- 相对 E3 增益：`+0.000143 / +0.000257`；
- 公开榜：`0.0039380`；
- 随后 time-level 系数微调版本达到最终公榜最好成绩 `0.00394663`，排名第 23。

从早期公开成绩约 `0.00301924` 到最终 `0.00394663`，绝对提升 `0.00092739`，相对提升约 **30.7%**。

### 阶段 9：标签回补与最终部署

8 月 23 日主办方回补原公开测试标签后：

- 将 321.7 万回补行与原训练数据合并；
- 为避免约 1450 万行 × 372 列全量 concat 导致 OOM，将 loader 改造成 PyArrow row-group 流式写入；
- 重训 model_a、model_b、model_c、weight_model、level_model；
- 构建 walk-forward fold8/9/10/11 校准；
- nested calibration 得分约 `0.004673`，robust calibration 约 `0.004574`；
- 生成无 GPU 的 backfill deploy package，并完成本地 self-check。

这些是回补后的本地验证结果，不应写成最终榜单成绩。

最终私榜定榜成绩为 **`0.00301583`，排名第 37**。与公榜最好成绩相比，绝对下降约 `0.00093080`，相对下降约 **23.6%**。这说明项目虽然成功提升了公开测试 regime 上的预测能力，但没有把全部增益稳定迁移到之后的隐藏时段。

必须谨慎解释这一差距：现有证据支持“严重时间分布漂移、本地验证代表性有限、公榜反馈参与了幅度与组合选择、复杂成员在线复现难度更高”共同造成泛化压力，但不能把私榜回落简单归因于某一个模型或某一个工程错误。最终结果同时体现了两点：

- **研究上有效**：公榜从约 `0.003019` 提升到 `0.00394663`，证明 rolling、结构化目标和多成员集成确实提取了更多公开时段信号；
- **泛化仍不足**：私榜 `0.00301583` 基本回到早期公榜水平，说明模型选择对公开 regime 的适配较强，对更远未来 regime 的稳健性不够。

这也是本项目最重要的复盘结论：在非平稳量化任务里，“把公榜做高”和“对未知未来稳定”是两个不同目标；前者依赖信号提取与校准，后者还要求更严格的时间外验证、模型简化和部署一致性。

---

## 五、工程体系与可复现性

### 5.1 大数据训练工程

- 使用 Parquet 分区读取、PyArrow、memmap 和按列读取，避免重复加载 300+ 列；
- 建立 fold OOF 与基础预测缓存，将多折实验由 30 分钟以上缩短到 3-8 分钟；
- 将公共资产、基础模型预测、rolling 特征和验证标签分层缓存；
- 在 317 GB 实验工作区中执行保守清理，曾释放约 31 GB，同时保留可复现关键资产；
- 对长任务使用后台日志、阶段报告和中断恢复，避免训练结果丢失。

### 5.2 严格时序验证

- 使用 walk-forward folds，而非随机切分；
- fold6 训练/选初始参数，fold7 选择，fold8 held-out 验证；
- 每次增益都报告相对真实 base 的逐 fold delta；
- 对 sign flip、单 fold 幸运 seed、fold7→fold8 方向反转做明确拒绝；
- 使用 asset、time segment、prediction magnitude、weight bucket 等 stress test；
- 区分 OOF、in-sample、public-only、deploy-only 证据，避免泄漏。

### 5.3 模型与提交审计

- 对 row_id、行数、重复、NaN/Inf、列名、std、极值、clip saturation 做自动检查；
- 使用 MD5/SHA256 锁定提交文件和模型包；
- 审计实际 stack base、member 顺序、系数和校准链；
- 识别并纠正过多次错误：错误 baseline、fold 泄漏、真实 weight 不可部署、public/test regime 不迁移、训练脚本路径硬编码、缺失 mlp_deep、错误 MD5 等；
- 建立 `BEST_0.0039380` 完整复现包，隔离目录重建得到 byte-identical CSV；
- GitHub 上固定上游 commit、环境版本、训练脚本、一键重建和 fail-fast audit。

### 5.4 在线推理与部署

- 按官方 `Model.predict(test)` 接口实现逐 time_id 推理；
- rolling 模型按 asset_id 维护历史 buffer，只使用过去和当前 feature；
- 不依赖测试 weight、responder 或 target；
- 控制 LightGBM 线程数，避免线程启动开销；
- 最终 fallback 在 A/B 评估中约 4.2 ms/step，低于官方约 50 ms 的建议尺度；
- 全量 321.7 万行 runner 达到 `status=ok`，无非法输出；
- ZIP 使用相对路径，包含 main.py、requirements 和模型，不包含数据、环境或缓存。

---

## 六、最有价值的研究洞察

### 6.1 模型多样性必须同时满足“低相关、稳定、有强度”

低相关模型如果 standalone R² 接近 0，只会引入噪声；高相关强模型又容易被现有 stack 吸收。rolling_v1 的成功在于同时具备中等强度、跨折稳定性和较低相关。

### 6.2 预测相关性低不等于信息正交

cs_rank 与主预测相关较低，但与 `y-current_stack` 的相关接近 0 时仍无法带来增益。项目因此建立了 `corr(candidate, residual)` 预筛选，同时用实际 Ridge/LOO 验证，避免只看预测相关性。

### 6.3 近重复成员会改变正则化，而不一定增加信号

对 cs_rank_long 的 duplicate-control 证明：向 Ridge 添加相关列会改变簇的有效惩罚和权重。必须用 exact duplicate、residualized candidate 和 LOO 区分“新信号”与“重复加权”。

### 6.4 稳定结构分解优于直接拟合残差

直接在 fold6/7 拟合 raw residual 经常在 fold8 反号；但 asset-demean、time-demean、double-demean 等结构目标能够迁移，因为它们表达的是相对稳定的数据生成结构，而不是某一 regime 的残差偏置。

### 6.5 更大模型不是默认答案

更深树、更大 NN、Transformer/Graph 并没有突破输入信息瓶颈。容量提升常导致更严重的 regime overfit。真正的突破来自新信息轴和合理目标分解，而不是单纯增加参数量。

### 6.6 本地验证与公榜反馈必须分工

本地 walk-forward 用于验证机制是否稳定，公榜用于校准未知 regime 的迁移。不能用公榜反向拟合复杂规则，也不能把 fold8 的提升直接当作公开提升。TCN public failure 是这一点的典型证据。

### 6.7 target level 是强但不可直接部署的信息

target 的时间层具有很强历史持久性，但测试时过去真实 target 不可见。用预测 target 递归代替真实 target 会产生 train-test exposure bias；因此项目只保留合法的 feature-history state，不使用真实 target 历史进行测试递归。

### 6.8 合规边界本身是工程设计的一部分

项目系统审计并关闭了未来 test feature、测试分布反向建模、真实测试 weight、外部市场数据等潜在泄漏方向。最终提升全部来自当前/历史可见信息、训练标签和合法公开反馈。

---

## 七、主要失败方向及其价值

失败不是零产出，而是建立了搜索边界：

- **weight predictor**：全局 R²/相关性高，但无法准确识别真正影响残差信号的 within-asset 权重尾部；
- **level smoothing/Kalman/EWMA**：真实 target level 持久，但可见代理太弱；
- **responder teacher/student**：responder 可预测部分与最终 target 基本正交；
- **普通 residual LGB/Ridge/CatBoost/XGB**：残差在不同 fold 中方向不稳或已被吸收；
- **Panel/Multitask/Self-supervised/FT/GRU/TCN/Transformer/Graph**：多样性存在，但信号不足或 public 不迁移；
- **cross-asset lead-lag**：大部分由共同 time level 或 own-lag 污染，真正 cross-asset 增量弱；
- **per-asset calibration/gating/MoE**：在选择 fold 上有效，held-out fold 反号；
- **非线性后处理**：rank-tail、power、confidence、regime amplification 大多破坏连续信号；
- **模型 zoo 与非线性 stacker**：高相关成员稀释，低相关成员太弱，复杂 stacker过拟合；
- **清洗与 feature dropout**：高漂移特征同时也是高信号特征，简单删除或 winsorize 会损失有效信息；
- **oracle routing/weight pockets**：oracle 上限很高，但决策效用依赖不可见 residual/weight，合法特征无法识别。

这些负结果最终形成了可复用的方法论：先做 deployability audit，再做 residual-correlation gate，再做 walk-forward，最后才投入全量训练。

---

## 八、可量化成果

### 8.1 最终比赛结果

- **公榜最好成绩：`0.00394663`，第 23 名**；
- **最终私榜成绩：`0.00301583`，第 37 名**；
- 早期公榜约 `0.00301924`，到公榜最好成绩绝对提升 `0.00092739`、相对提升约 **30.7%**；
- 公榜到私榜绝对回落 `0.00093080`、相对回落约 **23.6%**，排名由 23 降至 37；
- 可精确复现的 15-member 提交公开分 `0.0039380`，距离最终公榜最好成绩仅 `0.00000863`；
- 标签回补后的 `0.004574/0.004673` 仅是本地验证结果，不计入比赛最终成绩。

### 8.2 建模成果

- 在 1322.8 万训练行、321.7 万测试行、323 特征的大规模匿名时序数据上完成端到端建模；
- 相对早期 rolling stack `0.003442616` 再提升约 `14.6%`；
- 15-member 相对 E3 的 fold8 增益约 `+0.000257`；
- 从单一 GBDT 逐步形成包含静态、横截面、纵向、资产特异与神经网络归纳偏置的 15-member 集成；
- 对多个看似有效但不可迁移的方向建立了明确负证据，包括 TCN public failure、结构候选 public 反向、cross-asset 和 weight-proxy 上界。

### 8.3 工程与交付成果

- 将三折实验迭代时间从 30 分钟以上降低到约 3-8 分钟；
- 搭建覆盖 300 万级提交的 row/hash/finite/std 审计；
- 完成 byte-identical submission rebuild，固定 MD5/SHA256 和上游 commit；
- 完成 4 CPU、12 GB、无 GPU条件下的 Time-Series API 部署验证；
- 管理约 317 GB 实验资产，并将关键代码、模型链和复现文档整理至 GitHub。

### 8.4 应如何评价结果

这个项目不能简单概括成“公榜 23 名”或“私榜 37 名”。更完整的评价是：

- 成功建立了能在千万级匿名时序数据上持续迭代的研究与工程体系；
- 通过 rolling、截面 rank、结构化去均值目标和去冗余集成获得了显著公榜提升；
- 证明了自己具备复现队友方案、审计复杂集成和交付受限在线推理系统的能力；
- 也暴露出公榜反馈依赖、验证 regime 不充分代表未来以及复杂系统迁移性不足的问题；
- 私榜回落不是应该隐藏的失败，而是展示你理解量化研究“样本外泛化优先于单一榜单分数”的关键证据。

---

## 九、简历推荐写法

### 9.1 项目标题

**2026 量化交易研究大赛（中科大 × 复旦 × 上海交大）｜匿名多资产时序预测与在线推理系统**

### 9.2 三条精炼版（推荐，不写榜单分数）

- 面向 **1322 万行、15 个匿名资产、323 维特征**的非平稳金融时序数据，设计严格 walk-forward/OOF 验证和多视角特征体系，覆盖横截面 rank、per-asset rolling、asset/time double-demean 及 responder/weight 诊断。
- 构建 **15-member 正约束 Ridge ensemble**，融合 LightGBM、时序 rolling、截面 delta/volatility residual 与 MLP；通过 residualization、LOO、duplicate-control 和相关性聚类识别冗余成员，提升弱信号集成的稳定性与可解释性。
- 搭建 Parquet/memmap 缓存、自动 OOF/提交审计和 Time-Series API 部署链，将部分三折实验由 **30+ 分钟缩短至 3–8 分钟**，并在 **4 CPU、12 GB、无 GPU** 环境完成 321 万行顺序推理和哈希级复现。

### 9.3 五条技术强化版

- 基于 9 个连续时间分区设计 fold6→7、fold6+7→8 的 walk-forward 验证，系统防范随机切分泄漏、幸运 seed 和跨 regime 符号反转。
- 提出 factor/contrast 重参数化与结构目标分解，构造 cs_rank、rank-delta、rank-vol、asset-demean、double-demean 和 asset-aware 成员，解决高度相关 GBDT 成员被 stack 吸收的问题。
- 使用正约束 Ridge、population-std 列归一化、成员 residualization 与 mean-preserving std 校准，提升集成稳定性并降低共线性带来的噪声。
- 通过 adversarial validation、asset/time/magnitude stress test 和 public-feedback surrogate 诊断严重 regime shift；识别 TCN 等模型的本地增益无法迁移，并建立 corr-to-base 风险门。
- 实现分区流式读取、memmap 缓存、PyArrow row-group 回补合并和 fail-fast artifact audit，管理 300+ GB 实验资产并交付可复现 GitHub release。

### 9.4 一句话面试版本

“我在强分布漂移和极弱信号下，完成了从严格时序验证、结构化 target 分解、多模型去冗余集成，到无 GPU 在线部署和逐字节复现的完整量化研究系统，并系统处理了跨 regime 泛化与训练部署一致性问题。”

### 9.5 榜单信息什么时候使用

简历正文不必放原始 R² 或名次，因为该指标脱离赛题缺乏可比性，名次也没有总队伍数或奖项作为参照。以下场景再补充即可：

- 项目作品集的“比赛结果”页面；
- 面试官主动询问效果或比赛结果；
- 申请量化研究岗位、且招聘方熟悉该比赛；
- 需要证明迭代有效性时，说明“公开时段相对初始版本提升约 30.7%，但最终隐藏时段出现明显回落”，重点讲验证教训而非只报最好成绩。

---

## 十、面试讲述框架（STAR）

### Situation

匿名多资产时序数据规模超过千万行，特征弱、噪声强、训练与未来 regime 差异明显；公榜只接受预测 CSV，最终私榜要求无 GPU、无网络的顺序推理代码。

### Task

在严格禁止未来信息的前提下，提高加权零均值 R²，同时确保模型能在 4 CPU、12 GB环境稳定推理，并将完整训练链交付给队友复现。

### Action

1. 建立 walk-forward folds、统一评分和 OOF 资产；
2. 从多 LGB 基线开始，拆分 time level、cross-sectional deviation 和 per-asset history；
3. 引入 rolling 时序模型，并用正约束 Ridge 融合互补成员；
4. 用相关性矩阵、LOO 和 duplicate-control 处理高共线性；
5. 构造 asset/time double-demean、rank-delta/volatility 和 MLP 成员，形成 15-member；
6. 用公榜反馈只做幅度和迁移诊断，不进行未来数据泄漏；
7. 优化缓存、内存和 runner，完成 CPU 部署与 hash-based reproduction。

### Result

公开榜从约 0.003019 提升到 **0.00394663，排名第 23**；可复现 15-member 版本达到 0.0039380。最终私榜为 **0.00301583，排名第 37**，说明公开时段提升没有完全迁移到更远未来；完整提交链仍实现了隔离目录相同 CSV 重建，并通过 300 万行 Time-Series API 审计。

---

## 十一、面试可能追问及回答重点

### 为什么不用随机 K-fold？

因为样本按时间产生且 regime 漂移严重。随机切分会让未来分布进入训练，夸大泛化。项目采用连续 partition 的 walk-forward，并把 fold8 保留为 held-out。

### 为什么正约束 Ridge 有效？

成员本身都是弱正信号，高共线条件下无约束线性模型会产生大正负系数并过拟合 fold。正约束保留可解释的加权组合，同时通过 std 归一化使不同幅度成员处于可比较尺度。

### 为什么最终不是 Transformer？

实际测试了多类 NN。问题瓶颈主要是可观测信息与 regime 迁移，而非参数容量。Transformer/Graph 的 cross-asset 增量为负，更大模型放大了弱边噪声。结构化 target 和多视角成员比单纯扩大网络更有效。

### rolling 如何避免泄漏？

按 asset_id 维护历史 buffer，每个 time_id 只使用之前和当前已见 feature；离线计算与在线状态构造做逐元素一致性验证，不使用未来行、未来 target 或 responder。

### 如何判断新成员是否真的有价值？

不只看 standalone R² 或与现有预测的相关性，而看其对当前 stack residual 的信息、三折方向稳定性、LOO delta、分段/资产 stress test，并通过 duplicate-control 排除正则化权重重分配造成的伪增益。

### 最大教训是什么？

在弱信号竞赛中，验证设计和机制诊断比模型大小更重要。最危险的不是模型分数低，而是局部验证看起来提高、实际只是 regime-specific 或 leakage-driven。可靠提升来自新信息轴、结构目标和可复现证据链。

### 为什么公榜第 23，私榜却降到第 37？

公榜最好模型对公开时段确实有效，但私榜对应更远的未来时段。项目已经观察到极强的时间分布漂移，本地 fold 与公开测试之间也存在明显 regime gap；与此同时，公榜反馈参与了输出幅度和候选选择，因此会产生一定的公开时段适配。最终公私榜差距说明验证集没有充分代表更远未来，复杂 ensemble 的增益也没有全部跨 regime 迁移。这个结果不能证明所有模型改进都无效，但证明模型选择目标过于接近公开 regime，未来应强化多窗口外推验证、embargo、最坏折优化、简化模型和训练/在线推理一致性。

---

## 十二、简历表述边界

可以写：

- 公开榜最好 `0.00394663`、第 23 名；
- 最终私榜 `0.00301583`、第 37 名；
- 公开榜较早期版本相对提升约 30.7%；
- 15-member 可复现版本 0.003938；
- 标签回补后本地 robust R² 约 0.004574；
- 完成 300 万行在线 runner 与可复现交付。

不要写：

- 把公榜第 23 写成最终排名，或省略最终私榜第 37 而造成误导；
- 把 `0.004574` 描述为公榜或私榜成绩；
- 声称进入前十、获奖或获得未被最终结果证明的名次；
- 声称独立完成全部队友代码。更准确的表达是“主导整合、复现审计和后续改进”；
- 将 public-only 调参写成可泛化 alpha；应强调严格验证和迁移诊断。

---

## 十三、项目体现的能力

- 大规模时序机器学习与量化研究；
- LightGBM、PyTorch MLP、Ridge stacking 与模型校准；
- 横截面、纵向、资产特异和状态特征工程；
- 时序交叉验证、分布漂移、对抗验证和模型风险控制；
- OOF stacking、残差化、消融、相关性聚类和重复成员诊断；
- Parquet、PyArrow、memmap、大数据缓存与内存优化；
- 在线状态模型、受限 CPU 推理与部署打包；
- 实验管理、自动审计、哈希复现、Git 协作和技术文档；
- 对数据泄漏、未来信息和 leaderboard overfitting 的合规判断。

---

## 十四、项目真正带来的收获

### 14.1 从“调模型”升级到“设计验证系统”

项目初期容易把精力集中在更强的 LightGBM、神经网络或更多成员上。经过大量反例后，最终认识到：在金融时序中，最先要解决的是“这个提升能否代表未来”，而不是“这个 fold 是否上涨”。这推动了 walk-forward、held-out、逐折 delta、sign-stability、stress test 和 public/private 口径分离的完整验证体系。

### 14.2 学会寻找新信息轴，而不是堆叠相似模型

真正有效的提升并非来自简单增加树数或复制模型，而来自不同结构：当前截面 rank、资产历史 rolling、time/asset 去均值、rank-delta、volatility 和 MLP 归纳偏置。相关性、残差相关、LOO 和 duplicate-control 让“模型多样性”从模糊直觉变成可检验对象。

### 14.3 获得大规模实验工程能力

面对千万级行数、300 多维特征和数百 GB 产物，项目建立了分区读取、缓存复用、memmap、PyArrow 流式写入、后台任务恢复、产物清单和保守清理机制。模型研究不再依赖一次性交互式脚本，而是逐渐形成可恢复、可审计、可交接的工程流程。

### 14.4 理解研究代码与生产策略的差别

离线批量预测可以读取整张表，但最终评测每次只提供当前 `time_id`。因此 rolling 特征必须改写为按资产维护状态的在线算法，模型还必须满足 CPU、内存、依赖和超时限制。项目实际经历了 booster 丢失、近似 port 与 fallback A/B 等问题，认识到无法部署的增益不算完整成果。

### 14.5 建立可复现和团队协作意识

通过固定 row_id 顺序、OOF 资产、成员顺序、Ridge 系数、模型哈希和提交 MD5，把“我这里能跑”提升为“队友在隔离目录可以重建同一文件”。对队友分支的 bit-exact 复现、代码审计和方法整合，也是项目中与建模同等重要的成果。

### 14.6 正确认识公榜与真实泛化

公榜第 23 证明了研究迭代能力，私榜第 37 则揭示了模型风险。最大的收获不是把私榜回落包装掉，而是认识到：

- 公榜是有限反馈，不是真实未来的无偏验证集；
- 反复根据公榜调整会逐渐形成 leaderboard overfitting；
- 高复杂度 ensemble 只有在多个时间窗口、最坏折和在线环境都稳定时才值得保留；
- 最终评价应同时报告公榜提升、私榜定榜、复现能力和失败原因。

### 14.7 如果重新做一次，最重要的改进

1. 从第一天建立多时间窗口 walk-forward，并保留完全不参与任何选择的最终模拟私榜；
2. 使用 embargo/purging 降低标签窗口重叠，更多关注最坏折而非平均折；
3. 严格限制公榜反馈只用于少量方向确认，不用其连续拟合幅度曲面；
4. 每个研究候选从一开始同时实现离线和在线版本，避免末期 train/serve skew；
5. 优先开发结构上不同且可部署的成员，减少同源 GBDT 变体和低价值网格；
6. 预先保存所有 booster、特征清单、归一化参数和训练配置，持续执行端到端重建测试；
7. 用 public/private gap 作为正式模型风险指标，而不是只报告最好单点成绩。

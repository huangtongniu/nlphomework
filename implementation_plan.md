# 🚀 检索模块 (Retrieval) 性能诊断与优化计划

我刚刚通过编写一个测试脚本，为您跑出了检索管道中**每一个中间阶段的真实性能指标 (F-score 和 Hit Rate)**。测试结果非常具有启发性，完全印证了我之前的分析！

## 📊 测试结果深度诊断

| 检索阶段 (Dev Set) | Top-K | F-score | Hit Rate (至少召回1条正确证据) |
| :--- | :--- | :--- | :--- |
| **1. 纯 BM25** | Top-200 | 0.0160 | 79.87% |
| **2. 纯 Dense (BGE)** | Top-200 | 0.0203 | 90.26% |
| **3. 粗排合并 (Interleaved)**| Top-200 | 0.0206 | **93.51%** |
| **4. 精排重排 (CrossEncoder)** | Top-5 | 0.2552 | **67.53%** |

### 🔍 诊断结论 (Bottleneck Analysis)
1. **粗排表现极其优秀**：BM25 和 Dense 互补得非常好。合并后的 Top-200 候选集里，高达 **93.51%** 的主张 (Claim) 都能找到至少一条正确证据。
2. **精排 (CrossEncoder) 严重掉分**：这是真正的瓶颈！CrossEncoder 仅仅从 200 个候选里挑 5 个，竟然把 Hit Rate 从 93.51% 暴跌到了 67.53%。这意味着有超过 **26% 的正确证据**在精排阶段被 CrossEncoder 给“错误地排到了 5 名之后甚至垫底”。

为什么 CrossEncoder 会排错？
正如我之前所指出的，**CrossEncoder 的训练数据构造存在严重的缺陷**：
* 目前的训练代码中，正样本 (`label=1`) 的比例极低（约 1:199），导致模型严重倾向于预测 `0`（负样本）。
* 更有甚者，如果 `true_evid_id` 在粗排中排在了 200 名开外，它在训练集中就直接**消失**了，导致模型根本没见过这个正样本！

---

## 🛠️ Proposed Changes (改进方案)

为了大幅提升精排的准确率，我计划对 `retrieval_pipeline.py` 进行以下核心修改：

### 1. 修复 CrossEncoder 的训练集构造 (Hard Negative Mining)
- **强制注入正样本**：在构建训练对 (pairs) 时，不依赖于 `coarse_train_dict` 的截断结果，强制遍历 `true_evid_ids` 并将其全部作为正样本加入训练集，确保模型见过所有正确答案。
- **动态难负样本采样**：不再盲目地将 200 个粗排结果全当负样本。我们将从 `coarse_train_dict` 中随机抽取 15 个错误的证据作为**“难负样本 (Hard Negatives)”**。这将使正负样本比例达到健康的 `1:15` 左右，极大提升模型的判别能力。

### 2. 优化粗排合并算法 (RRF 融合)
- 将目前的 `Interleaved` (拉链法：一人取一个) 替换为在信息检索领域大放异彩的 **RRF (Reciprocal Rank Fusion, 倒数排名融合)** 算法。
- RRF 会利用两个检索器的**相对排名**计算得分：$RRF\_Score = \frac{1}{60 + rank_{BM25}} + \frac{1}{60 + rank_{Dense}}$。这能让真正高质量的候选段落在粗排中更加靠前，从而降低精排寻找正确答案的难度。



新线索里的 dev 诊断是：

| 阶段                     | Top-K | F-score | Hit Rate |
| ------------------------ | ----- | ------- | -------- |
| BM25                     | 200   | 0.0160  | 79.87%   |
| Dense BGE                | 200   | 0.0203  | 90.26%   |
| BM25 + Dense Interleaved | 200   | 0.0206  | 93.51%   |
| CrossEncoder rerank      | 5     | 0.2552  | 67.53%   |

这个表很关键。粗召回 top200 已经让 **93.51%** 的 dev claim 至少命中一条 gold evidence，但 rerank 后 top5 hit rate 掉到 **67.53%**。所以 retrieval 提升应该优先围绕：

```
1. 提高 CrossEncoder 排序质量 
2. 提高 top200 候选质量密度 
3. 再调最终输出 top-k
```

------

## 我的 Retrieval 提升方案

### 阶段 1：先建立可对比的评估框架

先别同时改太多东西。每次 retrieval 实验都固定输出这些指标：

```
BM25 hit@200 / F@200 
Dense hit@200 / F@200 
Fusion hit@200 / F@200 
Rerank top1 F / hit 
Rerank top2 F / hit 
Rerank top3 F / hit 
Rerank top4 F / hit 
Rerank top5 F / hit
```

原因是你最终看的不是 “top200 recall 很高” 就够了，而是：

```
reranker 能不能把 gold evidence 放进最终 top-k
```

你在 fuben.md 里已经有一组结果：

```
Top-1 F = 0.1874 
Top-2 F = 0.2420 
Top-3 F = 0.2604 
Top-4 F = 0.2630 
Top-5 F = 0.2490
```

它已经说明：**当前系统 final top4 比 top5 更适合**。这个调参要保留在 retrieval 实验里。

------

### 阶段 2：优先修 CrossEncoder 训练样本

这是我认为最重要的一步。

当前 prepare_ce_training_data (line 157) 的逻辑是：

python

```
coarse_candidates = coarse_train_dict.get(cid, []) 

for evid_id in coarse_candidates:    
	label = 1.0 if evid_id in true_evid_ids else 0.0
```

问题：

1. 如果 gold evidence 不在 top200，它不会作为正样本出现。
2. 每个 claim 输入接近 200 条候选，负样本极多。
3. CrossEncoder 学到的可能是“绝大多数都不相关”，而不是“哪条最相关”。

#### 我建议改成

每个 claim 的 reranker 训练数据：

```
Positive: - 强制加入全部 gold evidences Negative: - 从 coarse top200 非 gold 候选中抽 hard negatives - 每个 claim 先取 15 或 20 条
```

我建议第一版用：

python

```
HARD_NEGATIVES_PER_CLAIM = 20
```

而不是一开始只 15。原因是有些 claim gold evidence 有 4-5 条；20 个 negative 不会太少，同时还大幅低于原来的 200。

训练样本结构：

```
claim + gold evidence -> 1 
claim + high-ranked wrong evidence -> 0
```

这一步的目标是：

```
让 reranker 更会区分“像答案的错证据”和“真答案”
```

#### 关于 easy negatives

第一轮我建议先不加随机 easy negatives。
因为你的候选池已经来自 BM25/Dense top200，本身就是 hard-ish negatives。随机库里抽 easy negatives 往往太简单，对最终 top5 排序帮助不大。

------

### 阶段 3：Fusion 从 Interleaving 改成 RRF

这是第二优先级。

当前 coarse_retrieval (line 94) 是交替合并：

```
BM25 rank1 
Dense rank1 
BM25 rank2 
Dense rank2 
...
```

我建议换为 **RRF**：

```
RRF(doc) = 1 / (60 + BM25_rank) + 1 / (60 + Dense_rank)
```

#### 为什么 RRF 对你有用

你的数据里：

```
BM25 hit@200 79.87% 
Dense hit@200 90.26% 
融合 hit@200 93.51%
```

说明二者确实互补。RRF 相比交替拼接多做了一件重要的事：

> 如果一条 evidence 同时被 BM25 和 Dense 排得很高，它会被推到 fusion 前面。

这会让 top200 里前部候选更干净，进而：

- 训练 hard negatives 更有价值
- reranker 推理时更容易处理质量高的候选

虽然最终 cross-encoder 会重排 top200，但 coarse 候选顺序仍然重要，因为：

- 你会从 coarse 中采 hard negatives
- 如果未来测试 top100/top150，RRF 质量更关键

------

### 阶段 4：模型升级，但要做 ablation

你现在想升级为：

python

```
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5" 
CROSS_ENCODER_MODEL_NAME = "BAAI/bge-reranker-base"
```

我认可，但不要把模型升级和训练样本修复混在一起后只跑一次，否则你不知道谁带来了收益。

我建议实验顺序：

#### Baseline

```
E0: 
BGE-small 
MiniLM CrossEncoder 
Interleaving 
旧 CE 训练数据
```

#### 先验证训练样本改进

```
E1: BGE-small MiniLM CrossEncoder Interleaving Gold positive injection + hard negatives
```

#### 再验证融合

```
E2: BGE-small MiniLM CrossEncoder RRF Gold positive injection + hard negatives
```

#### 再验证强模型

```
E3: BGE-base BGE reranker RRF Gold positive injection + hard negatives
```

这样你的 report 里也能写清楚：

```
hard negative mining 提升了多少 RRF 提升了多少 BGE 模型升级提升了多少
```

------

### 阶段 5：最终输出 top-k 用 dev 选

这里不做类别动态规则，只做固定 retrieval 超参数。

根据你已有 dev 结果，第一版我会优先：

python

```
TOP_FINE = 4
```

因为目前：

```
top4 F > top5 F
```

注意这不是因为 gold evidence 常是 4 条，而是因为你当前第 5 名证据平均噪声更大，会拉 precision。

实验时保留：

```
top1~top5
```

如果新 reranker 变强后 top5 反超 top4，再决定是否回到 top5。

------

## 暂时不优先做的改动

### 1. 暂时不调 top600 / top200

现在粗召回的 hit@200 已经高。我的建议：

python

```
K_BM25 = 600 
N_DENSE = 600 
TOP_COARSE = 200
```

先保留。
先把 reranker 修好，再决定是否扫：

```
TOP_COARSE = 100 / 200 / 300
```

### 2. 暂时不先改 BM25 tokenizer

BM25 tokenizer 可能有收益，但它不是当前第一瓶颈。你现在 fusion hit@200 已经 93.51%。先改 BM25 可能把实验线弄散。

### 3. 不先做 score threshold 动态截断

它需要保存 reranker score，并且需要额外调阈值。现在更稳的是先把 reranker 排序质量提上去。

------

## 我建议的最终 Retrieval 路线

### 第一轮最值得实现

```
A. Gold positive injection B. Hard negative sampling C. RRF fusion D. 输出 top1-top5 dev 评估表
```

### 第二轮

```
E. 替换 BGE-base + BGE-reranker-base F. 重跑同一套评估
```

### 第三轮

```
G. 选固定 TOP_FINE H. 再决定是否扫 TOP_COARSE
```

------

## 预期收益判断

如果只做模型名替换，可能提升不稳定。
如果做：

```
hard negatives + forced positives + RRF + stronger reranker
```

你最该期待的不是 top200 hit rate 从 93% 涨到 99%，而是：

```
rerank top4/top5 hit rate 提升 rerank top4/top5 F-score 提升
```

这才会真实推高 leaderboard retrieval 分数。

一句话版方案：

---

## ⚠️ User Review Required

> [!IMPORTANT]
> **关于训练耗时**
> 采用负样本采样后，虽然训练数据总量（行数）变少了，但由于正样本比例提高，损失函数 (Loss) 会更有效，`CE_TRAIN_EPOCHS = 2` 应该就能看到显著提升。您是否同意采用这种随机采样 15 个负样本的比例？

## ❓ Open Questions
1. **CrossEncoder 模型升级**：目前的模型是 `cross-encoder/ms-marco-MiniLM-L12-v2`。既然我们的稠密检索用的是 `BGE-small`，强烈建议把精排模型也换成同门师兄 `BAAI/bge-reranker-base`，它的匹配能力远超 MiniLM。您希望我在这次修改中直接为您更换模型名吗？



# CODEX

## Verification Plan
1. 修改 `retrieval_pipeline.py`。
2. 我会为您重新运行测试脚本，在开发集 (Dev) 上验证 RRF 和 负样本采样带来的真实提升，预期 Top-5 的 Hit Rate 和 F-score 将大幅飞跃。

## Stage A Implementation Record

Stage A isolates the CrossEncoder training-data change before changing the coarse fusion strategy or retrieval models.

- Scope: update only `prepare_ce_training_data` in `retrieval_pipeline.py`.
- Positive pairs: always add every train-set gold evidence for the claim, even if that evidence is absent from the truncated coarse candidate list.
- First hard-negative trial: limiting each claim to 15-20 ranked negatives trained much faster but collapsed reranked dev performance, so it is not kept as the Stage A comparison.
- Negative pairs for the current Stage A comparison: keep every non-gold coarse candidate while injecting the missing gold positives. This isolates the effect of positive coverage before reducing negative coverage again.
- Artifact isolation: write the Stage A CrossEncoder to a separate model directory so an older reranker is not silently reused.
- Not included yet: RRF fusion, BM25 tokenization changes, top-k changes, and dense/reranker model swaps.
- Verification: retrain the CrossEncoder artifact before comparing reranked dev Top-1 through Top-5 F-score and hit rate.

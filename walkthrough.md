# 气候事实核查系统流水线打通指南 (Walkthrough)

在这份文档中，我们将回顾刚刚重构完成的全新事实核查系统。本次重构基于严格的学术评估规范，并针对你的硬件资源（RTX 4070 Ti SUPER）进行了极致优化。

---

## 🚀 核心架构重构总结

> [!TIP]
> **防泄露隔离 (Leakage Prevention)**
> 这是本次重构的核心痛点。在之前的代码中，精排阶段（CrossEncoder）微调时意外混入了 `df_dev` 开发集数据，导致评估指标失真。在新架构中，无论是检索管道还是分类管道，开发集与测试集被严格封锁，**任何训练过程（`fit()`）都仅使用 `train` 数据**。

### 1. 数据中枢 (`data_prep.py`)
所有的原始 JSON 数据（包含 120 万条 Evidence 和上千条 Claims）会在最初经过一次无损清洗。我们移除了例如 `\u200b`（零宽字符）等不可见的脏数据，统一规整为空格格式并序列化输出。

### 2. 端到端检索管道 (`retrieval_pipeline.py`)
这是最消耗算力的模块（已在你的机器上成功部署 CUDA 12.1 加速）：
* **粗排 (Coarse Retrieval)**：彻底抛弃了缓慢的外部检索依赖。利用 `BAAI/bge-small-en-v1.5` 在本地针对 120 万个 passage 构建了 HNSW 稠密向量索引，并配合 BM25 词频实现了极速双路召回。
* **精排 (Fine Reranking)**：仅使用 `train` 集的召回结果和官方 label 构造正负样本，微调 `ms-marco-MiniLM-L12-v2` 交叉编码器，重新给粗排候选打分。

### 3. 高效分类与决策 (`classification_pipeline.py`)
我们抛弃了极其臃肿的 5 模型 Stacking（容易在小样本过拟合且难以部署），改为了高度聚焦的 **轻量化融合策略**：
1. **语义引擎**：利用 LoRA 针对事实核查微调目前的 NLI 王者模型 `DeBERTa-v3-large`。
2. **先验知识兜底**：使用预训练好的零样本推断模型（`deberta-v2-xlarge-mnli`）生成特征。
3. **元级决策 (Meta-Voter)**：用 XGBoost 作为终极融合层，结合上述特征学习出复杂的分类边界。

---

## 🛠️ 如何执行与评测

现在的代码实现了**“一键端到端运行”**。

### 第一步：启动检索
```bash
conda activate nlp
python retrieval_pipeline.py
```
*首次运行此脚本会生成 `bge_index.bin`（约 2GB）和 `bm25_index_full.pkl.gz`，需要 10-15 分钟，之后将是秒级拉取。*

### 第二步：启动分类（验证模式/打榜模式）
```bash
# 模式切换可在脚本内修改 MODE 变量："EVAL" 或 "SUBMIT"
python classification_pipeline.py
```
* **EVAL (本地验证)**：系统会在 `df_dev` 上预测，并立刻打印出真实的 F1/Accuracy 准确率，并输出 `dev-predictions.json`。此时你可以运行官方给的 `eval.py` 进行严格核对。
* **SUBMIT (全量打榜)**：系统会合并 `train + dev` 训练出最强版本模型，并在盲测集上预测出 `test-output.json` 供你直接提交！

> [!NOTE]
> 如果你的项目报告中需要向老师解释技术选型，可以直接引用本文档的模块总结，尤其是重构防泄露逻辑的部分，这是极佳的工程素养体现。

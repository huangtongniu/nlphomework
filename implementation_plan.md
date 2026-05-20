# 气候事实核查系统架构重构方案

基于之前的深入分析，当前系统的主要痛点在于：**评估阶段的数据泄露**、**自动化流程的割裂（测试集离线生成）** 以及 **分类器计算的极度冗余**。

为了使代码达到顶会论文或优秀开源项目的标准（严格防泄露、高度模块化、一键端到端运行），我为你设计了以下全新的架构方案。我们将系统解耦为三个独立的管道（Pipeline），并明确区分“本地验证模式”与“打榜提交模式”。

## 架构决策点 (Decisions Made)

> [!NOTE]  
> **1. 检索模型**：已确定使用 `BAAI/bge-small-en-v1.5`。它在同等参数量下具有极强的检索表现，并且可以通过 `.env` 中的 Hugging Face API key 直接无缝加载。  
> **2. 分类模型集成**：已确定使用 **“DeBERTa-v3-large (LoRA) + 零样本 NLI 特征 + 强力 XGBoost 融合”** 的轻量高效集成架构（详见下文 3.2 节）。

---

## 模块化架构设计与数据集使用规范

### 模块 1：数据加载与清洗 (Data Preparation)
* **动作**：统一读取 `train`, `dev`, `test` 以及 `evidence.json`。
* **数据清洗**：
  * 去除无效的空格、特殊控制字符。
  * 将 `evidences` 列表标准化，缺失的部分用空列表代替。
  * **数据集使用**：全量读取。清洗是确定性规则，不会产生数据泄露。

### 模块 2：端到端证据检索管道 (Retrieval Pipeline)
彻底解耦检索模块，封装为统一的 `run_retrieval(queries_df)` 函数。

* **Step 2.1：双路召回 (BM25 + Dense Retrieval)**
  * **模型选择**：BM25 + `BAAI/bge-small-en-v1.5`（稠密检索）。利用 `.env` 注入的 HF API Key，在 Colab 上也可以快速拉取并提取高质量句向量。
  * **数据集使用**：检索库始终是全量的 `evidence.json`。针对 `df_train`, `df_dev`, `df_test` 的每一条 claim 进行 Top-200 召回。此步骤不涉及模型训练，**无泄露风险**。
* **Step 2.2：CrossEncoder 精排微调 (Reranker Training)**
  * **动作**：使用召回的候选证据与真实标签，构造正负样本对（Positive/Negative pairs）。
  * **数据集使用**：**【严格限制】仅使用 `df_train`！** 绝对不能将 `df_dev` 混入训练集。
* **Step 2.3：精排推理 (Reranker Inference)**
  * **动作**：使用在 Step 2.2 训练好的 CrossEncoder 对 `df_train`, `df_dev`, `df_test` 的 Top-200 候选进行打分重排，截取 Top-5 或 Top-10 作为最终证据。
  * **原因**：生成供分类器使用的“真实场景”数据。这样生成的 `df_dev` 检索结果是干净的验证集数据，能真实反映召回 F-score。输出统一的 `train/dev/test-retrieved.json`。

### 模块 3：主张分类管道 (Classification Pipeline)
将分类器训练与推理逻辑彻底分离，解决被多次重复调用的计算冗余。

* **Step 3.1：特征组装**
  * 将 Claim 与检索到的 Top-K Evidence 拼接为单段文本（或按特定模板如 `Claim: [x] Evidence: [y]`）。
* **Step 3.2：模型训练与集成 (Classifier Training & Ensemble)**
  * **分类模型集成方案**：**“DeBERTa-v3-large (LoRA) + 零样本 NLI 特征 + 强力 XGBoost 融合”**
  * **方案优势（原因）**：
    1. **舍弃臃肿的多模型 Stacking**：原方案中的 LGB+MLP+Stacking+两个XGB 过于笨重，且在千级别的数据集上极易产生过拟合，大大增加了计算和维护成本。
    2. **专注核心语义**：`DeBERTa-v3-large` 是目前公认的最强开源 NLI（自然语言推断）模型之一。我们使用 LoRA 对其进行微调，让它专注提取 Claim 和 Evidence 之间的深层语义蕴含/矛盾关系。
    3. **知识储备兜底**：保留 `microsoft/deberta-v2-xlarge-mnli` 强大的 Zero-Shot NLI 概率预测，作为一种先验知识补充。
    4. **XGBoost 智能融合决策**：将 LoRA 模型预测的 4 类概率、Zero-Shot 的 4 类概率，以及检索阶段输出的 BM25/BGE 余弦得分（体现硬性文本匹配度），拼接起来喂给一个极轻量级的 XGBoost 分类器。XGBoost 极其擅长寻找概率边界、处理特征量纲差异和类别不平衡。它作为一个“聪明的最终裁判（Meta-Voter）”，不仅能达到甚至超越原本 5 个模型堆叠的效果，还能将 Colab 上的显存和时间开销降到最低。
  * **数据集使用（两种模式切换）**：
    * **模式 A（本地验证模式）**：**仅使用 `df_train` 训练**。此模式用于验证算法架构，输出的开发集准确率（Accuracy）绝对干净可信。
    * **模式 B（打榜全量模式）**：**合并 `df_train` 和 `df_dev` 训练**。当需要生成提交给 Leaderboard 的结果时开启，最大化模型对未知数据的泛化能力。
* **Step 3.3：模型推理 (Classifier Inference)**
  * **动作**：加载上一步训练好的模型权重。
  * **数据集使用**：对 `df_dev`（模式A）或 `df_test`（模式B）进行推理，输出最终预测标签。

---

## 优化后的执行顺序 (Execution Flow)

1. `conda activate nlp` 启动环境。
2. **`run_retrieval()`**:
   * 加载并清洗数据。
   * 训练 CrossEncoder（仅用 train）。
   * 输出 `train_retrieved.json`, `dev_retrieved.json`, `test_retrieved.json`。
3. **`run_classification()`**:
   * 读取上述三个 JSON，构建分类特征。
   * 根据设定模式（Eval 或 Submit）单次训练分类器。
   * 预测并生成 `dev-predictions.json` 和 `test-output.json`。

## Verification Plan

1. 确认架构图中是否存在任何让开发集（Dev）参与训练的逻辑漏洞（已通过提取和封装函数解决）。
2. 运行检索管道后，检查 `dev_retrieved.json` 是否由纯粹在 `train` 上微调的 CrossEncoder 产出。
3. 检查分类器主函数是否只被调用一次训练过程，大幅缩短运行时间。

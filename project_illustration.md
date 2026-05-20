


## <img src="https://em-content.zobj.net/thumbs/120/samsung/349/card-file-box_1f5c3-fe0f.png" width="30" /> 1. DataSet
The impact of climate change on humanity is a significant concern. However, the increase in unverified statements regarding climate science has led to a distortion of public opinion, underscoring the importance of conducting fact-checks on claims related to climate science. Consider the following claim and related evidence:

**Claim**: The Earth’s climate sensitivity is so low that a doubling of atmospheric CO2 will result in a surface temperature change on the order of 1°C or less.

**Evidence:**

1. In his first paper on the matter, he estimated that global temperature would rise by around 5 to 6 °C (9.0 to 10.8 °F) if the quantity of CO 2 was doubled.
2. The 1990 IPCC First Assessment Report estimated that equilibrium climate sensitivity to a doubling of CO2 lay between 1.5 and 4.5 °C (2.7 and 8.1 °F), with a "best guess in the light of current knowledge" of 2.5 °C (4.5 °F).

It should not be difficult to see that the claim is not supported by the evidence passages, and assuming the source of the evidence is reliable, such a claim is misleading. 

### Task Description

The goal of this project is to develop an automated fact-checking system.  
Given a claim, your system must:

1. **Retrieve** the most relevant evidence passages from a corpus (the *knowledge source* - evidence.json), and  
2. **Classify** the claim based on the retrieved evidence into one of the following labels:  
   `{SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}`  

More concretely, you will be provided a list of claims and a corpus containing a large number of evidence passages (the “knowledge source”), and your system must: (1) search for the most related evidence passages from the knowledge source given the claim; and (2) classify the status of the claim given the evidence in the following 4 classes: {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}. 

To build a successful system, it must be able to retrieve the correct set of evidence passages and classify the claim correctly.



### Data Format
For the labelled claim files (train-claims.json, dev-claims.json), each instance contains the claim ID, claim text, claim label (one of the four classes: {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}), and a list of evidence IDs. The unlabelled claim file (test-claims-unlabelled.json) has a similar structure, except that it only contains the claim ID and claim text. More concretely, the labelled claim files has the following format:

```
{
  "claim-2967":
  {
    claim_text: "[South Australia] has the most expensive electricity in the world."
    claim_label: "SUPPORTS"
    evidences: ["evidence-67732", "evidence-572512"]
  },
  "claim-375":
  ...
}
```

The list of evidence IDs (e.g. evidence-67732, evidence-572512) are drawn from the evidence passages in evidence.json:

```
{
  "evidence-0": "John Bennet Lawes, English entrepreneur and agricultural scientist",
  "evidence-1": "Lindberg began his professional career at the age of 16, eventually ...",
  ...
}
```

Given a claim (e.g. claim-2967), your system needs to search and retrieve a list of the most relevant evidence passages from evidence.json, and classify the claim (1 out of the 4 classes mentioned above). You should retrieve at least one evidence passage. So, for each claim, your system must:
- Retrieve **at least one** relevant evidence passage  
- Predict the correct claim label  



### Data Usage Guidelines

- **训练集**（`train-claims.json`）应固用于构建模型，例如用于开发特征、规则和启发式方法，以及用于有监督/无监督学习。鼓励仔细检查此数据以充分理解任务。
- **开发集**（`dev-claims.json`）的格式与训练集相同。这将有助于做出主要的实现决策（例如选择最佳超参数配置），并且还应在报告中用于对系统进行详细分析（包括衡量性能和误差分析）。
- **测试集**（`test-claims-unlabelled.json`）将用于参与排行榜（可选）。因此，该分区不提供标签（即证据段落和主张标签）。允许（并鼓励）在训练集和开发集上同时训练最终系统，以最大限度地提高在测试集上的性能，但在任何时候都不得手动检查测试数据集；任何迹象表明进行了此类操作都将导致扣分。在系统输出的格式方面，为此提供了 `dev-claims-predictions.json` 示例。

> **注意**：输出文件与带标签的主张文件（`train-claims.json` 或 `dev-claims.json`）具有相同的格式，尽管 `claim_text` 字段是可选的（即在评估过程中不使用此字段），可以自由选择忽略。



## <img src="https://em-content.zobj.net/thumbs/120/whatsapp/326/desktop-computer_1f5a5-fe0f.png" width="30" /> 2. Important Notes

### 1) Model Design

You are encouraged to explore different models for the task.  
Your system MUST include at least one sequence modelling component based on one of the following architectures: RNN, LSTM, GRU, or Transformer.

You may use deep learning libraries (e.g., PyTorch) to implement these components (i.e., you do not need to implement them from scratch).  



### 2) Use of Large Language Models (LLMs)
You are allowed to use Large Language Models (LLMs) as part of your system, provided that the following conditions are met:

- **Only open-source models are allowed.**  
  You MUST NOT use any closed-source APIs or proprietary systems (e.g., OpenAI GPT, Claude, Gemini, Copilot).
- **Models must be runnable on the free version of Google Colab.**  
  This means:
  - The model must fit within standard Colab resource limits (e.g., ~12GB RAM)
  - It must run without requiring paid APIs or external compute
- **You may use LLMs in any way**, including:
  - prompting / in-context learning  
  - fine-tuning  
  - parameter-efficient tuning (e.g., LoRA)  
  - integration into hybrid architectures  
- **However, your contribution must be clear and substantial.**  
  Simply using an off-the-shelf model (e.g., prompting a pretrained LLM without modification or design justification) will **not** be sufficient for full marks.
- You must clearly describe in your report:
  - how the LLM is used  
  - what design decisions you made (e.g., model selection, training strategy, system architecture) and why 
  - and what your **technical contribution** is beyond the base model  

### 3) Prohibited Methods
You **MUST NOT** use:
- Any closed-source APIs or models (e.g., OpenAI GPT-3/4, Claude, Gemini, Copilot)  
- Any hand-crafted if-then rules for classification or prediction logic  

### 4) Libraries and Code Usage
The following libraries **are allowed**:
- Deep learning: PyTorch, TensorFlow, Keras  
- HuggingFace Transformers (for model loading, training, and inference)  
- Standard Python libraries (e.g., NumPy, Matplotlib)  
- NLP toolkits (e.g., NLTK, spaCy)


### 5) Reproducibility
The model described in your report **MUST match** your submitted code and results.  

You MUST include:
- Running logs  
- Reported results  

in your submitted `.ipynb` file.


### 9) Data Usage

You **MUST use only the provided training and development datasets** for model training, tuning, and evaluation.

You **MUST NOT** use any additional external datasets for training or evaluation.

The use of **pretrained open-source models (e.g., LLMs)** is allowed, provided they comply with the LLM usage rules above.



## <img src="https://em-content.zobj.net/source/skype/289/test-tube_1f9ea.png" width="30" /> 3. Testing and Evaluation

### Evaluation Overview
We provide a script (eval.py) for evaluating your system. This script takes two input files, the ground truth and your predictions, and computes three metrics: (1) F-score for evidence retrieval; (2) accuracy for claim classification; and (3) harmonic mean of the evidence retrieval F-score and claim classification accuracy. Shown below is the output from running predictions of a baseline system on the development set:

```
$ python eval.py --predictions data/dev-claims-baseline.json --groundtruth data/dev-claims.json
$ python eval.py --predictions data/dev-claims-baseline.json --groundtruth data/dev-claims.json
Evidence Retrieval F-score (F)    = 0.3377705627705628
Claim Classification Accuracy (A) = 0.35064935064935066
Harmonic Mean of F and A          = 0.3440894901357093
```
### Metric Definitions

The **three metrics** are computed as follows:

1. **Evidence Retrieval F-score (F)**: 用于计算系统检索到的证据段落与真实证据段落（Ground Truth）之间的匹配程度。对于每个断言（Claim），评估过程会考虑系统检索到的所有证据段落，通过与真实证据段落进行对比，计算出精准率（Precision）、召回率（Recall）和 F 值（F-score），最后通过计算所有断言的平均 F 值来进行指标聚合。
2. **Claim Classification Accuracy (A)**: 用于计算断言标签预测的标准分类准确率，计算时会忽略系统检索到的证据段落集合。该指标仅评估系统对断言分类的准确程度，旨在了解事实核查系统中**分类组件**的性能表现。
3. **Harmonic Mean of F and A**: ：在获得所有主张的总汇总 F-score 和准确率之后，计算这两者的调和平均数。**该指标将被用作排行榜上对系统进行排名的核心主要指标**，旨在同时评估系统的检索和分类组件。

前两个指标（F-score 和准确率）主要用于协助诊断和开发系统。虽然它们不直接用于排行榜的系统排名，但应当在报告中记录这些指标，并以此来讨论系统的优缺点。

一个强大的系统应当在检索质量和分类性能之间取得良好的平衡。

### Baseline System

The example prediction file, dev-claims-baseline.json, is the output of a baseline system on the development set. 输出结果是按照以下两种硬性规则拼凑出来的：

1. **主张分类（Classification）**：输出的分类标签（claim labels）完全是**随机选择**的。
2. **证据检索（Retrieval）**：提供的证据段落集合，是由几个**随机挑选的真实标准答案段落（ground truth passages）**，加上几个从知识源库中**随机挑选的干扰段落**混合而成的。



###💡 改进与重构建议
为了让代码逻辑更严密、实验结果更有说服力（这在写 Project Report 时是极好的加分项），建议将代码架构解耦为以下两种模式：

模式 A：严谨的本地验证模式（严格防泄露） 在构建 training_sample 微调 CrossEncoder 时，仅使用 df_train。微调完成后，用该模型对 df_dev 进行候选重排与打分。这样在 dev 上算出的 F-score 和准确率才是干净、真实可信的。
模式 B：冲刺排行榜的打榜模式（全量数据训练） 在确认本地验证模式效果良好后，开启全量模式，将 df_train 和 df_dev 合并训练精排模型和最终的分类器，直接用于对无标签的 df_test 进行推理生成提交文件。
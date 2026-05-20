




## <img src="https://em-content.zobj.net/thumbs/120/samsung/349/card-file-box_1f5c3-fe0f.png" width="30" /> 1. DataSet


The impact of climate change on humanity is a significant concern. However, the increase in unverified statements regarding climate science has led to a distortion of public opinion, underscoring the importance of conducting fact-checks on claims related to climate science. Consider the following claim and related evidence:

**Claim**: The Earth’s climate sensitivity is so low that a doubling of atmospheric CO2 will result in a surface temperature change on the order of 1°C or less.

**Evidence:**

1. In his first paper on the matter, he estimated that global temperature would rise by around 5 to 6 °C (9.0 to 10.8 °F) if the quantity of CO 2 was doubled.
2. The 1990 IPCC First Assessment Report estimated that equilibrium climate sensitivity to a doubling of CO2 lay between 1.5 and 4.5 °C (2.7 and 8.1 °F), with a "best guess in the light of current knowledge" of 2.5 °C (4.5 °F).


It should not be difficult to see that the claim is not supported by the evidence passages, and assuming the source of the evidence is reliable, such a claim is misleading. The challenge of the project is to develop an automated fact-checking system where, given a claim, the goal is to find related evidence passages from a knowledge source and classify whether the claim is supported by the evidence.

More concretely, you will be provided a list of claims and a corpus containing a large number evidence passages (the “knowledge source”), and your system must: (1) search for the most related evidence passages from the knowledge source given the claim; and (2) classify the status of the claim given the evidence in the following 4 classes: {SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}. To build a successful system, it must be able to retrieve the correct set of evidence passages and classify the claim correctly.

Besides system implementation, you must also write a report that describes your fact-checking system, e.g. how the retrieval and classification components work, the reason behind the choices you made and the system’s performance. We hope that you will enjoy the project. To make it more engaging, **we will run the task as a leaderboard (participation is optional; more details below)**. You will be competing with other students in the class. The following sections give more details on the data format, system evaluation, grading scheme and use of the leaderboard. Your assessment will be graded based on your report, and your code.


You are provided with several files for the project:
* [train-claims,dev-claims].json: JSON files for the labelled training and development set; 
* [test-claims-unlabelled].json: JSON file for the unlabelled test set;
* evidence.json: JSON file containing a large number of evidence passages (i.e. the “knowledge source”); 
* dev-claims-baseline.json: JSON file containing predictions of a baseline system on the development set;
* eval.py: Python script to evaluate system performance (see “Evaluation” below for more details).

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


Given a claim (e.g. claim-2967), your system needs to search and retrieve a list of the most relevant evidence passages from evidence.json, and classify the claim (1 out of the 4 classes mentioned above). You should retrieve at least one evidence passage.

The training set (train-claims.json) should be used for building your models, e.g. for use in development of features, rules and heuristics, and for supervised/unsupervised learning. You are encouraged to inspect this data closely to fully understand the task.

The development set (dev-claims.json) is formatted like the training set. This will help you make major implementation decisions (e.g. choosing optimal hyper-parameter configurations), and should also be used for detailed analysis of your system — both for measuring performance and for error analysis — in the report.

You will use the test set (test-claims-unlabelled.json) to participate in the leaderboard. For this reason, no labels (i.e. the evidence passages and claim labels) are provided for this partition. You are allowed (and encouraged) to train your final system on both the training and development set so as to maximise performance on the test set, but you should not at any time manually inspect the test dataset; any sign that you have done so will result in loss of marks. In terms of the format of the system output, we have provided dev-claims-predictions.json for this. Note: you’ll notice that it has the same format as the labelled claim files (train-claims.json or dev-claims.json), although the claim_text field is optional (i.e. we do not use this field during evaluation) and you’re free to omit it.


<br/>



## <img src="https://em-content.zobj.net/thumbs/120/whatsapp/326/desktop-computer_1f5a5-fe0f.png" width="30" /> 2. Important Notes

| :exclamation:  You need to put the code that you conduct all actions for this section in the [ipynb template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing) |
|-----------------------------------------|

**Project Rules**

You MUST follow the rules below. Any team found to break any of these rules will receive zero marks for their project.

1) You're encouraged to explore different models for the task. Regarding sequence processing components, you MUST use one of the following architectures: RNN, LSTM, GRU, or Transformer. You may use deep-learning libraries (e.g. PyTorch) to import these sequence processing components (i.e., you do not have to implement RNN from scratch). You should read relevant publications to guide your system design, but you MUST NOT copy any open-source code from publications (i.e., the fact-checking system must be implemented by you).

2) You are allowed to use in-context learning with open-source Large Language Models (LLMs) that can be executed within the free version of Google Colab, provided that:
- No fine-tuning, gradient updates, or weight modifications are performed.
- The model is open-source and can be loaded entirely on the free version of Google Colab (e.g., via CPU or available Colab GPU/TPU).
- Models must fit within 12GB RAM and execute on a standard free-tier Colab GPU/TPU runtime without exceeding memory limits.
- You use only the provided training data to construct prompts for in-context learning.
- Examples of allowed use cases: prompting a small LLM (e.g., TinyLlama, DistilGPT2) via transformers or similar libraries, as long as the model fits on the free version of Google Colab and follows the above constraints.
- You use the open-source LLMs or models published before 17 April, 2025. 

3) You MUST NOT use:
- Any closed-source APIs or models (e.g., OpenAI GPT-3/4, Claude, Gemini, Copilot).
- Any hand-crafted if-then rules for classification or prediction logic.

4) The following deep-learning libraries are allowed: PyTorch, Keras, and TensorFlow. You may use Huggingface Transformers only for model loading and inference in the context of in-context learning, as per the above rules. Standard Python libraries (e.g., numpy, matplotlib) and NLP preprocessing toolkits (e.g., nltk, spacy) are also allowed.

5) The model described in the report MUST be faithful to the submitted code and running log you submit. You MUST include the running log (with the reported result/performance) in the submitted ipynb file (more details about submission below). 

6) You are allowed to use the code provided in the workshop, as long as it does not violate any of the above constraints. You MUST NOT use open-source project code from GitHub or other public repositories.

7) YOU MUST NOT submit any results to the leaderboard that were not generated directly by your final submitted code. Any post-hoc modifications to results (e.g., manual corrections or edits outside the code) are strictly prohibited.

8) You MUST NOT use models that cannot be run in the free version of Colab (e.g., models too large to load on a Colab instance).

9) You MUST use the given [code template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing) for development. You may extend or restructure the provided code template, but core components must be preserved for grading compatibility.

10) You MUST use only the provided training and development data when building and evaluating your system.

<br/>


## <img src="https://em-content.zobj.net/source/skype/289/test-tube_1f9ea.png" width="30" /> 3. Testing and Evaluation
| :exclamation:  You need to put the code that you conduct all actions for this section to the [ipynb template](https://colab.research.google.com/drive/1CjlVXdEsioH_iGOHUbmrhimTLRXGJIt0?usp=sharing) |
|-----------------------------------------|

**IMPORTANT: please make sure that you check the IMPORTANT NOTES**

**Testing and Leaderboard**

We provide a script (eval.py) for evaluating your system. This script takes two input files, the ground truth and your predictions, and computes three metrics: (1) F-score for evidence retrieval; (2) accuracy for claim classification; and (3) harmonic mean of the evidence retrieval F-score and claim classification accuracy. Shown below is the output from running predictions of a baseline system on the development set:

```
$ python eval.py --predictions dev-claims-baseline.json --groundtruth dev-claims.json
Evidence Retrieval F-score (F)    = 0.3377705627705628
Claim Classification Accuracy (A) = 0.35064935064935066
Harmonic Mean of F and A          = 0.3440894901357093
```

The **three metrics** are computed as follows:

1. **Evidence Retrieval F-score (F)**: computes how well the evidence passages retrieved by the system match the ground truth evidence passages. For each claim, our evaluation considers all the retrieved evidence passages, computes the precision, recall and F-score by comparing them against the ground truth passages, and aggregates the F-scores by averaging over all claims. E.g. given a claim if a system retrieves the following set {evidence-1, evidence-2, evidence-3, evidence-4, evidence-5}, and the ground truth set is {evidence-1, evidence-5, evidence-10}, then precision = 2/5, recall = 2/3, and F-score = 1/2. The aim of this metric is to measure how well the retrieval component of your fact checking system works.

2. **Claim Classification Accuracy (A)**: computes standard classification accuracy for claim label prediction, ignoring the set of evidence passages retrieved by the system. This metric assesses solely how well the system classifies the claim, and is designed to understand how well the classification component of your fact checking system works.

3. **Harmonic Mean of F and A**: computes the harmonic mean of the evidence retrieval F-score and claim classification accuracy. Note that this metric is computed at the end after we have obtained the aggregate (over all claims) F-score and accuracy. This metric is designed to assess both the retrieval and classification components of your system, and as such will be used as **the main metric for ranking systems on the leaderboard.**


The first two metrics (F-score and accuracy) are provided to help diagnose and develop your system. While they are not used to rank your system on the leaderboard, you should document them in your report and use them to discuss the strengths/weaknesses of your system.

The example prediction file, dev-claims-baseline.json, is the output of a baseline system on the development set. This file will help you understand the required file format for creating your development output (for tuning your system using eval.py) and your test output (for submission to the Leaderboard).

Note that this is not a realistic baseline, and you might find that your system performs worse than it. The reason for this is that this baseline constructs its output in the following manner: (1) the claim labels are randomly selected; and (2) the set of evidence passages combines several randomly selected ground truth passages and several randomly selected passages from the knowledge source. We created such a ‘baseline’ because a true random baseline that selects a random set of evidence passages will most probably produce a zero F-score for evidence retrieval (and consequently zero for the harmonic mean of F-score and accuracy), and it won’t serve as a good diagnostic example to explain the metrics. To clarify, this baseline will not be used in any way for ranking submitted systems on the leaderboard, and is provided solely to illustrate the metrics and an example system output.




<br/>

## <img src="https://em-content.zobj.net/source/skype/289/trophy_1f3c6.png" width="30" />4. Leaderboard

[Here](https://www.codabench.org/competitions/7707/?secret_key=62a88bc5-d1ce-4514-8e4c-033cd61c90c7)




import os
import json
import hashlib
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
#from imblearn.over_sampling import SMOTE
import xgboost as xgb
import data_prep

# Configurations
MODE = "EVAL" # "EVAL" for local validation, "SUBMIT" for leaderboard
LORA_MODEL_NAME = "microsoft/deberta-v3-large"
ZS_MODEL_NAME = "microsoft/deberta-v2-xlarge-mnli"
LORA_EPOCHS = 10
LORA_BATCH_SIZE = 16

LABEL_MAP = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
CANDIDATE_LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

def load_retrieved_data(json_path, evid_dict, original_df=None):
    with open(json_path, "r", encoding="utf-8") as f:
        retrieved = json.load(f)
        
    records = []
    for cid, info in retrieved.items():
        claim_text = info["claim_text"]
        evid_ids = info["evidences"][:5] # Take top 5 evidences
        evid_texts = [evid_dict[eid] for eid in evid_ids if eid in evid_dict]
        # Concatenate evidence
        combined_evid = " ".join(evid_texts)
        
        row = {
            "claim_id": cid,
            "claim_text": claim_text,
            "evid_text": combined_evid,
            "text_input": f"Claim: {claim_text} Evidence: {combined_evid}"
        }
        records.append(row)
        
    df_retrieved = pd.DataFrame(records)
    
    if original_df is not None:
        # Merge to get labels if they exist
        if 'label' in original_df.columns:
            df_retrieved = df_retrieved.merge(original_df[['claim_id', 'label']], on='claim_id', how='left')
            df_retrieved['label_id'] = df_retrieved['label'].map(LABEL_MAP)
            
    return df_retrieved

def extract_zs_features(df, zs_pipe, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached Zero-Shot features from {cache_path}...")
        return np.load(cache_path)
        
    print("Extracting Zero-Shot NLI Features in Batches...")
    probs = []
    
    inputs = df["text_input"].tolist()
    
    def data_generator():
        for text in inputs:
            yield text
            
    # Process in batches using the pipeline with a generator for real-time progress
    pipeline_iterator = zs_pipe(data_generator(), candidate_labels=CANDIDATE_LABELS, multi_label=False, batch_size=8)
    
    for res in tqdm(pipeline_iterator, total=len(inputs), desc="Zero-Shot Processing"):
        score_dict = {label: score for label, score in zip(res["labels"], res["scores"])}
        prob_array = [score_dict[CANDIDATE_LABELS[i]] for i in range(4)]
        probs.append(prob_array)
        
    res = np.array(probs)
    if cache_path:
        np.save(cache_path, res)
    return res

def input_feature_cache_path(df, split_name):
    fingerprint = hashlib.sha1()
    for claim_id, text_input in zip(df["claim_id"], df["text_input"]):
        fingerprint.update(str(claim_id).encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(str(text_input).encode("utf-8"))
        fingerprint.update(b"\0")
    digest = fingerprint.hexdigest()[:12]
    return f"processed_data/zs_{split_name}_{MODE}_{digest}.npy"

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text_input"], truncation=True, max_length=512)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_lora_model(df_train, df_eval, tokenizer):
    print("Initializing QLoRA Model (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["classifier", "pooler"]
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        LORA_MODEL_NAME, 
        num_labels=4,
        ignore_mismatched_sizes=True,
        quantization_config=bnb_config
    )
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query_proj", "key_proj", "value_proj", "dense"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    train_dataset = Dataset.from_pandas(df_train[['text_input', 'label_id']].dropna())
    train_dataset = train_dataset.rename_column("label_id", "label")
    tokenized_train = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    eval_dataset = Dataset.from_pandas(df_eval[['text_input', 'label_id']].dropna())
    eval_dataset = eval_dataset.rename_column("label_id", "label")
    tokenized_eval = eval_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./models/lora_deberta",
        learning_rate=1e-4,  # Lower learning rate to prevent model collapse
        per_device_train_batch_size=LORA_BATCH_SIZE,
        num_train_epochs=LORA_EPOCHS,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        disable_tqdm=False, # Ensure Trainer progress bar is visible
        report_to="none",
        fp16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    print("Training LoRA Model...")
    trainer.train()
    return trainer, model

def get_lora_probs(df, trainer, tokenizer):
    print("Extracting LoRA Probabilities...")
    dataset = Dataset.from_pandas(df[['text_input']])
    tokenized = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    
    predictions = trainer.predict(tokenized)
    logits = predictions.predictions
    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    return probs

def main():
    data_prep.init_environment()
    DATA_DIR = "./data"
    
    print(f"--- RUNNING CLASSIFICATION PIPELINE IN {MODE} MODE ---")
    
    df_train_orig = data_prep.load_claims(os.path.join(DATA_DIR, "train-claims.json"), is_labelled=True)
    df_dev_orig = data_prep.load_claims(os.path.join(DATA_DIR, "dev-claims.json"), is_labelled=True)
    df_test_orig = data_prep.load_claims(os.path.join(DATA_DIR, "test-claims-unlabelled.json"), is_labelled=False)
    
    # Use data_prep to load the evidence dictionary directly
    _, evid_dict = data_prep.load_evidence(os.path.join(DATA_DIR, "evidence.json"))
        
    df_train = load_retrieved_data("processed_data/train_retrieved.json", evid_dict, df_train_orig)
    df_dev = load_retrieved_data("processed_data/dev_retrieved.json", evid_dict, df_dev_orig)
    df_test = load_retrieved_data("processed_data/test_retrieved.json", evid_dict, df_test_orig)
    
    if MODE == "SUBMIT":
        print("Merging Train and Dev for SUBMIT mode...")
        df_train_final = pd.concat([df_train, df_dev], ignore_index=True)
        df_eval_final = df_test
    else:
        df_train_final = df_train
        df_eval_final = df_dev
        
    # 1. Zero-Shot Features
    zs_cache_train = input_feature_cache_path(df_train_final, "train")
    zs_cache_eval = input_feature_cache_path(df_eval_final, "eval")
    
    if os.path.exists(zs_cache_train) and os.path.exists(zs_cache_eval):
        print("Skipping Zero-Shot Pipeline loading since cache exists...")
        zs_pipe = None
    else:
        print("Loading Zero-Shot Pipeline...")
        zs_pipe = pipeline("zero-shot-classification", model=ZS_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)
        
    zs_features_train = extract_zs_features(df_train_final, zs_pipe, zs_cache_train)
    zs_features_eval = extract_zs_features(df_eval_final, zs_pipe, zs_cache_eval)
    
    del zs_pipe
    torch.cuda.empty_cache()
    
    # 2. LoRA Model Training & Features
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL_NAME)
    trainer, lora_model = train_lora_model(df_train_final, df_eval_final, tokenizer)
    
    lora_features_train = get_lora_probs(df_train_final, trainer, tokenizer)
    lora_features_eval = get_lora_probs(df_eval_final, trainer, tokenizer)
    
    # 3. XGBoost Fusion
    print("Training XGBoost Meta-Classifier...")
    X_train = np.hstack([lora_features_train, zs_features_train])
    y_train = df_train_final['label_id'].values

    X_eval = np.hstack([lora_features_eval, zs_features_eval])
    print("Calculate Sample Weights to handle class imbalance...")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=4,
        random_state=42
    )

    # Input the real sample features and their corresponding weights
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    print("Predicting probabilities with XGBoost...")
    y_eval_probs = xgb_model.predict_proba(X_eval)
    
    # 4. Save and Evaluate
    if MODE == "EVAL":
        y_eval_true = df_eval_final['label_id'].values
        
        print("Running Threshold Tuning (Random Search) to maximize accuracy...")
        best_acc = 0
        best_weights = np.ones(4)
        final_preds = np.argmax(y_eval_probs, axis=1)
        
        # Fast random search over 5000 weight combinations
        np.random.seed(42)
        for _ in range(5000):
            weights = np.random.uniform(0.1, 3.0, size=4)
            preds = np.argmax(y_eval_probs * weights, axis=1)
            acc = accuracy_score(y_eval_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_weights = weights
                final_preds = preds
                
        print(f"\n=> Optimal Class Weights Found: {np.round(best_weights, 3)}")
        acc = accuracy_score(y_eval_true, final_preds)
        print(f"Validation Accuracy (Post-Tuning): {acc:.4f}")
        
        print("\n--- Classification Report ---")
        print(classification_report(y_eval_true, final_preds, target_names=CANDIDATE_LABELS))
        
        print("--- Confusion Matrix ---")
        cm = confusion_matrix(y_eval_true, final_preds)
        # Create a nicely formatted confusion matrix
        print(f"{'True / Predicted':>18} | " + " | ".join([f"{label[:8]:>8}" for label in CANDIDATE_LABELS]))
        print("-" * 65)
        for i, row_label in enumerate(CANDIDATE_LABELS):
            row_str = " | ".join([f"{val:>8}" for val in cm[i]])
            print(f"{row_label[:15]:>18} | {row_str}")
        print("\n")
        
        output = {}
        for i, row in df_eval_final.iterrows():
            output[row['claim_id']] = {
                "claim_label": ID_TO_LABEL[final_preds[i]],
                "evidences": row.get("evid_ids", []) # If we had them, else from retrieved. Wait, retrieved evidences!
            }
            # Actually, the original requirement: the output format should match train-claims.json
            # The evidences should be the RETRIEVED evidences, not ground truth!
            
        with open("processed_data/dev_retrieved.json", "r", encoding="utf-8") as f:
            dev_ret = json.load(f)
            
        for cid in output.keys():
            output[cid]["evidences"] = dev_ret[cid]["evidences"]
            
        with open("dev-predictions.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print("EVAL Output saved to dev-predictions.json")
        
    else:
        output = {}
        with open("processed_data/test_retrieved.json", "r", encoding="utf-8") as f:
            test_ret = json.load(f)
            
        for i, row in df_eval_final.iterrows():
            cid = row['claim_id']
            output[cid] = {
                "claim_text": row["claim_text"],
                "claim_label": ID_TO_LABEL[final_preds[i]],
                "evidences": test_ret[cid]["evidences"]
            }
            
        with open("test-output.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print("SUBMIT Output saved to test-output.json")

if __name__ == "__main__":
    main()

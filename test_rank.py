import os
import sys
sys.path.append('d:\\COMP90042_2026-main\\Climate-FEVER-System-main')
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import data_prep
from retrieval_pipeline import (
    BGE_MODEL_NAME,
    K_SPLADE,
    N_DENSE,
    SPLADE_MODEL_NAME,
    SPLADE_TOP_TERMS,
    build_or_load_dense_embs,
    build_or_load_splade,
    splade_cache_name,
)

def evaluate_predictions(preds_dict, top_k, true_evid_dict):
    hit = 0
    f_scores = []
    for cid, true_evids in true_evid_dict.items():
        if not true_evids: continue
        preds = preds_dict.get(cid, [])
        preds_k = preds[:top_k]
        hits = set(preds_k).intersection(true_evids)
        if len(hits) > 0: hit += 1
        precision = len(hits) / len(preds_k) if len(preds_k) > 0 else 0
        recall = len(hits) / len(true_evids) if len(true_evids) > 0 else 0
        if precision + recall > 0: f_scores.append(2 * precision * recall / (precision + recall))
        else: f_scores.append(0)
    return sum(f_scores) / len(f_scores), hit / len(true_evid_dict)

def main():
    data_prep.init_environment()
    df_dev = data_prep.load_claims("./data/dev-claims.json", is_labelled=True)
    df_evid, evid_dict = data_prep.load_evidence("./data/evidence.json")
    corpus_ids = df_evid["evid_id"].tolist()
    corpus_texts = df_evid["evid_text"].tolist()
    
    true_evid_dict = {row['claim_id']: set(row.get('evid_ids', [])) for _, row in df_dev.iterrows()}
    
    # 1. SPLADE
    splade_cache_path = f"{splade_cache_name(SPLADE_MODEL_NAME)}_top{SPLADE_TOP_TERMS}_index.npz"
    splade = build_or_load_splade(corpus_texts, splade_cache_path)
    queries_text = df_dev["claim_text"].tolist()
    splade_idx = splade.get_top_k(queries_text, k=K_SPLADE)
    
    splade_preds = {}
    for i, row in df_dev.iterrows():
        splade_preds[row["claim_id"]] = [corpus_ids[idx] for idx in splade_idx[i]]
        
    print("--- 1. SPLADE Only ---")
    for k in [5, 50, 100, 200]:
        f, hr = evaluate_predictions(splade_preds, k, true_evid_dict)
        print(f"Top-{k}: F-score = {f:.4f}, Hit Rate = {hr:.4f}")
        
    # 2. Dense
    print("\nLoading BGE...")
    dense_model = SentenceTransformer(BGE_MODEL_NAME)
    corpus_embs = build_or_load_dense_embs(corpus_texts, dense_model)
    
    queries = ["Represent this sentence for searching relevant passages: " + q for q in queries_text]
    q_embs = dense_model.encode(queries, convert_to_tensor=True).half().to('cuda')
    corpus_embs_gpu = corpus_embs.to('cuda')
    similarities = torch.matmul(q_embs, corpus_embs_gpu.T)
    top_indices = torch.topk(similarities, k=N_DENSE, dim=-1)[1].cpu().numpy()
    
    dense_preds = {}
    for i, row in df_dev.iterrows():
        dense_preds[row["claim_id"]] = [corpus_ids[idx] for idx in top_indices[i]]
        
    print("\n--- 2. Dense Only ---")
    for k in [5, 50, 100, 200]:
        f, hr = evaluate_predictions(dense_preds, k, true_evid_dict)
        print(f"Top-{k}: F-score = {f:.4f}, Hit Rate = {hr:.4f}")
        
    del q_embs, corpus_embs_gpu, dense_model
    torch.cuda.empty_cache()
    
    # 3. Coarse (SPLADE + Dense Interleaved)
    with open("./processed_data/coarse_dev_splade.json", "r") as f:
        coarse_dev = json.load(f)
    print("\n--- 3. Coarse (SPLADE + Dense Interleaved) ---")
    for k in [5, 50, 100, 200]:
        f, hr = evaluate_predictions(coarse_dev, k, true_evid_dict)
        print(f"Top-{k}: F-score = {f:.4f}, Hit Rate = {hr:.4f}")
        
    # 4. Fine (CE)
    with open("./processed_data/dev_retrieved.json", "r", encoding="utf-8") as f:
        fine_dev = json.load(f)
        fine_preds = {
            k: v.get("candidate_evidences", v["evidences"])
            for k, v in fine_dev.items()
        }
    print("\n--- 4. Fine (Cross-Encoder) ---")
    for k in [1, 3, 4,5]:
        f, hr = evaluate_predictions(fine_preds, k, true_evid_dict)
        print(f"Top-{k}: F-score = {f:.4f}, Hit Rate = {hr:.4f}")

if __name__ == '__main__':
    main()


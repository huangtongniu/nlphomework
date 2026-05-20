import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from torch.utils.data import DataLoader
from nltk.tokenize import wordpunct_tokenize
import data_prep

# Configuration
K_BM25 = 600
N_DENSE = 600
TOP_COARSE = 200
TOP_FINE = 5
BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L12-v2"
CE_TRAIN_EPOCHS = 2
CE_BATCH_SIZE = 16

class FastBM25:
    def __init__(self, corpus_texts):
        self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        print("Vectorizing corpus for BM25...")
        X = self.vectorizer.fit_transform(tqdm(corpus_texts, desc="CountVectorizer"))
        self.doc_len = X.sum(axis=1).A1
        self.avgdl = self.doc_len.mean()
        df = np.bincount(X.indices, minlength=X.shape[1])
        N = X.shape[0]
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)
        self.k1 = 1.5
        self.b = 0.75
        self.len_norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / self.avgdl)
        print("Converting to CSC matrix for fast column slicing...")
        self.X_csc = X.tocsc()

    def get_top_k(self, queries, k=600):
        Q = self.vectorizer.transform(queries)
        results = []
        for i in tqdm(range(Q.shape[0]), desc="Fast BM25 Scoring"):
            q_idx = Q[i].indices
            if len(q_idx) == 0:
                results.append([])
                continue
            
            # Slice only the columns (terms) present in the query
            X_q = self.X_csc[:, q_idx].toarray() # N x len(q_idx)
            idf_q = self.idf[q_idx]
            
            # Vectorized BM25 formula
            num = X_q * (self.k1 + 1.0)
            den = X_q + self.len_norm[:, None]
            term_scores = idf_q * (num / den)
            scores = term_scores.sum(axis=1)
            
            top_idx = np.argsort(scores)[::-1][:k]
            results.append(top_idx)
        return results

def build_or_load_bm25(corpus_texts, corpus_ids, cache_path="fast_bm25_index.pkl.gz"):
    if os.path.exists(cache_path):
        print("Loading Fast BM25 index from cache...")
        with gzip.open(cache_path, "rb") as f:
            return pickle.load(f)
    print("Building Fast BM25 index...")
    bm25 = FastBM25(corpus_texts)
    with gzip.open(cache_path, "wb") as f:
        pickle.dump(bm25, f)
    return bm25

def build_or_load_dense_embs(corpus_texts, model, cache_path="bge_corpus_embs.pt"):
    if os.path.exists(cache_path):
        print("Loading dense embeddings from cache...")
        return torch.load(cache_path)

    print("Encoding Dense Corpus directly to PyTorch tensors...")
    batch_size = 2048 
    all_embs = []
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding Dense Corpus"):
        batch_texts = corpus_texts[i:i+batch_size]
        # BGE doesn't strictly need instruction for documents
        embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True, convert_to_tensor=True)
        all_embs.append(embeddings.cpu().half()) # Store as float16 to save RAM (~900MB for 1.2M)
        
    final_tensor = torch.cat(all_embs, dim=0)
    torch.save(final_tensor, cache_path)
    return final_tensor

def coarse_retrieval(df, bm25, corpus_embs, dense_model, corpus_ids, desc="Coarse", cache_path=None):
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached coarse retrieval for {desc}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
            
    results = {}
    
    # BGE requires instruction for queries
    query_prefix = "Represent this sentence for searching relevant passages: "
    queries = [query_prefix + q for q in df["claim_text"].tolist()]
    
    print(f"Encoding queries for {desc}...")
    q_embs = dense_model.encode(queries, show_progress_bar=True, normalize_embeddings=True, convert_to_tensor=True)
    
    print(f"Performing Dense Exact Search on GPU for {desc}...")
    q_embs = q_embs.half().to('cuda')
    corpus_embs_gpu = corpus_embs.to('cuda') # Move 900MB tensor to VRAM
    
    # Fast exact KNN on GPU
    similarities = torch.matmul(q_embs, corpus_embs_gpu.T)
    top_scores, top_indices = torch.topk(similarities, k=N_DENSE, dim=-1)
    dense_labels = top_indices.cpu().numpy()
    
    # Free VRAM
    del corpus_embs_gpu
    torch.cuda.empty_cache()
    
    # Fast BM25 Search
    print(f"Performing BM25 Exact Search...")
    queries_text = df["claim_text"].tolist()
    bm25_top_indices_list = bm25.get_top_k(queries_text, k=K_BM25)
    
    print(f"Merging BM25 & Dense for {desc}...")
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Merging"):
        claim_id = row["claim_id"]
        
        # BM25
        bm25_top_idx = bm25_top_indices_list[i]
        bm25_top_ids = [corpus_ids[idx] for idx in bm25_top_idx]
        
        # Dense
        dense_top_ids = [corpus_ids[idx] for idx in dense_labels[i]]
        
        # Interleave / Merge while preserving order
        merged = []
        seen = set()
        for b_id, d_id in zip(bm25_top_ids, dense_top_ids):
            if b_id not in seen:
                merged.append(b_id)
                seen.add(b_id)
            if d_id not in seen:
                merged.append(d_id)
                seen.add(d_id)
        
        results[claim_id] = merged[:TOP_COARSE]
        
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
            
    return results

def prepare_ce_training_data(df_train, coarse_train_dict, evid_dict):
    print("Preparing CrossEncoder training data (Strictly Train Only)...")
    train_samples = []
    
    for _, row in tqdm(df_train.iterrows(), total=len(df_train)):
        cid = row["claim_id"]
        claim_text = row["claim_text"]
        true_evid_ids = set(row.get("evid_ids", []))
        
        coarse_candidates = coarse_train_dict.get(cid, [])
        
        for evid_id in coarse_candidates:
            label = 1.0 if evid_id in true_evid_ids else 0.0
            evid_text = evid_dict[evid_id]
            train_samples.append(InputExample(texts=[claim_text, evid_text], label=label))
            
    return train_samples

def rerank_candidates(df, coarse_dict, ce_model, evid_dict, desc="Reranking"):
    print(f"Reranking candidates for {desc}...")
    final_results = {}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        cid = row["claim_id"]
        claim_text = row["claim_text"]
        cands = coarse_dict.get(cid, [])
        
        if not cands:
            final_results[cid] = {"claim_text": claim_text, "evidences": []}
            continue
            
        pairs = [[claim_text, evid_dict[c]] for c in cands]
        scores = ce_model.predict(pairs, show_progress_bar=False)
        
        # Sort descending
        ranked_indices = np.argsort(scores)[::-1]
        top_k_ids = [cands[i] for i in ranked_indices[:TOP_FINE]]
        
        final_results[cid] = {
            "claim_text": claim_text,
            "evidences": top_k_ids
        }
    return final_results

def main():
    data_prep.init_environment()
    DATA_DIR = "./data"
    
    df_train = data_prep.load_claims(os.path.join(DATA_DIR, "train-claims.json"), is_labelled=True)
    df_dev = data_prep.load_claims(os.path.join(DATA_DIR, "dev-claims.json"), is_labelled=True)
    df_test = data_prep.load_claims(os.path.join(DATA_DIR, "test-claims-unlabelled.json"), is_labelled=False)
    df_evid, evid_dict = data_prep.load_evidence(os.path.join(DATA_DIR, "evidence.json"))
    
    corpus_ids = df_evid["evid_id"].tolist()
    corpus_texts = df_evid["evid_text"].tolist()
    
    # 1. BM25
    bm25 = build_or_load_bm25(corpus_texts, corpus_ids)
    
    # 2. Dense (BGE) via Fast Exact GPU Search
    print("Loading BGE model...")
    dense_model = SentenceTransformer(BGE_MODEL_NAME)
    corpus_embs = build_or_load_dense_embs(corpus_texts, dense_model)
    
    # Ensure processed_data exists early for caching
    os.makedirs("./processed_data", exist_ok=True)
    
    # 3. Coarse Retrieval
    coarse_train = coarse_retrieval(df_train, bm25, corpus_embs, dense_model, corpus_ids, "Train", "./processed_data/coarse_train.json")
    coarse_dev = coarse_retrieval(df_dev, bm25, corpus_embs, dense_model, corpus_ids, "Dev", "./processed_data/coarse_dev.json")
    coarse_test = coarse_retrieval(df_test, bm25, corpus_embs, dense_model, corpus_ids, "Test", "./processed_data/coarse_test.json")
    
    # 4. CrossEncoder Training (STRICTLY ON TRAIN ONLY)
    ce_model_path = "./models/ce_trained"
    if os.path.exists(ce_model_path):
        print(f"Loading trained CrossEncoder from {ce_model_path}...")
        ce_model = CrossEncoder(ce_model_path)
    else:
        print("Initializing CrossEncoder for training...")
        ce_model = CrossEncoder(CROSS_ENCODER_MODEL_NAME, num_labels=1)
        train_samples = prepare_ce_training_data(df_train, coarse_train, evid_dict)
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=CE_BATCH_SIZE)
        
        print("Training CrossEncoder...")
        os.makedirs("./models", exist_ok=True)
        ce_model.fit(
            train_dataloader=train_dataloader,
            epochs=CE_TRAIN_EPOCHS,
            warmup_steps=100,
            output_path=ce_model_path,
            show_progress_bar=True
        )
    
    # 5. Reranking
    fine_train = rerank_candidates(df_train, coarse_train, ce_model, evid_dict, "Train")
    fine_dev = rerank_candidates(df_dev, coarse_dev, ce_model, evid_dict, "Dev")
    fine_test = rerank_candidates(df_test, coarse_test, ce_model, evid_dict, "Test")
    
    # 6. Save results
    with open("./processed_data/train_retrieved.json", "w", encoding="utf-8") as f:
        json.dump(fine_train, f, indent=2, ensure_ascii=False)
    with open("./processed_data/dev_retrieved.json", "w", encoding="utf-8") as f:
        json.dump(fine_dev, f, indent=2, ensure_ascii=False)
    with open("./processed_data/test_retrieved.json", "w", encoding="utf-8") as f:
        json.dump(fine_test, f, indent=2, ensure_ascii=False)
        
    print("Retrieval pipeline complete!")

if __name__ == "__main__":
    main()

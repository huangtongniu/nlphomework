import json
import numpy as np

def evidence_f1(pred_evidences, gold_evidences):
    pred_set = set(pred_evidences)
    gold_set = set(gold_evidences)

    if len(pred_set) == 0 or len(gold_set) == 0:
        return 0.0

    tp = len(pred_set & gold_set)

    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(gold_set)

    return 2 * precision * recall / (precision + recall)

def evaluate_threshold(truth, retrieved, threshold, max_k):
    f1_scores = []
    hit_count = 0

    for claim_id, gold_info in truth.items():
        gold_evidences = gold_info["evidences"]
        
        info = retrieved.get(claim_id, {})
        cands = info.get("evidences", [])
        scores = info.get("scores", [])
        
        # Apply dynamic thresholding
        filtered_cands = []
        for cand, score in zip(cands, scores):
            if score >= threshold or len(filtered_cands) == 0:
                filtered_cands.append(cand)
            if len(filtered_cands) >= max_k:
                break
                
        # If no candidates at all
        if not filtered_cands and cands:
            filtered_cands = cands[:1]
            
        pred_evidences = filtered_cands
        f1_scores.append(evidence_f1(pred_evidences, gold_evidences))

        if set(pred_evidences) & set(gold_evidences):
            hit_count += 1

    avg_f1 = sum(f1_scores) / len(f1_scores)
    hit_rate = hit_count / len(truth)
    return avg_f1, hit_rate

def main():
    truth_path = "data/dev-claims.json"
    retrieved_path = "processed_data/dev_retrieved.json"

    with open(truth_path, "r", encoding="utf-8") as f:
        truth = json.load(f)

    with open(retrieved_path, "r", encoding="utf-8") as f:
        retrieved = json.load(f)

    print(f"{'Threshold':<10} | {'Max_K':<6} | {'F-score':<8} | {'Hit Rate':<8}")
    print("-" * 45)
    
    # Test baseline (Top-3 and Top-5 without threshold)
    f1, hr = evaluate_threshold(truth, retrieved, -999.0, 3)
    print(f"{'None':<10} | {3:<6} | {f1:.4f}   | {hr:.4f}   (Baseline Top-3)")
    f1, hr = evaluate_threshold(truth, retrieved, -999.0, 5)
    print(f"{'None':<10} | {5:<6} | {f1:.4f}   | {hr:.4f}   (Baseline Top-5)")
    
    print("-" * 45)
    # Grid search
    best_f1 = 0
    best_params = None
    
    for max_k in [3, 4, 5, 10]:
        for threshold in np.arange(-5.0, 5.0, 0.5):
            f1, hr = evaluate_threshold(truth, retrieved, threshold, max_k)
            if f1 > best_f1:
                best_f1 = f1
                best_params = (threshold, max_k, hr)
            # Only print promising ones
            if f1 > 0.25:
                print(f"{threshold:<10.1f} | {max_k:<6} | {f1:.4f}   | {hr:.4f}")
                
    print("-" * 45)
    print(f"Optimal Configuration: Threshold = {best_params[0]:.1f}, Max_K = {best_params[1]}")
    print(f"Maximum F-score: {best_f1:.4f} (Hit Rate: {best_params[2]:.4f})")

if __name__ == "__main__":
    main()

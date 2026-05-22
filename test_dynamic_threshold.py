import json
import numpy as np
from collections import Counter
from retrieval_pipeline import select_evidences

MAX_K_GRID = [3, 4, 5, 8, 10]
THRESHOLD_GRID = np.linspace(0.0, 1.0, 101)
MIN_DYNAMIC_GAIN = 0.001

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

def get_candidates(info):
    cands = info.get("candidate_evidences", info.get("evidences", []))
    scores = info.get("candidate_scores", info.get("scores", []))
    return cands, scores

def validate_retrieved(retrieved):
    for claim_id, info in retrieved.items():
        cands, scores = get_candidates(info)
        if len(cands) != len(scores):
            raise ValueError(
                f"{claim_id} has {len(cands)} candidates but {len(scores)} scores. "
                "Rerun retrieval_pipeline.py to regenerate sigmoid candidate scores."
            )

        final_evidences = info.get("evidences", [])
        final_scores = info.get("scores", [])
        if final_scores and len(final_evidences) != len(final_scores):
            raise ValueError(
                f"{claim_id} has {len(final_evidences)} final evidences "
                f"but {len(final_scores)} final scores."
            )

        for score in scores + final_scores:
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(f"{claim_id} has score outside [0, 1]: {score}")

def evaluate_policy(truth, retrieved, mode, max_k, threshold=0.0, gap=0.0, fixed_k=4):
    f1_scores = []
    hit_count = 0
    selected_counts = []

    for claim_id, gold_info in truth.items():
        gold_evidences = gold_info["evidences"]
        
        info = retrieved.get(claim_id, {})
        cands, scores = get_candidates(info)
        pred_evidences, _ = select_evidences(
            cands,
            scores,
            mode=mode,
            fixed_k=fixed_k,
            max_k=max_k,
            threshold=threshold,
            gap=gap,
        )
        selected_counts.append(len(pred_evidences))
        f1_scores.append(evidence_f1(pred_evidences, gold_evidences))

        if set(pred_evidences) & set(gold_evidences):
            hit_count += 1

    return {
        "mode": mode,
        "max_k": max_k,
        "threshold": threshold,
        "gap": gap,
        "fixed_k": fixed_k,
        "f1": sum(f1_scores) / len(f1_scores),
        "hit_rate": hit_count / len(truth),
        "avg_count": sum(selected_counts) / len(selected_counts),
        "count_dist": dict(sorted(Counter(selected_counts).items())),
    }

def policy_label(result):
    if result["mode"] == "fixed":
        return f"Fixed Top{result['fixed_k']}"
    if result["mode"] == "absolute":
        return f"Absolute >= {result['threshold']:.2f}, max_k={result['max_k']}"
    return f"Relative gap <= {result['gap']:.2f}, max_k={result['max_k']}"

def print_result(result):
    print(
        f"{policy_label(result):<34} | "
        f"{result['f1']:.4f} | {result['hit_rate']:.4f} | "
        f"{result['avg_count']:.2f} | {result['count_dist']}"
    )

def main():
    truth_path = "data/dev-claims.json"
    retrieved_path = "processed_data/dev_retrieved.json"

    with open(truth_path, "r", encoding="utf-8") as f:
        truth = json.load(f)

    with open(retrieved_path, "r", encoding="utf-8") as f:
        retrieved = json.load(f)

    validate_retrieved(retrieved)

    print(f"{'Policy':<34} | {'F-score':<7} | {'Hit':<6} | {'Avg K':<5} | Evidence count distribution")
    print("-" * 110)

    fixed_results = [
        evaluate_policy(truth, retrieved, mode="fixed", max_k=k, fixed_k=k)
        for k in range(1, 6)
    ]
    for result in fixed_results:
        print_result(result)

    absolute_results = []
    relative_results = []
    for max_k in MAX_K_GRID:
        for value in THRESHOLD_GRID:
            absolute_results.append(
                evaluate_policy(
                    truth,
                    retrieved,
                    mode="absolute",
                    max_k=max_k,
                    threshold=float(value),
                )
            )
            relative_results.append(
                evaluate_policy(
                    truth,
                    retrieved,
                    mode="relative_gap",
                    max_k=max_k,
                    gap=float(value),
                )
            )

    best_absolute = max(absolute_results, key=lambda result: result["f1"])
    best_relative = max(relative_results, key=lambda result: result["f1"])
    best_dynamic = max([best_absolute, best_relative], key=lambda result: result["f1"])
    fixed_top4 = fixed_results[3]

    print("-" * 110)
    print_result(best_absolute)
    print_result(best_relative)
    print("-" * 110)
    gain = best_dynamic["f1"] - fixed_top4["f1"]
    if gain > MIN_DYNAMIC_GAIN:
        print(f"Recommended dynamic policy: {policy_label(best_dynamic)}")
        print(f"Dev F-score gain over Fixed Top4: {gain:.4f}")
    else:
        print("Recommended final policy: Fixed Top4")
        print(f"Best dynamic gain over Fixed Top4 ({gain:.4f}) does not exceed {MIN_DYNAMIC_GAIN:.4f}.")

if __name__ == "__main__":
    main()

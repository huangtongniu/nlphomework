import json

truth_path = "data/dev-claims.json"
retrieved_path = "processed_data/dev_retrieved.json"

with open(truth_path, "r", encoding="utf-8") as f:
    truth = json.load(f)

with open(retrieved_path, "r", encoding="utf-8") as f:
    retrieved = json.load(f)


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


for k in range(1, 6):
    f1_scores = []
    hit_count = 0

    for claim_id, gold_info in truth.items():
        gold_evidences = gold_info["evidences"]
        pred_evidences = retrieved[claim_id]["evidences"][:k]

        f1_scores.append(evidence_f1(pred_evidences, gold_evidences))

        if set(pred_evidences) & set(gold_evidences):
            hit_count += 1

    avg_f1 = sum(f1_scores) / len(f1_scores)
    hit_rate = hit_count / len(truth)

    print(f"Top-{k}: F-score = {avg_f1:.4f}, Hit Rate = {hit_rate:.4f}")
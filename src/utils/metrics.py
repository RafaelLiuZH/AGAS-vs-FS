def calculate_precision_recall_f1(flagged_ids, dataset):
    tp, fp, fn = 0, 0, 0
    bot_review_ids = set([r["review_id"] for r in dataset if r.get("is_bot", False)])

    for fid in flagged_ids:
        if fid in bot_review_ids:
            tp += 1
        else:
            fp += 1

    for bid in bot_review_ids:
        if bid not in flagged_ids:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

# evaluation/metrics.py
import os
import pandas as pd

def exact_match(pred, gt):
    return pred["start"] == gt["start"] and pred["end"] == gt["end"]

def spans_overlap(pred, gt):
    return pred["start"] < gt["end"] and pred["end"] > gt["start"]


def span_iou(pred, gt):
    intersection_start = max(pred["start"], gt["start"])
    intersection_end = min(pred["end"], gt["end"])

    if intersection_end <= intersection_start:
        return 0.0

    intersection = intersection_end - intersection_start
    union = (pred["end"] - pred["start"]) + (gt["end"] - gt["start"]) - intersection

    return intersection / union

def token_jaccard(pred, gt):
    """
    Word-level Jaccard similarity between the text of two spans.
    Requires each entity dict to have a 'term' key.
    """
    pred_tokens = set(pred["term"].lower().split())
    gt_tokens   = set(gt["term"].lower().split())

    if not pred_tokens and not gt_tokens:
        return 1.0

    intersection = pred_tokens & gt_tokens
    union        = pred_tokens | gt_tokens

    return len(intersection) / len(union)


def combined_score(pred, gt, alpha=0.5):
    """
    Weighted combination of span IoU and token Jaccard.
    alpha controls the weight of span_iou vs token_jaccard.
    """
    iou     = span_iou(pred, gt)
    jaccard = token_jaccard(pred, gt)
    return alpha * iou + (1 - alpha) * jaccard


def match_entities(predicted, ground_truth, iou_threshold=0.5, alpha=0.5):
    matched_gt = set()

    prediction_results = []

    for i, pred in enumerate(predicted):
        best_match = None
        best_iou = 0.0

        for j, gt in enumerate(ground_truth):

            if j in matched_gt:
                continue

            if not spans_overlap(pred, gt):
                continue

            iou = combined_score(pred, gt, alpha)

            if iou > best_iou:
                best_iou = iou
                best_match = j

        matched = best_iou >= iou_threshold

        if matched:
            matched_gt.add(best_match)

        prediction_results.append({
            "pred_idx": i,
            "gt_idx": best_match,
            "iou": best_iou,
            "matched": matched
        })

    tp = sum(r["matched"] for r in prediction_results)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp

    return tp, fp, fn, prediction_results

def match_entity(pred_entity, ground_truth, iou_threshold=0.5, alpha=0.5):

    best_match = None
    best_iou = 0.0

    for gt in ground_truth:

        # Skip if spans do not overlap
        if not spans_overlap(pred_entity, gt):
            continue

        iou = combined_score(pred_entity, gt, alpha)

        if iou > best_iou:
            best_iou = iou
            best_match = gt

    if best_iou >= iou_threshold:
        return {
            "matched": True,
            "ground_truth": best_match,
            "iou": best_iou
        }

    return {
        "matched": False,
        "ground_truth": None,
        "iou": best_iou
    }

def compute_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    f1 = 0 if (precision + recall == 0) else 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def debug_matches(predicted, ground_truth, prediction_results):

    matched_pred = set()
    matched_gt = set()

    print("\n=== TRUE POSITIVES (MATCHES) ===")

    for r in prediction_results:

        if not r["matched"]:
            continue

        pred_idx = r["pred_idx"]
        gt_idx = r["gt_idx"]
        iou = r["iou"]

        pred = predicted[pred_idx]
        gt = ground_truth[gt_idx]

        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)

        print(
            f"PRED: {pred['term']} [{pred['start']},{pred['end']}]",
            f"GT: {gt['term']} [{gt['start']},{gt['end']}]",
            f"IoU: {iou:.3f}"
        )

    print("-" * 40)

    print("\n=== FALSE POSITIVES (UNMATCHED PREDICTIONS) ===")

    for r in prediction_results:

        if r["matched"]:
            continue

        pred = predicted[r["pred_idx"]]

        print(
            f"PRED: {pred['term']} [{pred['start']},{pred['end']}]",
            f"Best IoU: {r['iou']:.3f}"
        )

    print("-" * 40)

    print("\n=== FALSE NEGATIVES (MISSED GROUND TRUTH) ===")

    for j, gt in enumerate(ground_truth):

        if j not in matched_gt:

            print(
                f"GT: {gt['term']} [{gt['start']},{gt['end']}]"
            )

    print("-" * 40)


def save_metrics_csv(
    model_name,
    threshold,
    tp,
    fp,
    fn,
    precision,
    recall,
    f1,
    results_dir="results"
):
    os.makedirs(results_dir, exist_ok=True)

    filename = f"metrics_{model_name}_{threshold:.2f}.csv"
    path = os.path.join(results_dir, filename)

    df = pd.DataFrame([{
        "model": model_name,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }])

    df.to_csv(path, index=False)

    return path
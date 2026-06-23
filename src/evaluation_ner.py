import pickle
import pandas as pd

from evaluation.metrics import match_entities, compute_metrics, debug_matches

ANNOTATIONS_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"

NER_MODEL = "en_core_sci_sm"
NOTE_KEYWORDS_CACHE_PATH = f"cache/note_keywords_{NER_MODEL}.pkl"

IOU_THRESHOLD = 0.05    


def compute_overall_metrics(note_keywords, gt, iou_threshold=IOU_THRESHOLD):
    """Aggregate TP/FP/FN across all notes and compute precision/recall/F1."""
    all_tp, all_fp, all_fn = 0, 0, 0
    all_ious = []

    for note_id, predicted in note_keywords.items():
        ground_truth = gt.get(note_id, [])
        tp, fp, fn, prediction_results = match_entities(
            predicted, ground_truth, iou_threshold, alpha=1
        )
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_ious.extend([r["iou"] for r in prediction_results])

    precision, recall, f1 = compute_metrics(all_tp, all_fp, all_fn)

    return {
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ious": all_ious,
    }

def print_overall_metrics(metrics):
    all_ious = metrics["ious"]
    print(min(all_ious), max(all_ious))
    print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")


def print_per_note_debug(note_keywords, gt, iou_threshold=IOU_THRESHOLD):
    """Print a detailed match breakdown for every note."""
    for note_id, predicted in note_keywords.items():
        ground_truth = gt.get(note_id, [])
        tp, fp, fn, prediction_results = match_entities(predicted, ground_truth, iou_threshold)

        print(f"\n{'=' * 60}")
        print(f"NOTE: {note_id}")
        print(f"{'=' * 60}")
        debug_matches(predicted, ground_truth, prediction_results)


def main():
    print("Loading ground truth annotations...")
    annotations_df = pd.read_csv(ANNOTATIONS_PATH, dtype={"concept_id": str})
    print(f"Loaded {len(annotations_df)} ground truth annotations.")

    gt = {}
    for note_id, group in annotations_df.groupby("note_id"):
        gt[note_id] = [
            {
                "term": row.span.lower().strip(),
                "start": row.start,
                "end": row.end,
                "concept_id": row.concept_id,
            }
            for row in group.itertuples()
        ]
    
    print(f"Loading cached NER predictions from: {NOTE_KEYWORDS_CACHE_PATH}")
    with open(NOTE_KEYWORDS_CACHE_PATH, "rb") as f:
        note_keywords = pickle.load(f)

    metrics = compute_overall_metrics(note_keywords, gt)
    print_overall_metrics(metrics)

    print_per_note_debug(note_keywords, gt)

    return metrics


if __name__ == "__main__":
    main()
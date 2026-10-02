import json
import argparse

import pandas as pd

from ontology.snomed_loader import SNOMEDHierarchy

# --- adjust these to match your main script ---
SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"

RESULTS_JSON = "llm_classification_en_core_sci_sm_level_one_shot_attempt2.json"  
OUTPUT_CSV = "analysis_results_one_shot_attempt2.csv"
OUTPUT_SUMMARY = "analysis_summary_one_shot_attempt2.json"


def compute_lca_and_distance(snomed, id_a, id_b):

    if id_a == id_b:
        return id_a, 0, 0, 0

    result = snomed.get_distance_to_lca(id_a, id_b)
    lca = result["lca"]
    if not lca:
        return None, None, None, None

    return lca, result["distance_1_to_lca"], result["distance_2_to_lca"], result["total_distance"]


# ---------------------------------------------------------------------------
# Relationship categorization (uses get_all_ancestors / get_all_descendants,
# which your codebase already relies on elsewhere)
# ---------------------------------------------------------------------------
def categorize_relationship(snomed, pred_id, gt_id):
    if pred_id is None:
        return "null_prediction"
    if pred_id == gt_id:
        return "exact_match"

    gt_ancestors = snomed.get_all_ancestors(gt_id)
    if pred_id in gt_ancestors:
        return "too_broad"          # predicted is an ancestor of the true concept

    gt_descendants = snomed.get_all_descendants(gt_id)
    if pred_id in gt_descendants:
        return "too_specific"       # predicted is a descendant of the true concept

    return "unrelated"

# ---------------------------------------------------------------------------
# Candidate-based failure diagnosis
# ---------------------------------------------------------------------------
def diagnose_candidate_failure(r):
    """
    Distinguishes retrieval failures (ground truth never made it into the
    candidate set) from LLM choice failures (ground truth was offered but
    the LLM picked something else, or failed to produce a valid choice).
    Requires "candidate_ids" on the result (added by the classifier) — older
    result files without it are reported as unknown.
    """
    gt_id = r.get("gt_id")
    pred_id = r.get("concept_id")
    candidate_ids = r.get("candidate_ids")
 
    if gt_id is None:
        return "no_ground_truth"
    if pred_id == gt_id:
        return "exact_match"
    if candidate_ids is None:
        return "unknown_no_candidate_data"
    if gt_id not in candidate_ids:
        return "gt_missing_from_candidates"
    if pred_id is None:
        return "llm_failed_to_choose"
    return "llm_wrong_choice"

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def analyze(results, snomed):
    rows = []
 
    for r in results:
        pred_id = r.get("concept_id")
        gt_id = r.get("gt_id")
        candidate_ids = r.get("candidate_ids")
 
        row = {
            "note_id": r.get("note_id"),
            "term": r.get("term"),
            "start": r.get("start"),
            "end": r.get("end"),
            "predicted_category": r.get("category"),
            "predicted_concept_id": pred_id,
            "gt_concept_id": gt_id,
            "is_null": pred_id is None,
            "num_candidates": len(candidate_ids) if candidate_ids is not None else None,
            "candidate_diagnosis": diagnose_candidate_failure(r),
        }
 
        if gt_id is None:
            row.update({
                "relationship": "no_ground_truth",
                "lca_id": None, "dist_pred_to_lca": None,
                "dist_gt_to_lca": None, "total_distance": None,
            })
        elif pred_id is None:
            row.update({
                "relationship": "null_prediction",
                "lca_id": None, "dist_pred_to_lca": None,
                "dist_gt_to_lca": None, "total_distance": None,
            })
        else:
            relationship = categorize_relationship(snomed, pred_id, gt_id)
            lca, d_pred, d_gt, total = compute_lca_and_distance(snomed, pred_id, gt_id)
 
            row.update({
                "relationship": relationship,
                "lca_id": lca,
                "dist_pred_to_lca": d_pred,
                "dist_gt_to_lca": d_gt,
                "total_distance": total,
            })
 
        rows.append(row)
 
    return pd.DataFrame(rows)

def summarize(df: pd.DataFrame) -> dict:
    total = len(df)
    n_null = int(df["is_null"].sum())
    n_with_gt = int((df["relationship"] != "no_ground_truth").sum())
    n_exact = int((df["relationship"] == "exact_match").sum())

    summary = {
        "total_predictions": total,
        "null_predictions": n_null,
        "null_rate": (n_null / total) if total else None,
        "with_ground_truth": n_with_gt,
        "exact_matches": n_exact,
        "accuracy_on_scored": (n_exact / n_with_gt) if n_with_gt else None,
        "relationship_breakdown": df["relationship"].value_counts(dropna=False).to_dict(),
    }

    scored = df[df["total_distance"].notna()]
    if len(scored):
        summary["mean_total_distance"] = float(scored["total_distance"].mean())
        summary["median_total_distance"] = float(scored["total_distance"].median())
        summary["max_total_distance"] = int(scored["total_distance"].max())

    distance_by_relationship = {}
    for rel, group in df.groupby("relationship"):
        entry = {"count": int(len(group))}
        group_scored = group[group["total_distance"].notna()]
        if len(group_scored):
            entry.update({
                "mean_distance": float(group_scored["total_distance"].mean()),
                "median_distance": float(group_scored["total_distance"].median()),
                "min_distance": int(group_scored["total_distance"].min()),
                "max_distance": int(group_scored["total_distance"].max()),
            })
        distance_by_relationship[rel] = entry
    summary["distance_by_relationship"] = distance_by_relationship

    summary["candidate_diagnosis_breakdown"] = (
        df["candidate_diagnosis"].value_counts(dropna=False).to_dict()
    )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=RESULTS_JSON)
    parser.add_argument("--out-csv", default=OUTPUT_CSV)
    parser.add_argument("--out-summary", default=OUTPUT_SUMMARY)
    args = parser.parse_args()

    print(f"Loading classification results from: {args.results}")
    with open(args.results, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Loaded {len(results)} predictions.")

    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=SNOMED_CONCEPT_FILE,
        relationship_file=SNOMED_RELATIONSHIP_FILE,
        description_file=SNOMED_DESCRIPTION_FILE,
    )
    print("SNOMED CT hierarchy loaded.")

    df = analyze(results, snomed)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"Wrote per-prediction analysis to: {args.out_csv}")

    summary = summarize(df)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to: {args.out_summary}")

    print("\n--- Summary ---")
    for k, v in summary.items():
        if k != "relationship_breakdown":
            print(f"{k}: {v}")
    print("Relationship breakdown:")
    for k, v in summary["relationship_breakdown"].items():
        print(f"  {k}: {v}")

    print("\nDistance by relationship:")
    for k, v in summary["distance_by_relationship"].items():
        print(f"  {k}: {v}")

    print("\nCandidate diagnosis (wrong LLM choice vs. GT missing from candidates):")
    for k, v in summary["candidate_diagnosis_breakdown"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

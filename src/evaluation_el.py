import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from evaluation.metrics import token_jaccard
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.classification_metrics import (
    build_analysis_dataframe,
    plot_categorical_distribution,
    plot_correlation,
    plot_distribution,
)
import pickle

SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"

ANNOTATIONS_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"

NER_MODEL = "en_core_sci_sm"
TARGET_LEVEL = "hierarchy_pruning_attempt1"
CLASSIFICATION_RESULTS_PATH = f"llm_classification_{NER_MODEL}_level_{TARGET_LEVEL}.json"
CACHE_PATH = f"cache/note_keywords_{NER_MODEL}.pkl"
LEVEL_1_TARGET_NAMES = ["Body structure", "Clinical finding", "Procedure"]    
    
def row_distance(row):
    pred_label = row["predicted_label"]
    gt_label = row["gt_label"]

    if pred_label is None or gt_label is None:
        return float("nan")
    if pd.isna(pred_label) or pd.isna(gt_label):
        return float("nan")

    pred = {"term": pred_label}
    gt = {"term": gt_label}


    return 1 - token_jaccard(pred, gt)
   

def plot_context_length_by_level(analysis_df, failures_only=False):
    df = analysis_df[~analysis_df["gt_reached"]] if failures_only else analysis_df
    levels = sorted(df["level_reached"].dropna().unique())
    data = [df.loc[df["level_reached"] == level, "context_length"].dropna() for level in levels]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=levels)
    plt.xlabel("Level Reached")
    plt.ylabel("Context Length (characters)")
    plt.title("Context Length by Failure Level" if failures_only else "Context Length by Level Reached")
    plt.show()
    
def main():
    
    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=SNOMED_CONCEPT_FILE,
        relationship_file=SNOMED_RELATIONSHIP_FILE,
        description_file=SNOMED_DESCRIPTION_FILE,
    )
    print("SNOMED CT hierarchy loaded.")

    print(f"Loading classification results from: {CLASSIFICATION_RESULTS_PATH}")
    with open(CLASSIFICATION_RESULTS_PATH, "r", encoding="utf-8") as f:
        classif_results = json.load(f)

    print("Loading ground truth annotations...")
    annotations_df = pd.read_csv(ANNOTATIONS_PATH, dtype={"concept_id": str})
    print(f"Loaded {len(annotations_df)} ground truth annotations.")

    print("Loading entity extraction results...")
    print(f"Loading cached keywords from: {CACHE_PATH}")
    
    with open(CACHE_PATH, "rb") as f:
        note_keywords = pickle.load(f)

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


    analysis_df = build_analysis_dataframe(classif_results, snomed, gt, note_keywords)

    print("Classification results analysis:")
    print(analysis_df.describe(include="all").transpose())

    print("Total predictions:", len(analysis_df))
    print("Successful predictions:", analysis_df["gt_reached"].sum())
    print("Failed predictions:", len(analysis_df) - analysis_df["gt_reached"].sum())
    print("Accuracy:", analysis_df["gt_reached"].mean())
    print("Null predictions:", analysis_df["null_prediction"].sum())

    print("Distribution of levels reached:")
    print(analysis_df["level_reached"].describe())

    print("Distribution of level of failure:")
    print(analysis_df[~analysis_df["gt_reached"]]["level_reached"].describe())

    print("Distribution of distance to GT:")
    print(analysis_df[~analysis_df["gt_reached"]]["distance_to_gt"].describe())

    print("Distribution of possible paths failed:")
    print(analysis_df[~analysis_df["gt_reached"]]["num_remaining_paths"].describe())

    print("Correct Path Covered")
    print(analysis_df["correct_prefix_ratio"].describe())

    print("Number of predictions failed at level 1:", (analysis_df["level_reached"] == 1).sum())

    print("Number of times stopped at a parent node:", analysis_df["stopped_at_parent"].sum())
    # Distribution / correlation plots
    # plot_categorical_distribution(analysis_df[~analysis_df["gt_reached"]], "level_reached")  # Failure level distribution
    # plot_correlation(analysis_df, "level_reached", "distance_to_gt")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "distance_to_gt", "correct_prefix_ratio")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "level_reached", "correct_prefix_ratio")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "num_remaining_paths", "correct_prefix_ratio")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "candidate_count", "correct_prefix_ratio")

    # plot_distribution(analysis_df, "correct_prefix_ratio")

    # Bubble plot of hierarchy traversal performance (uncomment to use)
    # grouped = (
    #     analysis_df.groupby(["candidate_count", "level_reached"])
    #     .agg({"correct_prefix_ratio": "mean", "gt_reached": "count"})
    #     .reset_index()
    # )
    # plt.figure(figsize=(12, 7))
    # scatter = plt.scatter(
    #     grouped["level_reached"],
    #     grouped["candidate_count"],
    #     s=grouped["gt_reached"] * 5,
    #     c=grouped["correct_prefix_ratio"],
    #     cmap="viridis",
    #     alpha=0.7,
    # )
    # cbar = plt.colorbar(scatter)
    # cbar.set_label("Mean Path Coverage")
    # plt.xlabel("Level Reached")
    # plt.ylabel("Candidate Count")
    # plt.title("Hierarchy Traversal Performance\nBubble size = Number of Samples")
    # plt.show()

    # plot_level_1_confusion_matrix(analysis_df, snomed)

    failures_level_1 = analysis_df[analysis_df["level_reached"] == 1]

    # non_null = analysis_df[~analysis_df["null_prediction"]]
    # y_true = [snomed.get_ancestor_at_level(gt_id, 1) for gt_id in non_null["gt_id"]]
    # y_pred = [snomed.get_ancestor_at_level(pred_id, 1) for pred_id in non_null["predicted_id"]]

    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LEVEL_1_TARGET_NAMES)
    # disp.plot(cmap=plt.cm.Blues)
    # plt.title("Confusion Matrix at Level 1")
    # plt.show()

    # analysis_df = analysis_df.copy()
    # analysis_df["jaccard_distance"] = analysis_df.apply(row_distance, axis=1)

    # df = analysis_df
    # df = df[~df["gt_reached"]]
 
    # levels = sorted(df["level_reached"].dropna().unique())
    # data = [df.loc[df["level_reached"] == level, "jaccard_distance"].dropna() for level in levels]
 
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(data, labels=levels)
    # plt.xlabel("Level Reached")
    # plt.ylabel("Jaccard Distance (predicted vs. ground truth label)")
    # title =  "Jaccard Distance by Level Reached"
    # plt.title(title)
    # plt.show()

    # summary = analysis_df.groupby("distance_to_gt").agg(
    #     n=("exact_term_match", "size"),
    #     exact_match_rate=("exact_term_match", "mean"),
    #     mean_term_jaccard=("term_jaccard", "mean"),
    # )
    # print("\nNER term accuracy by classification level reached:")
    # print(summary)

    # df = analysis_df
    # df = df[~df["gt_reached"]]
 
    # levels = sorted(df["distance_to_gt"].dropna().unique())
    # data = [df.loc[df["distance_to_gt"] == level, "term_jaccard"].dropna() for level in levels]
 
    # plt.figure(figsize=(10, 6))
    # plt.boxplot(data, labels=levels)
    # plt.xlabel("Distance to Ground Truth")
    # plt.ylabel("NER Term Jaccard Similarity (predicted vs. ground truth text)")
    # title = "NER Term Match by Failure Level"
    # plt.title(title)
    # plt.show()

    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "correct_prefix_ratio", "term_jaccard")

    # plot_context_length_by_level(analysis_df, failures_only=True)

    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "correct_prefix_ratio", "context_length")

    return analysis_df, failures_level_1


if __name__ == "__main__":
    main()
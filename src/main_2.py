import pandas as pd
import os 
import pickle
import matplotlib.pyplot as plt
import json
from preprocessing.text_chunking import preprocess_notes
import numpy as np
from extraction.kwextractor_scispacy import SciSpaCyExtractor
from ontology.ollama_snomed_evaluation import SNOMEDClassifier
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.metrics import match_entities, compute_metrics, debug_matches, save_metrics_csv, match_entity
from evaluation.visualization import plot_confusion_matrix, plot_iou_distribution, plot_precision_recall_curve, visualize_spans
from collections import defaultdict
from extraction.utils import merge_adjacent_entities
from evaluation.scoring import macro_character_iou, support_weighted_character_iou, iou_per_class
import polars as pl
import time
from evaluation.classification_metrics import (
    build_analysis_dataframe,
    plot_categorical_distribution,
    plot_correlation,
    plot_distribution
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

COLUMNS = ["note_id", "start", "end", "concept_id"]
DTYPES = {
    "note_id": pl.String,
    "start": pl.Float64,
    "end": pl.Float64,
    "concept_id": pl.String,
}


def lift_annotations_to_level(
    annotations: pl.DataFrame,
    snomed,
    level: int,
    concept_column: str = "concept_id",
):
    """
    Replace concept IDs with ancestor IDs at a specified hierarchy level.
    """

    def lift(concept_id):
        return snomed.get_ancestor_at_level(concept_id, level)

    return annotations.with_columns(
        pl.col(concept_column)
        .map_elements(lift, return_dtype=pl.String)
        .alias(concept_column)
    )


if __name__ == "__main__":
    
    # LOADING DATA
    print("Loading data...")
    training_path = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_notes.csv" # CHANGE TO ALL DATA?
    notes_df = pd.read_csv(training_path)
    print(f"Loaded {len(notes_df)} notes.")

    print("Loading ground truth annotations...")
    annotations_path = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"
    annotations_df = pd.read_csv(annotations_path, dtype={"concept_id": str})
    print(f"Loaded {len(annotations_df)} ground truth annotations.")

    # LOAD SNOMED HIERARCHY
    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt",
        relationship_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt",
        description_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"
    )
    print("SNOMED CT hierarchy loaded.")

    # PREPROCESSING
    processed_df = preprocess_notes(notes_df)

    gt = {}

    for note_id, group in annotations_df.groupby("note_id"):
        gt[note_id] = [
            {
                "term": row.span.lower().strip(),
                "start": row.start,
                "end": row.end,
                "concept_id": row.concept_id
            }
            for row in group.itertuples()
        ]

    # EXTRACTION

    model="en_core_sci_sm"
    CACHE_PATH = f"cache/note_keywords_{model}.pkl"

    if os.path.exists(CACHE_PATH):

        print(f"Loading cached keywords from: {CACHE_PATH}")

        with open(CACHE_PATH, "rb") as f:
            note_keywords = pickle.load(f)
    
    else:

        model="en_core_sci_sm"
        note_keywords = defaultdict(list)

        extractor = SciSpaCyExtractor(model=model)
        OFFSET_FIX = 2
        CONTEXT_WINDOW = 50

        for row in processed_df.itertuples(index=False):
            note_id = row.note_id
            text = row.text
            chunk_start = row.chunk_start
            chunk_end = row.chunk_end
            
            keywords = extractor.extract(text)


            for kw in keywords:
                note_keywords[note_id].append({
                    "term": kw["text"].lower().strip(),
                    "start": kw["start"] + chunk_start + OFFSET_FIX,
                    "end": kw["end"] + chunk_start + OFFSET_FIX,
                    "label": kw["label"],
                    "context": text[max(0, kw["start"]-CONTEXT_WINDOW):kw["end"]+CONTEXT_WINDOW]
                })
        
        note_keywords = dict(note_keywords)
        
        for note_id in note_keywords:
            note_keywords[note_id] = merge_adjacent_entities(note_keywords[note_id], text=processed_df[processed_df.note_id == note_id].text.iloc[0])

        os.makedirs("cache", exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(note_keywords, f)
        print(f"Saved extracted keywords to cache: {CACHE_PATH}")

    # EVALUATION-CLASSIFICATION

    TARGET_LEVEL = "medgemma27b_comparison"
    output_path = f"llm_classification_{model}_level_{TARGET_LEVEL}"

    if os.path.exists(f"{output_path}.csv"):
        print(f"Loading cached classification results from: {output_path}")


        user_annotations = pl.read_csv(
            f"{output_path}.csv", schema_overrides=DTYPES
        ).select(COLUMNS)

        with open(f"{output_path}.json", "r", encoding="utf-8") as f:
            classif_results = json.load(f)
    
    else:
        classifier = SNOMEDClassifier()

        for i, item in enumerate(note_keywords.items()):

            note_id = item[0]
            predicted = item[1]

            ground_truth = gt.get(note_id, [])

            print(f"\nProcessing note {i+1}/{len(note_keywords)}: {note_id}")

            for term_data in predicted: # CHANGE TO ALL DATA

                match = match_entity(pred_entity=term_data, ground_truth=ground_truth, iou_threshold=0.05)
                
                if match["matched"]:

                    gt_concept_id = match["ground_truth"]["concept_id"]
                    gt_ancestors = snomed.get_all_ancestors(gt_concept_id)
                    gt_ancestors.add(gt_concept_id)

                    result = classifier.classify_hierarchical(
                        term=term_data["term"],
                        context=term_data["context"],
                        snomed=snomed,
                        all_ancestors=gt_ancestors,
                        gt_id=gt_concept_id,
                        parent_id=None,
                        start=term_data["start"],
                        end=term_data["end"],
                        note_id=note_id
                    )

            classifier.save_results(f"{output_path}.json")
            classifier.save_results_csv(f"{output_path}.csv")       

        print(f"\nSaved LLM classification results to: {output_path}")

        with open(f"{output_path}.json", "r", encoding="utf-8") as f:
            classif_results = json.load(f)

    analysis_df = build_analysis_dataframe(
        classif_results,
        snomed
    )
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

    # plot_categorical_distribution(analysis_df[~analysis_df["gt_reached"]], "level_reached") # Failure level distribution
    # plot_correlation(analysis_df, "level_reached", "distance_to_gt")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "distance_to_gt", "correct_prefix_ratio")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "level_reached", "correct_prefix_ratio")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "num_remaining_paths", "correct_prefix_ratio")
    # plot_correlation(analysis_df[~analysis_df["gt_reached"]], "candidate_count", "correct_prefix_ratio")

    plot_distribution(analysis_df, "correct_prefix_ratio")

    # grouped = (
    # analysis_df.groupby(["candidate_count", "level_reached"])
    # .agg({
    #     "correct_prefix_ratio": "mean",
    #     "gt_reached": "count"
    # })
    # .reset_index()
    # )
    
    # plt.figure(figsize=(12, 7))

    # scatter = plt.scatter(
    #     grouped["level_reached"],
    #     grouped["candidate_count"],
    #     s=grouped["gt_reached"] * 5,
    #     c=grouped["correct_prefix_ratio"],
    #     cmap="viridis",
    #     alpha=0.7
    # )

    # cbar = plt.colorbar(scatter)
    # cbar.set_label("Mean Path Coverage")

    # plt.xlabel("Level Reached")
    # plt.ylabel("Candidate Count")

    # plt.title(
    #     "Hierarchy Traversal Performance\n"
    #     "Bubble size = Number of Samples"
    # )

    # plt.show()
    
    # # PLOT LEVEL 1 CONFUSION MATRIX

    # y_true = [snomed.get_ancestor_at_level(gt_id, 1) for gt_id in analysis_df[~analysis_df["null_prediction"]]["gt_id"]]
    # y_pred = [snomed.get_ancestor_at_level(pred_id, 1) for pred_id in analysis_df[~analysis_df["null_prediction"]]["predicted_id"]]
    # cm = confusion_matrix(y_true, y_pred)

    # target_names = ["Body structure", "Clinical finding", "Procedure"]
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = target_names)
    # disp.plot(cmap = plt.cm.Blues)
    # plt.title("Confusion Matrix at Level 1")
    # plt.show()

    print("Number of predictions failed at level 1:", (analysis_df["level_reached"] == 1).sum())

    failures_level_1 = analysis_df[analysis_df["level_reached"] == 1]
    
import pandas as pd
import os 
import pickle
from preprocessing.text_chunking import preprocess_notes
import numpy as np
from extraction.kwextractor_scispacy import SciSpaCyExtractor
from ontology.ollama_snomed_classifier import OllamaSNOMEDClassifier
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.metrics import match_entities, compute_metrics, debug_matches, save_metrics_csv
from evaluation.visualization import plot_confusion_matrix, plot_iou_distribution, plot_precision_recall_curve, visualize_spans
from collections import defaultdict
from extraction.utils import merge_adjacent_entities
from evaluation.scoring import macro_character_iou, support_weighted_character_iou, iou_per_class
import polars as pl
import time

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
    annotations_df = pd.read_csv(annotations_path)
    print(f"Loaded {len(annotations_df)} ground truth annotations.")

    # LOAD SNOMED HIERARCHY
    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260401.txt",
        relationship_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260401.txt",
        description_file=r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260401T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260401.txt"
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
                "end": row.end
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

    # EVALUATION

    # iou_threshold = 0.05
    # all_tp, all_fp, all_fn = 0, 0, 0
    # all_predicted, all_ground_truth, all_matches = [], [], []
    # all_ious = []

    # for note_id, predicted in note_keywords.items():
    #     ground_truth = gt.get(note_id, [])

    #     tp, fp, fn, prediction_results  = match_entities(predicted, ground_truth, iou_threshold)

    #     all_tp += tp
    #     all_fp += fp
    #     all_fn += fn
    #     all_ious.extend([r["iou"] for r in prediction_results])

    # precision, recall, f1 = compute_metrics(all_tp, all_fp, all_fn)
    # print(min(all_ious), max(all_ious))
    # print(f"TP: {all_tp}, FP: {all_fp}, FN: {all_fn}")
    # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # csv_path = save_metrics_csv(model, iou_threshold, all_tp, all_fp, all_fn, precision, recall, f1)

    # print(f"Saved metrics to: {csv_path}")

    # # Confusion matrix
    # plot_confusion_matrix(all_tp, all_fp, all_fn, model_name=model, threshold=iou_threshold)

    # plot_iou_distribution(all_ious, threshold=iou_threshold, model_name=model)

    # # Precision-Recall curve
    # thresholds = np.linspace(0, 1, 20)
    # precisions, recalls = [], []

    # for t in thresholds:
    #     tp_sum, fp_sum, fn_sum = 0, 0, 0

    #     for note_id, predicted in note_keywords.items():
    #         ground_truth = gt.get(note_id, [])
    #         tp, fp, fn, _ = match_entities(predicted, ground_truth, iou_threshold=t)

    #         tp_sum += tp
    #         fp_sum += fp
    #         fn_sum += fn

    #     p, r, _ = compute_metrics(tp_sum, fp_sum, fn_sum)
    #     precisions.append(p)
    #     recalls.append(r)

    # plot_precision_recall_curve(thresholds, precisions, recalls, selected_t=iou_threshold, model_name=model)

    # CLASSIFICATION
    TARGET_LEVEL = 3
    CACHE_PATH_2 = f"llm_classification_{model}_level_{TARGET_LEVEL}.csv"

    if os.path.exists(CACHE_PATH_2):
        print(f"Loading cached classification results from: {CACHE_PATH_2}")

        user_annotations = pl.read_csv(
            CACHE_PATH_2, schema_overrides=DTYPES
        ).select(COLUMNS)
    
    else:
        classifier = OllamaSNOMEDClassifier()

        for i, item in enumerate(note_keywords.items()):

            note_id = item[0]
            predicted = item[1]

            print(f"\nProcessing note {i+1}/{len(note_keywords)}: {note_id}")

            for term_data in predicted[0:2]: # CHANGE TO ALL DATA

                result = classifier.classify_hierarchical(
                    term=term_data["term"],
                    context=term_data["context"],
                    snomed=snomed,
                    target_level=TARGET_LEVEL,
                    parent_id=None,
                    start=term_data["start"],
                    end=term_data["end"],
                    note_id=note_id
                )    

            if i >= 0: # CHANGE TO ALL DATA
                break
        
        output_path = f"llm_classification_{model}_level_{TARGET_LEVEL}"
        classifier.save_results(f"{output_path}.json")
        classifier.save_results_csv(f"{output_path}.csv")

        print(f"\nSaved LLM classification results to: {output_path}")
        
        # EVALUATION

        user_annotations_path = f"{output_path}.csv"
        
        user_annotations = pl.read_csv(
            user_annotations_path, schema_overrides=DTYPES
        ).select(COLUMNS)

    target_annotations = pl.read_csv(
        annotations_path, schema_overrides=DTYPES
    ).select(COLUMNS)

    user_annotations = user_annotations.with_columns([
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
    ])

    target_annotations = target_annotations.with_columns([
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
    ])

    target_annotations = target_annotations.filter(
    pl.col("note_id") == target_annotations["note_id"][0]
    )  

    target_annotations = lift_annotations_to_level(target_annotations, snomed, level=TARGET_LEVEL)

    ious = iou_per_class(user_annotations, target_annotations)
    for concept, score in ious.items():
        print(f"{concept}: {score:.4f}")

    macro_iou = macro_character_iou(user_annotations, target_annotations)
    weighted_iou = support_weighted_character_iou(user_annotations, target_annotations)

    print(f"macro-averaged character IoU: {macro_iou:0.4f}")
    print(f"support-weighted character IoU: {weighted_iou:0.4f}")

        
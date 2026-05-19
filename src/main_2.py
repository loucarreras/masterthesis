import pandas as pd
import os 
import pickle
from preprocessing.text_chunking import preprocess_notes
import numpy as np
from extraction.kwextractor_scispacy import SciSpaCyExtractor
from ontology.ollama_snomed_evaluation import OllamaSNOMEDClassifier
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
    print_global_metrics,
    analyze_failure_nodes,
    plot_level_distribution,
    plot_failure_levels,
    plot_shortest_path_distance,
    plot_semantic_similarity,
    plot_candidate_failure,
    correlation_candidates_success
)

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

    TARGET_LEVEL = "all_2"
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

            ground_truth = gt.get(note_id, [])

            print(f"\nProcessing note {i+1}/{len(note_keywords)}: {note_id}")

            for term_data in predicted[0:100]: # CHANGE TO ALL DATA

                match = match_entity(pred_entity=term_data, ground_truth=ground_truth)
                
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

            if i >= 0: # CHANGE TO ALL DATA
                break
        
        output_path = f"llm_classification_{model}_level_{TARGET_LEVEL}"
        classifier.save_results(f"{output_path}.json")
        classifier.save_results_csv(f"{output_path}.csv")

        print(f"\nSaved LLM classification results to: {output_path}")


    analysis_df = build_analysis_dataframe(
        classifier.results,
        snomed
    )

    print_global_metrics(analysis_df)

    failure_nodes_df = analyze_failure_nodes(
        classifier.results
    )

    print("\nTop failure nodes:")
    print(failure_nodes_df.head(20))

    correlation_candidates_success(analysis_df)
    plot_level_distribution(analysis_df)

    plot_failure_levels(analysis_df)

    plot_shortest_path_distance(analysis_df)

    plot_semantic_similarity(analysis_df)

    plot_candidate_failure(analysis_df)

    analysis_df.to_csv(
        f"{output_path}_analysis.csv",
        index=False
    )

    failure_nodes_df.to_csv(
        f"{output_path}_failure_nodes.csv",
        index=False
    )

    print("\nSaved analysis CSV files.")
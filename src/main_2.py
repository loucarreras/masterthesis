import os
import pickle
import json
from collections import defaultdict

import pandas as pd
import polars as pl
import torch

from preprocessing.text_chunking import preprocess_notes
from extraction.kwextractor_scispacy import SciSpaCyExtractor
from extraction.utils import merge_adjacent_entities
from ontology.ollama_snomed_evaluation import SNOMEDClassifier
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.metrics import match_entity


COLUMNS = ["note_id", "start", "end", "concept_id"]
DTYPES = {
    "note_id": pl.String,
    "start": pl.Float64,
    "end": pl.Float64,
    "concept_id": pl.String,
}

TRAINING_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_notes.csv"  # CHANGE TO ALL DATA?
ANNOTATIONS_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"

SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"

NER_MODEL = "en_core_sci_sm"
OFFSET_FIX = 2
CONTEXT_WINDOW = 50
IOU_THRESHOLD = 0.05

TARGET_LEVEL = "gemma4_exploratory"
OUTPUT_PATH = f"llm_classification_{NER_MODEL}_level_{TARGET_LEVEL}"
CACHE_PATH = f"cache/note_keywords_{NER_MODEL}.pkl"


def extract_entities(processed_df, model=NER_MODEL, cache_path=CACHE_PATH):
    """
    Entity recognition step: run the NER extractor over each chunk and merge
    adjacent entities back into note-level spans.
    
    """
    if os.path.exists(cache_path):
        print(f"Loading cached keywords from: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    note_keywords = defaultdict(list)
    extractor = SciSpaCyExtractor(model=model)

    for row in processed_df.itertuples(index=False):
        note_id = row.note_id
        text = row.text
        chunk_start = row.chunk_start

        keywords = extractor.extract(text)

        for kw in keywords:
            note_keywords[note_id].append(
                {
                    "term": kw["text"].lower().strip(),
                    "start": kw["start"] + chunk_start + OFFSET_FIX,
                    "end": kw["end"] + chunk_start + OFFSET_FIX,
                    "label": kw["label"],
                    "context": text[max(0, kw["start"] - CONTEXT_WINDOW):kw["end"] + CONTEXT_WINDOW],
                }
            )

    note_keywords = dict(note_keywords)

    for note_id in note_keywords:
        note_keywords[note_id] = merge_adjacent_entities(
            note_keywords[note_id],
            text=processed_df[processed_df.note_id == note_id].text.iloc[0],
        )

    os.makedirs("cache", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(note_keywords, f)
    print(f"Saved extracted keywords to cache: {cache_path}")

    return note_keywords


def classify_entities(note_keywords, gt, snomed, output_path=OUTPUT_PATH):
    """
    Entity classification step: hierarchically classify each extracted entity
    into the SNOMED CT ontology using an LLM.

    """
    if os.path.exists(f"{output_path}.csv"):
        print(f"Loading cached classification results from: {output_path}")

        user_annotations = pl.read_csv(
            f"{output_path}.csv", schema_overrides=DTYPES
        ).select(COLUMNS)

        with open(f"{output_path}.json", "r", encoding="utf-8") as f:
            classif_results = json.load(f)

        return user_annotations, classif_results

    classifier = SNOMEDClassifier()

    for i, (note_id, predicted) in enumerate(note_keywords.items()):
        ground_truth = gt.get(note_id, [])

        print(f"\nProcessing note {i + 1}/{len(note_keywords)}: {note_id}")

        for term_data in predicted:
            match = match_entity(
                pred_entity=term_data, ground_truth=ground_truth, iou_threshold=IOU_THRESHOLD
            )

            if match["matched"]:
                gt_concept_id = match["ground_truth"]["concept_id"]
                gt_ancestors = snomed.get_all_ancestors(gt_concept_id)
                gt_ancestors.add(gt_concept_id)

                classifier.classify_hierarchical(
                    term=term_data["term"],
                    context=term_data["context"],
                    snomed=snomed,
                    all_ancestors=gt_ancestors,
                    gt_id=gt_concept_id,
                    parent_id=None,
                    start=term_data["start"],
                    end=term_data["end"],
                    note_id=note_id,
                )

        classifier.save_results(f"{output_path}.json")
        classifier.save_results_csv(f"{output_path}.csv")

    print(f"\nSaved LLM classification results to: {output_path}")

    with open(f"{output_path}.json", "r", encoding="utf-8") as f:
        classif_results = json.load(f)

    return None, classif_results


def main():
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))

    print("Loading data...")
    notes_df = pd.read_csv(TRAINING_PATH)
    print(f"Loaded {len(notes_df)} notes.")

    print("Loading ground truth annotations...")
    annotations_df = pd.read_csv(ANNOTATIONS_PATH, dtype={"concept_id": str})
    print(f"Loaded {len(annotations_df)} ground truth annotations.")

    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=SNOMED_CONCEPT_FILE,
        relationship_file=SNOMED_RELATIONSHIP_FILE,
        description_file=SNOMED_DESCRIPTION_FILE,
    )
    print("SNOMED CT hierarchy loaded.")

    processed_df = preprocess_notes(notes_df)
    
    gt = {} # Group GT annotations by note_id for fast lookup during classification.
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

    note_keywords = extract_entities(processed_df)

    _, classif_results = classify_entities(note_keywords, gt, snomed)

    print(f"\nDone. Classification results available at: {OUTPUT_PATH}.json / {OUTPUT_PATH}.csv")
    return classif_results


if __name__ == "__main__":
    main()
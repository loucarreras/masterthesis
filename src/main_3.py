import os
import pickle
import json
from collections import defaultdict
import time
import pandas as pd
import polars as pl
import torch
from pathlib import Path

from preprocessing.text_chunking import preprocess_notes
from extraction.kwextractor_scispacy import SciSpaCyExtractor
from extraction.utils import merge_adjacent_entities
from ontology.ollama_one_shot_classifier import SNOMEDClassifier
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.metrics import match_entity
from pruning.dense_index import DenseIndex, SapBERTEncoder
from pruning.lexical_index import LexicalIndex
from pruning.subtree_pruner import build_allowed_subtree, get_roots, render_subtree, refine_subtree


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
CONTEXT_MODE = "sentence"  # "full_text" | "paragraph" | "sentence"

RUN_ID = os.environ.get("RUN_ID", "0")
NOTES_LIMIT = int(os.environ.get("NOTES_LIMIT", "0"))
TARGET_LEVEL = "one_shot_gemma24_20notes"
OUTPUT_PATH = f"llm_classification_{NER_MODEL}_level_{TARGET_LEVEL}"
CACHE_PATH = f"cache/note_keywords_{NER_MODEL}_{CONTEXT_MODE}.pkl"
CACHE_DIR = Path("./cache")

# Candidate generation
BM25_TOP_N = 10
DENSE_TOP_N = 10
DESCENDANT_DEPTH = 2

LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure",
}

SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

def get_allowed_concepts(hierarchy: SNOMEDHierarchy, root_ids: dict[str, str]) -> set[str]:
    """
    Returns the set of concept_ids consisting of the given roots plus all
    their descendants, using the hierarchy's own child_map (active is-a only).
    """
    allowed = set(root_ids.keys())
    for root_id, label in root_ids.items():
        t0 = time.time()
        descendants = hierarchy.get_all_descendants(root_id)
        allowed |= descendants
        print(f"  {label} ({root_id}): {len(descendants)} descendants in {time.time() - t0:.1f}s")
    return allowed

def extract_entities(processed_df, note_full_text=None, model=NER_MODEL, cache_path=CACHE_PATH):
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
            if CONTEXT_MODE == "full_text":
                context = note_full_text[note_id]
            elif CONTEXT_MODE == "paragraph":
                context = f"{row.section}:\n{text}".strip()
            elif CONTEXT_MODE == "sentence":
                context =  text[max(0, kw["start"] - CONTEXT_WINDOW):kw["end"] + CONTEXT_WINDOW]
            else:
                raise ValueError(f"Unknown CONTEXT_MODE: {CONTEXT_MODE}")
            
            note_keywords[note_id].append(
                {
                    "term": kw["text"].lower().strip(),
                    "start": kw["start"] + chunk_start + OFFSET_FIX,
                    "end": kw["end"] + chunk_start + OFFSET_FIX,
                    "label": kw["label"],
                    "context": context,
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


def classify_entities(note_keywords, gt, snomed, lexical_index, dense_index, output_path=OUTPUT_PATH):
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
                pred_entity=term_data, ground_truth=ground_truth, iou_threshold=IOU_THRESHOLD, alpha = 1
            )

            if match["matched"]:
                mention = term_data["term"]
                gt_concept_id = match["ground_truth"]["concept_id"]
                gt_ancestors = snomed.get_all_ancestors(gt_concept_id)
                gt_ancestors.add(gt_concept_id)

                results_lexical=lexical_index.search(mention, top_n=BM25_TOP_N)
                candidate_ids_lexical = [cid for cid, _ in results_lexical]

                results_dense = dense_index.search(mention, top_n=DENSE_TOP_N)
                candidate_ids_dense = [cid for cid, _ in results_dense]

                union_ids = list(set(candidate_ids_lexical) | set(candidate_ids_dense))

                allowed, parent_map, child_map = build_allowed_subtree(snomed, union_ids, descendant_depth=DESCENDANT_DEPTH)
                allowed, parent_map, child_map = refine_subtree(allowed, parent_map, child_map, set(union_ids))
                roots = get_roots(allowed, parent_map)
                
                classifier.classify(
                    term=mention,
                    context=term_data["context"],
                    snomed=snomed,
                    candidates=union_ids,
                    gt_id=gt_concept_id,
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
    if NOTES_LIMIT > 0:
        notes_df = notes_df.head(NOTES_LIMIT).copy()
        print(f"[run {RUN_ID}] Limiting to first {NOTES_LIMIT} notes: {list(notes_df.note_id)}")

    allowed_note_ids = set(notes_df.note_id)
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

    note_full_text = dict(zip(notes_df.note_id, notes_df.text))
    note_keywords = extract_entities(processed_df, note_full_text = note_full_text)


    print("Restricting to allowed top-level branches...")
    t0 = time.time()
    allowed_concepts = get_allowed_concepts(snomed, LEVEL1)
    print(f"Total allowed concepts: {len(allowed_concepts)} in {time.time() - t0:.1f}s")
    
    print("Building synonym map...")
    t0 = time.time()
    synonyms = snomed.build_full_synonym_map()
    before = len(synonyms)
    synonyms = {cid: terms for cid, terms in synonyms.items() if cid in allowed_concepts}
    print(f"Built synonym map for {before} concepts, restricted to {len(synonyms)} "
        f"in {time.time() - t0:.1f}s")

    print("Building BM25 index...")
    t0 = time.time()
    lexical_index = LexicalIndex(synonyms)
    print(f"Indexed {len(lexical_index.doc_terms)} description strings in {time.time() - t0:.1f}s")

    print("Building SapBERT encoder + dense index (embeds/caches every synonym string)...")
    t0 = time.time()
    encoder = SapBERTEncoder(model_name=SAPBERT_MODEL_NAME)
    dense_index = DenseIndex(synonyms, encoder=encoder, cache_dir=CACHE_DIR)
    print(f"Built dense index over {len(dense_index.doc_terms)} synonym strings "
                  f"in {time.time() - t0:.1f}s")

    
    if NOTES_LIMIT > 0:
        before = len(note_keywords)
        note_keywords = {nid: v for nid, v in note_keywords.items() if nid in allowed_note_ids}
        print(f"[run {RUN_ID}] note_keywords filtered to allowed notes: {len(note_keywords)} of {before} kept")
        assert set(note_keywords.keys()) <= allowed_note_ids
        assert len(note_keywords) <= NOTES_LIMIT
    
    _, classif_results = classify_entities(note_keywords, gt, snomed, lexical_index, dense_index)
    
    # --- run it ---
    print(f"\nDone. Classification results available at: {OUTPUT_PATH}.json / {OUTPUT_PATH}.csv")
    return classif_results


if __name__ == "__main__":
    main()
import os
import pickle
import json
from collections import defaultdict
import time
import pandas as pd
import polars as pl
import torch

from preprocessing.text_chunking import preprocess_notes
from extraction.kwextractor_scispacy import SciSpaCyExtractor
from extraction.utils import merge_adjacent_entities
from ontology.ollama_snomed_evaluation import SNOMEDClassifier
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.metrics import match_entity

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

TARGET_LEVEL = "hierarchy_pruning_attempt2"
OUTPUT_PATH = f"llm_classification_{NER_MODEL}_level_{TARGET_LEVEL}"
CACHE_PATH = f"cache/note_keywords_{NER_MODEL}_{CONTEXT_MODE}.pkl"

# Candidate generation
BM25_TOP_N = 5
DESCENDANT_DEPTH = 2

LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure",
}

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
                context =  text[max(0, kw["start"] - CONTEXT_WINDOW):kw["end"] + CONTEXT_WINDOW],
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


def classify_entities(note_keywords, gt, snomed, lexical_index, output_path=OUTPUT_PATH):
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

                results=lexical_index.search(mention, top_n=BM25_TOP_N)
                candidate_ids = [cid for cid, _ in results]

                SNOMED_ROOT = "138875005"
                allowed, parent_map, child_map = build_allowed_subtree(snomed, candidate_ids, descendant_depth=DESCENDANT_DEPTH)
                # allowed.discard(SNOMED_ROOT)
                # parent_map.pop(SNOMED_ROOT, None)
                # for node, parents in parent_map.items():
                #     parent_map[node] = [p for p in parents if p != SNOMED_ROOT]
                # child_map.pop(SNOMED_ROOT, None)  # its children's parent_map entries above already drop it
                allowed, parent_map, child_map = refine_subtree(allowed, parent_map, child_map, set(candidate_ids))
                roots = get_roots(allowed, parent_map)
                prompt_context = render_subtree(snomed, parent_map, child_map, roots, candidate_ids)
                
                print(f"Evaluating... {mention}")
                print(prompt_context)
                print("\n")
                
                break
            break
        break
        #         classifier.classify_hierarchical(
        #             term=mention,
        #             context=term_data["context"],
        #             snomed=snomed,
        #             all_ancestors=gt_ancestors,
        #             gt_id=gt_concept_id,
        #             parent_id=None,
        #             start=term_data["start"],
        #             end=term_data["end"],
        #             note_id=note_id,
        #             child_map=child_map,
        #             roots=roots,
        #             )
        # classifier.save_results(f"{output_path}.json")
        # classifier.save_results_csv(f"{output_path}.csv")

    print(f"\nSaved LLM classification results to: {output_path}")

    with open(f"{output_path}.json", "r", encoding="utf-8") as f:
        classif_results = json.load(f)

    return None, classif_results

import matplotlib.pyplot as plt
from collections import deque


def evaluate_subtrees(note_keywords, gt, snomed, lexical_index,
                       descendant_depth=DESCENDANT_DEPTH, bm25_top_n=BM25_TOP_N):
    rows = []

    for note_id, predicted in note_keywords.items():
        ground_truth = gt.get(note_id, [])

        for term_data in predicted:
            match = match_entity(pred_entity=term_data, ground_truth=ground_truth,
                                  iou_threshold=IOU_THRESHOLD, alpha=1)
            if not match["matched"]:
                continue

            mention = term_data["term"]
            gt_concept_id = match["ground_truth"]["concept_id"]

            candidate_ids = [cid for cid, _ in lexical_index.search(mention, top_n=bm25_top_n)]
            allowed, parent_map, child_map = build_allowed_subtree(
                snomed, candidate_ids, descendant_depth=descendant_depth
            )
            roots = get_roots(allowed, parent_map)

            # --- root depth in the FULL hierarchy (distance from SNOMED root) ---
            root_depths = [snomed.get_depths(r)["min_depth"] for r in roots]

            # --- subtree "height": BFS depth of every allowed node from the nearest root,
            #     walking the pruned child_map only ---
            depth_from_root = {}
            queue = deque([(r, 0) for r in roots])
            while queue:
                node, d = queue.popleft()
                if node in depth_from_root and depth_from_root[node] <= d:
                    continue
                depth_from_root[node] = d
                for child in child_map.get(node, []):
                    queue.append((child, d + 1))
            subtree_height = max(depth_from_root.values()) if depth_from_root else 0

            # --- where does the gold concept sit relative to the candidates? ---
            gt_is_candidate = gt_concept_id in candidate_ids
            gt_in_subtree = gt_concept_id in allowed

            if gt_is_candidate:
                gt_relation = "candidate"
            elif gt_in_subtree and any(
                gt_concept_id in snomed.get_all_ancestors(cid) for cid in candidate_ids
            ):
                gt_relation = "ancestor_of_candidate"
            elif gt_in_subtree:
                gt_relation = "descendant_of_candidate"
            else:
                gt_relation = "not_in_subtree"

            gt_depth_in_subtree = depth_from_root.get(gt_concept_id)  # None if not in subtree

            rows.append({
                "note_id": note_id,
                "mention": mention,
                "gt_concept_id": gt_concept_id,
                "gt_label": snomed.get_label(gt_concept_id),
                "n_candidates": len(candidate_ids),
                "subtree_size": len(allowed),
                "n_roots": len(roots),
                "root_depth_min": min(root_depths) if root_depths else None,
                "root_depth_max": max(root_depths) if root_depths else None,
                "subtree_height": subtree_height,
                "gt_is_candidate": gt_is_candidate,
                "gt_in_subtree": gt_in_subtree,
                "gt_relation": gt_relation,
                "gt_depth_in_subtree": gt_depth_in_subtree,
            })

    return pd.DataFrame(rows)



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
        
    _, classif_results = classify_entities(note_keywords, gt, snomed, lexical_index)
    
    # --- run it ---
    # subtree_eval_df = evaluate_subtrees(note_keywords, gt, snomed, lexical_index)

    # # --- headline metrics ---
    # print(f"Mentions evaluated: {len(subtree_eval_df)}")
    # print(f"GT reachable in subtree (recall): {subtree_eval_df['gt_in_subtree'].mean():.3f}")
    # print(subtree_eval_df["gt_relation"].value_counts(normalize=True))
    # print("\nSubtree size:")
    # print(subtree_eval_df["subtree_size"].describe())
    # print("\nSubtree height (BFS depth from root within pruned tree):")
    # print(subtree_eval_df["subtree_height"].describe())
    # print("\nRoot depth in full hierarchy:")
    # print(subtree_eval_df["root_depth_min"].describe())

    # # --- plots ---
    # fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # axes[0, 0].hist(subtree_eval_df["subtree_size"], bins=30)
    # axes[0, 0].set_title("Subtree size (# allowed concepts)")
    # axes[0, 0].set_xlabel("nodes")

    # axes[0, 1].hist(subtree_eval_df["subtree_height"], bins=range(0, subtree_eval_df["subtree_height"].max() + 2))
    # axes[0, 1].set_title("Subtree height (BFS depth from root)")
    # axes[0, 1].set_xlabel("levels")

    # subtree_eval_df["gt_relation"].value_counts().plot(kind="bar", ax=axes[1, 0])
    # axes[1, 0].set_title("Where does the gold concept sit?")
    # axes[1, 0].tick_params(axis="x", rotation=30)

    # axes[1, 1].hist(subtree_eval_df["root_depth_min"].dropna(), bins=20)
    # axes[1, 1].set_title("Root depth in full SNOMED hierarchy")
    # axes[1, 1].set_xlabel("depth from SNOMED root")

    # plt.tight_layout()
    # plt.savefig("cache/subtree_eval_plots.png", dpi=150)
    # plt.show()

    # # --- worst offenders, for manual inspection ---
    # print("\nMentions where GT is unreachable in the subtree:")
    # print(subtree_eval_df[subtree_eval_df["gt_relation"] == "not_in_subtree"]
    #     [["mention", "gt_label", "n_candidates"]].head(20))

    # print("\nLargest subtrees (check for fan-out blowups):")
    # print(subtree_eval_df.sort_values("subtree_size", ascending=False)
    #     [["mention", "gt_label", "subtree_size", "n_roots"]].head(10))

    # print(f"\nDone. Classification results available at: {OUTPUT_PATH}.json / {OUTPUT_PATH}.csv")
    return classif_results


if __name__ == "__main__":
    main()
import os
from pathlib import Path
import logging
import time
#from pruning.candidate_generator import CandidateGenerator
#from pruning.dense_index import DenseIndex
from main_2 import NER_MODEL
from pruning.lexical_index import LexicalIndex
from ontology.snomed_loader import SNOMEDHierarchy
# from subtree_pruner import build_allowed_subtree, get_allowed_children
import pickle
import pandas as pd
from evaluation.metrics import match_entity

SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"


CACHE_DIR = Path("./cache")
CACHE_PATH = f"cache/note_keywords_{NER_MODEL}.pkl"
ANNOTATIONS_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"
out_dir = "cache/lexical_eval_df_20.pkl"

# Candidate generation
BM25_TOP_N = 20
DENSE_TOP_N = 20
FINAL_TOP_K = 10

IOU_THRESHOLD = 0.05
 
# Subtree pruning: how many levels below each candidate to keep, so the LLM
# can still refine to something more specific than the candidate itself.
DESCENDANT_DEPTH = 2
 
# SapBERT model (biomedical entity linking encoder)
SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

def print_results(hierarchy, mention: str, results: list[dict]):
    print(f"\n=== '{mention}' ===")
    if not results:
        print("  (no matches)")
        return
    for r in results:
        label = hierarchy.get_label(r["concept_id"])
        print(f"  {r['score']:6.2f}  {r['concept_id']:12s}  {label:45s}  matched on: \"{r['matched_term']}\"")
 
def main():

    if os.path.exists(out_dir):
        print(f"Loading cached evaluation dataframe from: {out_dir}")
        eval_df = pd.read_pickle(out_dir)
    
    else:
        print("Loading NER results...")
        with open(CACHE_PATH, "rb") as f:
            note_keywords = pickle.load(f)
        
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

        print("Loading SNOMED hierarchy...")
        t0 = time.time()
        hierarchy = SNOMEDHierarchy(
            concept_file=SNOMED_CONCEPT_FILE,
            relationship_file=SNOMED_RELATIONSHIP_FILE,
            description_file=SNOMED_DESCRIPTION_FILE,
        )
        print(f"Loaded in {time.time() - t0:.1f}s "
            f"({len(hierarchy.parent_map)} concepts with parents, {len(hierarchy.descriptions)} descriptions)")
    
        print("Building synonym map...")
        t0 = time.time()
        synonyms = hierarchy.build_full_synonym_map()
        print(f"Built synonym map for {len(synonyms)} concepts in {time.time() - t0:.1f}s")
    
        print("Building BM25 index...")
        t0 = time.time()
        lexical_index = LexicalIndex(synonyms)
        print(f"Indexed {len(lexical_index.doc_terms)} description strings in {time.time() - t0:.1f}s")
        
        eval_rows = []

        for i, (note_id, predicted) in enumerate(note_keywords.items()):
            
            print(f"Analyzing note: {note_id} out of {len(note_keywords)} notes")
            t0 = time.time()
            ground_truth = gt.get(note_id, [])

            for term_data in predicted:
                
                match = match_entity(
                    pred_entity=term_data, ground_truth=ground_truth, iou_threshold=IOU_THRESHOLD
                )

                if match["matched"]:
                    gt_concept_id = match["ground_truth"]["concept_id"]
                    mention = term_data["term"]
                    results = lexical_index.search(mention, top_n=BM25_TOP_N)
                    candidate_ids = [cid for cid, _ in results]
                    
                    rank = candidate_ids.index(gt_concept_id) + 1 if gt_concept_id in candidate_ids else None

                    eval_rows.append({
                        "note_id": note_id,
                        "mention": mention,
                        "gold_concept_id": gt_concept_id,
                        "gold_label": hierarchy.get_label(gt_concept_id),
                        "n_candidates": len(candidate_ids),
                        "gt_among_top_candidates": gt_concept_id in candidate_ids,
                        "rank": rank,
                        "top_candidates": candidate_ids,
                    })
        
            
            print(f"Evaluated {len(eval_rows)} matched mentions in {time.time() - t0:.1f}s")
    
        eval_df = pd.DataFrame(eval_rows)

        print(f"\nBuilt evaluation DataFrame with {len(eval_df)} matched instances.")

        out_path = Path(out_dir)
        out_path.parent.mkdir(exist_ok=True)
        eval_df.to_pickle(out_path)
        print(f"\nSaved evaluation dataframe to {out_path}")

    print("Evaluating BM25 candidate generation performance...")
    print(eval_df.describe(include="all").transpose())
    
    print(f"Zero candidates returned: {(eval_df['n_candidates'] == 0).sum()} "
        f"({(eval_df['n_candidates'] == 0).mean():.1%})")
    
    mrr = (1.0/ eval_df["rank"]).fillna(0).mean()
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    for k in [1, 3, 5, 10, 20]:
        top_k = (eval_df["rank"] <= k).fillna(False).mean()
        print(f"Recall@{k}: {top_k:.4%}")

    





                
        

 
 
if __name__ == "__main__":
    main()


# def main():

#     snomed = SNOMEDHierarchy(
#         concept_file=SNOMED_CONCEPT_FILE,
#         relationship_file=SNOMED_RELATIONSHIP_FILE,
#         description_file=SNOMED_DESCRIPTION_FILE,
#     )

#     # 2. Flatten descriptions into a full conceptId -> [terms] map (one bulk
#     #    groupby, not a per-concept filter), then build the two candidate
#     #    generators. BM25 index is cheap; SapBERT embeddings are cached to
#     #    disk after the first run since encoding every synonym is the slow part.
#     synonyms = snomed.build_full_synonym_map()
#     lexical_index = LexicalIndex(synonyms)
#     # dense_index = DenseIndex(synonyms)
#     # generator = CandidateGenerator(lexical_index, dense_index)

#     # 3. Generate candidates for a mention
# #     mention = "SOB"  # e.g. shortness of breath, MIMIC-style abbreviation
# #     candidates = generator.generate(mention, top_k=FINAL_TOP_K)
# #     print(f"\nCandidates for mention '{mention}':")
# #     for c in candidates:
# #         print(f"  {c.concept_id}  {snomed.get_label(c.concept_id):50s} sources={c.sources}")

# #     # 4. Prune the SNOMED hierarchy to the subtree induced by these candidates
# #     candidate_ids = [c.concept_id for c in candidates]
# #     allowed = build_allowed_subtree(snomed, candidate_ids)
# #     print(f"\nPruned subtree size: {len(allowed)} concepts")

# #     # 5. Use `get_allowed_children` in place of `hierarchy.get_children`
# #     #    inside your existing level-by-level LLM navigation loop, e.g.:
# #     node = config.SNOMED_ROOT
# #     children = get_allowed_children(snomed, node, allowed)
# #     print(f"\nAllowed children of ROOT within pruned subtree: {len(children)}")
# #     for child in children[:10]:
# #         print(f"  {child}  {snomed.get_label(child)}")

# #     # 6. Before adopting pruning in your pipeline, validate recall on a dev
# #     #    set of (mention, gold_concept_id) pairs from the challenge annotations:
# #     #
# #     # from subtree_pruner import recall_at_k
# #     # dev_mentions = [...]
# #     # dev_gold = [...]
# #     # candidate_lists = [
# #     #     [c.concept_id for c in generator.generate(m)] for m in dev_mentions
# #     # ]
# #     # print(recall_at_k(hierarchy, candidate_lists, dev_gold))


# if __name__ == "__main__":
#     main()
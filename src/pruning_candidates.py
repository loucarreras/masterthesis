import os
from pathlib import Path
import logging
import time
#from pruning.candidate_generator import CandidateGenerator
#from pruning.dense_index import DenseIndex
from main_2 import NER_MODEL
from pruning.lexical_index import LexicalIndex, char_ngram_tokenizer
from pruning.fuzzy_index import FuzzyIndex, BlockedFuzzyIndex
from ontology.snomed_loader import SNOMEDHierarchy
# from subtree_pruner import build_allowed_subtree, get_allowed_children
import pickle
import pandas as pd
from evaluation.metrics import match_entity
import matplotlib.pyplot as plt
from functools import partial

SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"


CACHE_DIR = Path("./cache")
CACHE_PATH = f"cache/note_keywords_{NER_MODEL}.pkl"
ANNOTATIONS_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"
out_dir = "cache/lexical_eval_df_20_tokenizer.pkl"

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

# Pruning to only three top-level hierarchies
LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure",
}

CANDIDATE_METHOD = "bm25"      # "bm25" | "fuzzy" | "fuzzy_exhaustive"
FUZZY_SHORTLIST_SIZE = 500      # only used by "fuzzy" (blocked); tune for speed/recall trade-off


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

        print("Loading SNOMED hierarchy...")
        t0 = time.time()
        hierarchy = SNOMEDHierarchy(
            concept_file=SNOMED_CONCEPT_FILE,
            relationship_file=SNOMED_RELATIONSHIP_FILE,
            description_file=SNOMED_DESCRIPTION_FILE,
        )
        print(f"Loaded in {time.time() - t0:.1f}s "
            f"({len(hierarchy.parent_map)} concepts with parents, {len(hierarchy.descriptions)} descriptions)")
    
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
        
        print("Restricting to allowed top-level branches...")
        t0 = time.time()
        allowed_concepts = get_allowed_concepts(hierarchy, LEVEL1)
        print(f"Total allowed concepts: {len(allowed_concepts)} in {time.time() - t0:.1f}s")
    
        print("Building synonym map...")
        t0 = time.time()
        synonyms = hierarchy.build_full_synonym_map()
        before = len(synonyms)
        synonyms = {cid: terms for cid, terms in synonyms.items() if cid in allowed_concepts}
        print(f"Built synonym map for {before} concepts, restricted to {len(synonyms)} "
            f"in {time.time() - t0:.1f}s")
    
        print("Building BM25 index...")
        t0 = time.time()
        if CANDIDATE_METHOD == "bm25":
            lexical_index = LexicalIndex(synonyms, tokenizer=partial(char_ngram_tokenizer, n=3))
        elif CANDIDATE_METHOD == "fuzzy":
            lexical_index = BlockedFuzzyIndex(synonyms, shortlist_size=FUZZY_SHORTLIST_SIZE)
        elif CANDIDATE_METHOD == "fuzzy_exhaustive":
            lexical_index = FuzzyIndex(synonyms)
        else:
            raise ValueError(f"Unknown CANDIDATE_METHOD: {CANDIDATE_METHOD}")
        print(f"Built candidate index in {time.time() - t0:.1f}s")

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
    
    n_total = len(eval_df)

    print("EVALUATING BM25 INDEX PERFORMANCE...")
    print(eval_df.describe(include="all").transpose())
    
    # Zero candidates returned
    zero_df = eval_df[eval_df["n_candidates"] == 0]
    print(f"ZERO CANDIDATES RETURNED: {(eval_df['n_candidates'] == 0).sum()} "
        f"({(eval_df['n_candidates'] == 0).mean():.1%})")
    
    if len(zero_df):
        sample = zero_df[["mention", "gold_concept_id", "gold_label", "top_candidates"]].sample(
            min(15, len(zero_df)), random_state=0
        )
        print(sample.to_string(index=False))

    # Gold concept not among top candidates
    print("MISSES")
    print(f"Gold concept not among top {BM25_TOP_N} candidates: "
        f"{(eval_df['gt_among_top_candidates'] == False).sum()} "
        f"({(eval_df['gt_among_top_candidates'] == False).mean():.1%})")
    
    miss_df = eval_df[(~eval_df["gt_among_top_candidates"]) & (eval_df["n_candidates"] > 0)].copy()
    
    if len(miss_df):
        sample = miss_df[["mention", "gold_concept_id", "gold_label", "top_candidates"]].sample(
            min(15, len(miss_df)), random_state=0
        )
        print(sample.to_string(index=False))

    # Recall@k
    for k in [1, 3, 5, 10, 20]:
        top_k = (eval_df["rank"] <= k).fillna(False).mean()
        print(f"Recall@{k}: {top_k:.4%}")

    
    ks = list(range(1, BM25_TOP_N + 1))
    recall_curve = [(eval_df["rank"] <= k).fillna(False).mean() for k in ks]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, recall_curve, marker="o")
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("BM25 lexical candidate generation: Recall@k")
    plt.grid(True)
    plt.xticks(ks)
    plt.tight_layout()
    curve_path = "cache/recall_curve.png"
    plt.savefig(curve_path)
    print(f"\nSaved recall curve to {curve_path}")

    # Rank distribution of hits
    hits_df = eval_df[eval_df["rank"].notna()]
    print(f"\n=== Hits: {len(hits_df)} ({len(hits_df) / n_total:.1%}) ===")
    print(hits_df["rank"].describe())
    print("\nRank distribution (count of hits at each rank):")
    print(hits_df["rank"].value_counts().sort_index())

    plt.figure(figsize=(8, 6))
    plt.hist(eval_df["rank"].dropna(), bins=range(1, BM25_TOP_N + 2), align="left", rwidth=0.8) 
    plt.xticks(range(1, BM25_TOP_N + 1))
    plt.xlabel("Rank of Gold Concept in BM25 Candidates")
    plt.ylabel("Frequency")
    plt.title("Distribution of Gold Concept Ranks in BM25 Candidates")
    plt.show()

    # Mean reciprocal rank (MRR)
    rr = (1.0 / hits_df["rank"]).fillna(0)
    mrr = rr.mean()
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    print(f"Reciprocal Rank Statistics:")
    print(rr.describe())

    plt.figure(figsize=(8, 6))
    plt.violinplot(rr.dropna(), vert=True)
    plt.xlabel("Reciprocal Rank")
    plt.title("Distribution of Reciprocal Ranks for Gold Concepts in BM25 Candidates")
    plt.show()

    # Additional analysis: analysis of the mentions for which the gold concept was not among the top candidates

    misses_df = eval_df[~eval_df["gt_among_top_candidates"]]
    misses_df = misses_df[misses_df["n_candidates"] > 0]  # Only consider cases where candidates were returned
    print(f"\n=== Misses: {len(misses_df)} ({len(misses_df) / n_total:.1%}) ===")

    # For each miss, analyze the candidates returned. Is the candidate an ancestor of the gold concept? Is it a descendant? Distance?
    miss_rows = []
    
    for idx, row in misses_df.iterrows():
        gold_concept_id = row["gold_concept_id"]
        top_candidates = row["top_candidates"]

        # Analyze the relationship of each candidate to the gold concept
        golden_ancestors = hierarchy.get_all_ancestors(gold_concept_id)
        golden_descendants = hierarchy.get_all_descendants(gold_concept_id)
        
        for rank_position, candidate_id in enumerate(top_candidates, start=1):
            if candidate_id in golden_ancestors:
                relationship = "ancestor"
            elif candidate_id in golden_descendants:
                relationship = "descendant"
            else:
                relationship = "unrelated"
            
            raw_distance = hierarchy.get_shortest_path_length(candidate_id, gold_concept_id)
            distance = None if raw_distance == float("inf") else raw_distance


            miss_rows.append({
                "mention": row["mention"],
                "ground_truth": gold_concept_id,
                "gold_label": row["gold_label"],
                "candidate_id": candidate_id,
                "candidate_label": hierarchy.get_label(candidate_id),
                "candidate_rank_position": rank_position,  # 1 = BM25's top guess, etc.
                "relationship": relationship,
                "distance": distance,
            })
 
    miss_analysis_df = pd.DataFrame(miss_rows)
 
    print(f"\n=== Miss Analysis: {len(misses_df)} misses, {len(miss_analysis_df)} candidate-level rows ===")
    print(miss_analysis_df.groupby("relationship").size())
 
    # --- distance stats, overall and split by relationship type ---
    print("\nDistance distribution overall:")
    print(miss_analysis_df["distance"].describe())
 
    print("\nDistance distribution by relationship type:")
    print(miss_analysis_df.groupby("relationship")["distance"].describe())
 
    # --- per-miss summary: what's the BEST (closest) candidate for each miss? ---
    best_per_miss = (
        miss_analysis_df.sort_values("distance", na_position="last")
        .groupby(["mention", "ground_truth"], as_index=False)
        .first()
    )
    print(f"\n=== Best (closest) candidate per miss, n={len(best_per_miss)} ===")
    print(best_per_miss["distance"].describe())
    print(best_per_miss["relationship"].value_counts())
    print(best_per_miss.groupby("relationship")["distance"].describe())
    print("\nBest-candidate distance distribution:")
    print(best_per_miss["distance"].value_counts(dropna=False).sort_index())
    
    plt.figure(figsize=(8, 6))
    plt.hist(best_per_miss["distance"].value_counts(dropna=False).sort_index(), align="left", rwidth=0.8) 
    plt.xlabel("Distance to ground truth")
    plt.ylabel("Frequency")
    plt.title("Distribution of Distance for best-misses per mention")
    plt.show()
    
    # near misses: was ANY candidate within 2 hops, even if not exact?
    near_miss_threshold = 2
    near_misses = best_per_miss[best_per_miss["distance"].notna() & (best_per_miss["distance"] <= near_miss_threshold)]
    print(f"\nNear misses (closest candidate within {near_miss_threshold} hops): "
          f"{len(near_misses)} ({len(near_misses) / len(best_per_miss):.1%} of misses)")
 
    truly_unrelated = best_per_miss[best_per_miss["distance"].isna()]
    print(f"Truly unrelated (no candidate connected within search depth): "
          f"{len(truly_unrelated)} ({len(truly_unrelated) / len(best_per_miss):.1%} of misses)")
    if len(truly_unrelated):
        print(truly_unrelated[["mention", "gold_label"]].sample(min(15, len(truly_unrelated)), random_state=0).to_string(index=False))
 
    # save for later comparison (e.g. against dense retrieval's miss analysis)
    miss_analysis_df.to_pickle("cache/miss_analysis_df.pkl")
    best_per_miss.to_pickle("cache/miss_analysis_best_per_miss.pkl")

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
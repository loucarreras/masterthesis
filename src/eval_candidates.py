import os
import time
import pickle
import argparse
from pathlib import Path
 
import pandas as pd
import matplotlib.pyplot as plt
 
from main_2 import NER_MODEL
from pruning.lexical_index import LexicalIndex
from pruning.fuzzy_index import FuzzyIndex, BlockedFuzzyIndex
from pruning.dense_index import DenseIndex, SapBERTEncoder
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.metrics import match_entity

# Paths
SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"
 
CACHE_DIR = Path("./cache")
CACHE_PATH = f"cache/note_keywords_{NER_MODEL}.pkl"
ANNOTATIONS_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"
 
IOU_THRESHOLD = 0.05
 
# Pruning to only three top-level hierarchies
LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure",
}

# Method
METHOD = "lexical" # "lexical" | "dense"
TOP_N = 20
# lexical-only settings
CANDIDATE_METHOD = "bm25"       # "bm25" | "fuzzy" | "fuzzy_exhaustive"
FUZZY_SHORTLIST_SIZE = 500      # only used by "fuzzy" (blocked)
 
# semantic-only settings
SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

METHOD_CONFIG = {
    "lexical": {
        "top_n": TOP_N,
        "out_dir": f"cache/{METHOD}_eval_df_{TOP_N}.pkl",
        "label": "BM25 lexical",
        "recall_curve_path": "cache/recall_curve.png",
        "miss_analysis_path": "cache/miss_analysis_df.pkl",
        "best_per_miss_path": "cache/miss_analysis_best_per_miss.pkl",
    },
    "dense": {
        "top_n": TOP_N,
        "out_dir": f"cache/{METHOD}_eval_df_{TOP_N}.pkl",
        "label": "SapBERT dense",
        "recall_curve_path": "cache/dense_recall_curve.png",
        "miss_analysis_path": "cache/dense_miss_analysis_df.pkl",
        "best_per_miss_path": "cache/dense_miss_analysis_best_per_miss.pkl",
    },
}


def get_allowed_concepts(hierarchy: SNOMEDHierarchy, root_ids: dict[str, str]) -> set[str]:
    allowed = set(root_ids.keys())
    for root_id, label in root_ids.items():
        t0 = time.time()
        descendants = hierarchy.get_all_descendants(root_id)
        allowed |= descendants
        print(f"  {label} ({root_id}): {len(descendants)} descendants in {time.time() - t0:.1f}s")
    return allowed

def build_index(method: str, synonyms: dict):
    """Builds and returns the candidate index for the chosen method.
 
    Both LexicalIndex/FuzzyIndex/BlockedFuzzyIndex and DenseIndex expose the
    same interface used below: `.search(mention, top_n)` and `.doc_terms`.
    """
    if method == "lexical":
        if CANDIDATE_METHOD == "bm25":
            index = LexicalIndex(synonyms)
        elif CANDIDATE_METHOD == "fuzzy":
            index = BlockedFuzzyIndex(synonyms, shortlist_size=FUZZY_SHORTLIST_SIZE)
        elif CANDIDATE_METHOD == "fuzzy_exhaustive":
            index = FuzzyIndex(synonyms)
        else:
            raise ValueError(f"Unknown CANDIDATE_METHOD: {CANDIDATE_METHOD}")
        return index
 
    elif method == "dense":
        encoder = SapBERTEncoder(model_name=SAPBERT_MODEL_NAME)
        return DenseIndex(synonyms, encoder=encoder, cache_dir=CACHE_DIR)
 
    else:
        raise ValueError(f"Unknown METHOD: {method}")


def main(method: str = METHOD, num_candidates: int | None = None):
    if method not in METHOD_CONFIG:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(METHOD_CONFIG)}.")
    cfg = METHOD_CONFIG[method]

    top_n = num_candidates if num_candidates is not None else cfg["top_n"]
    cfg["top_n"] = top_n
    cfg["out_dir"] = f"cache/{method}_eval_df_{top_n}.pkl"

    print("Loading SNOMED hierarchy...")
    t0 = time.time()
    hierarchy = SNOMEDHierarchy(
        concept_file=SNOMED_CONCEPT_FILE,
        relationship_file=SNOMED_RELATIONSHIP_FILE,
        description_file=SNOMED_DESCRIPTION_FILE,
    )
    print(f"Loaded in {time.time() - t0:.1f}s "
        f"({len(hierarchy.parent_map)} concepts with parents, {len(hierarchy.descriptions)} descriptions)")
 
    if os.path.exists(cfg["out_dir"]):
        print(f"Loading cached evaluation dataframe from: {cfg['out_dir']}")
        eval_df = pd.read_pickle(cfg["out_dir"])

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
    
        print(f"Building {cfg['label']} index...")
        t0 = time.time()
        index = build_index(method, synonyms)
        print(f"Built candidate index over {len(index.doc_terms)} synonym strings "
            f"in {time.time() - t0:.1f}s")
    
        top_n = cfg["top_n"]
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
                    results = index.search(mention, top_n=top_n)
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
    
        out_path = Path(cfg["out_dir"])
        out_path.parent.mkdir(exist_ok=True)
        eval_df.to_pickle(out_path)
        print(f"\nSaved evaluation dataframe to {out_path}")
    

    # EVALUATION
    top_n = cfg["top_n"]
    label = cfg["label"]
    n_total = len(eval_df)
 
    print(f"EVALUATING {label.upper()} INDEX PERFORMANCE...")
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
    print(f"Gold concept not among top {top_n} candidates: "
          f"{(eval_df['gt_among_top_candidates'] == False).sum()} "
          f"({(eval_df['gt_among_top_candidates'] == False).mean():.1%})")
 
    miss_df = eval_df[(~eval_df["gt_among_top_candidates"]) & (eval_df["n_candidates"] > 0)].copy()
 
    if len(miss_df):
        sample = miss_df[["mention", "gold_concept_id", "gold_label", "top_candidates"]].sample(
            min(15, len(miss_df)), random_state=0
        )
        print(sample.to_string(index=False))
 
    # Recall@k
    for k in [1, 3, 5, 10, 20, 40]:
        if k > top_n:
            continue
        top_k = (eval_df["rank"] <= k).fillna(False).mean()
        print(f"Recall@{k}: {top_k:.4%}")
 
    ks = list(range(1, top_n + 1))
    recall_curve = [(eval_df["rank"] <= k).fillna(False).mean() for k in ks]
 
    plt.figure(figsize=(8, 5))
    plt.plot(ks, recall_curve, marker="o")
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title(f"{label} candidate generation: Recall@k")
    plt.grid(True)
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(cfg["recall_curve_path"])
    print(f"\nSaved recall curve to {cfg['recall_curve_path']}")
 
    # Rank distribution of hits
    hits_df = eval_df[eval_df["rank"].notna()]
    print(f"\n=== Hits: {len(hits_df)} ({len(hits_df) / n_total:.1%}) ===")
    print(hits_df["rank"].describe())
    print("\nRank distribution (count of hits at each rank):")
    print(hits_df["rank"].value_counts().sort_index())
 
    plt.figure(figsize=(8, 6))
    plt.hist(eval_df["rank"].dropna(), bins=range(1, top_n + 2), align="left", rwidth=0.8)
    plt.xticks(range(1, top_n + 1))
    plt.xlabel("Rank of Gold Concept in Candidates")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Gold Concept Ranks in {label} Candidates")
    plt.show()
 
    # Mean reciprocal rank (MRR)
    rr = (1.0 / hits_df["rank"]).fillna(0)
    mrr = rr.mean()
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
 
    print("Reciprocal Rank Statistics:")
    print(rr.describe())
 
    plt.figure(figsize=(8, 6))
    plt.violinplot(rr.dropna(), vert=True)
    plt.xlabel("Reciprocal Rank")
    plt.title(f"Distribution of Reciprocal Ranks for Gold Concepts in {label} Candidates")
    plt.show()
 
    # Miss analysis: for each miss, how does the candidate relate to the gold concept?
    misses_df = eval_df[~eval_df["gt_among_top_candidates"]]
    misses_df = misses_df[misses_df["n_candidates"] > 0]
    print(f"\n=== Misses: {len(misses_df)} ({len(misses_df) / n_total:.1%}) ===")
 
    miss_rows = []
 
    for idx, row in misses_df.iterrows():
        gold_concept_id = row["gold_concept_id"]
        top_candidates = row["top_candidates"]
 
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
                "candidate_rank_position": rank_position,
                "relationship": relationship,
                "distance": distance,
            })
 
    miss_analysis_df = pd.DataFrame(miss_rows)
 
    print(f"\n=== Miss Analysis: {len(misses_df)} misses, {len(miss_analysis_df)} candidate-level rows ===")
    print(miss_analysis_df.groupby("relationship").size())
 
    print("\nDistance distribution overall:")
    print(miss_analysis_df["distance"].describe())
 
    print("\nDistance distribution by relationship type:")
    print(miss_analysis_df.groupby("relationship")["distance"].describe())
 
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
    plt.title(f"Distribution of Distance for best-misses per mention ({label})")
    plt.show()
 
    near_miss_threshold = 2
    near_misses = best_per_miss[best_per_miss["distance"].notna() & (best_per_miss["distance"] <= near_miss_threshold)]
    print(f"\nNear misses (closest candidate within {near_miss_threshold} hops): "
          f"{len(near_misses)} ({len(near_misses) / len(best_per_miss):.1%} of misses)")
 
    truly_unrelated = best_per_miss[best_per_miss["distance"].isna()]
    print(f"Truly unrelated (no candidate connected within search depth): "
          f"{len(truly_unrelated)} ({len(truly_unrelated) / len(best_per_miss):.1%} of misses)")
    if len(truly_unrelated):
        print(truly_unrelated[["mention", "gold_label"]].sample(
            min(15, len(truly_unrelated)), random_state=0
        ).to_string(index=False))
 
    # save for later comparison (e.g. lexical vs semantic miss analysis)
    miss_analysis_df.to_pickle(cfg["miss_analysis_path"])
    best_per_miss.to_pickle(cfg["best_per_miss_path"])

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=list(METHOD_CONFIG), default=METHOD,
        help="Candidate generation method to evaluate (default: %(default)s)",
    )

    parser.add_argument(
        "--num-candidates", type=int, default=None,
        help="Number of candidates to retrieve per mention (top_n). "
             "Defaults to :" f"{TOP_N}.",
    )

    args = parser.parse_args()
    main(args.method, args.num_candidates)


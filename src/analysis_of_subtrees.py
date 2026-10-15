import time
from collections import deque
from pathlib import Path
 
import matplotlib.pyplot as plt
import pandas as pd
 
from ontology.snomed_loader import SNOMEDHierarchy
from pruning.subtree_pruner import build_allowed_subtree, refine_subtree, get_roots
 
SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"
 
LEXICAL_EVAL_CACHE = "cache/lexical_eval_df_20.pkl"
SUBTREE_EVAL_CACHE = "cache/subtree_eval_df_20.pkl"
 
DESCENDANT_DEPTH = 2
SNOMED_ROOT = "138875005"

 
def evaluate_subtrees(eval_df: pd.DataFrame, snomed: SNOMEDHierarchy,
                       descendant_depth: int = DESCENDANT_DEPTH) -> pd.DataFrame:
    rows = []
 
    for i, row in enumerate(eval_df.itertuples(index=False)):
        if i % 500 == 0:
            print(f"Analyzing mention {i}/{len(eval_df)}")
 
        mention = row.mention
        gt_concept_id = row.gold_concept_id
        candidate_ids = list(row.top_candidates)
 
        if not candidate_ids:
            rows.append({
                "note_id": row.note_id,
                "mention": mention,
                "gt_concept_id": gt_concept_id,
                "gt_label": row.gold_label,
                "n_candidates": 0,
                "subtree_size": None,
                "n_roots": None,
                "root_depth_min": None,
                "root_depth_max": None,
                "subtree_height": None,
                "gt_is_candidate": False,
                "gt_in_subtree": False,
                "gt_relation": "no_candidates",
                "gt_depth_in_subtree": None,
            })
            continue
 
        # Build then refine the allowed subtree around the BM25 candidates.
        allowed, parent_map, child_map = build_allowed_subtree(
            snomed, candidate_ids, descendant_depth=descendant_depth
        )
        allowed, parent_map, child_map = refine_subtree(
            allowed, parent_map, child_map, set(candidate_ids)
        )
        roots = get_roots(allowed, parent_map)
 
        # Root depth in the FULL hierarchy (distance from SNOMED root)
        root_depths = [snomed.get_depths(r)["min_depth"] for r in roots]
 
        # Subtree height
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
            "note_id": row.note_id,
            "mention": mention,
            "gt_concept_id": gt_concept_id,
            "gt_label": row.gold_label,
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
    import os
 
    if os.path.exists(SUBTREE_EVAL_CACHE):
        print(f"Loading cached subtree eval dataframe from: {SUBTREE_EVAL_CACHE}")
        subtree_eval_df = pd.read_pickle(SUBTREE_EVAL_CACHE)
    else:
        print(f"Loading cached candidate-generation eval dataframe from: {LEXICAL_EVAL_CACHE}")
        eval_df = pd.read_pickle(LEXICAL_EVAL_CACHE)
        print(f"Loaded {len(eval_df)} matched mentions (BM25 search already done).")
 
        print("Loading SNOMED hierarchy...")
        t0 = time.time()
        hierarchy = SNOMEDHierarchy(
            concept_file=SNOMED_CONCEPT_FILE,
            relationship_file=SNOMED_RELATIONSHIP_FILE,
            description_file=SNOMED_DESCRIPTION_FILE,
        )
        print(f"Loaded in {time.time() - t0:.1f}s "
              f"({len(hierarchy.parent_map)} concepts with parents, {len(hierarchy.descriptions)} descriptions)")
 
        subtree_eval_df = evaluate_subtrees(eval_df, hierarchy)
 
        out_path = Path(SUBTREE_EVAL_CACHE)
        out_path.parent.mkdir(exist_ok=True)
        subtree_eval_df.to_pickle(out_path)
        print(f"\nSaved subtree evaluation dataframe to {out_path}")
 
    # --- headline metrics ---
    n_no_candidates = (subtree_eval_df["n_candidates"] == 0).sum()
    has_candidates_df = subtree_eval_df[subtree_eval_df["n_candidates"] > 0]
    print(f"Mentions with zero BM25 candidates (excluded from size/height/depth "
          f"stats below): {n_no_candidates} ({n_no_candidates / len(subtree_eval_df):.1%})")
 
    # --- headline metrics ---
    print(f"Mentions evaluated: {len(subtree_eval_df)}")
    print(f"GT reachable in subtree (recall, all mentions): {subtree_eval_df['gt_in_subtree'].mean():.3f}")
    print(subtree_eval_df["gt_relation"].value_counts(normalize=True))
    print(f"\nSubtree size (n={len(has_candidates_df)}, candidates > 0 only):")
    print(has_candidates_df["subtree_size"].describe())
    print("\nSubtree height (BFS depth from root within pruned tree):")
    print(has_candidates_df["subtree_height"].describe())
    print("\nRoot depth in full hierarchy:")
    print(has_candidates_df["root_depth_min"].describe())
    print("\nNumber of roots (first level decisions)")
    print(has_candidates_df["n_roots"].describe())
    print("\nGT depth in the subtree")
    print(has_candidates_df["gt_depth_in_subtree"].describe())

    # --- plots (candidates > 0 subset only, except the gt_relation panel) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
 
    axes[0, 0].hist(has_candidates_df["subtree_size"].dropna(), bins=30)
    axes[0, 0].set_title(f"Subtree size (# allowed concepts), n={len(has_candidates_df)}")
    axes[0, 0].set_xlabel("nodes")

    axes[0, 1].hist(has_candidates_df["subtree_height"].dropna(),
                     bins=range(0, int(has_candidates_df["subtree_height"].max()) + 2))
    axes[0, 1].set_title("Subtree height (BFS depth from root)")
    axes[0, 1].set_xlabel("levels")
 

    axes[1,0].hist(has_candidates_df["gt_depth_in_subtree"].dropna())
    axes[1, 0].set_title("Where does the GT sit in the subtree?")
    axes[1, 0].tick_params(axis="x", rotation=30)
 
    axes[1, 1].hist(has_candidates_df["root_depth_min"].dropna(), bins=11)
    axes[1, 1].set_title("Root depth in full SNOMED hierarchy")
    axes[1, 1].set_xlabel("depth from SNOMED root")
 
    plt.tight_layout()
    Path("cache").mkdir(exist_ok=True)
    plt.savefig("cache/subtree_eval_plots.png", dpi=150)
    plt.show()
 
    # --- worst offenders, for manual inspection ---
    print("\nMentions where GT is unreachable in the subtree (candidates > 0 only):")
    print(has_candidates_df[has_candidates_df["gt_relation"] == "not_in_subtree"]
          [["mention", "gt_label", "n_candidates"]].head(20))
 
    print("\nMentions with zero BM25 candidates at all:")
    print(subtree_eval_df[subtree_eval_df["n_candidates"] == 0]
          [["mention", "gt_label"]].head(20))
 
    print("\nLargest subtrees (check for fan-out blowups):")
    print(has_candidates_df.sort_values("subtree_size", ascending=False)
          [["mention", "gt_label", "subtree_size", "n_roots"]].head(10))
 
 
if __name__ == "__main__":
    main()

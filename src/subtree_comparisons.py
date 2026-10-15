import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ontology.snomed_loader import SNOMEDHierarchy
from pruning.subtree_pruner import build_allowed_subtree, refine_subtree, get_roots
from pruning.dense_index import SapBERTEncoder

SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"

COMPARISON_PATH = "cache/bm25_vs_sapbert_comparison_df.pkl"
OUT_PATH = "cache/tree_comparison_df.pkl"

DESCENDANT_DEPTH = 2
SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
COMPUTE_EMBEDDING_COHERENCE = False  # set False to skip the (slower) semantic metric

LEVEL1 = {
    "404684003": "Clinical finding",
    "71388002": "Procedure",
    "123037004": "Body structure",
}


def top_level_category(candidate_id: str, hierarchy: SNOMEDHierarchy) -> str | None:
    if candidate_id in LEVEL1:
        return candidate_id
    ancestors = hierarchy.get_all_ancestors(candidate_id)
    hits = [root for root in LEVEL1 if root in ancestors]
    return hits[0] if hits else None  # SNOMED is poly-hierarchical; take the first hit


def topcat_entropy(candidate_ids: list[str], hierarchy: SNOMEDHierarchy) -> float | None:
    cats = [top_level_category(cid, hierarchy) for cid in candidate_ids]
    cats = [c for c in cats if c is not None]
    if not cats:
        return None
    counts = Counter(cats)
    n = len(cats)
    probs = [c / n for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(len(LEVEL1))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def build_tree(candidate_ids: list[str], hierarchy: SNOMEDHierarchy,
               descendant_depth: int = DESCENDANT_DEPTH) -> dict:
    if not candidate_ids:
        return {
            "allowed": set(), "subtree_size": None, "n_roots": None,
            "root_depth_min": None, "subtree_height": None,
            "candidate_density": None, "topcat_entropy": None,
        }

    allowed, parent_map, child_map = build_allowed_subtree(
        hierarchy, candidate_ids, descendant_depth=descendant_depth
    )
    allowed, parent_map, child_map = refine_subtree(
        allowed, parent_map, child_map, set(candidate_ids)
    )
    roots = get_roots(allowed, parent_map)
    root_depths = [hierarchy.get_depths(r)["min_depth"] for r in roots]

    depth_from_root = {}
    from collections import deque
    queue = deque([(r, 0) for r in roots])
    while queue:
        node, d = queue.popleft()
        if node in depth_from_root and depth_from_root[node] <= d:
            continue
        depth_from_root[node] = d
        for child in child_map.get(node, []):
            queue.append((child, d + 1))
    subtree_height = max(depth_from_root.values()) if depth_from_root else 0

    return {
        "allowed": allowed,
        "subtree_size": len(allowed),
        "n_roots": len(roots),
        "root_depth_min": min(root_depths) if root_depths else None,
        "subtree_height": subtree_height,
        "candidate_density": len(candidate_ids) / len(allowed) if allowed else None,
        "topcat_entropy": topcat_entropy(candidate_ids, hierarchy),
    }


def embed_coherence(candidate_ids: list[str], embeddings: dict[str, np.ndarray]) -> float | None:
    vecs = [embeddings[cid] for cid in candidate_ids if cid in embeddings]
    if len(vecs) < 2:
        return None
    mat = np.stack(vecs)  # already L2-normalized by SapBERTEncoder.encode
    sims = mat @ mat.T
    n = len(vecs)
    upper = sims[np.triu_indices(n, k=1)]
    return float(upper.mean())


def main():
    print(f"Loading candidate comparison dataframe from: {COMPARISON_PATH}")
    comparison_df = pd.read_pickle(COMPARISON_PATH)

    print("Loading SNOMED hierarchy...")
    hierarchy = SNOMEDHierarchy(
        concept_file=SNOMED_CONCEPT_FILE,
        relationship_file=SNOMED_RELATIONSHIP_FILE,
        description_file=SNOMED_DESCRIPTION_FILE,
    )

    embeddings: dict[str, np.ndarray] = {}
    if COMPUTE_EMBEDDING_COHERENCE:
        print("Collecting unique concept ids to embed (one encode pass, not per-mention)...")
        all_ids = set()
        for col in ["top_candidates_bm25", "top_candidates_dense"]:
            for lst in comparison_df[col]:
                all_ids.update(lst)
        all_ids = sorted(all_ids)
        print(f"Embedding {len(all_ids)} unique concept labels with SapBERT...")
        encoder = SapBERTEncoder(model_name=SAPBERT_MODEL_NAME)
        labels = [hierarchy.get_label(cid) for cid in all_ids]
        vecs = encoder.encode(labels)
        embeddings = dict(zip(all_ids, vecs))

    rows = []
    for i, row in enumerate(comparison_df.itertuples(index=False)):
        if i % 500 == 0:
            print(f"Building trees for mention {i}/{len(comparison_df)}")

        bm25_ids = list(row.top_candidates_bm25)
        dense_ids = list(row.top_candidates_dense)
        union_ids = list(set(bm25_ids) | set(dense_ids))

        bm25_tree = build_tree(bm25_ids, hierarchy)
        dense_tree = build_tree(dense_ids, hierarchy)
        union_tree = build_tree(union_ids, hierarchy)

        if COMPUTE_EMBEDDING_COHERENCE:
            bm25_tree["embed_coherence"] = embed_coherence(bm25_ids, embeddings)
            dense_tree["embed_coherence"] = embed_coherence(dense_ids, embeddings)
            union_tree["embed_coherence"] = embed_coherence(union_ids, embeddings)
        else:
            bm25_tree["embed_coherence"] = dense_tree["embed_coherence"] = union_tree["embed_coherence"] = None

        # structural overlap of the two individual trees' neighbourhoods
        a, b = bm25_tree["allowed"], dense_tree["allowed"]
        if a or b:
            allowed_set_jaccard = len(a & b) / len(a | b) if (a | b) else None
        else:
            allowed_set_jaccard = None

        # fragmentation risk if candidates were simply pooled before pruning
        indiv_sizes = [s for s in (bm25_tree["subtree_size"], dense_tree["subtree_size"]) if s]
        indiv_roots = [r for r in (bm25_tree["n_roots"], dense_tree["n_roots"]) if r]
        union_size_ratio = (union_tree["subtree_size"] / max(indiv_sizes)
                             if indiv_sizes and union_tree["subtree_size"] else None)
        union_roots_ratio = (union_tree["n_roots"] / max(indiv_roots)
                              if indiv_roots and union_tree["n_roots"] else None)

        gt_concept_id = row.gold_concept_id

        rows.append({
            "note_id": row.note_id,
            "mention": row.mention,
            "gold_concept_id": gt_concept_id,
            "gold_label": row.gold_label,

            "bm25_subtree_size": bm25_tree["subtree_size"],
            "bm25_n_roots": bm25_tree["n_roots"],
            "bm25_root_depth_min": bm25_tree["root_depth_min"],
            "bm25_subtree_height": bm25_tree["subtree_height"],
            "bm25_candidate_density": bm25_tree["candidate_density"],
            "bm25_topcat_entropy": bm25_tree["topcat_entropy"],
            "bm25_embed_coherence": bm25_tree["embed_coherence"],
            "bm25_gt_in_subtree": gt_concept_id in bm25_tree["allowed"],

            "dense_subtree_size": dense_tree["subtree_size"],
            "dense_n_roots": dense_tree["n_roots"],
            "dense_root_depth_min": dense_tree["root_depth_min"],
            "dense_subtree_height": dense_tree["subtree_height"],
            "dense_candidate_density": dense_tree["candidate_density"],
            "dense_topcat_entropy": dense_tree["topcat_entropy"],
            "dense_embed_coherence": dense_tree["embed_coherence"],
            "dense_gt_in_subtree": gt_concept_id in dense_tree["allowed"],

            "union_subtree_size": union_tree["subtree_size"],
            "union_n_roots": union_tree["n_roots"],
            "union_root_depth_min": union_tree["root_depth_min"],
            "union_subtree_height": union_tree["subtree_height"],
            "union_candidate_density": union_tree["candidate_density"],
            "union_topcat_entropy": union_tree["topcat_entropy"],
            "union_embed_coherence": union_tree["embed_coherence"],
            "union_gt_in_subtree": gt_concept_id in union_tree["allowed"],

            "allowed_set_jaccard": allowed_set_jaccard,
            "union_size_ratio": union_size_ratio,
            "union_roots_ratio": union_roots_ratio,
        })

    tree_comparison_df = pd.DataFrame(rows)

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(exist_ok=True)
    tree_comparison_df.to_pickle(out_path)
    print(f"\nSaved tree comparison dataframe to {out_path}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n=== Mean metrics per strategy ===")
    metric_names = ["subtree_size", "n_roots", "root_depth_min", "subtree_height",
                     "candidate_density", "topcat_entropy", "embed_coherence"]
    summary = pd.DataFrame({
        "bm25": [tree_comparison_df[f"bm25_{m}"].median() for m in metric_names],
        "dense": [tree_comparison_df[f"dense_{m}"].median() for m in metric_names],
        "union": [tree_comparison_df[f"union_{m}"].median() for m in metric_names],
    }, index=metric_names)
    print(summary)

    print("\nGT reachable in subtree, by strategy:")
    print(f"  BM25 tree:  {tree_comparison_df['bm25_gt_in_subtree'].mean():.3f}")
    print(f"  Dense tree: {tree_comparison_df['dense_gt_in_subtree'].mean():.3f}")
    print(f"  Union tree: {tree_comparison_df['union_gt_in_subtree'].mean():.3f}")

    print("\nAllowed-set Jaccard (BM25 tree vs. SapBERT tree neighbourhood overlap):")
    print(tree_comparison_df["allowed_set_jaccard"].describe())

    print("\nUnion fragmentation risk (ratio > 1 means union tree is bigger than either alone):")
    print("  size ratio:")
    print(tree_comparison_df["union_size_ratio"].describe())
    print("  roots ratio:")
    print(tree_comparison_df["union_roots_ratio"].describe())

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ax, metric in zip(axes.flat, metric_names[:6]):
        data = [tree_comparison_df[f"{s}_{metric}"].dropna() for s in ["bm25", "dense", "union"]]
        ax.boxplot(data, labels=["BM25", "SapBERT", "Union"])
        ax.set_title(metric)

    plt.tight_layout()
    plt.savefig("cache/tree_comparison_metrics.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(tree_comparison_df["allowed_set_jaccard"].dropna(), bins=30)
    axes[0].set_title("BM25 vs SapBERT tree overlap (Jaccard)")
    axes[0].set_xlabel("Jaccard of allowed-node sets")

    axes[1].hist(tree_comparison_df["union_size_ratio"].dropna(), bins=30)
    axes[1].axvline(1.0, color="red", linestyle="--")
    axes[1].set_title("Union tree size / larger individual tree")
    axes[1].set_xlabel("ratio")

    axes[2].hist(tree_comparison_df["union_roots_ratio"].dropna(), bins=30)
    axes[2].axvline(1.0, color="red", linestyle="--")
    axes[2].set_title("Union tree roots / larger individual tree's roots")
    axes[2].set_xlabel("ratio")

    plt.tight_layout()
    plt.savefig("cache/tree_comparison_fragmentation.png", dpi=150)
    plt.show()

    if COMPUTE_EMBEDDING_COHERENCE:
        plt.figure(figsize=(7, 5))
        plt.scatter(tree_comparison_df["allowed_set_jaccard"],
                    tree_comparison_df["union_embed_coherence"], alpha=0.3)
        plt.xlabel("Structural overlap (allowed_set_jaccard)")
        plt.ylabel("Semantic coherence of union candidates (embed_coherence)")
        plt.title("Do structurally overlapping trees also agree semantically?")
        plt.tight_layout()
        plt.savefig("cache/tree_comparison_structural_vs_semantic.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
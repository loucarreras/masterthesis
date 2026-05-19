import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import pearsonr


# ============================================================
# HELPERS
# ============================================================
def attach_ground_truth(results, gt):

    enriched = []

    for r in results:

        note_id = r["note_id"]

        ground_truth = gt.get(note_id, [])

        match = match_entity(
            pred_entity=r,
            ground_truth=ground_truth
        )

        if match["matched"]:

            r["gt_id"] = match["ground_truth"]["concept_id"]

        else:

            r["gt_id"] = None

        enriched.append(r)

    return enriched


def get_level_reached(trace):

    count = 0

    for step in trace:
        if step["valid"]:
            count += 1
        else:
            break

    return count


# ============================================================
# ONTOLOGY METRICS
# ============================================================

def ancestor_overlap(pred_id, gt_id, snomed):

    if pred_id is None or gt_id is None:
        return np.nan

    pred_anc = snomed.get_all_ancestors(pred_id)
    gt_anc = snomed.get_all_ancestors(gt_id)

    pred_anc.add(pred_id)
    gt_anc.add(gt_id)

    intersection = pred_anc.intersection(gt_anc)

    return len(intersection) / max(len(gt_anc), 1)


def hierarchical_precision(pred_id, gt_id, snomed):

    if pred_id is None or gt_id is None:
        return np.nan

    pred_anc = snomed.get_all_ancestors(pred_id)
    gt_anc = snomed.get_all_ancestors(gt_id)

    pred_anc.add(pred_id)
    gt_anc.add(gt_id)

    intersection = pred_anc.intersection(gt_anc)

    return len(intersection) / max(len(pred_anc), 1)


def hierarchical_recall(pred_id, gt_id, snomed):

    if pred_id is None or gt_id is None:
        return np.nan

    pred_anc = snomed.get_all_ancestors(pred_id)
    gt_anc = snomed.get_all_ancestors(gt_id)

    pred_anc.add(pred_id)
    gt_anc.add(gt_id)

    intersection = pred_anc.intersection(gt_anc)

    return len(intersection) / max(len(gt_anc), 1)


def shortest_path_distance(pred_id, gt_id, snomed):

    if pred_id is None or gt_id is None:
        return np.nan

    pred_path = snomed.get_path_to_root(pred_id)
    gt_path = snomed.get_path_to_root(gt_id)

    common = 0

    for p, g in zip(pred_path, gt_path):

        if p == g:
            common += 1
        else:
            break

    distance = (
        (len(pred_path) - common)
        + (len(gt_path) - common)
    )

    return distance


def semantic_similarity(pred_id, gt_id, snomed):

    dist = shortest_path_distance(pred_id, gt_id, snomed)

    if np.isnan(dist):
        return np.nan

    return 1 / (1 + dist)


# ============================================================
# BUILD DATAFRAME
# ============================================================

def build_analysis_dataframe(results, snomed):

    rows = []

    for r in results:

        trace = r["trace"]

        pred_id = r["concept_id"]
        gt_id = r.get("gt_id")

        level_reached = r["level_reached"]

        failure_level = get_failure_level(trace)

        correct_prefix = get_correct_prefix(trace)

        candidate_counts = [
            step["candidate_count"]
            for step in trace
        ]

        avg_candidates = np.mean(candidate_counts)

        final_candidate_count = candidate_counts[-1]

        success = (
            pred_id is not None
            and gt_id is not None
            and pred_id == gt_id
        )

        gt_depth = len(snomed.get_path_to_root(gt_id))

        normalized_depth = (
            level_reached / gt_depth
            if gt_depth > 0
            else np.nan
        )

        rows.append({
            "term": r["term"],
            "note_id": r["note_id"],
            "predicted_id": pred_id,
            "gt_id": gt_id,
            "success": success,
            "reason": r["reason"],
            "level_reached": level_reached,
            "failure_level": failure_level,
            "correct_prefix": correct_prefix,
            "avg_candidates": avg_candidates,
            "final_candidate_count": final_candidate_count,
            "ancestor_overlap": ancestor_overlap(pred_id, gt_id, snomed),
            "hierarchical_precision": hierarchical_precision(pred_id, gt_id, snomed),
            "hierarchical_recall": hierarchical_recall(pred_id, gt_id, snomed),
            "shortest_path_distance": shortest_path_distance(pred_id, gt_id, snomed),
            "semantic_similarity": semantic_similarity(pred_id, gt_id, snomed),
            "gt_depth": gt_depth,
            "normalized_depth": normalized_depth
        })

    return pd.DataFrame(rows)


# ============================================================
# METRICS
# ============================================================

def print_global_metrics(df):

    total = len(df)

    correct = df["success"].sum()

    failed = total - correct

    accuracy = correct / total

    print("\n==========================")
    print("GLOBAL METRICS")
    print("==========================")

    print(f"Total entities: {total}")
    print(f"Correct entities: {correct}")
    print(f"Failed entities: {failed}")

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nHierarchical Metrics")

    print(f"Ancestor overlap: {df['ancestor_overlap'].mean():.4f}")
    print(f"Hierarchical precision: {df['hierarchical_precision'].mean():.4f}")
    print(f"Hierarchical recall: {df['hierarchical_recall'].mean():.4f}")
    print(f"Shortest path distance: {df['shortest_path_distance'].mean():.4f}")
    print(f"Semantic similarity: {df['semantic_similarity'].mean():.4f}")


# ============================================================
# FAILURE ANALYSIS
# ============================================================

def analyze_failure_nodes(results):

    nodes = []

    for r in results:

        for step in r["trace"]:

            if not step["valid"]:
                nodes.append(step["chosen_label"])
                break

    counter = Counter(nodes)

    failure_df = pd.DataFrame(
        counter.items(),
        columns=["node", "failures"]
    ).sort_values(
        "failures",
        ascending=False
    )

    return failure_df


# ============================================================
# PLOTS
# ============================================================

def plot_level_distribution(df):

    plt.figure(figsize=(8, 5))

    df["level_reached"].hist(bins=20)

    plt.xlabel("Level Reached")
    plt.ylabel("Count")
    plt.title("Hierarchy Level Distribution")

    plt.tight_layout()
    plt.show()


def plot_failure_levels(df):

    plt.figure(figsize=(8, 5))

    df["failure_level"].dropna().hist(bins=20)

    plt.xlabel("Failure Level")
    plt.ylabel("Count")
    plt.title("Failure Level Distribution")

    plt.tight_layout()
    plt.show()


def plot_shortest_path_distance(df):

    plt.figure(figsize=(8, 5))

    df["shortest_path_distance"].dropna().hist(bins=30)

    plt.xlabel("Shortest Path Distance")
    plt.ylabel("Count")
    plt.title("Ontology Distance Distribution")

    plt.tight_layout()
    plt.show()


def plot_semantic_similarity(df):

    plt.figure(figsize=(8, 5))

    df["semantic_similarity"].dropna().hist(bins=30)

    plt.xlabel("Semantic Similarity")
    plt.ylabel("Count")
    plt.title("Semantic Similarity Distribution")

    plt.tight_layout()
    plt.show()


def plot_candidate_failure(df):

    bins = [0, 5, 10, 20, 50, 100, 200, 500]

    df["candidate_bin"] = pd.cut(
        df["final_candidate_count"],
        bins=bins
    )

    failure_rate = (
        df.groupby("candidate_bin")["success"]
        .apply(lambda x: 1 - x.mean())
    )

    plt.figure(figsize=(8, 5))

    failure_rate.plot(kind="bar")

    plt.ylabel("Failure Rate")
    plt.xlabel("Candidate Count Bin")
    plt.title("Failure Rate vs Candidate Count")

    plt.tight_layout()
    plt.show()


def correlation_candidates_success(df):

    df["success_numeric"] = (
        df["success"].astype(int)
    )

    corr, p = pearsonr(
        df["final_candidate_count"],
        df["success_numeric"]
    )

    print("\n==========================")
    print("CORRELATION")
    print("==========================")

    print(f"Correlation: {corr:.4f}")
    print(f"P-value: {p:.6f}")
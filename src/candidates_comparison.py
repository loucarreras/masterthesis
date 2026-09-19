from pathlib import Path
from matplotlib.colors import LogNorm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LEXICAL_EVAL_PATH = "cache/lexical_eval_df_20.pkl"
DENSE_EVAL_PATH = "cache/dense_eval_df_20.pkl"
OUT_PATH = "cache/bm25_vs_sapbert_comparison_df.pkl"

TOP_N = 20  # both caches were built with top_n=20


def _add_dup_key(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_occurrence"] = df.groupby(["note_id", "mention", "gold_concept_id"]).cumcount()
    return df


def build_comparison_df(lexical_df: pd.DataFrame, dense_df: pd.DataFrame) -> pd.DataFrame:
    lexical_df = _add_dup_key(lexical_df)
    dense_df = _add_dup_key(dense_df)

    merged = lexical_df.merge(
        dense_df,
        on=["note_id", "mention", "gold_concept_id", "_occurrence"],
        how="outer",
        suffixes=("_bm25", "_dense"),
        indicator=True,
    )

    if (merged["_merge"] != "both").any():
        n_left_only = (merged["_merge"] == "left_only").sum()
        n_right_only = (merged["_merge"] == "right_only").sum()
        print(f"WARNING: {n_left_only} mentions only in BM25 eval_df, "
              f"{n_right_only} only in dense eval_df -- the two caches don't "
              f"cover the exact same matched mentions. Comparison below is "
              f"restricted to the {(merged['_merge'] == 'both').sum()} mentions "
              f"present in both.")
    merged = merged[merged["_merge"] == "both"].drop(columns="_merge")

    # gold_label / n_candidates / etc. exist on both sides with suffixes; keep one
    # canonical gold_label column.
    merged["gold_label"] = merged["gold_label_bm25"].fillna(merged["gold_label_dense"])

    merged["bm25_hit"] = merged["gt_among_top_candidates_bm25"]
    merged["dense_hit"] = merged["gt_among_top_candidates_dense"]

    def _outcome(row):
        if row["bm25_hit"] and row["dense_hit"]:
            return "both_hit"
        if row["bm25_hit"] and not row["dense_hit"]:
            return "bm25_only"
        if row["dense_hit"] and not row["bm25_hit"]:
            return "dense_only"
        return "both_miss"

    merged["outcome"] = merged.apply(_outcome, axis=1)

    # candidate-set overlap (Jaccard), independent of whether the gold concept
    # was actually retrieved -- tells you how correlated the two retrievers are
    def _jaccard(row):
        a, b = set(row["top_candidates_bm25"]), set(row["top_candidates_dense"])
        if not a and not b:
            return None
        return len(a & b) / len(a | b)

    merged["candidate_jaccard"] = merged.apply(_jaccard, axis=1)

    # union recall: would the gold concept have been retrieved if we pooled
    # both candidate lists? (a simple ensemble upper bound)
    merged["union_hit"] = merged["bm25_hit"] | merged["dense_hit"]

    return merged


def main():
    print(f"Loading BM25 eval_df from: {LEXICAL_EVAL_PATH}")
    lexical_df = pd.read_pickle(LEXICAL_EVAL_PATH)
    print(f"Loading dense eval_df from: {DENSE_EVAL_PATH}")
    dense_df = pd.read_pickle(DENSE_EVAL_PATH)

    comparison_df = build_comparison_df(lexical_df, dense_df)
    n = len(comparison_df)
    print(f"\nCompared {n} matched mentions present in both caches.")

    # --- headline recall comparison ---
    print("\n=== Recall@k: BM25 vs SapBERT ===")
    rows = []
    for k in [1, 3, 5, 10, 20]:
        bm25_r = (comparison_df["rank_bm25"] <= k).fillna(False).mean()
        dense_r = (comparison_df["rank_dense"] <= k).fillna(False).mean()
        union_r = (
            (comparison_df["rank_bm25"] <= k).fillna(False)
            | (comparison_df["rank_dense"] <= k).fillna(False)
        ).mean()
        rows.append({"k": k, "bm25_recall": bm25_r, "dense_recall": dense_r, "union_recall": union_r})
        print(f"  Recall@{k}:  BM25 {bm25_r:.4%}   SapBERT {dense_r:.4%}   Union {union_r:.4%}")
    recall_comparison_df = pd.DataFrame(rows)

    plt.figure(figsize=(8, 5))
    plt.plot(recall_comparison_df["k"], recall_comparison_df["bm25_recall"], marker="o", label="BM25")
    plt.plot(recall_comparison_df["k"], recall_comparison_df["dense_recall"], marker="o", label="SapBERT")
    plt.plot(recall_comparison_df["k"], recall_comparison_df["union_recall"], marker="o",
             linestyle="--", label="Union (either retrieves)")
    plt.xlabel("k")
    plt.ylabel("Recall@k")
    plt.title("Candidate generation: BM25 vs SapBERT vs Union")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    Path("cache").mkdir(exist_ok=True)
    plt.savefig("cache/bm25_vs_sapbert_recall_curve.png")
    print("\nSaved comparison recall curve to cache/bm25_vs_sapbert_recall_curve.png")

    # --- MRR comparison ---
    for name, col in [("BM25", "rank_bm25"), ("SapBERT", "rank_dense")]:
        hits = comparison_df[comparison_df[col].notna()]
        mrr = (1.0 / hits[col]).mean() if len(hits) else float("nan")
        print(f"\n{name} MRR: {mrr:.4f} (over {len(hits)} hits)")

    # --- where do they agree / disagree? ---
    print("\n=== Outcome breakdown (did each retriever surface the gold concept?) ===")
    outcome_counts = comparison_df["outcome"].value_counts()
    print(outcome_counts)
    print((outcome_counts / n).apply(lambda x: f"{x:.1%}"))

    plt.figure(figsize=(6, 5))
    outcome_counts.reindex(["both_hit", "bm25_only", "dense_only", "both_miss"]).plot(kind="bar")
    plt.title("Which retriever(s) recovered the gold concept?")
    plt.ylabel("# mentions")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("cache/bm25_vs_sapbert_outcomes.png")
    plt.show()

    # --- candidate set overlap (agreement), independent of hit/miss ---
    print("\nCandidate-set Jaccard overlap (top_candidates_bm25 vs top_candidates_dense):")
    print(comparison_df["candidate_jaccard"].describe())

    plt.figure(figsize=(8, 5))
    plt.hist(comparison_df["candidate_jaccard"].dropna(), bins=30)
    plt.xlabel("Jaccard overlap of top-20 candidate sets")
    plt.ylabel("# mentions")
    plt.title("How much do BM25 and SapBERT candidate lists overlap?")
    plt.tight_layout()
    plt.savefig("cache/bm25_vs_sapbert_jaccard_hist.png")
    plt.show()

    # --- cases where the two disagree, for manual inspection ---
    bm25_only_df = comparison_df[comparison_df["outcome"] == "bm25_only"]
    dense_only_df = comparison_df[comparison_df["outcome"] == "dense_only"]
    both_miss_df = comparison_df[comparison_df["outcome"] == "both_miss"]

    print(f"\n=== BM25 hit, SapBERT missed: {len(bm25_only_df)} ===")
    if len(bm25_only_df):
        print(bm25_only_df[["mention", "gold_label", "rank_bm25"]]
              .sample(min(15, len(bm25_only_df)), random_state=0).to_string(index=False))

    print(f"\n=== SapBERT hit, BM25 missed: {len(dense_only_df)} ===")
    if len(dense_only_df):
        print(dense_only_df[["mention", "gold_label", "rank_dense"]]
              .sample(min(15, len(dense_only_df)), random_state=0).to_string(index=False))

    print(f"\n=== Both missed: {len(both_miss_df)} ({len(both_miss_df) / n:.1%}) ===")
    if len(both_miss_df):
        print(both_miss_df[["mention", "gold_label"]]
              .sample(min(15, len(both_miss_df)), random_state=0).to_string(index=False))

    # --- rank correlation for mentions both retrievers found ---
    both_hit_df = comparison_df[comparison_df["outcome"] == "both_hit"]
    if len(both_hit_df) > 1:
        corr = both_hit_df["rank_bm25"].corr(both_hit_df["rank_dense"], method="spearman")
        print(f"\nSpearman rank correlation (rank_bm25 vs rank_dense) on {len(both_hit_df)} "
              f"double-hit mentions: {corr:.3f}")
 
        bins = range(1, TOP_N + 2)  # integer bin edges 1..TOP_N+1 -> one bin per rank value
        plt.figure(figsize=(6.5, 6))
        h = plt.hist2d(
            both_hit_df["rank_bm25"], both_hit_df["rank_dense"],
            bins=[bins, bins], cmap="viridis", norm = LogNorm(),
        )
        plt.colorbar(h[3], label="# mentions")
        plt.xlabel("BM25 rank")
        plt.xticks(bins)
        plt.ylabel("SapBERT rank")
        plt.yticks(bins)
        plt.title("Rank agreement where both retrievers hit (n="
                   f"{len(both_hit_df)})")
        plt.gca().invert_yaxis()  # rank 1 (best) at top, matches reading order of the scatter version
        plt.tight_layout()
        plt.savefig("cache/bm25_vs_sapbert_rank_heatmap.png")
        plt.show()


    counts, xedges, yedges = np.histogram2d(
        both_hit_df["rank_bm25"],
        both_hit_df["rank_dense"],
        bins=[bins, bins],
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        counts.T,
        origin="upper",
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(1, counts.max())),
    )

    # Add count to each non-empty cell
    for y in range(TOP_N):
        for x in range(TOP_N):
            if counts[x, y] > 0:
                ax.text(
                    x, y,
                    f"{int(counts[x, y])}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    ax.set_xticks(range(TOP_N))
    ax.set_xticklabels(range(1, TOP_N + 1))
    ax.set_yticks(range(TOP_N))
    ax.set_yticklabels(range(1, TOP_N + 1))

    ax.set_xlabel("BM25 rank")
    ax.set_ylabel("SapBERT rank")
    ax.set_title(
        f"Rank agreement where both retrievers hit (n={len(both_hit_df)})"
    )

    fig.colorbar(im, ax=ax, label="# mentions (log scale)")
    plt.tight_layout()
    plt.savefig(
        "cache/bm25_vs_sapbert_rank_heatmap.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.show()    

    comparison_df.to_pickle(OUT_PATH)
    print(f"\nSaved full comparison dataframe to {OUT_PATH}")


if __name__ == "__main__":
    main()
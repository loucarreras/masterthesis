from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from evaluation.metrics import token_jaccard
from ontology.snomed_loader import SNOMEDHierarchy
from evaluation.classification_metrics import (
    build_analysis_dataframe,
    plot_categorical_distribution,
    plot_correlation,
    plot_distribution,
)
import pickle
from scipy.stats import friedmanchisquare, wilcoxon

SNOMED_CONCEPT_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Concept_Snapshot_INT_20260501.txt"
SNOMED_RELATIONSHIP_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Relationship_Snapshot_INT_20260501.txt"
SNOMED_DESCRIPTION_FILE = r"..\data\SnomedCT_InternationalRF2_PRODUCTION_20260501T120000Z\Snapshot\Terminology\sct2_Description_Snapshot-en_INT_20260501.txt"

ANNOTATIONS_PATH = r"..\data\snomed-ct-entity-linking-challenge-1.2.1\train_annotations.csv"

CONTEXT_MODES = ["comparison", "gtner_comparison"]

NER_MODEL = "en_core_sci_sm"
MODEL = "medgemma1_5"

CLASSIFICATION_RESULTS_PATHS = {
    mode: f"llm_classification_{NER_MODEL}_level_{MODEL}_{mode}.json" for mode in CONTEXT_MODES
}

CACHE_PATH = f"cache/note_keywords_{NER_MODEL}.pkl"


def load_analysis_df_for_mode(mode, snomed, gt):
    results_path = CLASSIFICATION_RESULTS_PATHS[mode]
    cache_path = CACHE_PATH

    print(f"Loading classification results for mode '{mode}' from: {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        classif_results = json.load(f)

    with open(cache_path, "rb") as f:
        note_keywords = pickle.load(f)

    df = build_analysis_dataframe(classif_results, snomed, gt, note_keywords)
    df["key"] = list(zip(df["note_id"], df["start"], df["end"]))
    df["failed_at_level_1"] = df["level_reached"] == 1

    # Columns to analyze in the following analysis
    metric_cols = [
        "gt_reached",
        "null_prediction",
        "stopped_at_parent",
        "failed_at_level_1",
        "level_reached",
        "distance_to_gt",
        "correct_prefix_ratio"
    ]
    renamed = {col: f"{col}_{mode}" for col in metric_cols}
    return df[["key"] + metric_cols].rename(columns=renamed)

def print_contingency_tables(merged, success_cols):
    
    print("\nContingency tables:")
    for i, a in enumerate(success_cols):
        for b in success_cols[i + 1:]:
            table = pd.crosstab(merged[a], merged[b])

            # Ensure both True/False are present even if one cell is empty
            table = table.reindex(index=[False, True], columns=[False, True], fill_value=0)

            both_fail = table.loc[False, False]
            both_success = table.loc[True, True]
            a_only = table.loc[True, False]   # succeeded in a, failed in b
            b_only = table.loc[False, True]   # failed in a, succeeded in b

            print(f"\n  {a} vs {b}:")
            print(f"    Both succeeded:        {both_success}")
            print(f"    Both failed:           {both_fail}")
            print(f"    Succeeded in {a} only: {a_only}")
            print(f"    Succeeded in {b} only: {b_only}")
            print(f"    Total discordant pairs: {a_only + b_only}")
            if (a_only + b_only) > 0:
                print(f"    Net direction: {'favors ' + b if b_only > a_only else 'favors ' + a if a_only > b_only else 'tied'}")

def run_pairwise_stats(merged, metric_name, type, failures_only=False):
    cols = [f"{metric_name}_{mode}" for mode in CONTEXT_MODES]

    df = merged
    if failures_only:
        # restrict to entities that failed in ALL conditions, so the metric is defined everywhere
        fail_mask = ~merged[[f"gt_reached_{mode}" for mode in CONTEXT_MODES]].any(axis=1)
        df = merged[fail_mask]

    df = df.dropna(subset=cols)

    print(f"\n{'='*60}")
    print(f"Metric: {metric_name}" + (" (failures only)" if failures_only else ""))
    print(f"N = {len(df)}")
    print(f"{'='*60}")

    if type == "binary":
    
        print(f"\nRate per context mode:")
        print(df[cols].mean())

        q_result = cochrans_q(df[cols].astype(int))
        print(f"\nCochran's Q test: statistic={q_result.statistic:.4f}, p-value={q_result.pvalue:.4f}")

        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
        pvals = []
        for a, b in pairs:
            table = pd.crosstab(df[a], df[b])
            pvals.append(mcnemar(table, exact=True).pvalue)

        _, pvals_corrected, _, _ = multipletests(pvals, method="bonferroni")

        print("\nPairwise McNemar tests (Bonferroni-corrected):")
        for (a, b), p_raw, p_corr in zip(pairs, pvals, pvals_corrected):
            sig = "significant" if p_corr < 0.05 else "not significant"
            print(f"  {a} vs {b}: raw p={p_raw:.4f}, corrected p={p_corr:.4f} ({sig})")

        print_contingency_tables(merged, cols)

    if type == "continuous":
        
        print("\nMean per context mode:")
        print(df[cols].mean())
        print("\nMedian per context mode:")
        print(df[cols].median())

        if len(df) < 3:
            print("Not enough matched samples to run tests.")
            return

        if len(cols) >= 3:
            stat, p = friedmanchisquare(*[df[c] for c in cols])
            print(f"\nFriedman test: statistic={stat:.4f}, p-value={p:.4f}")
        else:
            print("\nFriedman test requires at least 3 conditions; skipping.")

        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
        pvals = []
        for a, b in pairs:
            stat_w, p_w = wilcoxon(df[a], df[b])
            pvals.append(p_w)

        _, pvals_corrected, _, _ = multipletests(pvals, method="bonferroni")

        print("\nPairwise Wilcoxon signed-rank tests (Bonferroni-corrected):")
        for (a, b), p_raw, p_corr in zip(pairs, pvals, pvals_corrected):
            sig = "significant" if p_corr < 0.05 else "not significant"
            median_diff = (df[a] - df[b]).median()
            mean_diff = (df[a] - df[b]).mean()
            print(f"  {a} vs {b}: raw p={p_raw:.4f}, corrected p={p_corr:.4f} ({sig}), median diff={median_diff:.4f}, mean diff={mean_diff:.4f}")

    

def compare_context_modes(snomed, gt):
    dfs = [load_analysis_df_for_mode(mode, snomed, gt) for mode in CONTEXT_MODES]

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="key", how="inner")

    print(f"\nMatched entities across all {len(CONTEXT_MODES)} context modes: {len(merged)}")
    for mode, df in zip(CONTEXT_MODES, dfs):
        print(f"  (originally {len(df)} in '{mode}')")

    metrics_to_compare = ["gt_reached", "null_prediction", "stopped_at_parent", "failed_at_level_1"]
    for metric in metrics_to_compare:
        run_pairwise_stats(merged, metric, type ="binary", failures_only=False)

    continuous_metrics = ["level_reached", "distance_to_gt", "correct_prefix_ratio"]
    for metric in continuous_metrics:
        run_pairwise_stats(merged, metric, type="continuous", failures_only=False)

    return merged

def main():
    
    print("Loading SNOMED CT hierarchy...")
    snomed = SNOMEDHierarchy(
        concept_file=SNOMED_CONCEPT_FILE,
        relationship_file=SNOMED_RELATIONSHIP_FILE,
        description_file=SNOMED_DESCRIPTION_FILE,
    )
    print("SNOMED CT hierarchy loaded.")

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

    comparison_df = compare_context_modes(snomed, gt)

    return comparison_df


if __name__ == "__main__":
    main()
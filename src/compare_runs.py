"""
compare_runs.py

Compares N repeated runs of the SNOMED LLM classification pipeline to measure
how stochastic vs. deterministic the LLM's concept assignments are.

Loads files named like:
    llm_classification_en_core_sci_sm_level_one_shot_attempt1.csv
    llm_classification_en_core_sci_sm_level_one_shot_attempt2.csv
    ...
Each file must contain at least: note_id, start, end, concept_id, gt_id.
(term, category, reason, candidate_ids are ignored here -- they're not needed
for the stability/agreement metrics.)

Filters every run down to the same reference set of N_NOTES notes (some runs
may have been accidentally run on more notes than others), aligns mentions by
(note_id, start, end), and computes:

  1. Exact-match agreement rate (% of mentions where all runs agree)
  2. Per-mention Shannon entropy of the concept_id distribution (raw + normalized)
  3. Majority-vote deviation rate
  4. Majority-vote accuracy vs. gt_id (does "ensembling" the runs beat a single run?)
  5. Fleiss' kappa (overall multi-rater agreement)
  6. Pairwise Cohen's kappa between every pair of runs (+ heatmap)
  7. Hardest mentions (highest entropy) for manual inspection
  8. Per-run accuracy vs. gt_id, mean/std across runs

Usage:
    python compare_runs.py \
        --pattern "llm_classification_en_core_sci_sm_level_one_shot_attempt{i}" \
        --runs 1-10 \
        --n-notes 5 \
        --out-dir analysis_output
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy

try:
    from sklearn.metrics import cohen_kappa_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

COLUMNS = ["note_id", "start", "end", "concept_id", "gt_id"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def parse_run_spec(spec: str) -> list[int]:
    """Parse '1-10' or '1,2,3' or '1-5,8,10' into a list of ints."""
    runs = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            runs.extend(range(int(lo), int(hi) + 1))
        else:
            runs.append(int(part))
    return runs


def load_run(pattern: str, run_id: int) -> pd.DataFrame | None:
    """Load a single run's predictions from .csv (preferred) or .json."""
    base = pattern.format(i=run_id)
    csv_path = f"{base}.csv"
    json_path = f"{base}.json"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, dtype={"note_id": str, "concept_id": str, "gt_id": str})
    elif os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        records = raw["results"] if isinstance(raw, dict) and "results" in raw else raw
        df = pd.DataFrame.from_records(records)
        for col in ("note_id", "concept_id", "gt_id"):
            if col in df.columns:
                df[col] = df[col].astype(str)
    else:
        print(f"  [warn] no file found for run {run_id} (looked for {csv_path} / {json_path})")
        return None

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"run {run_id}: missing columns {missing}")
    return df[COLUMNS].copy()


def load_all_runs(pattern: str, run_ids: list[int]) -> dict[int, pd.DataFrame]:
    runs = {}
    for r in run_ids:
        df = load_run(pattern, r)
        if df is not None:
            runs[r] = df
            print(f"  run {r}: {len(df)} rows, {df['note_id'].nunique()} unique notes")
    if len(runs) < 2:
        sys.exit("Need at least 2 successfully loaded runs to compare. Aborting.")
    return runs


# --------------------------------------------------------------------------
# Restrict every run to the same first-N notes
# --------------------------------------------------------------------------

def find_target_notes(runs: dict[int, pd.DataFrame], n_notes: int) -> tuple[set, int]:
    """
    Use the run with the fewest unique notes (<= n_notes if possible) as the
    reference for "the first N notes", preserving row order of first
    appearance within that run's file.
    """
    counts = {r: df["note_id"].nunique() for r, df in runs.items()}
    small_enough = [r for r, c in counts.items() if c <= n_notes]
    ref_run = min(small_enough, key=lambda r: counts[r]) if small_enough else min(counts, key=lambda r: counts[r])

    ordered_notes = list(dict.fromkeys(runs[ref_run]["note_id"]))
    if len(ordered_notes) < n_notes:
        print(f"  [warn] reference run {ref_run} only has {len(ordered_notes)} notes (< {n_notes} requested)")
    target = set(ordered_notes[:n_notes])
    print(f"  Using run {ref_run} as reference. Target notes ({len(target)}): {sorted(target)}")
    return target, ref_run


def restrict_runs(runs: dict[int, pd.DataFrame], target_notes: set) -> dict[int, pd.DataFrame]:
    restricted = {}
    for r, df in runs.items():
        before = df["note_id"].nunique()
        filt = df[df["note_id"].isin(target_notes)].copy()
        after = filt["note_id"].nunique()
        if after < len(target_notes):
            missing = target_notes - set(filt["note_id"])
            print(f"  [warn] run {r}: only found {after}/{len(target_notes)} target notes; missing {missing}")
        elif before != after:
            print(f"  run {r}: trimmed from {before} to {after} notes")
        restricted[r] = filt
    return restricted


# --------------------------------------------------------------------------
# Align mentions across runs
# --------------------------------------------------------------------------

def build_wide_table(runs: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns:
      wide: DataFrame indexed by (note_id, start, end), one column per run_id
            holding that run's concept_id, restricted to mentions present in
            every run.
      gt:   Series indexed the same way, holding gt_id for each mention
            (taken from whichever run has it -- gt shouldn't vary by run).
    """
    frames = []
    gt_frames = []
    for r, df in runs.items():
        d = df.copy()
        d["start"] = d["start"].round().astype(int)
        d["end"] = d["end"].round().astype(int)
        d = d.drop_duplicates(subset=["note_id", "start", "end"])
        pred = d.rename(columns={"concept_id": r}).set_index(["note_id", "start", "end"])[[r]]
        frames.append(pred)
        gt_frames.append(d.set_index(["note_id", "start", "end"])["gt_id"])

    wide = pd.concat(frames, axis=1)
    n_total = len(wide)
    wide_complete = wide.dropna(how="any")
    n_complete = len(wide_complete)
    print(f"  {n_total} unique mentions found across runs; {n_complete} present in ALL runs "
          f"({n_total - n_complete} dropped due to partial coverage)")

    gt_all = pd.concat(gt_frames)
    gt_all = gt_all[~gt_all.index.duplicated(keep="first")]
    n_gt_conflicts = 0
    for gtf in gt_frames:
        common = gtf.index.intersection(gt_all.index)
        n_gt_conflicts += (gtf.loc[common] != gt_all.loc[common]).sum()
    if n_gt_conflicts:
        print(f"  [warn] {n_gt_conflicts} mention(s) have a gt_id that differs across runs -- "
              f"using the first run's value.")
    gt = gt_all.reindex(wide_complete.index)

    return wide_complete, gt


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def per_mention_metrics(wide: pd.DataFrame, gt: pd.Series) -> pd.DataFrame:
    run_cols = list(wide.columns)
    n_runs = len(run_cols)
    max_entropy = np.log2(n_runs) if n_runs > 1 else 1.0

    records = []
    for idx, row in wide.iterrows():
        counts = row.value_counts()
        mode_val = counts.idxmax()
        mode_count = counts.max()
        probs = counts / counts.sum()
        ent = shannon_entropy(probs, base=2)
        records.append({
            "note_id": idx[0],
            "start": idx[1],
            "end": idx[2],
            "gt_id": gt.loc[idx],
            "n_runs": n_runs,
            "n_distinct_answers": len(counts),
            "mode_concept_id": mode_val,
            "mode_count": mode_count,
            "agreement_rate": mode_count / n_runs,
            "is_unanimous": len(counts) == 1,
            "entropy_bits": ent,
            "entropy_normalized": ent / max_entropy if max_entropy > 0 else 0.0,
            "majority_vote_correct": mode_val == gt.loc[idx],
        })
    return pd.DataFrame(records).sort_values("entropy_bits", ascending=False).reset_index(drop=True)


def fleiss_kappa(wide: pd.DataFrame) -> float:
    """
    Manual Fleiss' kappa over the wide (subjects x raters) table.

    Caveat: Fleiss' kappa assumes a shared, fixed category set across all
    subjects. Here the "categories" are SNOMED concept_ids drawn from a huge,
    largely mention-specific vocabulary, so the chance-agreement term (P_e)
    will usually be tiny and kappa will sit very close to the raw agreement
    rate. Treat it as a rough overall summary, not a rigorous chance-corrected
    statistic -- the per-mention agreement/entropy numbers are more reliable.
    """
    run_cols = list(wide.columns)
    n_raters = len(run_cols)
    categories = pd.unique(wide.values.ravel())
    cat_index = {c: i for i, c in enumerate(categories)}
    N, k = len(wide), len(categories)
    count_matrix = np.zeros((N, k), dtype=int)

    for row_i, (_, row) in enumerate(wide.iterrows()):
        for val in row.values:
            count_matrix[row_i, cat_index[val]] += 1

    P_i = (np.sum(count_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()

    p_j = count_matrix.sum(axis=0) / (N * n_raters)
    P_e_bar = np.sum(p_j ** 2)

    if P_e_bar == 1.0:
        return 1.0
    return (P_bar - P_e_bar) / (1 - P_e_bar)


def pairwise_kappa_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    run_cols = list(wide.columns)
    mat = pd.DataFrame(index=run_cols, columns=run_cols, dtype=float)
    for a, b in itertools.combinations(run_cols, 2):
        k = cohen_kappa_score(wide[a], wide[b]) if HAVE_SKLEARN else np.nan
        mat.loc[a, b] = k
        mat.loc[b, a] = k
    for r in run_cols:
        mat.loc[r, r] = 1.0
    return mat


def pairwise_exact_match_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    run_cols = list(wide.columns)
    mat = pd.DataFrame(index=run_cols, columns=run_cols, dtype=float)
    for a, b in itertools.combinations(run_cols, 2):
        match_rate = (wide[a] == wide[b]).mean()
        mat.loc[a, b] = match_rate
        mat.loc[b, a] = match_rate
    for r in run_cols:
        mat.loc[r, r] = 1.0
    return mat


def plot_heatmap(mat: pd.DataFrame, title: str, out_path: Path):
    if not HAVE_MPL:
        print(f"  [warn] matplotlib not available, skipping {out_path.name}")
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(mat)), max(4, 0.6 * len(mat))))
    im = ax.imshow(mat.values.astype(float), vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            ax.text(j, i, f"{mat.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if mat.values[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def per_run_accuracy(runs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Per-run accuracy using the gt_id column already present in each run's file."""
    rows = []
    for r, df in runs.items():
        acc = (df["concept_id"] == df["gt_id"]).mean() if len(df) else np.nan
        rows.append({"run": r, "n_mentions": len(df), "accuracy": acc})
    return pd.DataFrame(rows).sort_values("run")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", default="llm_classification_en_core_sci_sm_level_one_shot_attempt{i}",
                     help="Filename pattern with {i} for the run number (no extension).")
    ap.add_argument("--runs", default="1-10", help="Run numbers, e.g. '1-10' or '1,2,3,5-8'.")
    ap.add_argument("--n-notes", type=int, default=5, help="Number of notes to restrict every run to.")
    ap.add_argument("--out-dir", default="analysis_output", help="Directory to write reports/plots to.")
    ap.add_argument("--top-k", type=int, default=20, help="How many highest-entropy mentions to report.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ids = parse_run_spec(args.runs)

    print(f"Loading {len(run_ids)} runs matching pattern '{args.pattern}'...")
    runs = load_all_runs(args.pattern, run_ids)

    print(f"\nFinding reference set of first {args.n_notes} notes...")
    target_notes, ref_run = find_target_notes(runs, args.n_notes)

    print("\nRestricting every run to the target notes...")
    runs = restrict_runs(runs, target_notes)

    print("\nAligning mentions across runs (note_id, start, end)...")
    wide, gt = build_wide_table(runs)
    if len(wide) == 0:
        sys.exit("No mentions are present in all runs after alignment. Nothing to compare.")

    wide.to_csv(out_dir / "wide_predictions.csv")

    print("\nComputing per-mention agreement/entropy/majority-vote accuracy...")
    per_mention = per_mention_metrics(wide, gt)
    per_mention.to_csv(out_dir / "per_mention_metrics.csv", index=False)

    unanimous_rate = per_mention["is_unanimous"].mean()
    mean_agreement_rate = per_mention["agreement_rate"].mean()
    mean_entropy = per_mention["entropy_bits"].mean()
    mean_entropy_norm = per_mention["entropy_normalized"].mean()
    majority_vote_acc = per_mention["majority_vote_correct"].mean()

    print("\nComputing Fleiss' kappa (overall multi-rater agreement)...")
    fk = fleiss_kappa(wide)

    print("\nComputing pairwise run agreement (Cohen's kappa + exact-match %)...")
    kappa_mat = pairwise_kappa_matrix(wide)
    match_mat = pairwise_exact_match_matrix(wide)
    kappa_mat.to_csv(out_dir / "pairwise_cohen_kappa.csv")
    match_mat.to_csv(out_dir / "pairwise_exact_match.csv")
    plot_heatmap(kappa_mat, "Pairwise Cohen's kappa between runs", out_dir / "pairwise_cohen_kappa.png")
    plot_heatmap(match_mat, "Pairwise exact-match rate between runs", out_dir / "pairwise_exact_match.png")

    hardest = per_mention.head(args.top_k)
    hardest.to_csv(out_dir / f"hardest_mentions_top{args.top_k}.csv", index=False)

    print("\nComputing per-run accuracy vs gt_id...")
    acc_df = per_run_accuracy(runs)
    acc_df.to_csv(out_dir / "per_run_accuracy.csv", index=False)

    # -------------------- summary --------------------
    summary_lines = [
        "=" * 60,
        "RUN COMPARISON SUMMARY",
        "=" * 60,
        f"Runs compared:              {sorted(runs.keys())}",
        f"Target notes ({args.n_notes}):          {sorted(target_notes)}",
        f"Mentions compared:          {len(wide)}",
        "",
        f"Unanimous agreement rate:   {unanimous_rate:.1%}  (all {len(runs)} runs picked the same concept)",
        f"Mean per-mention agreement: {mean_agreement_rate:.1%}  (share of runs matching the mode answer)",
        f"Mean entropy:               {mean_entropy:.3f} bits raw / {mean_entropy_norm:.3f} normalized "
        f"(0 = fully deterministic, 1 normalized = maximally split)",
        f"Fleiss' kappa:              {fk:.3f}  (see caveat in fleiss_kappa() docstring re: large label space)",
        "",
        "Pairwise Cohen's kappa (mean over all run pairs): "
        f"{kappa_mat.values[np.triu_indices_from(kappa_mat.values, k=1)].mean():.3f}",
        "Pairwise exact-match rate (mean over all run pairs): "
        f"{match_mat.values[np.triu_indices_from(match_mat.values, k=1)].mean():.1%}",
        "",
        f"Per-run accuracy vs gt_id:  mean = {acc_df['accuracy'].mean():.1%}   std = {acc_df['accuracy'].std():.1%}",
        f"Majority-vote accuracy:     {majority_vote_acc:.1%}  "
        f"({'beats' if majority_vote_acc > acc_df['accuracy'].mean() else 'does not beat'} "
        f"the average single run -- tells you whether ensembling runs would help)",
        "",
        f"Top {args.top_k} highest-entropy (most unstable) mentions written to "
        f"hardest_mentions_top{args.top_k}.csv",
        "",
        f"All outputs written to: {out_dir.resolve()}",
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")


if __name__ == "__main__":
    main()
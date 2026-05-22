import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import linregress
from ontology.snomed_loader import SNOMEDHierarchy

def build_analysis_dataframe(results, snomed):
    
    rows = []

    for r in results:

        trace = r.get("trace", [])
        gt_id = r.get("gt_id")
        pred_id = r.get("concept_id")
        level_reached = r.get("level_reached")
        candidate_count = trace[-1].get("candidate_count")

        success = gt_id == pred_id

        # GT paths evaluation
        gt_paths = snomed.get_all_ancestor_paths(gt_id)

        filtered_paths = gt_paths.copy()
        path_predicted = []

        for step in trace[:-1]:
            predicted_step = step.get("chosen_id")
            level = step.get("level")

            if predicted_step is not None:
                path_predicted.append(predicted_step)
            
            filtered_paths = [
                path for path in filtered_paths
                if len(path) > level and path[level] == predicted_step
            ]

        path_predicted.append(pred_id) # Predicted path

        gt_depth = min(len(p) for p in filtered_paths) if filtered_paths else 0
            
        gt_reached = False
        if gt_id in path_predicted:
            gt_reached = True
            

        distance = None
        if pred_id and gt_id:
            distance = gt_depth - level_reached

        correct_prefix_ratio = level_reached / gt_depth if gt_depth else 0

        rows.append({

            "note_id": r.get("note_id"),
            "term": r.get("term"),

            "gt_id": gt_id,
            "gt_label": snomed.get_label(gt_id) if gt_id else None,

            "predicted_id": pred_id,
            "predicted_label": snomed.get_label(pred_id) if pred_id else None,

            "success": success,
            "gt_reached": gt_reached,

            "level_reached": level_reached,
            "gt_depth": gt_depth,

            "num_remaining_paths": len(filtered_paths),

            # "failure_level": failure_level,

            "distance_to_gt": distance,

            "candidate_count": candidate_count,

            "null_prediction": pred_id is None,

            # "ancestor_overlap": ancestor_overlap,

            "correct_prefix_ratio": correct_prefix_ratio
        })

    return pd.DataFrame(rows)

def plot_categorical_distribution(df, column):
    values = df[column]

    plt.figure(figsize=(8,5))
    plt.hist(values, bins=range(0, values.max() + 1))
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    plt.show()

def plot_correlation(df, var1, var2):

    # Compute correlation coefficient
    correlation = df[[var1, var2]].corr().iloc[0, 1]
    print(f"Correlation between {var1} and {var2}: {correlation:.2f}")

    x = df[var1].values
    y = df[var2].values

    # Remove NaNs (important in real datasets)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # Scatter plot
    plt.scatter(x, y, alpha=0.6)

    # Fit regression line (1st degree polynomial)
    m, b = np.polyfit(x, y, 1)
    y_line = m * x + b

    # Sort for clean line drawing
    sorted_idx = np.argsort(x)

    plt.plot(x[sorted_idx], y_line[sorted_idx], color='red', linewidth=2)

    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f"Correlation between {var1} and {var2}")

    plt.show()

def plot_distribution(df, var):

    plt.figure(figsize=(8,5))
    plt.violinplot(df[var])
    plt.xlabel(var)
    plt.title(f"Distribution of {var}")
    plt.show()
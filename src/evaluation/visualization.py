# evaluation/visualization.py

import matplotlib.pyplot as plt
import numpy as np
from spacy import displacy
import os

def save_plot(plot_name, model_name, threshold, results_dir="results"):
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{plot_name}_{model_name}_{threshold:.2f}.png"
    path = os.path.join(results_dir, filename)
    
    return path

def plot_confusion_matrix(tp, fp, fn, threshold=0.5, model_name="scispacy"):
    cm = np.array([
        [tp, fp],
        [fn, 0]
    ])

    fig, ax = plt.subplots()

    im = ax.imshow(cm, cmap="Reds")

    # Add colorbar (scale reference)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(["GT Positive", "GT Negative"])
    ax.set_yticklabels(["Pred Positive", "Pred Negative"])

    # Annotate values with contrast-aware text color
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = "white" if value > cm.max() / 2 else "black"
            ax.text(j, i, value, ha="center", va="center", color=color)

    ax.set_title("Span Detection Confusion Matrix")
    plt.tight_layout()
    path = save_plot("confusion_matrix", model_name, threshold)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_iou_distribution(ious, threshold=0.5, model_name="scispacy"):
    plt.figure()

    plt.hist(ious, bins=20, alpha=0.7)

    # Threshold line
    plt.axvline(threshold, linestyle="--", linewidth=2)
    
    # Optional shading (accepted region)
    plt.axvspan(threshold, 1.0, alpha=0.1)

    plt.title("IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Frequency")

    plt.text(threshold + 0.02, plt.ylim()[1]*0.9, f"Threshold = {threshold}")

    plt.tight_layout()
    path = save_plot("iou_distribution", model_name, threshold)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_precision_recall_curve(thresholds, precisions, recalls, selected_t=0.5, model_name="scispacy"):
    f1s = [
        2*p*r/(p+r) if (p+r) > 0 else 0
        for p, r in zip(precisions, recalls)
        ]
    
    plt.figure()

    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1 Score", linestyle="--")

    # Mark selected threshold
    plt.axvline(selected_t, linestyle="--")

    # Find closest index
    idx = np.argmin(np.abs(thresholds - selected_t))
    plt.scatter([selected_t], [precisions[idx]])
    plt.scatter([selected_t], [recalls[idx]])

    plt.xlabel("IoU Threshold")
    plt.ylabel("Score")
    plt.title("Precision/Recall vs IoU Threshold")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    path = save_plot("precision_recall_curve", model_name, selected_t)
    plt.savefig(path, dpi=300)
    plt.close()


def visualize_spans(text, predicted, ground_truth):
    ents = []

    for p in predicted:
        ents.append({
            "start": p["start"],
            "end": p["end"],
            "label": "PRED"
        })

    for g in ground_truth:
        ents.append({
            "start": g["start"],
            "end": g["end"],
            "label": "GT"
        })

    doc = {"text": text, "ents": ents}
    displacy.render(doc, style="ent", manual=True)
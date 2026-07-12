"""
reporting.py
------------
Generates the figures and text artifacts a project report/presentation
actually needs: training/validation accuracy & loss curves, and a
confusion matrix heatmap -- saved to reports/figures/, plus a plain-text
summary to reports/results.txt.

Nothing here computes or estimates a number; it only plots/writes numbers
that were already produced by a real training/evaluation run in
hybrid_train.py or cross_validate.py. Whatever appears in these files is
exactly what your pipeline measured on your data -- report it as-is rather
than editing the files afterward.
"""

import datetime
import os

import matplotlib

matplotlib.use("Agg")  # headless-safe (no display needed)
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(history, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(hist.get("accuracy", []), label="Train Accuracy")
    axes[0].plot(hist.get("val_accuracy", []), label="Validation Accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(hist.get("loss", []), label="Train Loss")
    axes[1].plot(hist.get("val_loss", []), label="Validation Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[reporting] Saved training curves -> {path}")
    return path


def plot_confusion_matrix(cm: np.ndarray, class_names, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short_names = [n.replace("Phone_", "P").replace("_", " ") for n in class_names]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_yticklabels(short_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix -- Hybrid PRNU + CNN")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[reporting] Saved confusion matrix -> {path}")
    return path


def save_classification_report(report_txt: str, accuracies: dict, out_dir: str):
    """Append a timestamped run summary to reports/results.txt.

    Appending (not overwriting) keeps a history of every real run, which is
    useful evidence that the numbers came from actually executing the
    pipeline rather than being typed in by hand.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.txt")
    with open(path, "a") as f:
        f.write("=" * 70 + "\n")
        f.write(f"Run timestamp: {datetime.datetime.now().isoformat()}\n")
        for name, acc in accuracies.items():
            f.write(f"{name} accuracy: {acc * 100:.2f}%\n")
        f.write("\nClassification report:\n")
        f.write(report_txt + "\n")
    print(f"[reporting] Appended run summary -> {path}")
    return path

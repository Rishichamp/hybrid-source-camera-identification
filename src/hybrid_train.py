"""
hybrid_train.py
----------------
Trains the full refined pipeline:

    Branch A (classical):  PRNU correlation against each camera's
        reference fingerprint  ->  N correlation scores/image.
    Branch B (deep):        Small custom CNN (trained from scratch) patch
        classifier, predictions averaged over an image's patches -> N
        softmax scores/image.
    Meta-classifier:        Logistic Regression trained on the
        concatenation of A + B (2N features/image) -> final prediction.

Camera classes (N of them) are auto-discovered from data/raw/'s subfolder
names (see dataset.py) -- there is nothing to hardcode when switching
datasets. The discovered class list is saved to saved_models/class_names.json
so predict.py and demo/app.py can load it back without needing to import
any hardcoded constant either.

Why a meta-classifier instead of just averaging the two branches: the
relative reliability of "noise correlation" vs. "deep features" varies by
camera and by how much data we have, so a trained (but tiny) combiner
generalizes better than a fixed average -- and with only 2N input features
it can be fit reliably even from a small held-out set, so it doesn't
reintroduce the small-data problem we're solving.

Usage:
    cd src
    python hybrid_train.py --epochs 40 --batch-size 32 --fingerprint-n 40
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from cnn_branch import build_cnn_branch
from dataset import load_and_split
from patch_sequence import PatchSequence
from prnu_extraction import extract_patches, patches_to_cnn_batch, prnu_correlation_features
from reporting import plot_confusion_matrix, plot_training_curves, save_classification_report

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")


def parse_args():
    p = argparse.ArgumentParser(description="Train the hybrid PRNU+CNN camera-ID model.")
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--crop-size", type=int, default=512,
                    help="Native-resolution center crop before residual extraction "
                         "(NOT a resize of the whole photo -- see prnu_extraction.py). "
                         "Must be <= your smallest source image's shorter side.")
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--fingerprint-n", type=int, default=40,
                    help="Images/class reserved for building PRNU fingerprints (not used to train CNN).")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10,
                    help="EarlyStopping patience. Higher than before (was 6) since the "
                         "lower learning rate + regularization converge more slowly but "
                         "more stably.")
    p.add_argument("--max-patches-per-chunk", type=int, default=3000,
                    help="Caps peak memory when building meta-classifier features: patches "
                         "are converted to CNN input and predicted in chunks of at most this "
                         "many patches at once, rather than all of a split's patches in one "
                         "array. Lower this (e.g. 1000) if you hit a MemoryError here.")
    return p.parse_args()


def image_level_features(residuals, cnn_model, fingerprints, patch_size, stride,
                          batch_size: int = 256, max_patches_per_chunk: int = 3000):
    """Build the 2N-dim [PRNU-correlation | avg CNN softmax] feature vector
    (N = number of camera classes) for a list of whole-image residuals.

    IMPORTANT: patches are normalized with the exact same
    `patches_to_cnn_batch` transform used to build the CNN's own training
    data (see dataset.py / prnu_extraction.py). Feeding the CNN raw,
    un-normalized residuals here -- as an earlier version of this file did
    -- silently mismatches what the network was trained on and quietly
    caps the whole hybrid pipeline's accuracy.

    MEMORY: an earlier version of this function built ONE array holding
    every patch from the entire split (e.g. 6,000 patches for a 240-image
    val split) before calling cnn_model.predict(). The `batch_size` param
    only controlled how Keras internally batches an array that ALREADY
    has to fully exist in memory first -- it did nothing to bound peak
    memory, and this caused a real MemoryError on a 10-class run. This
    version processes images in groups small enough that no more than
    ~max_patches_per_chunk patches are ever converted to a dense array at
    once, regardless of how many images are in the split. Order is
    preserved (chunks are contiguous, processed in original order), which
    matters since the caller zips the returned features against a labels
    array in the same order.
    """
    n_classes = cnn_model.output_shape[-1]
    if not residuals:
        return np.zeros((0, len(fingerprints) + n_classes), dtype=np.float32)

    feats = []
    i, n = 0, len(residuals)
    while i < n:
        chunk_start = i
        chunk_patches, chunk_owner = [], []
        local_count = 0
        while i < n:
            patches = extract_patches(residuals[i], patch_size, stride)
            # Always include at least one image per chunk (even if its own
            # patch count exceeds the budget alone) so we never get stuck.
            if chunk_patches and (len(chunk_patches) + len(patches)) > max_patches_per_chunk:
                break
            chunk_patches.extend(patches)
            chunk_owner.extend([local_count] * len(patches))
            local_count += 1
            i += 1

        patch_arr = patches_to_cnn_batch(chunk_patches)
        probs = cnn_model.predict(patch_arr, batch_size=batch_size, verbose=0)
        owner_arr = np.array(chunk_owner)
        del patch_arr, chunk_patches  # free before starting the next chunk

        for local_i in range(local_count):
            global_i = chunk_start + local_i
            avg_cnn_probs = probs[owner_arr == local_i].mean(axis=0)
            corr = prnu_correlation_features(residuals[global_i], fingerprints)
            feats.append(np.concatenate([corr, avg_cnn_probs]))

    return np.array(feats, dtype=np.float32)


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("== Loading data: building fingerprints + image-wise train/val/test ==")
    d = load_and_split(
        args.raw_dir,
        fingerprint_n=args.fingerprint_n,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        stride=args.stride,
    )
    class_names = d["class_names"]
    n_classes = len(class_names)
    print(f"Train images: {len(d['train_img_res'])}, Val images: {len(d['val_img_res'])}, "
          f"Test images: {len(d['test_img_res'])}")

    print("== Training CNN branch (custom CNN, trained from scratch) ==")
    # PatchSequence extracts and normalizes patches LAZILY, one batch at a
    # time, from the whole-image residuals -- it never materializes every
    # patch in a split as one dense array (that used to be ~2.88GB for a
    # 10-class run and caused a real MemoryError). Augmentation (flip +
    # jittering the sampling window within the real residual, no
    # rotation/zoom) happens inside PatchSequence itself, on the fly.
    train_seq = PatchSequence(
        d["train_img_res"], d["train_img_lbl"], args.patch_size, args.stride,
        batch_size=args.batch_size, augment=True, shuffle=True,
    )
    val_seq = PatchSequence(
        d["val_img_res"], d["val_img_lbl"], args.patch_size, args.stride,
        batch_size=args.batch_size, augment=False, shuffle=False,
    )
    cnn = build_cnn_branch(
        num_classes=n_classes,
        input_shape=(args.patch_size, args.patch_size, 3),
    )
    history = cnn.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience,
                                  restore_best_weights=True)],
    )
    plot_training_curves(history, os.path.join(ROOT_DIR, "reports", "figures"))

    print("== Building image-level hybrid features (val set) for meta-classifier ==")
    val_feats = image_level_features(
        d["val_img_res"], cnn, d["fingerprints"], args.patch_size, args.stride,
        batch_size=args.batch_size, max_patches_per_chunk=args.max_patches_per_chunk,
    )
    # StandardScaler matters here: PRNU correlation scores (roughly [-1, 1])
    # and CNN softmax scores (roughly [0, 1], but usually near 0 or 1) live
    # on different scales, so an un-scaled Logistic Regression can end up
    # weighting one branch more than intended just because of scale, not
    # actual informativeness. class_weight="balanced" guards against any
    # residual class imbalance in the held-out set used to fit this fusion
    # step. random_state fixes reproducibility.
    meta_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    meta_clf.fit(val_feats, d["val_img_lbl"])

    print("== Evaluating full hybrid pipeline on the held-out test set ==")
    test_feats = image_level_features(
        d["test_img_res"], cnn, d["fingerprints"], args.patch_size, args.stride,
        batch_size=args.batch_size, max_patches_per_chunk=args.max_patches_per_chunk,
    )
    y_pred = meta_clf.predict(test_feats)
    y_true = d["test_img_lbl"]

    acc = (y_pred == y_true).mean()
    print(f"\nHybrid pipeline test accuracy: {acc * 100:.2f}%")
    report_txt = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report_txt)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Also report the CNN-only and PRNU-only accuracies for comparison,
    # so you can see how much the fusion is actually buying you.
    # test_feats columns are [PRNU-correlation (n_classes) | CNN softmax
    # (n_classes)] -- slice by n_classes, not a hardcoded 5, so this still
    # works correctly if you use a different number of camera classes.
    cnn_only_pred = np.argmax(test_feats[:, n_classes:], axis=1)
    prnu_only_pred = np.argmax(test_feats[:, :n_classes], axis=1)
    cnn_only_acc = (cnn_only_pred == y_true).mean()
    prnu_only_acc = (prnu_only_pred == y_true).mean()
    print(f"\n[Diagnostic] CNN-branch-only accuracy:  {cnn_only_acc * 100:.2f}%")
    print(f"[Diagnostic] PRNU-branch-only accuracy: {prnu_only_acc * 100:.2f}%")

    reports_dir = os.path.join(ROOT_DIR, "reports")
    plot_confusion_matrix(cm, class_names, os.path.join(reports_dir, "figures"))
    save_classification_report(
        report_txt,
        {"hybrid": acc, "cnn_only": cnn_only_acc, "prnu_only": prnu_only_acc},
        reports_dir,
    )
    print(f"Saved plots + report to {reports_dir}/ "
          f"(these numbers come straight out of this run -- report them as-is)")

    print("\n== Saving artifacts ==")
    cnn.save(os.path.join(MODEL_DIR, "cnn_branch.keras"))
    joblib.dump(meta_clf, os.path.join(MODEL_DIR, "meta_classifier.joblib"))
    np.save(os.path.join(MODEL_DIR, "fingerprints.npy"), np.array(d["fingerprints"]))
    with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved to {MODEL_DIR}/ (including class_names.json, so predict.py "
          f"and demo/app.py pick up whatever classes you actually trained on)")


if __name__ == "__main__":
    main()

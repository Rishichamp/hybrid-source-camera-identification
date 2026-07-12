"""
cross_validate.py
-------------------
With only 200 images/class, a single train/val/test split can give a
misleadingly optimistic (or pessimistic) accuracy number just by chance in
how the split fell. This script repeats the entire hybrid pipeline
(fingerprint building -> CNN training -> meta-classifier -> test
evaluation) across several random seeds/splits and reports mean +/- std
accuracy, which is a much more honest number to put in your report/README
than a single run.

IMPORTANT (memory): each split builds a brand-new CNN via
build_cnn_branch(). Keras/TensorFlow accumulate graph and session state
across repeated model-building calls within the same Python process if
that state is never explicitly released -- this caused later splits in a
real run to fail with a MemoryError even when the requested allocation
was small (a few hundred MB), while earlier splits succeeded fine. The fix
is to call keras.backend.clear_session() (and force garbage collection)
after every split, releasing the previous split's model/graph before the
next one is built -- see the end of main()'s loop below.

Usage:
    cd src
    python cross_validate.py --n-splits 5 --epochs 25
"""

import argparse
import gc
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping

from cnn_branch import build_cnn_branch
from dataset import load_and_split
from hybrid_train import image_level_features
from patch_sequence import PatchSequence

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(SRC_DIR, "..")
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")


def parse_args():
    p = argparse.ArgumentParser(description="Repeated-split robustness check.")
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--crop-size", type=int, default=512,
                    help="Native-resolution center crop before residual extraction.")
    p.add_argument("--patch-size", type=int, default=96)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--fingerprint-n", type=int, default=40)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-patches-per-chunk", type=int, default=3000,
                    help="Caps peak memory when building meta-classifier features. "
                         "Lower this (e.g. 1000) if you hit a MemoryError here.")
    return p.parse_args()


def run_one_split(args, seed):
    d = load_and_split(
        args.raw_dir, fingerprint_n=args.fingerprint_n, crop_size=args.crop_size,
        patch_size=args.patch_size, stride=args.stride, seed=seed,
    )
    # PatchSequence extracts+normalizes patches lazily, one batch at a time,
    # from the whole-image residuals -- never materializes an entire
    # split's patches as one dense array (that caused a real MemoryError
    # on a 10-class run). Same noise-preserving augmentation policy as
    # hybrid_train.py (flip + jitter the sampling window, no rotation/zoom).
    train_seq = PatchSequence(
        d["train_img_res"], d["train_img_lbl"], args.patch_size, args.stride,
        batch_size=args.batch_size, augment=True, shuffle=True, seed=seed,
    )
    val_seq = PatchSequence(
        d["val_img_res"], d["val_img_lbl"], args.patch_size, args.stride,
        batch_size=args.batch_size, augment=False, shuffle=False, seed=seed,
    )
    cnn = build_cnn_branch(num_classes=len(d["class_names"]),
                            input_shape=(args.patch_size, args.patch_size, 3))
    cnn.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
        verbose=0,
    )

    val_feats = image_level_features(d["val_img_res"], cnn, d["fingerprints"],
                                      args.patch_size, args.stride,
                                      batch_size=args.batch_size,
                                      max_patches_per_chunk=args.max_patches_per_chunk)
    meta_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    )
    meta_clf.fit(val_feats, d["val_img_lbl"])

    test_feats = image_level_features(d["test_img_res"], cnn, d["fingerprints"],
                                       args.patch_size, args.stride,
                                       batch_size=args.batch_size,
                                       max_patches_per_chunk=args.max_patches_per_chunk)
    y_pred = meta_clf.predict(test_feats)
    y_true = d["test_img_lbl"]
    acc = (y_pred == y_true).mean()

    # Explicitly drop references to this split's largest objects before
    # returning -- the actual release of TensorFlow's internal graph/session
    # state still happens via keras.backend.clear_session() in main()'s
    # loop (Python's own refcounting doesn't reach into that), but dropping
    # these here means nothing is kept alive by this function's own frame
    # any longer than necessary while the caller gets around to it.
    del cnn, train_seq, val_seq, d
    return acc


def main():
    args = parse_args()
    accs = []
    for i in range(args.n_splits):
        seed = 42 + i
        print(f"\n===== Split {i + 1}/{args.n_splits} (seed={seed}) =====")
        acc = run_one_split(args, seed)
        print(f"Split {i + 1} hybrid test accuracy: {acc * 100:.2f}%")
        accs.append(acc)

        # Release this split's model/graph state before building the next
        # one -- without this, TensorFlow/Keras accumulate memory across
        # repeated model-building calls in the same process, making later
        # splits progressively more likely to hit a MemoryError even when
        # each individual split's own memory needs are modest and constant.
        K.clear_session()
        gc.collect()

    accs = np.array(accs)
    print("\n================ Summary ================")
    print(f"Per-split accuracies: {[f'{a*100:.2f}%' for a in accs]}")
    print(f"Mean accuracy: {accs.mean() * 100:.2f}%  (+/- {accs.std() * 100:.2f}%)")

    reports_dir = os.path.join(ROOT_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "cross_validation.txt"), "a") as f:
        import datetime
        f.write("=" * 70 + "\n")
        f.write(f"Run timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"n_splits={args.n_splits}, epochs={args.epochs}\n")
        f.write(f"Per-split accuracies: {[f'{a*100:.2f}%' for a in accs]}\n")
        f.write(f"Mean accuracy: {accs.mean() * 100:.2f}% (+/- {accs.std() * 100:.2f}%)\n")
    print(f"Saved cross-validation summary -> {reports_dir}/cross_validation.txt "
          f"(use this mean +/- std in your report -- it's the honest, "
          f"defensible number, not a single lucky split)")


if __name__ == "__main__":
    main()

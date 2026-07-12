"""
dataset.py
----------
Loads a camera-identification dataset laid out as one folder per camera
under data/raw/, and prepares TWO different views of it, because the two
branches of the hybrid model need different things:

    1. "Fingerprint set" (held out): used ONLY to build each camera's
       averaged PRNU reference fingerprint. These images are never used to
       train or evaluate the CNN branch, and the fingerprint-building
       itself isn't "training" in the ML sense (see prnu_extraction
       .build_camera_fingerprint), so this doesn't cost us classifier data.

    2. "Classifier set": split image-wise into train/val/test (70/15/15).
       Each image is turned into several overlapping residual patches for
       the CNN branch (see prnu_extraction.extract_patches), multiplying
       the effective sample count. Patches from the same source image
       always stay together in the same split (train, val, or test) to
       avoid data leakage.

IMPORTANT (memory): this module returns whole-image RESIDUALS, not
pre-extracted patches. An earlier version pre-computed every patch for an
entire split into one dense array (e.g. 28,000 patches x 96x96x3 float32
= ~2.88GB for a 10-class run) before training even started, which caused
a real MemoryError. Residuals are much smaller (one crop_size x crop_size
array per image, not dozens of patch_size x patch_size x 3 arrays), and
patches are now extracted lazily, one training batch at a time, by
patch_sequence.PatchSequence -- see hybrid_train.py / cross_validate.py
for how these residuals feed into that.

Camera classes are auto-discovered from data/raw/'s subfolder names --
there is no hardcoded class list to edit. Drop in ANY camera dataset (your
own photos, a Kaggle set, or anything else) as one subfolder per camera
under data/raw/, and this module picks it up automatically. The discovered
class list is returned as d["class_names"] and saved alongside the trained
model (see hybrid_train.py) so predict.py/demo/app.py never need to
hardcode it either.
"""

import os

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from prnu_extraction import build_camera_fingerprint, extract_prnu_residual

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def discover_camera_classes(raw_dir: str):
    """Auto-discover camera classes: one class per subfolder of raw_dir
    that contains at least one valid image, sorted alphabetically for a
    stable, reproducible label ordering across runs."""
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"data/raw directory not found: {raw_dir}")
    classes = []
    for name in sorted(os.listdir(raw_dir)):
        path = os.path.join(raw_dir, name)
        if os.path.isdir(path) and _list_images(path):
            classes.append(name)
    if not classes:
        raise RuntimeError(
            f"No camera folders with images found under {raw_dir}. "
            f"Expected a layout like data/raw/<CameraName>/img1.jpg, ..."
        )
    return classes


def _list_images(camera_path: str):
    return [os.path.join(camera_path, f) for f in sorted(os.listdir(camera_path))
            if f.lower().endswith(VALID_EXTENSIONS)]


def load_and_split(raw_dir: str, class_names=None, fingerprint_n: int = 40,
                    crop_size: int = 512, patch_size: int = 96, stride: int = 64,
                    seed: int = 42):
    """Load everything and produce fingerprints + image-wise train/val/test.

    Args:
        class_names: explicit list of camera folder names to use, in the
            order you want as class labels. If None (default), classes are
            auto-discovered from raw_dir's subfolders (alphabetical order).
        crop_size: a crop_size x crop_size patch is taken from the CENTER
            of each image at its NATIVE resolution (see
            prnu_extraction.extract_prnu_residual) -- not a resize of the
            whole photo, which would smear out the very per-pixel noise
            PRNU relies on.
        patch_size, stride: NOT used to pre-extract patches here anymore --
            passed through only so callers building a PatchSequence don't
            need to duplicate these defaults. Kept as parameters for
            backward-compatible call signatures.

    Returns a dict with:
        class_names: the camera class names actually used, in label order
            (index i corresponds to label i everywhere else in the dict).
        fingerprints: list of (num_classes) reference fingerprints (2D arrays)
        train_img_res / train_img_lbl (and val_/test_ equivalents):
            whole-image residuals + integer labels, one entry per image
            (NOT per patch). Feed these into patch_sequence.PatchSequence
            for CNN training/validation, or into
            hybrid_train.image_level_features for the PRNU-correlation +
            averaged-CNN-softmax feature vector.
    """
    if class_names is None:
        class_names = discover_camera_classes(raw_dir)
        print(f"Auto-discovered {len(class_names)} camera classes: {class_names}")

    rng = np.random.RandomState(seed)

    fingerprints = []
    train_img_res, train_img_lbl = [], []
    val_img_res, val_img_lbl = [], []
    test_img_res, test_img_lbl = [], []

    for label, camera in enumerate(class_names):
        camera_path = os.path.join(raw_dir, camera)
        if not os.path.isdir(camera_path):
            print(f"[WARN] Folder not found, skipping: {camera_path}")
            continue

        files = _list_images(camera_path)
        if len(files) < fingerprint_n + 10:
            print(f"[WARN] {camera} has only {len(files)} images -- "
                  f"consider lowering fingerprint_n.")

        rng.shuffle(files)
        fp_files, clf_files = files[:fingerprint_n], files[fingerprint_n:]

        print(f"{camera}: {len(fp_files)} imgs -> fingerprint, "
              f"{len(clf_files)} imgs -> classifier set")

        # --- Build this camera's fingerprint ---
        fp_residuals = []
        for f in fp_files:
            img = cv2.imread(f)
            if img is None:
                continue
            fp_residuals.append(extract_prnu_residual(img, crop_size))
        if fp_residuals:
            fingerprints.append(build_camera_fingerprint(fp_residuals))
        else:
            raise RuntimeError(f"No fingerprint images could be read for {camera}")

        # --- Classifier set: image-wise split (patchifying happens later,
        # lazily, in PatchSequence -- NOT here) ---
        clf_train, clf_temp = train_test_split(clf_files, test_size=0.30, random_state=seed)
        clf_val, clf_test = train_test_split(clf_temp, test_size=0.50, random_state=seed)

        def process(file_list, img_res_bucket, img_lbl_bucket):
            for f in file_list:
                img = cv2.imread(f)
                if img is None:
                    continue
                img_res_bucket.append(extract_prnu_residual(img, crop_size))
                img_lbl_bucket.append(label)

        process(clf_train, train_img_res, train_img_lbl)
        process(clf_val, val_img_res, val_img_lbl)
        process(clf_test, test_img_res, test_img_lbl)

    return {
        "class_names": class_names,
        "fingerprints": fingerprints,
        "train_img_res": train_img_res, "train_img_lbl": np.array(train_img_lbl),
        "val_img_res": val_img_res, "val_img_lbl": np.array(val_img_lbl),
        "test_img_res": test_img_res, "test_img_lbl": np.array(test_img_lbl),
    }


if __name__ == "__main__":
    RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    d = load_and_split(RAW_DIR)
    print(f"Classes: {d['class_names']}")
    print(f"Train images: {len(d['train_img_res'])}, "
          f"Val: {len(d['val_img_res'])}, Test: {len(d['test_img_res'])}")
    print(f"Fingerprints built: {len(d['fingerprints'])}")

"""
tests/test_dataset.py
----------------------
Unit tests for dataset.py's camera-class auto-discovery -- the mechanism
that lets you swap datasets (your own photos, a Kaggle set, anything else)
without editing any code. Uses small synthetic images written to a temp
directory, so no real dataset is needed.
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataset import discover_camera_classes, load_and_split  # noqa: E402


def _write_camera_folder(raw_dir, camera, n, seed=0, size=(300, 400)):
    d = os.path.join(raw_dir, camera)
    os.makedirs(d, exist_ok=True)
    rng = np.random.RandomState(seed)
    for i in range(n):
        img = rng.randint(0, 255, size=(*size, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(d, f"img_{i}.jpg"), img)


def test_discover_camera_classes_finds_all_folders_with_images():
    with tempfile.TemporaryDirectory() as tmp:
        _write_camera_folder(tmp, "CameraA", 5, seed=0)
        _write_camera_folder(tmp, "CameraB", 5, seed=1)
        classes = discover_camera_classes(tmp)
        assert classes == ["CameraA", "CameraB"]  # alphabetical, stable order


def test_discover_camera_classes_ignores_empty_folders():
    with tempfile.TemporaryDirectory() as tmp:
        _write_camera_folder(tmp, "CameraA", 3, seed=0)
        os.makedirs(os.path.join(tmp, "EmptyFolder"))  # no images inside
        classes = discover_camera_classes(tmp)
        assert classes == ["CameraA"]


def test_discover_camera_classes_ignores_non_image_files():
    with tempfile.TemporaryDirectory() as tmp:
        _write_camera_folder(tmp, "CameraA", 2, seed=0)
        # A stray non-image file at the top level should not become a class.
        with open(os.path.join(tmp, "readme.txt"), "w") as f:
            f.write("not a camera")
        classes = discover_camera_classes(tmp)
        assert classes == ["CameraA"]


def test_discover_camera_classes_raises_if_none_found():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            discover_camera_classes(tmp)
            assert False, "expected RuntimeError for a dataset with no cameras"
        except RuntimeError:
            pass


def test_load_and_split_auto_discovers_and_labels_consistently():
    """More than 5 classes (e.g. a 10-class public dataset) should work
    exactly the same way as 5 -- labels should match discovered order."""
    with tempfile.TemporaryDirectory() as tmp:
        camera_names = [f"Camera{i}" for i in range(7)]
        for i, cam in enumerate(camera_names):
            _write_camera_folder(tmp, cam, 8, seed=i)

        d = load_and_split(tmp, fingerprint_n=3, crop_size=128,
                            patch_size=32, stride=24)

        assert d["class_names"] == camera_names
        assert len(d["fingerprints"]) == len(camera_names)
        # Every label that appears should be a valid index into class_names.
        all_labels = (set(d["train_img_lbl"].tolist())
                      | set(d["val_img_lbl"].tolist())
                      | set(d["test_img_lbl"].tolist()))
        assert all_labels.issubset(set(range(len(camera_names))))
        # Whole-image residuals (not patches) -- one entry per image, and
        # every residual's label list should be the same length as its
        # residual list.
        assert len(d["train_img_res"]) == len(d["train_img_lbl"])
        assert len(d["val_img_res"]) == len(d["val_img_lbl"])
        assert len(d["test_img_res"]) == len(d["test_img_lbl"])

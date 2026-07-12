"""
tests/test_prepare_camera_dataset.py
--------------------------------------
Unit tests for scripts/prepare_camera_dataset.py -- the dataset-prep
helper for turning a large public dataset (e.g. the Kaggle camera-model-
identification dataset) into a small, repo-friendly data/raw/ folder. No
real dataset needed; uses synthetic images written to a temp directory.
"""

import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from prepare_camera_dataset import (  # noqa: E402
    _crop_center,
    discover_folders_mode,
    discover_regex_mode,
)


def _write_synthetic(base, camera, n, size=(600, 800), flat=False, prefix_idx=0):
    rng = np.random.RandomState(prefix_idx)
    if flat:
        for i in range(n):
            img = rng.randint(0, 255, size=(*size, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(base, f"{camera}_0_{i:04d}.jpg"), img)
    else:
        d = os.path.join(base, camera)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            img = rng.randint(0, 255, size=(*size, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(d, f"img_{i}.jpg"), img)


def test_crop_center_shape_native_resolution():
    img = np.random.RandomState(0).randint(0, 255, size=(900, 1200, 3), dtype=np.uint8)
    cropped = _crop_center(img, 512)
    assert cropped.shape == (512, 512, 3)
    # Should be an exact pixel window, not resized/interpolated.
    h, w = img.shape[:2]
    top, left = (h - 512) // 2, (w - 512) // 2
    assert np.array_equal(cropped, img[top:top + 512, left:left + 512])


def test_crop_center_upscales_small_images():
    img = np.random.RandomState(0).randint(0, 255, size=(300, 400, 3), dtype=np.uint8)
    cropped = _crop_center(img, 512)
    assert cropped.shape == (512, 512, 3)


def test_discover_folders_mode():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, "CamA", 5, prefix_idx=0)
        _write_synthetic(tmp, "CamB", 3, prefix_idx=1)
        groups = discover_folders_mode(tmp)
        assert set(groups.keys()) == {"CamA", "CamB"}
        assert len(groups["CamA"]) == 5
        assert len(groups["CamB"]) == 3


def test_discover_regex_mode():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, "Nikon_D200", 4, flat=True, prefix_idx=0)
        _write_synthetic(tmp, "Canon_Ixus70", 6, flat=True, prefix_idx=1)
        groups = discover_regex_mode(tmp, r"(?P<camera>[A-Za-z]+_[A-Za-z0-9]+)_\d+_\d+\.jpg")
        assert set(groups.keys()) == {"Nikon_D200", "Canon_Ixus70"}
        assert len(groups["Nikon_D200"]) == 4
        assert len(groups["Canon_Ixus70"]) == 6


def test_discover_regex_mode_skips_unmatched_without_crashing():
    with tempfile.TemporaryDirectory() as tmp:
        _write_synthetic(tmp, "Nikon_D200", 2, flat=True, prefix_idx=0)
        # A file that won't match the pattern at all.
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(tmp, "random_unrelated_name.jpg"), img)
        groups = discover_regex_mode(tmp, r"(?P<camera>[A-Za-z]+_[A-Za-z0-9]+)_\d+_\d+\.jpg")
        assert "Nikon_D200" in groups
        assert len(groups["Nikon_D200"]) == 2

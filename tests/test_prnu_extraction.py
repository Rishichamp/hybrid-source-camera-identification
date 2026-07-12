"""
tests/test_prnu_extraction.py
------------------------------
Lightweight unit tests for the core PRNU math -- no dataset/images
required, so these run in CI on every commit. These specifically would
have caught the train/inference normalization mismatch bug (residuals fed
raw to the CNN in one code path, scaled in another) that was fixed in this
version: test_residual_to_cnn_input_matches_batch_version guards against
that regression re-appearing.

Run with:
    cd src && python -m pytest ../tests -v
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prnu_extraction import (  # noqa: E402
    build_camera_fingerprint,
    extract_patches,
    extract_prnu_residual,
    normalized_cross_correlation,
    patches_to_cnn_batch,
    prnu_correlation_features,
    residual_to_cnn_input,
)
from prnu_extraction import _center_crop_or_upscale as pe_center_crop  # noqa: E402


def _synthetic_image(seed=0, size=256):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_extract_prnu_residual_shape_and_zero_mean():
    img = _synthetic_image(size=800)
    residual = extract_prnu_residual(img, crop_size=256)
    assert residual.shape == (256, 256)
    # Row/column zero-meaning should leave the overall mean very close to 0.
    assert abs(residual.mean()) < 1e-3


def test_extract_prnu_residual_crops_native_resolution_not_resize():
    """A center crop should take the SAME pixel values as in the original
    image (just a windowed subset) -- NOT a resized/interpolated version.
    This guards against regressing to resizing the whole frame, which
    smears out per-pixel PRNU."""
    img = _synthetic_image(size=800, seed=7)
    gray = np.mean(img, axis=2).astype(np.uint8)
    cropped = pe_center_crop(gray, 256)
    h, w = gray.shape
    top, left = (h - 256) // 2, (w - 256) // 2
    expected = gray[top:top + 256, left:left + 256]
    assert np.array_equal(cropped, expected)


def test_extract_prnu_residual_handles_small_source_image():
    """Images smaller than crop_size should be upscaled to fit rather than
    crashing or silently returning a wrong-sized residual."""
    img = _synthetic_image(size=128)
    residual = extract_prnu_residual(img, crop_size=256)
    assert residual.shape == (256, 256)


def test_extract_patches_count_and_shape():
    residual = np.random.randn(256, 256).astype(np.float32)
    patches = extract_patches(residual, patch_size=96, stride=64)
    # (256-96)//64 + 1 = 3 steps per axis -> 9 patches
    assert len(patches) == 9
    for p in patches:
        assert p.shape == (96, 96)


def test_fingerprint_self_correlation_is_high():
    """A fingerprint built from noisy copies of the same underlying pattern
    should correlate strongly with a fresh residual sharing that pattern,
    and much less with unrelated noise."""
    rng = np.random.RandomState(0)
    true_pattern = rng.randn(64, 64).astype(np.float32) * 2.0

    residuals = [true_pattern + rng.randn(64, 64).astype(np.float32) * 5.0
                 for _ in range(30)]
    fingerprint = build_camera_fingerprint(residuals)

    matching_residual = true_pattern + rng.randn(64, 64).astype(np.float32) * 5.0
    unrelated_residual = rng.randn(64, 64).astype(np.float32) * 5.0

    corr_match = normalized_cross_correlation(matching_residual, fingerprint)
    corr_unrelated = normalized_cross_correlation(unrelated_residual, fingerprint)

    assert corr_match > corr_unrelated


def test_prnu_correlation_features_shape():
    rng = np.random.RandomState(1)
    fingerprints = [rng.randn(64, 64).astype(np.float32) for _ in range(5)]
    residual = rng.randn(64, 64).astype(np.float32)
    feats = prnu_correlation_features(residual, fingerprints)
    assert feats.shape == (5,)


def test_residual_to_cnn_input_output_range():
    patch = (np.random.randn(96, 96).astype(np.float32) * 8.0) + 3.0
    out = residual_to_cnn_input(patch)
    assert out.shape == (96, 96, 3)
    assert out.min() >= 0.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6
    # All three "RGB" channels must be identical (faked from one channel).
    assert np.allclose(out[..., 0], out[..., 1])
    assert np.allclose(out[..., 1], out[..., 2])


def test_residual_to_cnn_input_matches_batch_version():
    """Regression guard: the single-patch and batched normalization paths
    must agree, since predict.py/demo use one and training uses the other.
    If these ever diverge again, train/inference will silently mismatch."""
    rng = np.random.RandomState(2)
    patches = [rng.randn(96, 96).astype(np.float32) * 6.0 for _ in range(4)]

    single = np.stack([residual_to_cnn_input(p) for p in patches])
    batched = patches_to_cnn_batch(patches)

    assert single.shape == batched.shape
    assert np.allclose(single, batched, atol=1e-5)


def test_residual_to_cnn_input_not_naive_divide_by_255():
    """Guards against regressing to the original bug: a small, zero-mean
    residual divided by 255 would collapse to near-zero everywhere. The
    correct normalization should use much more of the [0, 1] range."""
    patch = (np.random.RandomState(3).randn(96, 96).astype(np.float32)) * 5.0
    out = residual_to_cnn_input(patch)
    naive = patch / 255.0
    assert out.std() > naive.std() * 5  # normalized version uses far more contrast

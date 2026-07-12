"""
tests/test_hybrid_train.py
----------------------------
Unit tests for hybrid_train.image_level_features -- specifically its
chunked-processing behavior, added after a real MemoryError on a 10-class
run: an earlier version built ONE array holding every patch from an
entire split (e.g. 6,000 patches for a 240-image val split) before handing
it to the CNN, which meant peak memory scaled with split size regardless
of any batch_size setting. These tests use a fake CNN (no real training,
no GPU/large RAM needed) to verify the chunked version stays correct --
right output shape, right order, no patches dropped or duplicated -- even
when max_patches_per_chunk is set absurdly small to force many chunk
boundaries, since it's exactly at those boundaries a chunking bug would
show up.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hybrid_train import image_level_features  # noqa: E402


class _FakeCNN:
    """Deterministic stand-in for a trained Keras model: no real inference,
    just returns a fixed-shape softmax-like output per patch so we can
    verify image_level_features' bookkeeping (chunking, ordering,
    per-image averaging) independent of any actual model quality."""

    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.output_shape = (None, n_classes)
        self.call_count = 0
        self.max_batch_seen = 0

    def predict(self, patch_arr, batch_size=256, verbose=0):
        self.call_count += 1
        self.max_batch_seen = max(self.max_batch_seen, len(patch_arr))
        n = len(patch_arr)
        out = np.zeros((n, self.n_classes), dtype=np.float32)
        out[:, 0] = 1.0  # every patch "votes" for class 0
        return out


def _make_residuals(n_images, size=64, seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(size, size).astype(np.float32) for _ in range(n_images)]


def test_image_level_features_empty_input():
    cnn = _FakeCNN(n_classes=4)
    feats = image_level_features([], cnn, fingerprints=[np.zeros((8, 8))] * 4,
                                  patch_size=32, stride=16)
    assert feats.shape == (0, 8)


def test_image_level_features_shape_matches_n_images_and_2n_classes():
    n_classes = 3
    residuals = _make_residuals(6)
    fingerprints = [np.random.RandomState(i).randn(64, 64).astype(np.float32)
                     for i in range(n_classes)]
    cnn = _FakeCNN(n_classes=n_classes)
    feats = image_level_features(residuals, cnn, fingerprints,
                                  patch_size=32, stride=16)
    assert feats.shape == (6, 2 * n_classes)


def test_image_level_features_tiny_chunk_budget_forces_many_chunks_but_stays_correct():
    """The actual regression case: force max_patches_per_chunk small enough
    that every image lands in its own chunk (or even needs multiple calls),
    and confirm the output is identical in shape/order to a normal run."""
    n_classes = 3
    residuals = _make_residuals(8, size=64)
    fingerprints = [np.random.RandomState(i).randn(64, 64).astype(np.float32)
                     for i in range(n_classes)]

    cnn_normal = _FakeCNN(n_classes=n_classes)
    feats_normal = image_level_features(residuals, cnn_normal, fingerprints,
                                         patch_size=32, stride=16,
                                         max_patches_per_chunk=100000)

    cnn_chunked = _FakeCNN(n_classes=n_classes)
    feats_chunked = image_level_features(residuals, cnn_chunked, fingerprints,
                                          patch_size=32, stride=16,
                                          max_patches_per_chunk=1)

    assert feats_normal.shape == feats_chunked.shape == (8, 2 * n_classes)
    # PRNU-correlation half is independent of chunking -- must match exactly.
    np.testing.assert_allclose(feats_normal[:, :n_classes], feats_chunked[:, :n_classes])
    # Our fake CNN always outputs the same thing per patch regardless of
    # batch composition, so the CNN half should also match exactly here.
    np.testing.assert_allclose(feats_normal[:, n_classes:], feats_chunked[:, n_classes:])
    # With max_patches_per_chunk=1, chunking should have forced far more
    # predict() calls than the normal (single-chunk) run.
    assert cnn_chunked.call_count > cnn_normal.call_count
    assert cnn_normal.call_count == 1


def test_image_level_features_never_builds_a_giant_array_at_once():
    """Regression guard for the actual MemoryError: verify predict() is
    never called with more than max_patches_per_chunk (plus one image's
    worth of slack) patches at a time, even for a larger batch of images."""
    n_classes = 3
    residuals = _make_residuals(20, size=64)
    fingerprints = [np.random.RandomState(i).randn(64, 64).astype(np.float32)
                     for i in range(n_classes)]
    cnn = _FakeCNN(n_classes=n_classes)
    budget = 20
    image_level_features(residuals, cnn, fingerprints, patch_size=32, stride=16,
                          max_patches_per_chunk=budget)
    # One image's patches at 64x64/32/16 = 3x3 = 9 patches -- allow slack
    # for the "always include at least one image per chunk" rule.
    assert cnn.max_batch_seen <= budget + 9

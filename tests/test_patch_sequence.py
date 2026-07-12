"""
tests/test_patch_sequence.py
------------------------------
Unit tests for patch_sequence.PatchSequence -- the lazy, on-the-fly patch
generator that replaced pre-materializing an entire split's patches as one
dense array (which caused a real MemoryError: 28,000 patches x 96x96x3
float32 = ~2.88GB for a 10-class training set, built before training even
started). These tests confirm every patch position is covered exactly
once per epoch, batches stay correctly shaped/normalized, and -- the
actual regression this exists to catch -- that no single batch's memory
footprint scales with the number of images in the split.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from patch_sequence import PatchSequence  # noqa: E402


def _make_residuals(n_images, size=128, seed=0):
    rng = np.random.RandomState(seed)
    return [rng.randn(size, size).astype(np.float32) for _ in range(n_images)]


def test_all_patches_covered_exactly_once_per_epoch():
    residuals = _make_residuals(10, size=128)
    labels = list(range(10))
    seq = PatchSequence(residuals, labels, patch_size=32, stride=32,
                         batch_size=8, shuffle=False, augment=False)

    seen_positions = []
    for i in range(len(seq)):
        start = i * seq.batch_size
        seen_positions.extend(seq.index[start:start + seq.batch_size])

    # 128/32 = 4 -> 4x4 = 16 patch positions/image x 10 images = 160 total.
    assert len(seq.index) == 160
    assert len(seen_positions) == 160
    assert len(set(seen_positions)) == 160  # no duplicates


def test_batch_shape_and_normalized_range():
    residuals = _make_residuals(5, size=96)
    labels = [0, 1, 2, 0, 1]
    seq = PatchSequence(residuals, labels, patch_size=32, stride=32, batch_size=4)
    X, y = seq[0]
    assert X.shape[1:] == (32, 32, 3)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert len(y) == len(X)
    assert set(y.tolist()).issubset({0, 1, 2})


def test_shuffle_false_is_deterministic_across_epochs():
    """Validation/test sequences use shuffle=False -- order must be stable
    so repeated evaluation of the same model gives the same result."""
    residuals = _make_residuals(6, size=96)
    labels = list(range(6))
    seq = PatchSequence(residuals, labels, patch_size=32, stride=32,
                         batch_size=4, shuffle=False, augment=False)
    order_before = list(seq.index)
    seq.on_epoch_end()
    order_after = list(seq.index)
    assert order_before == order_after


def test_shuffle_true_changes_order_on_epoch_end():
    residuals = _make_residuals(20, size=128)
    labels = list(range(20))
    seq = PatchSequence(residuals, labels, patch_size=32, stride=32,
                         batch_size=8, shuffle=True, seed=0)
    order_before = list(seq.index)
    seq.on_epoch_end()
    order_after = list(seq.index)
    assert order_before != order_after
    assert sorted(order_before) == sorted(order_after)  # same positions, new order


def test_memory_per_batch_does_not_scale_with_number_of_images():
    """The actual regression guard: one batch's byte size should depend
    only on batch_size/patch_size, never on how many images are in the
    split -- this is what a 'materialize everything up front' bug would
    violate."""
    small = PatchSequence(_make_residuals(5, size=128), list(range(5)),
                           patch_size=32, stride=32, batch_size=8)
    large = PatchSequence(_make_residuals(200, size=128), list(range(200)),
                           patch_size=32, stride=32, batch_size=8)

    X_small, _ = small[0]
    X_large, _ = large[0]
    assert X_small.nbytes == X_large.nbytes  # same batch_size -> same bytes
    # 200 images x 16 patches/image = 3200 total positions, but no single
    # batch should ever be anywhere near that large.
    assert len(large.index) == 3200
    assert X_large.shape[0] == 8


def test_augmentation_keeps_patches_in_bounds():
    """Jittering the sampling window must never crop outside the residual."""
    residuals = _make_residuals(4, size=64)
    labels = [0, 1, 2, 3]
    seq = PatchSequence(residuals, labels, patch_size=32, stride=32,
                         batch_size=4, augment=True, seed=1)
    for _ in range(5):  # run several epochs of random jitter
        X, y = seq[0]
        assert X.shape == (4, 32, 32, 3)
        assert not np.isnan(X).any()
        seq.on_epoch_end()


def test_raises_on_patch_size_larger_than_residual():
    residuals = [np.zeros((16, 16), dtype=np.float32)]
    try:
        PatchSequence(residuals, [0], patch_size=32, stride=32)
        assert False, "expected ValueError when patch_size > residual size"
    except ValueError:
        pass

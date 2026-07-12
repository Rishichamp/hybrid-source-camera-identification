"""
patch_sequence.py
------------------
A Keras Sequence that extracts and normalizes residual patches LAZILY,
one batch at a time, instead of pre-computing every patch for an entire
split and holding it all in memory as one dense array.

Why this exists: dataset.py used to build X_train/X_val/X_test as single
arrays holding every patch from every image in a split -- for a 10-class
dataset with ~1,120 training images at 25 patches/image, that's 28,000
patches x 96x96x3 float32 = ~2.88GB materialized before training even
starts. This caused a real MemoryError partway through a 5-split
cross-validation run. The fix: store only the (much smaller) whole-image
residuals -- e.g. 1,120 images x 512x512 float32 = ~1.1GB, a >2x
reduction, and shrinking further the larger --stride is set -- and
extract just the patches needed for the CURRENT batch, on demand, from
those residuals. Peak memory becomes O(batch_size) instead of
O(total patches in the split).

This also lets augmentation be more faithful than before: rather than
shifting an already-extracted patch and filling the resulting empty
border with reflected pixels (what the old ImageDataGenerator-based
shift augmentation did), this jitters the (row, col) SAMPLING POSITION
within the real residual before cropping -- so a "shifted" patch is
always genuine residual data, never a synthetic fill.
"""

import numpy as np
from tensorflow.keras.utils import Sequence

from prnu_extraction import patches_to_cnn_batch


class PatchSequence(Sequence):
    """Lazily yields (patch_batch, label_batch) pairs for Keras training.

    Args:
        residuals: list of 2D whole-image residual arrays (NOT patches).
        labels: per-image integer class labels, same length as residuals.
        patch_size, stride: patch grid geometry (same meaning as
            prnu_extraction.extract_patches).
        batch_size: patches per batch.
        augment: if True, randomly flip patches and jitter the sampling
            position by a few pixels each epoch (noise-preserving -- no
            rotation/zoom, which would interpolate and blur the very
            high-frequency signal this pipeline depends on).
        shuffle: shuffle patch order each epoch (should be True for
            training, False for validation/test to keep results stable).
        seed: RNG seed for shuffling/augmentation reproducibility.
    """

    def __init__(self, residuals, labels, patch_size: int, stride: int,
                 batch_size: int = 32, augment: bool = True, shuffle: bool = True,
                 seed: int = 0):
        self.residuals = residuals
        self.labels = np.asarray(labels)
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)

        # Precompute (image_idx, row, col) triples for every patch
        # position -- cheap (just integers), unlike precomputing the
        # patches themselves.
        self.index = []
        for img_idx, residual in enumerate(residuals):
            h, w = residual.shape
            if h < patch_size or w < patch_size:
                continue
            for row in range(0, h - patch_size + 1, stride):
                for col in range(0, w - patch_size + 1, stride):
                    self.index.append((img_idx, row, col))
        if not self.index:
            raise ValueError(
                "No patches could be extracted -- check that patch_size <= "
                "your residuals' size."
            )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.index) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.index)

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        batch_positions = self.index[start:start + self.batch_size]

        patches, labels = [], []
        for img_idx, row, col in batch_positions:
            residual = self.residuals[img_idx]
            h, w = residual.shape
            r, c = row, col

            if self.augment:
                # Jitter the sampling window itself (up to ~10% of patch
                # size), clipped to stay in-bounds -- this is always real
                # residual data, unlike shifting a patch and filling the
                # gap with reflected pixels.
                max_jitter = max(1, self.patch_size // 10)
                r = int(np.clip(row + self.rng.randint(-max_jitter, max_jitter + 1),
                                 0, h - self.patch_size))
                c = int(np.clip(col + self.rng.randint(-max_jitter, max_jitter + 1),
                                 0, w - self.patch_size))

            patch = residual[r:r + self.patch_size, c:c + self.patch_size]

            if self.augment:
                if self.rng.rand() < 0.5:
                    patch = patch[:, ::-1]
                if self.rng.rand() < 0.5:
                    patch = patch[::-1, :]

            patches.append(np.ascontiguousarray(patch))
            labels.append(self.labels[img_idx])

        X = patches_to_cnn_batch(patches)
        y = np.array(labels)
        return X, y

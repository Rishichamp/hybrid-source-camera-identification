"""
prnu_extraction.py
-------------------
Refined Sensor Pattern Noise (SPN / PRNU) extraction.

v1 of this project used a Gaussian-blur "high-pass filter" as a crude proxy
for sensor noise. This version replaces that with the classical,
forensics-grade approach from Lukas, Fridrich & Goljan (2006):

    1. Decompose the grayscale image with a multi-level Discrete Wavelet
       Transform (DWT).
    2. Estimate the noise standard deviation from the finest detail
       sub-band using a robust MAD estimator.
    3. Apply a local adaptive Wiener filter to each detail sub-band
       (this is what actually "denoises" the image).
    4. Reconstruct the denoised image with the inverse DWT.
    5. Residual = original - denoised.  This residual approximates the
       camera's sensor pattern noise far better than a simple Gaussian
       high-pass filter, because the Wiener step adapts to local texture
       instead of blurring indiscriminately.
    6. Zero-mean the residual row-wise and column-wise to suppress
       non-unique (JPEG-grid, demosaicing) artifacts that aren't part of
       the true sensor fingerprint.

On top of this, we add two things the original project didn't have:

    - Patch extraction, so a handful of images can yield many training
      samples (crucial when you only have ~200 images per camera).
    - Reference "camera fingerprint" construction + normalized
      cross-correlation, i.e. the classical PRNU matching approach, which
      needs very little data to work and complements the CNN branch.
"""

import cv2
import numpy as np
import pywt
from scipy.ndimage import uniform_filter


def _wavelet_denoise(gray: np.ndarray, wavelet: str = "db8", levels: int = 4,
                      window_size: int = 3) -> np.ndarray:
    """Adaptive Wiener-in-wavelet-domain denoising (Mihcak et al. style)."""
    gray = gray.astype(np.float64)
    coeffs = pywt.wavedec2(gray, wavelet, level=levels)

    # Robust noise sigma estimate from the finest-level diagonal sub-band.
    _, _, cD1 = coeffs[-1]
    sigma = np.median(np.abs(cD1)) / 0.6745
    sigma_sq = sigma ** 2

    new_coeffs = [coeffs[0]]
    for detail_level in coeffs[1:]:
        new_detail = []
        for sub in detail_level:
            local_var = uniform_filter(sub ** 2, size=window_size)
            wiener_gain = np.maximum(local_var - sigma_sq, 0) / (local_var + 1e-8)
            new_detail.append(sub * wiener_gain)
        new_coeffs.append(tuple(new_detail))

    denoised = pywt.waverec2(new_coeffs, wavelet)
    # waverec2 can pad to even dimensions -- crop back to original shape.
    return denoised[:gray.shape[0], :gray.shape[1]]


def _center_crop_or_upscale(gray: np.ndarray, size: int) -> np.ndarray:
    """Return a `size` x `size` crop from the center of `gray`.

    If the source image is smaller than `size` in either dimension (rare
    for modern phone photos, but possible), it is upscaled just enough to
    fit -- upscaling doesn't destroy PRNU the way downscaling does, since
    it doesn't average multiple original sensor pixels into one output
    pixel.
    """
    h, w = gray.shape
    if h < size or w < size:
        scale = size / min(h, w)
        gray = cv2.resize(gray, (int(np.ceil(w * scale)), int(np.ceil(h * scale))))
        h, w = gray.shape
    top = (h - size) // 2
    left = (w - size) // 2
    return gray[top:top + size, left:left + size]


def extract_prnu_residual(img_bgr: np.ndarray, crop_size: int = 512) -> np.ndarray:
    """Full refined PRNU residual pipeline for one image.

    Args:
        img_bgr: image as read by cv2.imread (BGR).
        crop_size: a `crop_size` x `crop_size` patch is taken from the
            CENTER of the image at its NATIVE resolution -- NOT a resize
            of the whole frame. This matters a lot: resizing a full photo
            (e.g. a 3000px-wide phone photo) down to a small target size
            averages dozens of original sensor pixels into each output
            pixel, which smears out exactly the fine-grained per-pixel
            noise that PRNU depends on. A center crop at native resolution
            preserves it. (An earlier version of this function resized the
            whole image instead, which is very likely why PRNU-branch-only
            accuracy was much lower than expected.)

    Returns:
        2D float32 residual, zero-meaned row-wise and column-wise.
    """
    if img_bgr is None:
        raise ValueError("Input image is None - check the file path.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _center_crop_or_upscale(gray, crop_size)

    denoised = _wavelet_denoise(gray)
    residual = gray.astype(np.float32) - denoised.astype(np.float32)

    # Suppress non-unique artifacts (row/column banding, JPEG blocking).
    residual = residual - residual.mean(axis=1, keepdims=True)
    residual = residual - residual.mean(axis=0, keepdims=True)
    return residual


def extract_patches(residual: np.ndarray, patch_size: int = 96, stride: int = 64):
    """Split a residual into overlapping square patches.

    With crop_size=512, patch_size=96, stride=64 this yields 49 patches
    per image (7x7 grid) -- i.e. 200 images/class becomes ~9,800 training
    samples/class for the CNN branch, at zero extra data-collection cost.
    """
    h, w = residual.shape
    patches = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(residual[y:y + patch_size, x:x + patch_size])
    return patches


def build_camera_fingerprint(residuals: list) -> np.ndarray:
    """Average a list of residuals into one reference fingerprint for a camera.

    This is the classical PRNU step: the average residual over many images
    converges to the camera's fixed sensor pattern, while scene content
    (which differs image to image) averages toward zero. Unlike training a
    classifier, this needs comparatively few images to work well -- the
    original Lukas et al. paper used as few as 50.
    """
    stacked = np.stack(residuals, axis=0)
    return np.mean(stacked, axis=0)


def normalized_cross_correlation(residual: np.ndarray, fingerprint: np.ndarray) -> float:
    """Normalized cross-correlation between a query residual and a camera
    fingerprint. Since both come from the same crop_size center crop and are
    not cropped/shifted relative to each other, plain NCC (no FFT shift-search)
    is sufficient and much cheaper than full PCE computation."""
    a = residual.flatten() - residual.mean()
    b = fingerprint.flatten() - fingerprint.mean()
    denom = (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2))) + 1e-8
    return float(np.sum(a * b) / denom)


def prnu_correlation_features(residual: np.ndarray, fingerprints: list) -> np.ndarray:
    """Correlate a residual against every camera's fingerprint.

    Returns a vector of length len(fingerprints) -- one correlation score
    per camera. This vector alone is often enough to classify the camera
    (argmax = highest-correlation camera), and it also serves as a compact,
    highly informative feature to feed into the meta-classifier alongside
    the CNN branch's predictions.
    """
    return np.array([normalized_cross_correlation(residual, fp) for fp in fingerprints],
                     dtype=np.float32)


def residual_to_cnn_input(patch: np.ndarray, clip_std: float = 4.0) -> np.ndarray:
    """Convert a raw PRNU residual/patch into normalized 3-channel CNN input.

    IMPORTANT: a PRNU residual is NOT a 0-255 pixel intensity map. It's a
    small, real-valued, zero-mean noise signal (typically in roughly
    [-15, +15] for an 8-bit image after Wiener denoising). Naively dividing
    it by 255.0 -- as if it were a normal image -- crushes almost the
    entire signal into a tiny band around zero and starves the CNN branch
    of contrast to learn from.

    This rescales each patch to zero mean / unit variance, clips outliers,
    and maps to [0, 1] (the range expected before an ImageNet backbone's
    own internal preprocessing), then fakes RGB by repeating the channel.

    Using this SAME function everywhere a residual patch is turned into
    CNN input (training, validation, testing, and single-image prediction)
    is what keeps the CNN branch's training and inference distributions
    consistent -- previously `image_level_features()` (used for the
    meta-classifier and predict.py) fed the CNN raw, un-normalized residual
    values, silently mismatching what the network was actually trained on.
    """
    x = patch.astype(np.float32)
    mean = x.mean()
    std = x.std() + 1e-8
    x = (x - mean) / std
    x = np.clip(x, -clip_std, clip_std) / clip_std   # now in [-1, 1]
    x = (x + 1.0) / 2.0                              # now in [0, 1]
    return np.repeat(x[..., np.newaxis], 3, axis=-1)  # fake RGB


def patches_to_cnn_batch(patches) -> np.ndarray:
    """Vectorized version of residual_to_cnn_input for a list/array of patches."""
    if len(patches) == 0:
        return np.zeros((0, 0, 0, 3), dtype=np.float32)
    arr = np.stack([np.asarray(p, dtype=np.float32) for p in patches])  # (N, H, W)
    mean = arr.mean(axis=(1, 2), keepdims=True)
    std = arr.std(axis=(1, 2), keepdims=True) + 1e-8
    arr = (arr - mean) / std
    arr = np.clip(arr, -4.0, 4.0) / 4.0
    arr = (arr + 1.0) / 2.0
    return np.repeat(arr[..., np.newaxis], 3, axis=-1)

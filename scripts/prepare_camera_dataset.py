"""
scripts/prepare_camera_dataset.py
----------------------------------
Turns a large public source-camera dataset into a small, repo-friendly,
pipeline-ready `data/raw/` folder, WITHOUT losing PRNU signal quality.

The key idea: this pipeline only ever needs a native-resolution CENTER
CROP of each image (see prnu_extraction.extract_prnu_residual) -- never
the whole multi-megapixel original. So instead of storing full-resolution
source photos (several MB each) and committing gigabytes to git, this
script crops each selected image ONCE, up front, to a size a little larger
than the pipeline's own --crop-size (so the pipeline's internal crop still
has room and never touches an edge), and saves just that crop at high
(but not lossless-huge) JPEG quality. A few hundred images at 768x768
typically comes out to a few hundred MB total, not several GB -- while
the residual extraction and everything downstream is IDENTICAL to using
the full original image, because that's the only part of the image the
pipeline would ever look at anyway.

Recommended dataset for this pipeline: Kaggle's "IEEE's Signal Processing
Society - Camera Model Identification" dataset (search "sp-society-camera
-model-identification" on kaggle.com) -- 10 distinct camera models, one
device per model, 275 full-resolution images/device, downloaded via
`kaggle competitions download -c sp-society-camera-model-identification`.
Its `train/` folder is already laid out as one subfolder per camera model
-- exactly what "folders mode" below expects, no reorganizing needed.

USAGE (folders mode -- images already sorted into one folder per camera,
e.g. the Kaggle dataset's train/ folder, or your own captured photos):

    python prepare_camera_dataset.py \\
        --input-dir /path/to/train \\
        --output-dir ../data/raw \\
        --per-camera 200 --crop-size 768 --jpeg-quality 95

USAGE (regex mode -- all images are in one flat folder, with the camera
identifiable from the filename -- some public datasets name files with the
camera brand/model/device embedded in the filename):

    python prepare_camera_dataset.py \\
        --input-dir /path/to/flat_folder \\
        --filename-regex "(?P<camera>[A-Za-z]+_[A-Za-z0-9]+)_\\d+_\\d+\\.(jpg|JPG)" \\
        --output-dir ../data/raw --per-camera 200

Either way, verify the actual camera/device folder or file names in the
source dataset yourself once downloaded -- this script doesn't know or
guess at any dataset's exact naming scheme, and dataset.py auto-discovers
whatever class names end up under data/raw/, so there's nothing to hardcode.
"""

import argparse
import os
import random
import re
from collections import defaultdict

import cv2
import numpy as np

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _crop_center(img: np.ndarray, size: int) -> np.ndarray:
    """Center crop (or minimal upscale-to-fit) at native resolution.

    Deliberately mirrors prnu_extraction._center_crop_or_upscale's exact
    formula (top/left = (dim - size) // 2) so that whatever the pipeline
    crops out of these pre-cropped images lines up with the same
    underlying pixels a from-scratch run against the originals would have
    used -- this script is a size optimization, not a different crop.
    """
    h, w = img.shape[:2]
    if h < size or w < size:
        scale = size / min(h, w)
        img = cv2.resize(img, (int(np.ceil(w * scale)), int(np.ceil(h * scale))))
        h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top:top + size, left:left + size]


def discover_folders_mode(input_dir: str):
    """camera -> [file paths], one subfolder per camera under input_dir."""
    groups = {}
    for name in sorted(os.listdir(input_dir)):
        sub = os.path.join(input_dir, name)
        if not os.path.isdir(sub):
            continue
        files = [os.path.join(sub, f) for f in sorted(os.listdir(sub))
                  if f.lower().endswith(VALID_EXTENSIONS)]
        if files:
            groups[name] = files
    return groups


def discover_regex_mode(input_dir: str, pattern: str):
    """camera -> [file paths], inferred from filenames via a regex with a
    named group 'camera'."""
    regex = re.compile(pattern)
    groups = defaultdict(list)
    unmatched = 0
    for root, _, files in os.walk(input_dir):
        for f in sorted(files):
            if not f.lower().endswith(VALID_EXTENSIONS):
                continue
            m = regex.search(f)
            if m and "camera" in m.groupdict():
                groups[m.group("camera")].append(os.path.join(root, f))
            else:
                unmatched += 1
    if unmatched:
        print(f"[WARN] {unmatched} files didn't match --filename-regex and were skipped. "
              f"Check the pattern against an actual filename if this seems too high.")
    return dict(groups)


def estimate_total_size(groups, cameras, per_camera, crop_size, jpeg_quality, sample_n=5):
    """Encode a handful of real crops in-memory to estimate the total
    output size before committing to writing everything to disk."""
    sizes = []
    for cam in cameras[:3]:
        for f in groups[cam][:sample_n]:
            img = cv2.imread(f)
            if img is None:
                continue
            cropped = _crop_center(img, crop_size)
            ok, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if ok:
                sizes.append(len(buf))
    if not sizes:
        return None
    avg_bytes = sum(sizes) / len(sizes)
    total_images = len(cameras) * per_camera
    return avg_bytes * total_images / (1024 ** 2)  # MB


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 ** 2)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                    help="Path to the downloaded source dataset.")
    p.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(__file__), "..", "data", "raw"))
    p.add_argument("--filename-regex", default=None,
                    help="If set, use flat/regex mode instead of folders mode. "
                         "Must contain a named group 'camera'.")
    p.add_argument("--cameras", nargs="*", default=None,
                    help="Optional: restrict to (and order) these camera names. "
                         "Default: use every camera discovered, alphabetically.")
    p.add_argument("--per-camera", type=int, default=200,
                    help="Total images to keep per camera (this pipeline's own "
                         "--fingerprint-n is carved out of this total later by "
                         "hybrid_train.py -- you don't need to split it here).")
    p.add_argument("--crop-size", type=int, default=768,
                    help="Stored crop size. Should be a bit LARGER than the "
                         "--crop-size you'll train with (default 512), so the "
                         "pipeline's own crop always lands away from any edge.")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-total-mb", type=float, default=1200,
                    help="Abort before writing anything if the estimated total "
                         "exceeds this. Use --force to proceed anyway.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.filename_regex:
        groups = discover_regex_mode(args.input_dir, args.filename_regex)
    else:
        groups = discover_folders_mode(args.input_dir)

    if not groups:
        raise SystemExit(f"No cameras discovered under {args.input_dir}. "
                          f"Check --input-dir and (if used) --filename-regex.")

    cameras = args.cameras if args.cameras else sorted(groups.keys())
    missing = [c for c in cameras if c not in groups]
    if missing:
        raise SystemExit(f"Requested cameras not found: {missing}. "
                          f"Discovered: {sorted(groups.keys())}")

    print(f"Discovered {len(groups)} camera(s); using {len(cameras)}: {cameras}")
    for cam in cameras:
        print(f"  {cam}: {len(groups[cam])} source images available")
        if len(groups[cam]) < args.per_camera:
            print(f"  [WARN] only {len(groups[cam])} available, "
                  f"fewer than --per-camera={args.per_camera}")

    est_mb = estimate_total_size(groups, cameras, args.per_camera,
                                  args.crop_size, args.jpeg_quality)
    if est_mb is not None:
        print(f"\nEstimated total output size: ~{est_mb:.0f} MB "
              f"(budget: {args.max_total_mb:.0f} MB)")
        if est_mb > args.max_total_mb and not args.force:
            raise SystemExit(
                "Estimated size exceeds --max-total-mb. Lower --per-camera, "
                "--crop-size, or --jpeg-quality, or re-run with --force to "
                "proceed anyway."
            )

    for cam in cameras:
        files = list(groups[cam])
        random.shuffle(files)
        files = files[:args.per_camera]

        out_dir = os.path.join(args.output_dir, cam)
        os.makedirs(out_dir, exist_ok=True)

        written = 0
        for f in files:
            img = cv2.imread(f)
            if img is None:
                print(f"    [skip] unreadable: {f}")
                continue
            cropped = _crop_center(img, args.crop_size)
            out_name = f"{written:04d}_{os.path.splitext(os.path.basename(f))[0]}.jpg"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            written += 1
        print(f"  {cam}: wrote {written} cropped images -> {out_dir}")

    actual_mb = dir_size_mb(args.output_dir)
    print(f"\nDone. Actual total size of {args.output_dir}: {actual_mb:.0f} MB")
    print("Next: just run hybrid_train.py -- dataset.py auto-discovers camera "
          "classes from these folder names, nothing to edit in code.")


if __name__ == "__main__":
    main()

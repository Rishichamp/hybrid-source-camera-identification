# Changelog

## v2.3 -- CNN branch stability fix + public-dataset support (accuracy push)

Requested goal: push toward 90%+ accuracy. This isn't something to
guarantee ahead of time -- real accuracy is an outcome of data and task
difficulty, not a dial -- but two concrete, legitimate changes make it a
realistic target rather than wishful thinking.

**Fixed: CNN branch instability.** The v2.2 run's own log showed the
problem plainly: train accuracy climbing smoothly (47%->63%) while
val_loss became wildly unstable (spiking above 28), and CNN-only accuracy
landing BELOW the random-guess baseline (19% on 5 classes). That pattern
means the network was overfitting to incidental, spurious detail in
training patches (leftover scene texture that happened to correlate with
camera label in that split) with nothing to stop it, not "not learning."
`cnn_branch.py` now adds:
- L2 weight regularization on every conv/dense layer.
- Label smoothing (0.1) on the loss, which caps how large the loss can get
  for any single confidently-wrong prediction -- directly targeting the
  val_loss-spike symptom.
- A lower learning rate (3e-4 vs. 1e-3) -- loss *exploding* between
  epochs, rather than just plateauing, usually means the step size itself
  is too large for the landscape.
- `hybrid_train.py`'s data augmentation dropped rotation/zoom (both
  require interpolating between pixels, which blurs the very
  high-frequency noise signal PRNU depends on) in favor of flips and small
  integer-ish shifts (no resampling).
- `EarlyStopping` patience raised 6 -> 10 to match the slower, steadier
  convergence a regularized, lower-LR network produces.

**Added: dataset flexibility + public-dataset support, sized for a portfolio repo.**
- Camera classes are now AUTO-DISCOVERED from `data/raw/`'s subfolder
  names (`dataset.discover_camera_classes`) instead of a hardcoded
  `PHONE_CLASSES`/`CAMERA_CLASSES` list -- drop in any dataset (your own
  photos, a Kaggle set, anything) as one folder per camera and it just
  works, with zero code edits. `hybrid_train.py` saves the discovered
  class list to `saved_models/class_names.json`; `predict.py` and
  `demo/app.py` load it back from there instead of importing any
  hardcoded constant, so a trained model is fully self-describing.
  `image_level_features()`'s feature-vector slicing was also generalized
  (it previously hardcoded a 5-class assumption, which would have quietly
  broken on a dataset with a different number of classes).
- `scripts/prepare_camera_dataset.py`: given a public dataset organized
  either as one folder per camera, or as a flat folder with the camera
  identifiable via a filename regex, this crops each selected image ONCE
  at native resolution (reusing the exact same center-crop geometry as
  `prnu_extraction.py`, just applied ahead of time) and writes a small,
  repo-ready `data/raw/` -- typically shrinking a multi-GB source down to
  a few hundred MB with **zero loss of PRNU signal quality**, since the
  crop is the only part of the original image the pipeline would ever
  look at anyway. Estimates total output size before writing anything and
  aborts over a `--max-total-mb` budget (default 1200MB) unless `--force`
  is passed.
- **Dataset recommendation corrected mid-project:** the Dresden Image
  Database is the traditional academic benchmark for this task, but
  multiple current reports describe its original download host as
  unreliable/dead, so the README instead recommends Kaggle's "IEEE's
  Signal Processing Society - Camera Model Identification" dataset (10
  camera models, 275 images/device, reliably hosted, and it's the same
  "different physical devices per model" task structure this project
  already uses).
- README now recommends NOT committing the prepared dataset directly to
  git (large repos are bad practice and GitHub discourages them) -- commit
  the prep script + code + results instead, so anyone can reproduce the
  exact dataset from the freely-downloadable source.
- 5 new unit tests for the prep script's cropping and camera-discovery
  logic (folder mode and regex mode).

## v2.2 -- root-caused a real ~50% accuracy result (methodology unchanged)

After running v2.1 on a real 200-images/camera dataset, the honest,
non-fabricated result came back at ~50% hybrid accuracy (5-fold CV:
52.00% +/- 4.17%), CNN-branch-only at ~33% (barely above the 20% random
baseline), PRNU-branch-only at ~47%. Both branches underperforming pointed
to two specific, fixable causes rather than "just needs more data":

**Fixed**
- **Residual extraction was resizing the whole photo, not cropping it.**
  `extract_prnu_residual()` called `cv2.resize(gray, (256, 256))` on the
  *entire* image regardless of its original resolution. For a modern (or
  even 2010s-era) phone photo at 2000px+ on its shorter side, this averages
  dozens of original sensor pixels into every output pixel -- which
  directly attenuates the fine, per-pixel sensor noise that PRNU is built
  on. This is a well-known trap in the PRNU literature (Lukas, Fridrich &
  Goljan and follow-up work always crop at native resolution). Fixed by
  taking a `crop_size` x `crop_size` (default 512) crop from the CENTER of
  the image at its native resolution (`_center_crop_or_upscale` in
  `prnu_extraction.py`), only upscaling (never downscaling) if a source
  image happens to be smaller than `crop_size`.
- **The CNN branch was a poor architectural fit for noise-residual data.**
  v2.1 used MobileNetV2 pretrained on ImageNet with only its last 4 layers
  unfrozen. Its pretrained filters encode natural-image edges/textures/
  colors -- a PRNU residual is deliberately the opposite of natural-image
  content, so those filters had little to latch onto, and training
  plateaued at ~33% (barely above chance) rather than overfitting.
  Replaced with a small custom CNN (`cnn_branch.py`), trained entirely
  from scratch: 4 conv+batchnorm+pool blocks, ~1/25th the parameters of
  MobileNetV2, sized appropriately for training from scratch on a few
  thousand patches/class. This follows the same approach used in
  camera-model-identification literature that classifies noise residuals
  directly (Bondi et al. 2017 and related work), rather than fine-tuning
  an ImageNet classifier.
- `--target-size` renamed to `--crop-size` everywhere (hybrid_train.py,
  cross_validate.py, predict.py, demo/app.py) to make clear this is a
  native-resolution crop, not a resize.
- With `crop_size=512`, `patch_size=96`, `stride=64`, each image now
  yields 49 patches (7x7 grid) instead of 9 -- ~5x more training samples
  per image, at zero extra data-collection cost, purely as a side effect
  of extracting the residual correctly.

**Added**
- Unit tests confirming the center crop takes an unmodified pixel window
  (not an interpolated/resized one), and that undersized source images are
  handled by upscaling rather than crashing or silently misbehaving.

**A note on what this does and doesn't fix:** these are structural,
literature-grounded corrections to real weaknesses, not a guarantee of any
specific accuracy number. Retrain and let `reports/results.txt` /
`reports/cross_validation.txt` tell you what you actually get.

## v2.1 -- bug fixes, evaluation tooling, tests (methodology unchanged)

The hybrid design itself (wavelet-domain PRNU + patch-based MobileNetV2 +
logistic-regression fusion) is exactly as in v2. Everything below is a
correctness fix or added tooling, not a change to the approach.

**Fixed**
- **CNN input normalization mismatch (the big one).** A PRNU residual is a
  small, real-valued, zero-mean noise signal, not a 0-255 image. The CNN
  branch was trained on residuals scaled by a plain `/255.0`, which crushes
  almost all of the signal into roughly `[-0.08, 0.08]`. Worse,
  `hybrid_train.image_level_features()` -- used to build the meta-classifier's
  training data and every prediction in `predict.py`/the demo -- fed the CNN
  **raw, unscaled** residuals with no normalization at all, a straight
  train/inference distribution mismatch. Both paths now go through a single
  shared function (`prnu_extraction.residual_to_cnn_input` /
  `patches_to_cnn_batch`): standardize to zero-mean/unit-variance, clip
  outliers, map to `[0, 1]`. Used consistently in `dataset.py`,
  `hybrid_train.py`, `cross_validate.py`, `predict.py`, and `demo/app.py`.
- Removed the now-deprecated `multi_class="auto"` argument to
  `LogisticRegression`.
- Meta-classifier is now a `StandardScaler` + `LogisticRegression` pipeline
  (`class_weight="balanced"`, fixed `random_state=42`) instead of an
  un-scaled classifier -- PRNU correlation scores and CNN softmax scores
  live on different numeric scales, which can otherwise bias the fusion
  step for reasons that have nothing to do with which branch is actually
  more informative.

**Added**
- `src/reporting.py`: saves `reports/figures/training_curves.png`,
  `reports/figures/confusion_matrix.png`, and appends a timestamped run
  summary to `reports/results.txt` (and `reports/cross_validation.txt` for
  `cross_validate.py`) -- generated directly from each real run's output.
- `tests/test_prnu_extraction.py`: unit tests for the core PRNU math that
  don't need the dataset, including a regression guard against the
  normalization bug above re-appearing.
- `.github/workflows/tests.yml`: runs the unit tests on every push/PR.
- `requirements-dev.txt` for test-only dependencies.

**Performance**
- `image_level_features()` now batches all of an image-set's patches into
  a single `cnn_model.predict()` call instead of one call per image, which
  is both faster and removes per-call Keras overhead.

## v2 -- hybrid PRNU + CNN redesign for a smaller dataset

See the main README for the full v1 -> v2 comparison (wavelet PRNU vs.
Gaussian-blur high-pass, MobileNetV2 vs. DenseNet121, patch-based sampling,
classical PRNU correlation branch, meta-classifier fusion, cross-validation).

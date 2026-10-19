# IMAGO_LAB repository audit — 2026-10-09

## Scope and safety
This is a read-only audit of `HdlArf/IMAGO_LAB` using GitHub code search and read-only file fetches. No GitHub writes were performed. Because the GitHub code-search index does not expose a complete repository tree, this is an evidence-based partial inventory, not a claim that every repository file has been enumerated.

## Lab catalogue found by repository search
- **LAB 01 — Basics:** `image_io.py`, `color_spaces.py`, `image_properties.py`.
- **LAB 02 — Thresholding / binary:** `grayscale.py`, `binary.py`, `threshold.py`, `manual_threshold.py`.
- **LAB 03 — Geometry:** `crop.py`, `resize.py`, `rotation.py`, `translation.py`.
- **LAB 04 — Arithmetic / drawing:** `addition.py`, `subtraction.py`, `multiplication.py`, `division.py`, `difference.py`, `blending.py`, `drawing.py`.
- **LAB 05 — Logical transforms:** `masks.py`, `log_transform.py`, `gamma_transform.py`, `object_extraction.py`, `logical_operations.py`.
- **LAB 06 — Histogram / filters:** `contrast.py`, `histogram.py`, `equalization.py`, `mean_filter.py`, `gaussian_filter.py`, `median_filter.py`, `spatial_filter.py`.
- **LAB 07 — Sharpening / edges:** `sharpening.py`, `canny.py`, `sobel.py`, `laplacian.py`.
- **LAB 08 — Detection:** `ball_detection.py`, `hough_lines.py`, `hough_circles.py`, `corners.py`.
- **LAB 09 — Segmentation / morphology:** `otsu.py`, `kmeans.py`, `watershed.py`, `threshold_segmentation.py`, `adaptive_threshold.py`, `color_masking.py`, `erosion.py`, `dilation.py`, `opening.py`, `closing.py`, `gradient.py`, `tophat.py`, `blackhat.py`.
- **LAB 10 — Model-based detection:** confirmed code files `labs/lab10_face_eye/face_detection.py` and `labs/lab10_face_eye/eye_detection.py`. These reference local frontal-face and eye cascades. Search also found smile-cascade and Russian license-plate-cascade assets at `models/smile/haarcascade_smile.xml` and `models/face/haarcascade_license_plate_rus_16stages.xml`, plus a nested copy of the plate cascade. Full-body, upper-body, lower-body, profile-face, cat-face, and other cascade names were **not found as assets in the repository search results**.

## Model availability distinction
The current Python environment's OpenCV installation includes cascade files for eye, frontalcatface, frontalcatface_extended, frontalface variants, fullbody, lowerbody, profileface, smile, Russian plate, and upperbody. These are **OpenCV-installed fallback assets**, not confirmed assets bundled in GitHub. They are traditional Haar Cascade detectors, not neural body-part segmentation models.

## Current PySide6 app audit
The current sidebar groups 64 tool labels into eight groups, but the labels are not equivalent to 54 independently validated implementations. Processing functions exist for many common operations in `imago/core/processing.py`. The old repository's LAB implementations have not yet all been copied verbatim into the new app; several functions are app-side implementations or wrappers.

Identified gaps/risks for the next implementation pass:
1. Additional LAB-inspired controls have since been added for color-space views, drawing primitives, logical operations, color masks, and threshold-based object extraction. A full one-to-one source-signature audit remains outstanding.
2. The generic model selector includes lower-body detection. This environment confirms that OpenCV supplies fullbody, upperbody, lowerbody, face, profile-face, eye, smile, Russian plate, and cat-face cascades; the new app uses those installed assets as fallbacks when local model files are absent.
3. The app uses a generic detector wrapper for all cascades rather than the repository's LAB 10 face/eye functions, so the exact original model paths and behavior are not yet preserved in the PySide6 package.
4. The package does not bundle all XML assets; nine detector choices were smoke-tested against this environment's `cv2.data.haarcascades` assets. Running requires OpenCV and may rely on `cv2.data.haarcascades` for fallbacks.
5. PySide6 is not installed in this audit environment, so the GUI cannot yet be launched here. Core tests are runnable.

## Tests run
Command: `python -m unittest discover -s tests -v` from `IMAGO_PySide6_v1`.
Result after the latest implementation pass: **10 tests passed**. These validate selected processing outputs, deterministic Canny previews from a stable input, basic arithmetic, and ImageState apply/undo/redo/reset. They do not validate GUI interactions or every tool end-to-end.

## Next verification steps
1. Enumerate remaining source files by additional read-only repository queries or a local read-only archive if available.
2. Inspect every LAB module's function signatures and dependencies before porting.
3. Map each confirmed source operation to a GUI control and a test.
4. Bundle or explicitly resolve each model file and record provenance.
5. Install requirements in a separate test environment and launch the GUI before calling the desktop build validated.

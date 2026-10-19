# IMAGO_LAB inventory — evidence-based working catalogue

This catalogue is a working inventory based on files found in the public `HdlArf/IMAGO_LAB` repository. It is not a claim that every operation below has already been copied verbatim or fully validated inside the PySide6 app. Old Gradio UI and backup files are intentionally excluded from the migration target.

## LAB 01 — Basics
- Image I/O (`image_io.py`)
- Color spaces (`color_spaces.py`)
- Image properties (`image_properties.py`)

## LAB 02 — Thresholding / binary
- Grayscale conversion (`grayscale.py`)
- Binary conversion (`binary.py`)
- Threshold (`threshold.py`)
- Manual threshold (`manual_threshold.py`)

## LAB 03 — Geometry
- Crop / ROI (`crop.py`)
- Resize (`resize.py`)
- Rotation (`rotation.py`)
- Translation (`translation.py`)

## LAB 04 — Arithmetic / drawing
- Addition, subtraction, multiplication, division, difference
- Blending
- Drawing primitives

## LAB 05 — Logical transforms
- Masks
- Log transform
- Gamma transform
- Object extraction
- Logical operations

## LAB 06 — Histogram / filters
- Contrast adjustment
- Histogram calculation
- Histogram equalization
- Mean filter
- Gaussian filter
- Median filter
- Spatial filter

## LAB 07 — Sharpening / edges
- Sharpening
- Canny edges
- Sobel
- Laplacian

## LAB 08 — Detection
- Ball-like circle detection
- Hough lines
- Hough circles
- Corner detection

## LAB 09 — Segmentation / morphology
- Otsu threshold
- K-means segmentation
- Watershed segmentation
- Threshold segmentation
- Adaptive threshold
- Color masking
- Erosion, dilation, opening, closing, gradient, top-hat, black-hat

## LAB 10 — Model-based detection
Repository paths confirmed by search:
- `labs/lab10_face_eye/face_detection.py`
- `labs/lab10_face_eye/eye_detection.py`
- Model references include frontal-face and eye Haar cascades. Search results also surfaced smile and Russian license-plate cascade files, including duplicates in nested model locations.

The repository code-search index did not reliably enumerate every XML asset. In the current app, nine detector choices are available: frontal face, profile face, eyes, full body, upper body, lower body, smile, Russian license plate, and cat face. These use local files if present and otherwise the installed OpenCV cascade data. This confirms runtime fallback availability, not that every XML is stored in the source repository. The new app's detector wrapper can use OpenCV's installed Haar Cascade assets as a fallback for full-body, upper-body, lower-body, profile-face, smile, license-plate, eye, and cat-face detection. These are cascade detectors, not modern neural segmentation models; detections are approximate and depend on image conditions.

## Existing Develop UI list
The old `ui/develop.py` surfaced 14 primary controls: Exposure, Contrast, Saturation, Tone, Grayscale, Gaussian Blur, Sharpen, Resize, Rotate, Translate, Threshold, Otsu, Edges/Canny, Equalize. This is not the total lab count.

## Migration status in this PySide6 package
- Processing implementations currently available in `imago/core/processing.py` include tone/color, geometry, thresholding, filters, edges, Hough-based detections, morphology, K-means, watershed, arithmetic, and a generic Haar-cascade detector.
- Main-window controls are being wired in batches. Presence in this inventory alone does not mean a GUI control or model asset has been validated.
- No source file in the original GitHub repository is modified by this work.

## Current GUI groups (64 tool entries)
- Develop (14): Exposure, Brightness, Contrast, Saturation, Tone, Grayscale, Gaussian Blur, Mean Filter, Median Filter, Sharpen, Resize, Rotate, Translate, Crop.
- Tone & Color (7): Equalize, Log Transform, Gamma Transform, Negative, HSV View, Lab View, YCrCb View.
- Image Arithmetic (6): Add, Subtract, Difference, Multiply, Divide, Blend Images.
- Masks & Logic (9): Logical NOT/AND/OR/XOR, Extract Object, Color Mask, Draw Line/Rectangle/Circle.
- Thresholding (4): Threshold, Inverse Threshold, Otsu, Adaptive Threshold.
- Edges & Detection (6): Canny, Sobel, Laplacian, Detect Lines/Circles/Corners.
- Segmentation & Morphology (9): K-means, Watershed, Erode, Dilate, Open, Close, Gradient, Top Hat, Black Hat.
- Face & Object Models (9): Face, Profile Face, Eyes, Full Body, Upper Body, Lower Body, Smile, License Plate, Cat Face.

## Validation after the latest implementation pass
- 10 core unit tests pass.
- Syntax compilation passes for the Python package.
- Nine OpenCV Haar cascade assets were loaded and exercised on a blank test image.
- GUI runtime testing remains unverified because PySide6 could not be installed in the isolated environment.
- The inventory remains evidence-based: available GitHub tools did not expose a complete repository tree, so this is not a mathematically exhaustive source-file count and the original source repository remains read-only.

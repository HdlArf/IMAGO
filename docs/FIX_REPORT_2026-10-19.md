# IMAGO Repair Pass 2 — 2026-10-19

## Changes in this pass

- Improved the HΛDΞL splash rendering with high-DPI text, a clean black canvas, and a restrained fade transition. The signature remains logo-only.
- Added thumbnails to Filmstrip items, a `Remove Selected` button, a right-click `Remove from Session` action, and the Delete-key shortcut. Removal affects the current IMAGO session only; it never deletes the original disk file.
- Added per-image `ImageState` storage so selecting another Filmstrip image no longer discards previously committed edits on the image being left.
- Made first previews run immediately after selecting a tool and when resetting a tool, instead of requiring an arbitrary slider movement to initialize operations.
- Fixed Inspector cleanup recursively: old sliders nested inside row layouts could survive tool switches and contaminate later tools' parameter reads.
- Set blur filter defaults to a non-identity 3×3 kernel.
- Removed the 4000-pixel Resize ceiling for images already larger than 4000 pixels; the controls now preserve the original dimensions as their initial values.
- Corrected Exposure so it uses exposure compensation (gain in stops) instead of duplicating Brightness.
- Corrected Brightness clipping for negative values, and Contrast now pivots around middle gray. These controls preserve RGBA alpha.
- Updated Sharpen, Log Transform, Gamma Transform, Negative, and K-means Segmentation to preserve the alpha channel instead of modifying opacity; fixed K-means on RGBA images, which previously failed on reshape.
- Expanded tests for all non-model tool families, alpha preservation, tool intensity controls, and all bundled Haar XML cascades.
- Kept Lab View and YCrCb View. The original GitHub repository was not modified.

## Automated verification

- `python -m compileall -q .`: passed.
- `python -m unittest discover -s tests -v`: 16 tests passed.
- 56 non-model processing tool operations were smoke-tested with synthetic RGB input.
- All 17 bundled Haar Cascade XML files loaded successfully and were exercised with a small blank-image inference smoke test.
- Face, eye, smile, body, and cat cascade families were also run against generated portrait/cat/smile contact sheets. These are smoke checks only; classical Haar cascades can produce false positives on contact sheets and do not guarantee detection accuracy on arbitrary photographs.

## Verification still pending

PySide6 is not installed in the current execution environment, so the actual GUI could not be launched here. Therefore the Filmstrip interaction, splash visual appearance, first-preview behavior inside Qt, and Apply/Reset/Undo/Redo interactions still need a GUI runtime test on a machine with the project's dependencies installed. The algorithm and model smoke tests do not substitute for that runtime test.

# IMAGO PySide6 — V2 development pass

## Added or improved
- Expanded the sidebar to 64 tool entries across 8 categories.
- Added manual Favorites and Arabic/English keyword search.
- Added color-space preview helpers for HSV, CIELAB, and YCrCb.
- Added threshold-based object extraction, HSV color masking, logical NOT/AND/OR/XOR, and drawing demonstrations (line/rectangle/circle).
- Added a lightweight histogram to the image Inspector when no tool is active.
- Fixed apply semantics so APPLY commits one checkpoint and returns the Inspector to image information.
- Fixed Reset Tool to restore the committed image before resetting controls.
- Hardened image arithmetic to handle grayscale/RGB/RGBA inputs more safely.
- Added tests for the additional processing functions and nine installed OpenCV cascade models.

## Verification
- `python -m compileall -q .`: passed.
- `python -m unittest discover -s tests -v`: 10 tests passed.
- GUI launch/interaction: not verified in this environment because PySide6 is absent and package installation could not access the network.
- Original GitHub repository: read-only; no writes were made.

## Remaining release gates
- Install requirements and run the GUI on a desktop environment; verify tool switching, drag/drop, image navigation, Favorites/search, Inspector/histogram, preview, Apply/Reset Tool, Undo/Redo, and export.
- Complete source-by-source parity review against every LAB module in `HdlArf/IMAGO_LAB`; current GitHub code search did not expose a complete tree, and not all original implementations have been copied verbatim.
- Validate detector behavior on representative real images. OpenCV Haar cascades are classical detectors and are not equivalent to neural body-part segmentation.

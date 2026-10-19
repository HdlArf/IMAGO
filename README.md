# IMAGO — PySide6 working build

This is the separate PySide6 desktop project. It does not write to or modify the original `HdlArf/IMAGO_LAB` GitHub repository.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```

## Current implementation

- Carbon dark interface with cyan accent picker; the IMAGO wordmark follows the selected accent color.
- Minimal black HΛDΞL signature splash at startup, then transition to the main interface.
- Manual Favorites at the top of the tool list and Arabic/English keyword search.
- Image import, folder import, drag/drop, filmstrip, zoom-to-fit and 100% view.
- Live preview computed from the committed image, not from the previous preview.
- Apply and Reset Tool buttons remain hidden unless the current preview differs from the committed image.
- Undo/redo history checkpoints and reset to original.
- 64 sidebar tool entries across eight groups: Develop, Tone & Color, Image Arithmetic, Masks & Logic, Thresholding, Edges & Detection, Segmentation & Morphology, and Face & Object Models. Entries are mapped to processing functions in the current app; not every item is a verbatim port of a source LAB file.
- Detector controls use local `models/` files if present, then OpenCV's bundled Haar cascades as fallback. Detection quality depends on the selected cascade and image conditions.
- `docs/TOOL_INVENTORY.md` and `docs/REPOSITORY_AUDIT.md` document the evidence-based source catalogue and its limits. The GitHub source repository remains unchanged.

## Validation status

- Python syntax compilation: passed.
- Core unit tests: passed (10 tests), including selected processing operations, histogram metadata, arithmetic, and loading nine Haar cascade models from the installed OpenCV distribution.
- The GUI itself was not launched in this environment because PySide6 is not installed and package installation was blocked by unavailable network access. Install requirements on a desktop environment and run `python main.py` for interactive verification.
- This is an expanded working build, not a final release claim: the repository tree is not fully enumerable through the available GitHub search tool, some original lab functions are reimplemented/wrapped rather than copied verbatim, and the full GUI workflow still needs a desktop smoke test. The nine detector choices use local model files if available and otherwise fall back to OpenCV cascades; these are classical cascades, not neural body-part segmentation models.

## Filmstrip behavior

- Select a thumbnail to switch the active image.
- Use **Remove Selected**, right-click a thumbnail and choose **Remove from Session**, or press **Delete** to remove it from the current session only. The original file on disk is not deleted.
- Each imported image keeps its own committed edit history while switching between Filmstrip thumbnails.

## Validation note

See `docs/FIX_REPORT_2026-10-19.md` for this repair pass and the distinction between processing/model smoke tests and GUI runtime testing. GUI interactions must still be verified on a system with PySide6 installed.

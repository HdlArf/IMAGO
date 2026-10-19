"""Image state with immutable original, preview isolation, and undo/redo checkpoints."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class ImageState:
    original: Optional[np.ndarray] = None
    current: Optional[np.ndarray] = None
    history: list[tuple[str, np.ndarray]] = field(default_factory=list)
    redo_stack: list[tuple[str, np.ndarray]] = field(default_factory=list)

    def load(self, image: np.ndarray) -> np.ndarray:
        arr = self._validate(image)
        self.original = arr.copy()
        self.current = arr.copy()
        self.history = [("Original", arr.copy())]
        self.redo_stack.clear()
        return self.current.copy()

    def preview_base(self) -> np.ndarray:
        if self.current is None:
            raise ValueError("Import an image first.")
        return self.current.copy()

    def apply(self, image: np.ndarray, label: str) -> np.ndarray:
        if self.current is None:
            raise ValueError("Import an image first.")
        arr = self._validate(image)
        self.current = arr.copy()
        self.history.append((label, arr.copy()))
        self.redo_stack.clear()
        return self.current.copy()

    def undo(self) -> Optional[np.ndarray]:
        if len(self.history) <= 1:
            return self.get_current()
        self.redo_stack.append(self.history.pop())
        self.current = self.history[-1][1].copy()
        return self.current.copy()

    def redo(self) -> Optional[np.ndarray]:
        if not self.redo_stack:
            return self.get_current()
        item = self.redo_stack.pop()
        self.history.append((item[0], item[1].copy()))
        self.current = item[1].copy()
        return self.current.copy()

    def reset_original(self) -> Optional[np.ndarray]:
        if self.original is None:
            return None
        self.current = self.original.copy()
        self.history = [("Original", self.original.copy())]
        self.redo_stack.clear()
        return self.current.copy()

    def get_current(self) -> Optional[np.ndarray]:
        return None if self.current is None else self.current.copy()

    @staticmethod
    def _validate(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image)
        if arr.size == 0 or arr.ndim not in (2, 3):
            raise ValueError("Image must be a non-empty 2D or 3D array.")
        if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
            raise ValueError("Only grayscale, RGB, and RGBA images are supported.")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

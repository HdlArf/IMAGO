"""PySide6 desktop shell for IMAGO v1."""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QSize, QPointF, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QAction, QPixmap, QImage, QColor, QDragEnterEvent, QDropEvent, QPainter, QPen, QPolygonF, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSplashScreen,
    QPushButton, QToolButton, QFrame, QSplitter, QListWidget, QListWidgetItem,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QSlider, QSpinBox, QScrollArea, QColorDialog, QInputDialog, QStatusBar, QLineEdit, QMenu, QCheckBox, QComboBox
)
from imago.core.image_state import ImageState
from imago.core import processing

DARK_QSS = """
QMainWindow, QWidget { background:#15191e; color:#e7edf2; font-family:'Segoe UI',Arial; font-size:10pt; }
QFrame#Topbar, QFrame#Panel { background:#1d232a; border:1px solid #2c343d; border-radius:8px; }
QLabel#Brand { font-size:17pt; font-weight:700; letter-spacing:2px; background: transparent; }
QLabel#Muted { color:#9aa7b3; }
QPushButton, QToolButton { background:#252d35; color:#eaf0f4; border:1px solid #35414b; border-radius:6px; padding:7px 10px; }
QPushButton:hover, QToolButton:hover { border-color:ACCENT; }
QPushButton:checked { background:ACCENT; color:#081116; font-weight:700; }
QListWidget { background:#171c21; border:0; outline:0; }
QListWidget::item { padding:8px; margin:2px 0; border-radius:5px; }
QListWidget::item:selected { background:ACCENT; color:#071116; }
QSlider::groove:horizontal { height:5px; background:#343e48; border-radius:2px; }
QSlider::sub-page:horizontal { background:ACCENT; border-radius:2px; }
QSlider::handle:horizontal { background:#f4f8fb; width:13px; margin:-5px 0; border-radius:6px; }
QSpinBox { background:#101418; border:1px solid #39434d; border-radius:4px; padding:4px; }
QStatusBar { color:#aab5bf; }
""".replace("ACCENT", "#22d3ee")

TOOLS = [
    ("Develop", ["Exposure", "Brightness", "Contrast", "Saturation", "Tone", "Grayscale", "Gaussian Blur", "Mean Filter", "Median Filter", "Sharpen", "Resize", "Rotate", "Translate", "Crop"]),
    ("Tone & Color", ["Equalize", "Log Transform", "Gamma Transform", "Negative", "HDR", "HSV View"]),
    ("Image Arithmetic", ["Add Images", "Subtract Images", "Difference Images", "Multiply Images", "Divide Images", "Blend Images"]),
    ("Masks & Logic", ["Logical NOT", "Logical AND", "Logical OR", "Logical XOR", "Extract Object", "Color Mask", "Draw Line", "Draw Rectangle", "Draw Circle"]),
    ("Thresholding", ["Threshold", "Inverse Threshold", "Otsu", "Adaptive Threshold"]),
    ("Edges & Detection", ["Canny Edges", "Sobel", "Laplacian", "Detect Lines", "Detect Circles", "Detect Corners"]),
    ("Segmentation & Morphology", ["K-means Segmentation", "Watershed", "Erode", "Dilate", "Morphological Open", "Morphological Close", "Morphological Gradient", "Top Hat", "Black Hat"]),
    ("Face & Object Models", ["Detect Face", "Detect Profile Face", "Detect Eyes", "Detect Full Body", "Detect Upper Body", "Detect Lower Body", "Detect Smile", "Detect License Plate", "Detect Cat Face"]),
]
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def array_to_pixmap(arr: np.ndarray) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        q = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    elif arr.shape[2] == 4:
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
    else:
        h, w, _ = arr.shape
        q = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(q)

class ImageCanvas(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pix_item: QGraphicsPixmapItem | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(QColor("#101418"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setAcceptDrops(True)
        self._fit = True
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def set_image(self, arr):
        self.scene().clear()
        self.pix_item = None
        if arr is None:
            self.scene().setSceneRect(0, 0, 800, 500)
            self.resetTransform()
            self.viewport().update()
            return
        self.pix_item = self.scene().addPixmap(array_to_pixmap(arr))
        self.scene().setSceneRect(self.pix_item.boundingRect())
        if self._fit:
            self.fit_image()

    def fit_image(self):
        self._fit = True
        if self.pix_item:
            self.fitInView(self.pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_100(self):
        self._fit = False
        if self.pix_item:
            self.resetTransform()
            self.centerOn(self.pix_item)

    def wheelEvent(self, event):
        self._fit = False
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit_image()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.window().import_paths(paths)
            event.acceptProposedAction()

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.setMinimumHeight(118)
        self.setMaximumHeight(145)

    def set_image(self, image):
        self.image = None if image is None else np.asarray(image).copy()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#14191e"))
        area = self.rect().adjusted(8, 8, -8, -18)
        painter.setPen(QPen(QColor("#34404a"), 1))
        painter.drawRect(area)
        if self.image is None or self.image.size == 0:
            painter.setPen(QColor("#8795a2")); painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "Histogram")
            painter.end(); return
        arr = self.image
        if arr.ndim == 2:
            channels = [(arr, QColor("#cbd5e1"))]
        else:
            arr = arr[:, :, :3]
            channels = [(arr[:, :, 0], QColor("#f87171")), (arr[:, :, 1], QColor("#4ade80")), (arr[:, :, 2], QColor("#60a5fa"))]
        max_count = max(float(np.bincount(ch.ravel(), minlength=256).max()) for ch, _ in channels) or 1.0
        for channel, color in channels:
            counts = np.bincount(channel.ravel(), minlength=256).astype(np.float64)
            points = []
            for idx, count in enumerate(counts):
                x = area.left() + (idx / 255.0) * area.width()
                y = area.bottom() - (count / max_count) * max(1, area.height() - 2)
                points.append(QPointF(x, y))
            painter.setPen(QPen(color, 1.2))
            painter.drawPolyline(QPolygonF(points))
        painter.setPen(QColor("#84919d")); painter.drawText(8, self.height()-3, "0")
        painter.drawText(self.width()-22, self.height()-3, "255")
        painter.end()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMAGO")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 650)
        self.state = ImageState()
        self.image_states: dict[str, ImageState] = {}
        self.paths: list[str] = []
        self.active_index = -1
        self.preview = None
        self.second_image = None
        self.active_tool = None
        self.tool_sliders: list[QSlider] = []
        self.accent = "#22d3ee"
        self._make_ui()
        self.setAcceptDrops(True)
        self.statusBar().showMessage("Ready · Import an image to get started")

    def _make_ui(self):
        root = QWidget()
        outer = QVBoxLayout(root); outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(8)
        top = QFrame(); top.setObjectName("Topbar")
        th = QHBoxLayout(top); th.setContentsMargins(12, 7, 12, 7)
        brand = QLabel("IMAGO")
        brand.setObjectName("Brand")
        brand.setStyleSheet(f"color: {self.accent}; background: transparent;")
        self.brand_label = brand
        th.addWidget(brand)
        th.addSpacing(18)
        for label, slot in [("Open Image", self.open_images), ("Open Folder", self.open_folder),
                            ("Undo", self.undo), ("Redo", self.redo), ("Export", self.export_image),
                            ("Accent", self.pick_accent), ("Reset Original", self.reset_original)]:
            b = QPushButton(label); b.setToolTip(label); b.clicked.connect(slot); th.addWidget(b)
        th.addStretch(1)
        outer.addWidget(top)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.sidebar = QFrame(); self.sidebar.setObjectName("Panel"); self.sidebar.setMinimumWidth(175); self.sidebar.setMaximumWidth(240)
        sl = QVBoxLayout(self.sidebar); sl.setContentsMargins(8, 10, 8, 10)
        sl.addWidget(QLabel("TOOLS"))
        self.favorite_tools = []
        self.tool_search = QLineEdit()
        self.tool_search.setPlaceholderText("Search tools / ابحث عن أداة")
        self.tool_search.textChanged.connect(self.filter_tools)
        sl.addWidget(self.tool_search)
        self.favorite_btn = QPushButton("☆ Add / Remove Favorite")
        self.favorite_btn.clicked.connect(self.toggle_favorite)
        sl.addWidget(self.favorite_btn)
        self.tool_list = QListWidget()
        self._populate_tool_list()
        self.tool_list.currentItemChanged.connect(self.select_tool)
        sl.addWidget(self.tool_list)
        split.addWidget(self.sidebar)

        center = QWidget(); cl = QVBoxLayout(center); cl.setContentsMargins(0, 0, 0, 0)
        self.canvas = ImageCanvas()
        cl.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        for label, slot in [("Fit", self.canvas.fit_image), ("100%", self.canvas.set_100)]:
            b=QPushButton(label); b.clicked.connect(slot); controls.addWidget(b)
        controls.addStretch(1)
        self.image_info = QLabel("No image"); self.image_info.setObjectName("Muted"); controls.addWidget(self.image_info)
        cl.addLayout(controls)
        split.addWidget(center)

        self.inspector = QFrame(); self.inspector.setObjectName("Panel"); self.inspector.setMinimumWidth(265); self.inspector.setMaximumWidth(330)
        self.il = QVBoxLayout(self.inspector); self.il.setContentsMargins(14, 14, 14, 14); self.il.setSpacing(10)
        self.inspector_title = QLabel("IMAGE INSPECTOR"); self.inspector_title.setStyleSheet("font-weight:700;font-size:12pt;")
        self.il.addWidget(self.inspector_title)
        self.inspector_desc = QLabel("Image information and tool parameters appear here.")
        self.inspector_desc.setWordWrap(True); self.inspector_desc.setObjectName("Muted"); self.il.addWidget(self.inspector_desc)
        self.controls_area = QWidget(); self.controls_layout = QVBoxLayout(self.controls_area); self.controls_layout.setContentsMargins(0,0,0,0)
        self.il.addWidget(self.controls_area)
        self.histogram = HistogramWidget()
        self.il.addStretch(1)
        self.apply_btn = QPushButton("✓  APPLY"); self.apply_btn.clicked.connect(self.apply_preview); self.il.addWidget(self.apply_btn)
        self.reset_btn = QPushButton("RESET TOOL"); self.reset_btn.clicked.connect(self.reset_tool); self.il.addWidget(self.reset_btn)
        self.apply_btn.hide(); self.reset_btn.hide()
        split.addWidget(self.inspector)
        split.setSizes([205, 930, 305])
        outer.addWidget(split, 1)

        bottom = QFrame(); bottom.setObjectName("Panel")
        bl = QHBoxLayout(bottom); bl.setContentsMargins(8, 6, 8, 6); bl.setSpacing(10)
        bl.addWidget(QLabel("FILMSTRIP"))
        self.filmstrip = QListWidget()
        self.filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setIconSize(QSize(52, 38))
        self.filmstrip.setMaximumHeight(82); self.filmstrip.setMinimumHeight(66)
        self.filmstrip.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filmstrip.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.filmstrip.customContextMenuRequested.connect(self._filmstrip_context_menu)
        self.filmstrip.currentRowChanged.connect(self.select_image)
        bl.addWidget(self.filmstrip, 1)
        self.remove_image_btn = QPushButton("Remove Selected")
        self.remove_image_btn.setToolTip("Remove the selected image from this IMAGO session. The file on disk is not deleted.")
        self.remove_image_btn.clicked.connect(self.remove_selected_image)
        bl.addWidget(self.remove_image_btn)
        self.remove_image_action = QAction("Remove from Session", self)
        self.remove_image_action.setShortcut("Delete")
        self.remove_image_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.remove_image_action.triggered.connect(self.remove_selected_image)
        self.addAction(self.remove_image_action)
        outer.addWidget(bottom)
        self.setCentralWidget(root)

    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open images", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
        if paths: self.import_paths(paths)

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open image folder")
        if folder:
            paths = [str(p) for p in sorted(Path(folder).iterdir()) if p.is_file() and p.suffix.lower() in EXTS]
            self.import_paths(paths)

    def import_paths(self, paths):
        added = 0
        for path in paths:
            p = str(Path(path))
            if Path(p).suffix.lower() not in EXTS: continue
            if p in self.paths: continue
            try:
                with Image.open(p) as im:
                    im.load()
                    arr = np.array(im.convert("RGBA" if "A" in im.getbands() else "RGB"))
                if arr.size == 0: continue
            except Exception as exc:
                QMessageBox.warning(self, "Cannot open image", f"{Path(p).name}\n{exc}")
                continue
            self.paths.append(p)
            item = QListWidgetItem(Path(p).name)
            item.setToolTip(p)
            try:
                thumb = array_to_pixmap(arr).scaled(self.filmstrip.iconSize(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                item.setIcon(QIcon(thumb))
            except Exception: pass
            item.setSizeHint(QSize(142, 56))
            self.filmstrip.addItem(item)
            added += 1
        if added and self.active_index < 0: self.filmstrip.setCurrentRow(0)
        elif added: self.statusBar().showMessage(f"Added {added} image(s)")

    def _filmstrip_context_menu(self, pos):
        item = self.filmstrip.itemAt(pos)
        if item is None: return
        self.filmstrip.setCurrentItem(item)
        menu = QMenu(self)
        remove_action = menu.addAction("Remove from Session")
        remove_action.triggered.connect(self.remove_selected_image)
        menu.exec(self.filmstrip.mapToGlobal(pos))

    def remove_selected_image(self):
        row = self.filmstrip.currentRow()
        if row < 0 or row >= len(self.paths): return
        removed_path = self.paths[row]
        previous_active = self.active_index
        active_removed = row == previous_active

        self.filmstrip.blockSignals(True)
        item = self.filmstrip.takeItem(row)
        if item: del item
        self.paths.pop(row)
        self.image_states.pop(removed_path, None)

        if not self.paths:
            self.active_index = -1; self.state = ImageState(); self.image_states.clear()
            self.preview = None; self.active_tool = None; self.second_image = None
            self.filmstrip.setCurrentRow(-1); self.filmstrip.blockSignals(False)
            self.canvas.set_image(None); self._update_info(); self._show_smart_inspector()
            return

        if active_removed:
            self.active_index = -1; target_row = min(row, len(self.paths) - 1)
            self.filmstrip.setCurrentRow(-1); self.filmstrip.blockSignals(False)
            self.filmstrip.setCurrentRow(target_row)
        else:
            self.active_index = previous_active - (1 if row < previous_active else 0)
            self.filmstrip.setCurrentRow(self.active_index)
            self.filmstrip.blockSignals(False)

    def select_image(self, row):
        if row < 0 or row >= len(self.paths): return
        path = self.paths[row]
        try:
            image_state = self.image_states.get(path)
            if image_state is None:
                with Image.open(path) as im:
                    im.load()
                    arr = np.array(im.convert("RGBA" if "A" in im.getbands() else "RGB"))
                image_state = ImageState(); image_state.load(arr); self.image_states[path] = image_state
            self.state = image_state
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open image", str(exc)); return
        self.active_index = row; self.preview = None; self.active_tool = None; self.second_image = None
        self.canvas._fit = True; self.canvas.set_image(self.state.get_current())
        self._update_info(); self._show_smart_inspector()

    def _populate_tool_list(self):
        selected = self.active_tool
        self.tool_list.blockSignals(True); self.tool_list.clear()
        if self.favorite_tools:
            header = QListWidgetItem("— FAVORITES —"); header.setFlags(Qt.ItemFlag.NoItemFlags); self.tool_list.addItem(header)
            for tool in self.favorite_tools:
                item = QListWidgetItem("★  " + tool); item.setData(Qt.ItemDataRole.UserRole, tool); self.tool_list.addItem(item)
        for group, tools in TOOLS:
            header = QListWidgetItem("— " + group.upper() + " —"); header.setFlags(Qt.ItemFlag.NoItemFlags); self.tool_list.addItem(header)
            for tool in tools:
                item = QListWidgetItem("   " + tool); item.setData(Qt.ItemDataRole.UserRole, tool); self.tool_list.addItem(item)
                if selected == tool and self.tool_search.text().strip() == "": self.tool_list.setCurrentItem(item)
        self.tool_list.blockSignals(False)
        self.filter_tools(self.tool_search.text())

    def toggle_favorite(self):
        item = self.tool_list.currentItem()
        tool = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not tool: return
        if tool in self.favorite_tools: self.favorite_tools.remove(tool)
        else: self.favorite_tools.append(tool)
        self._populate_tool_list()

    def filter_tools(self, query):
        q = (query or "").strip().lower()
        aliases = {"تباين": "contrast", "سطوع": "brightness", "تعريض": "exposure", "تشبع": "saturation", "رمادي": "grayscale", "ابيض واسود": "grayscale", "تمويه": "blur", "ضبابية": "blur", "حدة": "sharpen", "قص": "crop", "تدوير": "rotate", "حجم": "resize", "حواف": "edges", "كانى": "canny", "عتبة": "threshold", "تجزئة": "segmentation", "وجه": "face", "عيون": "eyes", "عين": "eyes", "جسم": "body", "ابتسامة": "smile", "دائرة": "circles", "خطوط": "lines", "الوان": "color", "لون": "color", "قناع": "mask", "منطقية": "logical", "مدرج": "equalize"}
        normalized = aliases.get(q, q)
        visible_by_group = {}
        for index in range(self.tool_list.count()):
            item = self.tool_list.item(index)
            tool = item.data(Qt.ItemDataRole.UserRole)
            if tool:
                visible = not normalized or normalized in tool.lower()
                item.setHidden(not visible)
                header_index = index - 1
                while header_index >= 0 and self.tool_list.item(header_index).data(Qt.ItemDataRole.UserRole): header_index -= 1
                if header_index >= 0: visible_by_group[header_index] = visible_by_group.get(header_index, False) or visible
            else: visible_by_group.setdefault(index, False)
        for index in range(self.tool_list.count()):
            item = self.tool_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) is None: item.setHidden(bool(normalized) and not visible_by_group.get(index, False))

    def _update_info(self):
        if self.state.current is None: self.image_info.setText("No image"); return
        a=self.state.current
        self.image_info.setText(f"{a.shape[1]} × {a.shape[0]} px  ·  {len(self.state.history)-1} edit(s)")

    def _show_smart_inspector(self):
        self.active_tool = None
        self.inspector_title.setText("IMAGE INSPECTOR")
        if self.state.current is None: self.inspector_desc.setText("Import an image first.")
        else:
            a=self.state.current
            channels = "Grayscale" if a.ndim == 2 else ("RGBA" if a.shape[2] == 4 else "RGB")
            self.inspector_desc.setText(f"{Path(self.paths[self.active_index]).name}\n{a.shape[1]} × {a.shape[0]} px\n{channels}")
        self._clear_controls()
        self.histogram = HistogramWidget()
        self.histogram.set_image(self.state.get_current())
        self.controls_layout.addWidget(self.histogram)
        self.apply_btn.hide(); self.reset_btn.hide()

    def _clear_controls(self):
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                child_layout = item.layout()
                if widget is not None:
                    widget.hide(); widget.setParent(None); widget.deleteLater()
                elif child_layout is not None:
                    clear_layout(child_layout); child_layout.deleteLater()
        self.tool_sliders = []
        for attr in ("_lock_aspect", "_interp_combo", "_resize_w", "_resize_h", "_orig_aspect", "_rotate_slider", "_mask_color"):
            try: delattr(self, attr)
            except AttributeError: pass
        clear_layout(self.controls_layout)

    def select_tool(self, current, previous):
        if current is None: return
        tool=current.data(Qt.ItemDataRole.UserRole)
        if not tool: return
        if self.state.current is None:
            QMessageBox.information(self, "IMAGO", "Import an image first."); return
        if self.preview is not None:
            box=QMessageBox(self); box.setWindowTitle("Unapplied changes"); box.setText("This tool has uncommitted preview changes.")
            apply_action=box.addButton("Apply & Switch", QMessageBox.ButtonRole.AcceptRole)
            discard_action=box.addButton("Discard & Switch", QMessageBox.ButtonRole.DestructiveRole)
            stay_action=box.addButton("Stay", QMessageBox.ButtonRole.RejectRole)
            box.exec(); clicked=box.clickedButton()
            if clicked == stay_action:
                if previous: self.tool_list.setCurrentItem(previous)
                return
            if clicked == apply_action: self.apply_preview()
            else: self.preview=None; self.canvas.set_image(self.state.get_current())
        self.active_tool=tool; self.second_image=None
        if tool in ("Add Images", "Subtract Images", "Difference Images", "Multiply Images", "Divide Images", "Blend Images", "Logical AND", "Logical OR", "Logical XOR"):
            path, _ = QFileDialog.getOpenFileName(self, "Choose second image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)")
            if not path: self._show_smart_inspector(); return
            try:
                with Image.open(path) as im: im.load(); self.second_image=np.array(im.convert("RGB"))
            except Exception as exc:
                QMessageBox.warning(self, "Cannot open second image", str(exc)); self._show_smart_inspector(); return
        self._build_tool_controls(tool)
        self._preview(tool)

    def _add_slider(self, label, minimum, maximum, value, callback, suffix=""):
        row=QVBoxLayout()
        head=QHBoxLayout(); lab=QLabel(label); val=QSpinBox(); val.setRange(minimum,maximum); val.setValue(value)
        head.addWidget(lab); head.addStretch(1); head.addWidget(val)
        slider=QSlider(Qt.Orientation.Horizontal); slider.setRange(minimum,maximum); slider.setValue(value)
        slider.valueChanged.connect(val.setValue); val.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(lambda v: callback(v))
        row.addLayout(head); row.addWidget(slider); self.controls_layout.addLayout(row)
        self.tool_sliders.append(slider)
        return slider, val

    def _build_tool_controls(self, tool):
        self._clear_controls()
        self.inspector_title.setText(tool.upper())
        self.inspector_desc.setText("Live preview · Apply commits one step to History.")
        self.apply_btn.hide(); self.reset_btn.hide()
        if tool in ("Exposure", "Brightness", "Contrast", "Saturation"):
            self._add_slider(tool, -100, 100, 0, lambda v, t=tool: self._preview(t, v))
        elif tool == "Tone": self._add_slider("Hue shift", -90, 90, 0, lambda v: self._preview(tool, v))
        elif tool == "HDR":
            self._add_slider("Local contrast", 0, 300, 100, lambda v: self._preview(tool, v))
            self._add_slider("Color saturation", 0, 200, 100, lambda v: self._preview(tool, v))
            self._add_slider("Warm ↔ Cool", -100, 100, 0, lambda v: self._preview(tool, v))
        elif tool == "Sharpen":
            self.inspector_desc.setText("Unsharp mask. Threshold protects flat areas from noise.")
            self._add_slider("Amount", 0, 300, 100, lambda v: self._preview(tool, v))
            self._add_slider("Radius × 10", 1, 100, 20, lambda v: self._preview(tool, v))
            self._add_slider("Threshold (0 = off)", 0, 50, 0, lambda v: self._preview(tool, v))
        elif tool == "Grayscale":
            self._add_slider("Brightness", -100, 100, 0, lambda v: self._preview(tool, v))
        elif tool == "Rotate":
            btn_row = QHBoxLayout()
            for label, angle in [("↶ 90°", -90), ("↷ 90°", 90), ("180°", 180)]:
                b = QPushButton(label); b.clicked.connect(lambda _=False, a=angle: self._set_rotate_angle(a)); btn_row.addWidget(b)
            self.controls_layout.addLayout(btn_row)
            slider, _ = self._add_slider("Fine angle", -180, 180, 0, lambda v: self._preview(tool, v))
            self._rotate_slider = slider
        elif tool == "Resize":
            h, w = self.state.current.shape[:2]; self._orig_aspect = w / h
            preset_row = QHBoxLayout()
            for label, pw, ph in [("HD", 1280, 720), ("FHD", 1920, 1080), ("4K", 3840, 2160), ("Original", w, h)]:
                b = QPushButton(label); b.setToolTip(f"{pw} × {ph}"); b.clicked.connect(lambda _=False, pw=pw, ph=ph: self._apply_resize_preset(pw, ph)); preset_row.addWidget(b)
            self.controls_layout.addLayout(preset_row)
            self._lock_aspect = QCheckBox(" Lock aspect ratio"); self._lock_aspect.setChecked(True); self.controls_layout.addWidget(self._lock_aspect)
            sw, _ = self._add_slider("Width", 1, max(w, 4000), w, lambda v: self._on_resize_slider("w")); self._resize_w = sw
            sh, _ = self._add_slider("Height", 1, max(h, 4000), h, lambda v: self._on_resize_slider("h")); self._resize_h = sh
            self._interp_combo = QComboBox(); self._interp_combo.addItems(["Nearest", "Linear", "Cubic", "Area", "Lanczos"]); self._interp_combo.setCurrentText("Cubic")
            self._interp_combo.currentTextChanged.connect(lambda _: self._preview("Resize")); self.controls_layout.addWidget(self._interp_combo)
        elif tool in ("Gaussian Blur", "Mean Filter", "Median Filter"):
            self._add_slider("Kernel size", 1, 31, 3, lambda v: self._preview(tool, v))
        elif tool == "Translate":
            h,w=self.state.current.shape[:2]
            self._add_slider("Horizontal", -w, w, 0, lambda v: self._preview(tool, v))
            self._add_slider("Vertical", -h, h, 0, lambda v: self._preview(tool, v))
        elif tool == "Crop":
            h,w=self.state.current.shape[:2]
            self._add_slider("X", 0, max(0,w-1), 0, lambda v: self._preview(tool, v))
            self._add_slider("Y", 0, max(0,h-1), 0, lambda v: self._preview(tool, v))
            self._add_slider("Width", 1, max(1,w), w, lambda v: self._preview(tool, v))
            self._add_slider("Height", 1, max(1,h), h, lambda v: self._preview(tool, v))
        elif tool in ("Threshold", "Inverse Threshold"):
            self._add_slider("Threshold", 0, 255, 127, lambda v: self._preview(tool, v))
        elif tool == "Canny Edges":
            self._add_slider("Low threshold", 0, 255, 50, lambda v: self._preview(tool, v))
            self._add_slider("High threshold", 1, 255, 150, lambda v: self._preview(tool, v))
        elif tool == "Adaptive Threshold":
            self._add_slider("Block size", 3, 51, 11, lambda v: self._preview(tool, v))
            self._add_slider("C", -20, 20, 2, lambda v: self._preview(tool, v))
        elif tool == "Gamma Transform": self._add_slider("Gamma × 10", 1, 50, 10, lambda v: self._preview(tool, v))
        elif tool == "K-means Segmentation": self._add_slider("Clusters", 2, 10, 3, lambda v: self._preview(tool, v))
        elif tool in ("Erode", "Dilate", "Morphological Open", "Morphological Close", "Morphological Gradient", "Top Hat", "Black Hat"):
            self._add_slider("Kernel size", 1, 21, 3, lambda v: self._preview(tool, v))
        elif tool == "Detect Lines": self._add_slider("Vote threshold", 10, 300, 80, lambda v: self._preview(tool, v))
        elif tool == "Detect Circles":
            self._add_slider("Minimum radius", 1, 100, 5, lambda v: self._preview(tool, v))
            self._add_slider("Maximum radius", 10, 300, 100, lambda v: self._preview(tool, v))
        elif tool == "Detect Corners": self._add_slider("Maximum corners", 10, 500, 100, lambda v: self._preview(tool, v))
        elif tool == "Extract Object":
            self._add_slider("Foreground threshold",0,255,127,lambda v:self._preview(tool,v))
            self._add_slider("Invert mask (0/1)",0,1,0,lambda v:self._preview(tool,v))
        elif tool == "Color Mask":
            self.inspector_desc.setText("Pick a target color and adjust tolerance.")
            self._mask_color = (255, 255, 0)
            btn = QPushButton("Pick Target Color"); btn.clicked.connect(self._pick_mask_color); self.controls_layout.addWidget(btn)
            self._add_slider("Tolerance", 1, 100, 30, lambda v: self._preview(tool, v))
        elif tool in ("Draw Line", "Draw Rectangle", "Draw Circle"):
            h,w=self.state.current.shape[:2]
            self.inspector_desc.setText("Set drawing coordinates and line thickness. Color defaults to cyan-yellow.")
            self.draw_color=(0,220,255)
            color_btn=QPushButton("Choose drawing color"); color_btn.clicked.connect(self._pick_draw_color); self.controls_layout.addWidget(color_btn)
            for label, maximum, default in [("X1",w-1,int(w*.2)),("Y1",h-1,int(h*.2)),("X2",w-1,int(w*.8)),("Y2",h-1,int(h*.8)),("Thickness",30,3)]:
                self._add_slider(label,1 if label=="Thickness" else 0,max(1,maximum),max(1,default) if label=="Thickness" else default,lambda v: self._preview(tool,v))
        elif tool in ("Add Images", "Subtract Images", "Multiply Images"):
            self._add_slider("Second image coefficient (%)",0,300,100,lambda v:self._preview(tool,v))
        elif tool == "Blend Images": self._add_slider("First image weight (%)", 0, 100, 50, lambda v: self._preview(tool,v))
        elif tool in ("Logical AND", "Logical OR", "Logical XOR"):
            self._add_slider("Binary threshold",0,255,127,lambda v:self._preview(tool,v))
        elif tool in ("Detect Face", "Detect Profile Face", "Detect Eyes", "Detect Full Body", "Detect Upper Body", "Detect Lower Body", "Detect Smile", "Detect License Plate", "Detect Cat Face"):
            self.inspector_desc.setText("Uses the bundled OpenCV Haar Cascade XML model. Detections are approximate and depend on image quality.")
            self._add_slider("Scale factor × 100", 105, 160, 110, lambda v: self._preview(tool, v))
            self._add_slider("Minimum neighbors", 0, 12, 5, lambda v: self._preview(tool, v))
        elif tool == "Equalize":
            self.inspector_desc.setText("Adaptive Histogram Equalization.")
            self._add_slider("Clip Limit", 1, 50, 20, lambda v: self._preview(tool, v))
            self._add_slider("Tile Size", 2, 16, 8, lambda v: self._preview(tool, v))
        elif tool == "HSV View":
            self.inspector_desc.setText("View individual HSV channels.")
            self._add_slider("Channel (0=H, 1=S, 2=V)", 0, 2, 0, lambda v: self._preview(tool, v))
        elif tool in ("Otsu", "Sobel", "Laplacian", "Log Transform", "Negative", "Watershed", "Difference Images", "Divide Images", "Logical NOT", "Logical AND", "Logical OR", "Logical XOR") or tool.startswith("Morphological") or tool in ("Erode", "Dilate", "Top Hat", "Black Hat"):
            self.controls_layout.addWidget(QLabel("Preview is computed from the committed image."))
        else:
            self.controls_layout.addWidget(QLabel("Select parameters to preview this operation."))

    def _pick_draw_color(self):
        initial=getattr(self,"draw_color",(0,220,255))
        color=QColorDialog.getColor(QColor(initial[0],initial[1],initial[2]),self,"Choose drawing color")
        if color.isValid():
            self.draw_color=(color.red(),color.green(),color.blue())
            if self.active_tool in ("Draw Line","Draw Rectangle","Draw Circle"): self._preview(self.active_tool)

    def _pick_mask_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self._mask_color = (color.red(), color.green(), color.blue())
            self._preview("Color Mask")

    def _set_rotate_angle(self, angle):
        slider = getattr(self, "_rotate_slider", None)
        if slider is not None:
            slider.setValue(angle); self._preview("Rotate", angle)

    def _apply_resize_preset(self, pw, ph):
        sw = getattr(self, "_resize_w", None); sh = getattr(self, "_resize_h", None)
        if sw is None or sh is None: return
        sw.blockSignals(True); sh.blockSignals(True)
        sw.setValue(min(pw, sw.maximum())); sh.setValue(min(ph, sh.maximum()))
        sw.blockSignals(False); sh.blockSignals(False); self._preview("Resize")

    def _on_resize_slider(self, which):
        sw = getattr(self, "_resize_w", None); sh = getattr(self, "_resize_h", None)
        if sw is None or sh is None: return
        lock = getattr(self, "_lock_aspect", None)
        if lock is not None and lock.isChecked():
            aspect = getattr(self, "_orig_aspect", 1.0) or 1.0
            if which == "w":
                new_h = max(1, min(int(sw.value() / aspect), sh.maximum()))
                sh.blockSignals(True); sh.setValue(new_h); sh.blockSignals(False)
            else:
                new_w = max(1, min(int(sh.value() * aspect), sw.maximum()))
                sw.blockSignals(True); sw.setValue(new_w); sw.blockSignals(False)
        self._preview("Resize")

    def _preview(self, tool, value=None):
        if self.state.current is None or self.active_tool is None: return
        if tool != self.active_tool: return
        base=self.state.preview_base()
        sliders=self.tool_sliders
        vals=[x.value() for x in sliders]
        try:
            if tool == "Exposure": out=processing.exposure(base, value if value is not None else (vals[0] if vals else 0))
            elif tool == "Brightness": out=processing.brightness(base, value if value is not None else (vals[0] if vals else 0))
            elif tool == "Contrast": out=processing.contrast(base, value if value is not None else (vals[0] if vals else 0))
            elif tool == "Saturation": out=processing.saturation(base, value if value is not None else (vals[0] if vals else 0))
            elif tool == "Tone": out=processing.tone(base, value if value is not None else (vals[0] if vals else 0))
            elif tool == "Grayscale": out=processing.brightness(processing.grayscale(base), value or 0)
            elif tool == "Gaussian Blur": out=processing.gaussian_blur(base, value if value is not None else (vals[0] if vals else 3))
            elif tool == "Mean Filter": out=processing.mean_blur(base, value if value is not None else (vals[0] if vals else 3))
            elif tool == "Median Filter": out=processing.median_blur(base, value if value is not None else (vals[0] if vals else 3))
            elif tool == "Sharpen":
                amount = (vals[0] if vals else 100) / 100.0; radius = (vals[1] if len(vals) > 1 else 20) / 10.0; thr = vals[2] if len(vals) > 2 else 0
                out = processing.sharpen(base, amount, radius, thr)
            elif tool == "Rotate": out=processing.rotate(base, value if value is not None else (vals[0] if vals else 0))
            elif tool == "Resize":
                h, w = base.shape[:2]
                interp_map = {"Nearest": "nearest", "Linear": "linear", "Cubic": "cubic", "Area": "area", "Lanczos": "lanczos"}
                combo = getattr(self, "_interp_combo", None)
                interp = interp_map.get(combo.currentText() if combo else "Cubic", "cubic")
                out = processing.resize(base, vals[0] if vals else w, vals[1] if len(vals) > 1 else h, interp)
            elif tool == "Translate": out=processing.translate(base, vals[0] if vals else 0, vals[1] if len(vals)>1 else 0)
            elif tool == "Crop":
                h,w=base.shape[:2]; out=processing.crop(base, vals[0] if vals else 0, vals[1] if len(vals)>1 else 0, vals[2] if len(vals)>2 else w, vals[3] if len(vals)>3 else h)
            elif tool == "Threshold": out=processing.threshold(base, value if value is not None else (vals[0] if vals else 127))
            elif tool == "Inverse Threshold": out=processing.binary_inverse(base, value if value is not None else (vals[0] if vals else 127))
            elif tool == "Otsu": out=processing.otsu(base)
            elif tool == "Adaptive Threshold": out=processing.adaptive_threshold(base, vals[0] if vals else 11, vals[1] if len(vals)>1 else 2)
            elif tool == "Canny Edges": out=processing.canny(base, vals[0] if vals else 50, vals[1] if len(vals)>1 else 150)
            elif tool == "Equalize": out=processing.equalize(base, (vals[0] if vals else 20), (vals[1] if len(vals) > 1 else 8))
            elif tool == "Log Transform": out=processing.log_transform(base)
            elif tool == "Gamma Transform": out=processing.gamma_transform(base, (value or 10)/10)
            elif tool == "Negative": out=processing.negative(base)
            elif tool == "HDR": out=processing.hdr_tone(base, (vals[0] if vals else 100)/100, (vals[1] if len(vals)>1 else 100)/100, (vals[2] if len(vals)>2 else 0)/100)
            elif tool == "HSV View": out=processing.color_space_view(base, "HSV", (vals[0] if vals else 0))
            elif tool in ("Add Images", "Subtract Images", "Difference Images", "Multiply Images", "Divide Images", "Blend Images"):
                op={"Add Images":"add","Subtract Images":"subtract","Difference Images":"difference","Multiply Images":"multiply","Divide Images":"divide","Blend Images":"blend"}[tool]
                coefficient=(vals[0]/100) if vals and tool in ("Add Images","Subtract Images","Multiply Images") else 1.0
                weight=(vals[0]/100) if vals and tool=="Blend Images" else .5
                out=processing.arithmetic(base, self.second_image, op, weight, coefficient)
            elif tool == "Logical NOT": out=processing.logical_operation(base, operation="not")
            elif tool in ("Logical AND", "Logical OR", "Logical XOR"):
                op={"Logical AND":"and","Logical OR":"or","Logical XOR":"xor"}[tool]
                out=processing.logical_operation(base, self.second_image, op, vals[0] if vals else 127)
            elif tool == "Extract Object": out=processing.extract_object(base, vals[0] if vals else 127, bool(vals[1]) if len(vals)>1 else False)
            elif tool == "Sobel": out=processing.sobel(base)
            elif tool == "Laplacian": out=processing.laplacian(base)
            elif tool == "Detect Lines": out=processing.detect_lines(base, value or 80)
            elif tool == "Detect Circles": out=processing.detect_circles(base, vals[0] if vals else 5, max((vals[1] if len(vals)>1 else 100), (vals[0] if vals else 5)+1))
            elif tool == "Detect Corners": out=processing.detect_corners(base, value or 100)
            elif tool == "K-means Segmentation": out=processing.kmeans_segmentation(base, value or 3)
            elif tool == "Watershed": out=processing.watershed(base)
            elif tool in ("Erode", "Dilate", "Morphological Open", "Morphological Close", "Morphological Gradient", "Top Hat", "Black Hat"):
                op={"Erode":"erode","Dilate":"dilate","Morphological Open":"open","Morphological Close":"close","Morphological Gradient":"gradient","Top Hat":"tophat","Black Hat":"blackhat"}[tool]
                out=processing.morphology(base, op, value or 3)
            elif tool == "Color Mask":
                target = getattr(self, "_mask_color", (255, 255, 0)); tol = vals[0] if vals else 30
                out=processing.color_mask(base, target, tol)
            elif tool in ("Draw Line", "Draw Rectangle", "Draw Circle"):
                v=vals+[0]*5; primitive={"Draw Line":"line","Draw Rectangle":"rectangle","Draw Circle":"circle"}[tool]
                rgb=getattr(self,"draw_color",(0,220,255)); bgr=(rgb[2],rgb[1],rgb[0])
                out=processing.draw_primitive(base,primitive,v[0],v[1],v[2],v[3],color=bgr,thickness=v[4])
            elif tool.startswith("Detect "):
                model_names={"Detect Face":"haarcascade_frontalface_default.xml","Detect Profile Face":"haarcascade_profileface.xml","Detect Eyes":"haarcascade_eye.xml","Detect Full Body":"haarcascade_fullbody.xml","Detect Upper Body":"haarcascade_upperbody.xml","Detect Lower Body":"haarcascade_lowerbody.xml","Detect Smile":"haarcascade_smile.xml","Detect License Plate":"haarcascade_russian_plate_number.xml","Detect Cat Face":"haarcascade_frontalcatface.xml"}
                local_model=Path(__file__).resolve().parents[2]/"models"/"haar"/model_names[tool]
                if not local_model.is_file(): raise FileNotFoundError(f"Bundled detection model is missing: {local_model}")
                out=processing.detect_haar(base, local_model, tool.replace("Detect ",""), (vals[0] if vals else 110)/100, vals[1] if len(vals)>1 else 5)
            else: return
            changed=(out.shape != base.shape or not np.array_equal(out,base))
            self.preview=out if changed else None
            self.apply_btn.setVisible(changed); self.reset_btn.setVisible(changed)
            self.canvas.set_image(out if changed else base)
            self._update_info()
        except Exception as exc:
            self.preview=None; self.apply_btn.hide(); self.reset_btn.hide()
            self.canvas.set_image(base)
            self.statusBar().showMessage(f"Preview failed: {exc}")

    def apply_preview(self):
        if self.preview is None or self.active_tool is None: return
        self.state.apply(self.preview, self.active_tool); self.preview=None
        self.canvas.set_image(self.state.get_current()); self._update_info()
        self._show_smart_inspector()

    def reset_tool(self):
        if self.active_tool is None: return
        self.preview = None; tool = self.active_tool
        self.canvas.set_image(self.state.get_current()); self._build_tool_controls(tool); self._preview(tool)

    def undo(self):
        arr=self.state.undo()
        if arr is not None: self.preview=None; self.canvas.set_image(arr); self._update_info(); self._show_smart_inspector()

    def redo(self):
        arr=self.state.redo()
        if arr is not None: self.preview=None; self.canvas.set_image(arr); self._update_info(); self._show_smart_inspector()

    def reset_original(self):
        arr=self.state.reset_original()
        if arr is not None: self.preview=None; self.canvas.set_image(arr); self._update_info(); self._show_smart_inspector()

    def export_image(self):
        arr=self.preview if self.preview is not None else self.state.get_current()
        if arr is None: return
        path, selected=QFileDialog.getSaveFileName(self, "Export image", "imago_export.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path: return
        if Path(path).suffix.lower() not in (".png",".jpg",".jpeg"): path += ".png"
        try:
            out=arr
            if Path(path).suffix.lower() in (".jpg",".jpeg"):
                if out.ndim == 3 and out.shape[2] == 4: out=out[:,:,:3]
                if out.ndim == 2: Image.fromarray(out).convert("RGB").save(path, quality=95)
                else: Image.fromarray(out).convert("RGB").save(path, quality=95)
            else: Image.fromarray(out).save(path)
        except Exception as exc: QMessageBox.critical(self, "Export failed", str(exc))

    def pick_accent(self):
        color=QColorDialog.getColor(QColor(self.accent), self, "Choose accent color")
        if not color.isValid(): return
        self.accent = color.name(); self.setStyleSheet(DARK_QSS.replace("#22d3ee", self.accent))
        self.brand_label.setStyleSheet(f"color: {self.accent}; background: transparent;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event):
        paths=[u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths: self.import_paths(paths)

def _make_signature_splash() -> QSplashScreen:
    screen = QApplication.primaryScreen()
    canvas = QPixmap(screen.size())
    canvas.fill(QColor("#000000"))
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    font = QFont("Segoe UI")
    font.setPointSizeF(60.0); font.setWeight(QFont.Weight.DemiBold); font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 15.0)
    painter.setFont(font); painter.setPen(QColor("#F5F7FA"))
    painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, "HΛDΞL")
    painter.end()
    splash = QSplashScreen(canvas, Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
    splash.setWindowOpacity(0.0)
    return splash

def run():
    app = QApplication(sys.argv)
    app.setApplicationName("IMAGO"); app.setOrganizationName("HΛDΞL")
    window = MainWindow(); window.setStyleSheet(DARK_QSS.replace("#22d3ee", window.accent))
    splash = _make_signature_splash(); splash.show(); app.processEvents()
    
    fade_in = QPropertyAnimation(splash, b"windowOpacity", splash)
    fade_in.setDuration(600); fade_in.setStartValue(0.0); fade_in.setEndValue(1.0); fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
    splash._fade_in_animation = fade_in; fade_in.start()
    
    def fade_out_signature():
        fade_out = QPropertyAnimation(splash, b"windowOpacity", splash)
        fade_out.setDuration(800); fade_out.setStartValue(splash.windowOpacity()); fade_out.setEndValue(0.0); fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        fade_out.finished.connect(lambda: (window.show(), splash.finish(window)))
        splash._fade_out_animation = fade_out; fade_out.start()
        
    QTimer.singleShot(2000, fade_out_signature)
    return app.exec()
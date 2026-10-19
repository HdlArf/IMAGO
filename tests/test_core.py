import unittest
import numpy as np
import cv2
from pathlib import Path
from imago.core.image_state import ImageState
from imago.core import processing

class ImageStateTests(unittest.TestCase):
    def test_preview_base_is_copy(self):
        state=ImageState()
        image=np.zeros((8,9,3),dtype=np.uint8)
        state.load(image)
        base=state.preview_base()
        base[:]=255
        self.assertEqual(int(state.current.max()),0)

    def test_apply_undo_redo_reset(self):
        state=ImageState()
        original=np.zeros((5,6,3),dtype=np.uint8)
        state.load(original)
        changed=np.full_like(original,90)
        state.apply(changed,"Exposure")
        self.assertEqual(int(state.current.max()),90)
        state.undo()
        self.assertEqual(int(state.current.max()),0)
        state.redo()
        self.assertEqual(int(state.current.max()),90)
        state.undo()
        state.apply(np.full_like(original, 33), "Branch")
        self.assertEqual(len(state.redo_stack), 0)
        state.reset_original()
        self.assertEqual(int(state.current.max()),0)
        self.assertEqual(len(state.history),1)

    def test_empty_state_fails_cleanly(self):
        with self.assertRaisesRegex(ValueError,"Import an image first"):
            ImageState().preview_base()

class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.image=np.zeros((24,32,3),dtype=np.uint8)
        self.image[5:18,7:23]=[200,100,50]

    def test_shape_and_dtype(self):
        for result in [
            processing.exposure(self.image,10),
            processing.contrast(self.image,20),
            processing.grayscale(self.image),
            processing.gaussian_blur(self.image,3),
            processing.sharpen(self.image),
            processing.rotate(self.image,5),
            processing.threshold(self.image,120),
            processing.canny(self.image,50,150),
            processing.equalize(self.image),
            processing.brightness(self.image, 10),
            processing.saturation(self.image, 20),
            processing.tone(self.image, 10),
            processing.mean_blur(self.image, 3),
            processing.median_blur(self.image, 3),
            processing.laplacian(self.image),
            processing.sobel(self.image),
            processing.resize(self.image, 16, 12),
            processing.translate(self.image, 2, 3),
            processing.crop(self.image, 1, 1, 10, 10),
            processing.binary_inverse(self.image, 120),
            processing.otsu(self.image),
            processing.adaptive_threshold(self.image, 11, 2),
            processing.log_transform(self.image),
            processing.gamma_transform(self.image, 1.5),
            processing.negative(self.image),
            processing.morphology(self.image, 'open', 3),
            processing.kmeans_segmentation(self.image, 3),
            processing.watershed(self.image),
            processing.detect_lines(self.image),
            processing.detect_circles(self.image),
            processing.detect_corners(self.image),
            processing.color_space_view(self.image, 'HSV'),
            processing.color_space_view(self.image, 'LAB'),
            processing.color_space_view(self.image, 'YCrCb'),
            processing.color_mask(self.image),
            processing.extract_object(self.image, 120),
            processing.logical_operation(self.image, operation='not'),
            processing.logical_operation(self.image, operation='and'),
            processing.draw_primitive(self.image, 'line'),
            processing.draw_primitive(self.image, 'rectangle'),
            processing.draw_primitive(self.image, 'circle'),
        ]:
            self.assertEqual(result.dtype,np.uint8)
            self.assertGreater(result.size,0)

    def test_exposure_brightness_and_contrast_are_distinct(self):
        gray = np.array([[20, 80, 128, 200]], dtype=np.uint8)
        exp = processing.exposure(gray, 50)
        bright = processing.brightness(gray, 50)
        cont = processing.contrast(gray, 50)
        self.assertFalse(np.array_equal(exp, bright))
        self.assertFalse(np.array_equal(cont, bright))
        self.assertEqual(int(processing.brightness(np.array([[20]], dtype=np.uint8), -100)[0, 0]), 0)
        np.testing.assert_array_equal(processing.exposure(gray, 0), gray)
        np.testing.assert_array_equal(processing.brightness(gray, 0), gray)
        np.testing.assert_array_equal(processing.contrast(gray, 0), gray)

    def test_tone_controls_preserve_rgba_alpha(self):
        rgba = np.dstack((self.image, np.full(self.image.shape[:2], 77, dtype=np.uint8)))
        for result in (
            processing.exposure(rgba, 30),
            processing.brightness(rgba, -25),
            processing.contrast(rgba, 35),
        ):
            self.assertEqual(result.shape, rgba.shape)
            np.testing.assert_array_equal(result[:, :, 3], rgba[:, :, 3])

    def test_all_processing_tool_families_smoke(self):
        other = np.flip(self.image, axis=1).copy()
        operations = {
            "Exposure": lambda: processing.exposure(self.image, 20),
            "Brightness": lambda: processing.brightness(self.image, 20),
            "Contrast": lambda: processing.contrast(self.image, 20),
            "Saturation": lambda: processing.saturation(self.image, 20),
            "Tone": lambda: processing.tone(self.image, 20),
            "Grayscale": lambda: processing.grayscale(self.image),
            "Gaussian Blur": lambda: processing.gaussian_blur(self.image, 3),
            "Mean Filter": lambda: processing.mean_blur(self.image, 3),
            "Median Filter": lambda: processing.median_blur(self.image, 3),
            "Sharpen": lambda: processing.sharpen(self.image, 1.2, 1.5),
            "Resize": lambda: processing.resize(self.image, 16, 12),
            "Rotate": lambda: processing.rotate(self.image, 15),
            "Translate": lambda: processing.translate(self.image, 2, 3),
            "Crop": lambda: processing.crop(self.image, 1, 1, 10, 10),
            "Equalize": lambda: processing.equalize(self.image),
            "Log Transform": lambda: processing.log_transform(self.image),
            "Gamma Transform": lambda: processing.gamma_transform(self.image, 1.2),
            "Negative": lambda: processing.negative(self.image),
            "HDR": lambda: processing.hdr_tone(self.image, 1.2, 1.1, 0.1),
            "HSV View": lambda: processing.color_space_view(self.image, "HSV"),
            "Lab View": lambda: processing.color_space_view(self.image, "LAB"),
            "YCrCb View": lambda: processing.color_space_view(self.image, "YCrCb"),
            "Add Images": lambda: processing.arithmetic(self.image, other, "add", coefficient=0.8),
            "Subtract Images": lambda: processing.arithmetic(self.image, other, "subtract", coefficient=0.8),
            "Difference Images": lambda: processing.arithmetic(self.image, other, "difference"),
            "Multiply Images": lambda: processing.arithmetic(self.image, other, "multiply", coefficient=0.8),
            "Divide Images": lambda: processing.arithmetic(self.image, other, "divide"),
            "Blend Images": lambda: processing.arithmetic(self.image, other, "blend", weight=0.6),
            "Logical NOT": lambda: processing.logical_operation(self.image, operation="not"),
            "Logical AND": lambda: processing.logical_operation(self.image, other, "and", 127),
            "Logical OR": lambda: processing.logical_operation(self.image, other, "or", 127),
            "Logical XOR": lambda: processing.logical_operation(self.image, other, "xor", 127),
            "Extract Object": lambda: processing.extract_object(self.image, 120, False),
            "Color Mask": lambda: processing.color_mask(self.image, (0, 0, 0), (179, 255, 255)),
            "Draw Line": lambda: processing.draw_primitive(self.image, "line", 1, 2, 20, 15),
            "Draw Rectangle": lambda: processing.draw_primitive(self.image, "rectangle", 1, 2, 20, 15),
            "Draw Circle": lambda: processing.draw_primitive(self.image, "circle", 4, 4, 12, 12),
            "Threshold": lambda: processing.threshold(self.image, 127),
            "Inverse Threshold": lambda: processing.binary_inverse(self.image, 127),
            "Otsu": lambda: processing.otsu(self.image),
            "Adaptive Threshold": lambda: processing.adaptive_threshold(self.image, 11, 2),
            "Canny Edges": lambda: processing.canny(self.image, 50, 150),
            "Sobel": lambda: processing.sobel(self.image),
            "Laplacian": lambda: processing.laplacian(self.image),
            "Detect Lines": lambda: processing.detect_lines(self.image, 30),
            "Detect Circles": lambda: processing.detect_circles(self.image, 1, 12),
            "Detect Corners": lambda: processing.detect_corners(self.image, 20),
            "K-means Segmentation": lambda: processing.kmeans_segmentation(self.image, 3),
            "Watershed": lambda: processing.watershed(self.image),
            "Erode": lambda: processing.morphology(self.image, "erode", 3),
            "Dilate": lambda: processing.morphology(self.image, "dilate", 3),
            "Morphological Open": lambda: processing.morphology(self.image, "open", 3),
            "Morphological Close": lambda: processing.morphology(self.image, "close", 3),
            "Morphological Gradient": lambda: processing.morphology(self.image, "gradient", 3),
            "Top Hat": lambda: processing.morphology(self.image, "tophat", 3),
            "Black Hat": lambda: processing.morphology(self.image, "blackhat", 3),
        }
        for tool_name, operation in operations.items():
            with self.subTest(tool=tool_name):
                result = operation()
                self.assertIsInstance(result, np.ndarray)
                self.assertGreater(result.size, 0)
                self.assertEqual(result.dtype, np.uint8)

    def test_preview_operations_are_deterministic_from_same_base(self):
        first = processing.canny(self.image, 30, 120)
        second = processing.canny(self.image.copy(), 30, 120)
        np.testing.assert_array_equal(first, second)

    def test_image_arithmetic(self):
        other = np.full_like(self.image, 30)
        self.assertEqual(processing.arithmetic(self.image, other, 'add').shape, self.image.shape)
        self.assertEqual(processing.arithmetic(self.image, other, 'blend').shape, self.image.shape)

    def test_bundled_haar_models_are_present_and_loadable(self):
        root = Path(__file__).resolve().parents[1] / "models" / "haar"
        required = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_profileface.xml",
            "haarcascade_eye.xml",
            "haarcascade_fullbody.xml",
            "haarcascade_upperbody.xml",
            "haarcascade_lowerbody.xml",
            "haarcascade_smile.xml",
            "haarcascade_russian_plate_number.xml",
            "haarcascade_frontalcatface.xml",
        ]
        bundled = sorted(root.glob("*.xml"))
        self.assertEqual(len(bundled), 17, "Expected 17 bundled Haar Cascade XML models")
        for filename in required:
            self.assertTrue((root / filename).is_file(), f"Missing bundled model: {filename}")
        for model in bundled:
            cascade = cv2.CascadeClassifier(str(model))
            self.assertFalse(cascade.empty(), f"Could not load bundled model: {model.name}")
        blank = np.zeros((96, 128, 3), dtype=np.uint8)
        for model in bundled:
            result = processing.detect_haar(blank, model, model.stem)
            self.assertEqual(result.shape, blank.shape, f"Model inference smoke test failed: {model.name}")

    def test_new_lab_operations(self):
        other = np.full_like(self.image, 180)
        for op in ('and', 'or', 'xor'):
            result = processing.logical_operation(self.image, other, op, 100)
            self.assertEqual(result.ndim, 2)
        props = processing.image_properties(self.image)
        self.assertEqual(props['width'], 32)
        self.assertEqual(props['height'], 24)
        hist = processing.histogram_channels(self.image)
        self.assertEqual(set(hist), {'Red', 'Green', 'Blue'})
        self.assertTrue(all(len(v) == 256 for v in hist.values()))

    def test_rgba_intensity_transforms_preserve_alpha(self):
        rgba = np.dstack((self.image, np.full(self.image.shape[:2], 83, dtype=np.uint8)))
        for result in (
            processing.sharpen(rgba, 1.2, 1.5),
            processing.log_transform(rgba),
            processing.gamma_transform(rgba, 1.4),
            processing.negative(rgba),
            processing.kmeans_segmentation(rgba, 3),
        ):
            self.assertEqual(result.shape, rgba.shape)
            np.testing.assert_array_equal(result[:, :, 3], rgba[:, :, 3])

    def test_gamma_and_crop(self):
        self.assertEqual(processing.gamma_transform(self.image, 1.2).shape, self.image.shape)
        self.assertEqual(processing.crop(self.image, 2, 3, 8, 9).shape[:2], (9, 8))


    def test_hdr_sharpen_mask_and_draw_controls(self):
        hdr = processing.hdr_tone(self.image, 1.5, 1.2, 0.25)
        sharp = processing.sharpen(self.image, 1.8, 1.5)
        mask = processing.color_mask(self.image, (0, 0, 0), (179, 255, 255))
        line = processing.draw_primitive(self.image, 'line', 1, 2, 20, 15, thickness=4)
        rect = processing.draw_primitive(self.image, 'rectangle', 2, 3, 18, 17, thickness=2)
        circle = processing.draw_primitive(self.image, 'circle', 10, 10, 16, 16, thickness=2)
        for result in (hdr, sharp, mask, line, rect, circle):
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(result.shape, self.image.shape)

    def test_arithmetic_coefficients_are_manual(self):
        other = np.full_like(self.image, 50)
        add_low = processing.arithmetic(self.image, other, 'add', coefficient=0.5)
        add_high = processing.arithmetic(self.image, other, 'add', coefficient=2.0)
        self.assertFalse(np.array_equal(add_low, add_high))
        sub = processing.arithmetic(self.image, other, 'subtract', coefficient=0.5)
        mul = processing.arithmetic(self.image, other, 'multiply', coefficient=1.5)
        self.assertEqual(sub.shape, self.image.shape)
        self.assertEqual(mul.shape, self.image.shape)

    def test_rotation_expands_canvas(self):
        out=processing.rotate(self.image,45)
        self.assertGreaterEqual(out.shape[0],self.image.shape[0])
        self.assertGreaterEqual(out.shape[1],self.image.shape[1])

if __name__ == "__main__":
    unittest.main()

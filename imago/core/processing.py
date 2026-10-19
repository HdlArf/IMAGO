"""Image processing operations. App-facing arrays use RGB/RGBA or grayscale."""
from __future__ import annotations
import cv2
import numpy as np

def _gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2: return arr.copy()
    if arr.shape[2] == 4: return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2GRAY)

def _bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2: return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4: return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_RGB2BGR)

def _rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2: return arr
    if arr.shape[2] == 4: return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(np.ascontiguousarray(arr), cv2.COLOR_BGR2RGB)

def exposure(image, value):
    stops = float(np.clip(value, -100, 100)) / 50.0
    gain = 2.0 ** stops
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[:, :, :3])
        out = np.clip(color.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return np.dstack((out, arr[:, :, 3]))
    return np.clip(np.ascontiguousarray(arr).astype(np.float32) * gain, 0, 255).astype(np.uint8)

def contrast(image, value):
    factor = max(0.0, 1.0 + float(value) / 100.0)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[:, :, :3])
        out = np.clip((color.astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)
        return np.dstack((out, arr[:, :, 3]))
    return np.clip((np.ascontiguousarray(arr).astype(np.float32) - 127.5) * factor + 127.5, 0, 255).astype(np.uint8)

def brightness(image, value):
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[:, :, :3])
        out = np.clip(color.astype(np.float32) + float(value), 0, 255).astype(np.uint8)
        return np.dstack((out, arr[:, :, 3]))
    return np.clip(np.ascontiguousarray(arr).astype(np.float32) + float(value), 0, 255).astype(np.uint8)

def saturation(image, value):
    if image.ndim == 2: return image.copy()
    alpha = image[:,:,3].copy() if image.shape[2] == 4 else None
    rgb = np.ascontiguousarray(image[:,:,:3]) if alpha is not None else np.ascontiguousarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * (1 + float(value)/100), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.dstack((out, alpha)) if alpha is not None else out

def tone(image, value):
    if image.ndim == 2: return image.copy()
    alpha = image[:,:,3].copy() if image.shape[2] == 4 else None
    rgb = np.ascontiguousarray(image[:,:,:3]) if alpha is not None else np.ascontiguousarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:,:,0] = (hsv[:,:,0] + int(value)) % 180
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.dstack((out, alpha)) if alpha is not None else out

def grayscale(image): 
    return _gray(image)

def gaussian_blur(image, kernel):
    k = max(1, int(kernel)); k += (k % 2 == 0)
    return image.copy() if k == 1 else cv2.GaussianBlur(image, (k, k), 0)

def mean_blur(image, kernel):
    k = max(1, int(kernel)); k += (k % 2 == 0)
    return cv2.blur(image, (k, k))

def median_blur(image, kernel):
    k = max(1, int(kernel)); k += (k % 2 == 0)
    return image.copy() if k == 1 else cv2.medianBlur(image, k)

def sharpen(image, amount=1.0, radius=2.0, threshold=0):
    amount = max(0.0, float(amount))
    radius = max(0.1, float(radius))
    threshold = max(0.0, float(threshold))
    if amount == 0: return image.copy()
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[:, :, :3])
        alpha = arr[:, :, 3]
    else:
        color = np.ascontiguousarray(arr)
        alpha = None
    blurred = cv2.GaussianBlur(color, (0, 0), radius)
    sharpened = cv2.addWeighted(color, 1.0 + amount, blurred, -amount, 0)
    if threshold > 0:
        diff = cv2.absdiff(color, sharpened)
        if diff.ndim == 3: diff = diff.max(axis=2)
        mask = (diff >= threshold).astype(np.uint8) * 255
        if color.ndim == 3: mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        out = np.where(mask > 0, sharpened, color).astype(np.uint8)
    else:
        out = sharpened
    return np.dstack((out, alpha)) if alpha is not None else out

def hdr_tone(image, strength=1.0, saturation=1.0, warmth=0.0):
    strength = float(np.clip(strength, 0.0, 3.0))
    saturation = float(np.clip(saturation, 0.0, 2.0))
    warmth = float(np.clip(warmth, -1.0, 1.0))
    arr = np.asarray(image)
    if arr.ndim == 2:
        base = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        alpha = None
    else:
        alpha = arr[:,:,3].copy() if arr.shape[2] == 4 else None
        base = np.ascontiguousarray(arr[:,:,:3]) if alpha is not None else np.ascontiguousarray(arr)
    lab = cv2.cvtColor(base, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0+strength*2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(l)
    mixed = cv2.addWeighted(l, max(0.0, 1.0-strength*.45), enhanced, min(1.0, strength*.45), 0)
    out = cv2.cvtColor(cv2.merge((mixed, a, b)), cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1]*saturation, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.int16)
    out[:,:,0] = np.clip(out[:,:,0]+int(24*warmth), 0, 255)
    out[:,:,2] = np.clip(out[:,:,2]-int(24*warmth), 0, 255)
    out = np.uint8(out)
    if arr.ndim == 2: return cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    return np.dstack((out, alpha)) if alpha is not None else out

def laplacian(image):
    g = _gray(image); edges = cv2.Laplacian(g, cv2.CV_16S, ksize=3)
    return cv2.convertScaleAbs(edges)

def sobel(image, direction='both'):
    g = _gray(image); x = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); y = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    if direction == 'x': out = np.abs(x)
    elif direction == 'y': out = np.abs(y)
    else: out = cv2.magnitude(x, y)
    return cv2.convertScaleAbs(out)

def rotate(image, angle):
    h, w = image.shape[:2]; center = (w/2.0, h/2.0); matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    c, s = abs(matrix[0,0]), abs(matrix[0,1]); nw, nh = max(1, int(h*s+w*c)), max(1, int(h*c+w*s))
    matrix[0,2] += nw/2 - center[0]; matrix[1,2] += nh/2 - center[1]
    border = (0,0,0,0) if image.ndim == 3 and image.shape[2] == 4 else (0,0,0)
    return cv2.warpAffine(np.ascontiguousarray(image), matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border)

def resize(image, width, height, interpolation='cubic'):
    modes = {'nearest':cv2.INTER_NEAREST,'linear':cv2.INTER_LINEAR,'cubic':cv2.INTER_CUBIC,'area':cv2.INTER_AREA,'lanczos':cv2.INTER_LANCZOS4}
    return cv2.resize(np.ascontiguousarray(image), (max(1, int(width)), max(1, int(height))), interpolation=modes.get(interpolation, cv2.INTER_CUBIC))

def translate(image, x, y):
    h, w = image.shape[:2]; m = np.float32([[1,0,float(x)],[0,1,float(y)]])
    return cv2.warpAffine(np.ascontiguousarray(image), m, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0) if image.ndim==3 and image.shape[2]==4 else (0,0,0))

def crop(image, x, y, width, height):
    h, w = image.shape[:2]; x = max(0, min(int(x), w-1)); y = max(0, min(int(y), h-1))
    x2 = max(x+1, min(w, x+int(width))); y2 = max(y+1, min(h, y+int(height)))
    return image[y:y2, x:x2].copy()

def threshold(image, value): return cv2.threshold(_gray(image), int(value), 255, cv2.THRESH_BINARY)[1]
def binary_inverse(image, value=127): return cv2.threshold(_gray(image), int(value), 255, cv2.THRESH_BINARY_INV)[1]
def otsu(image): return cv2.threshold(_gray(image), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
def adaptive_threshold(image, block_size=11, c=2):
    b = max(3, int(block_size)); b += b % 2 == 0
    return cv2.adaptiveThreshold(_gray(image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, b, int(c))

def canny(image, low, high): return cv2.Canny(_gray(image), int(low), int(max(int(low)+1, int(high))))

def equalize(image, clip_limit=2.0, tile_size=8):
    clip_limit = max(0.1, float(clip_limit))
    tile_size = max(1, int(tile_size))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    arr = np.asarray(image)
    if arr.ndim == 2: return clahe.apply(arr)
    color = np.ascontiguousarray(arr[:, :, :3]) if arr.shape[2] == 4 else np.ascontiguousarray(arr)
    yuv = cv2.cvtColor(color, cv2.COLOR_RGB2YCrCb)
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    out = cv2.cvtColor(yuv, cv2.COLOR_YCrCb2RGB)
    return np.dstack((out, arr[:, :, 3])) if arr.shape[2] == 4 else out

def log_transform(image):
    arr = np.asarray(image)
    scale = 255.0 / np.log1p(255.0)
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[:, :, :3])
        out = np.clip(scale * np.log1p(color.astype(np.float32)), 0, 255).astype(np.uint8)
        return np.dstack((out, arr[:, :, 3]))
    return np.clip(scale * np.log1p(np.ascontiguousarray(arr).astype(np.float32)), 0, 255).astype(np.uint8)

def gamma_transform(image, gamma=1.0):
    gamma = max(0.01, float(gamma))
    table = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255, 0, 255).astype(np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[:, :, :3])
        out = cv2.LUT(color, table)
        return np.dstack((out, arr[:, :, 3]))
    return cv2.LUT(np.ascontiguousarray(arr), table)

def negative(image):
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 4:
        color = np.ascontiguousarray(arr[:, :, :3])
        return np.dstack((255 - color, arr[:, :, 3]))
    return 255 - np.ascontiguousarray(arr)

def bitwise_not(image): return cv2.bitwise_not(np.ascontiguousarray(image))

def morphology(image, operation='open', kernel_size=3, iterations=1):
    gray = _gray(image); k = max(1, int(kernel_size)); kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k,k))
    op = {'erode':cv2.MORPH_ERODE,'dilate':cv2.MORPH_DILATE,'open':cv2.MORPH_OPEN,'close':cv2.MORPH_CLOSE,'gradient':cv2.MORPH_GRADIENT,'tophat':cv2.MORPH_TOPHAT,'blackhat':cv2.MORPH_BLACKHAT}.get(operation, cv2.MORPH_OPEN)
    return cv2.morphologyEx(gray, op, kernel, iterations=max(1, int(iterations)))

def kmeans_segmentation(image, clusters=3):
    arr = np.asarray(image)
    alpha = arr[:, :, 3].copy() if arr.ndim == 3 and arr.shape[2] == 4 else None
    if arr.ndim == 3:
        color = np.ascontiguousarray(arr[:, :, :3])
        pixels = color.reshape((-1, 3)).astype(np.float32)
    else:
        color = np.ascontiguousarray(arr)
        pixels = arr.reshape((-1, 1)).astype(np.float32)
    k = max(2, min(int(clusters), len(pixels)))
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    if arr.ndim == 2:
        out = centers.astype(np.uint8)[labels.flatten()].reshape(arr.shape)
    else:
        out = centers.astype(np.uint8)[labels.flatten()].reshape(arr.shape[:2] + (color.shape[2],))
    return np.dstack((out, alpha)) if alpha is not None else out

def watershed(image):
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4): raise ValueError('Watershed requires a color image.')
    bgr = _bgr(arr); gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY); _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8); opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2); sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5); _, sure_fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0); sure_fg = np.uint8(sure_fg); unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg); markers += 1; markers[unknown==255] = 0; markers = cv2.watershed(bgr, markers); bgr[markers==-1] = (255,0,0)
    return _rgb(bgr)

def detect_lines(image, threshold_value=80):
    g = _gray(image); edges = cv2.Canny(g, 50, 150); lines = cv2.HoughLinesP(edges, 1, np.pi/180, int(threshold_value), minLineLength=30, maxLineGap=8); out = _bgr(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]: cv2.line(out, (x1,y1), (x2,y2), (0,220,255), 2)
    return _rgb(out)

def detect_circles(image, min_radius=5, max_radius=100):
    g = cv2.GaussianBlur(_gray(image), (9,9), 2); circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=25, param1=100, param2=28, minRadius=int(min_radius), maxRadius=int(max_radius)); out = _bgr(image)
    if circles is not None:
        for x,y,r in np.round(circles[0]).astype(int): cv2.circle(out, (x,y), r, (0,255,0), 2); cv2.circle(out, (x,y), 2, (0,0,255), -1)
    return _rgb(out)

def detect_corners(image, max_corners=100):
    g = _gray(image); corners = cv2.goodFeaturesToTrack(g, maxCorners=int(max_corners), qualityLevel=.01, minDistance=8); out = _bgr(image)
    if corners is not None:
        for x,y in np.int32(corners).reshape(-1,2): cv2.circle(out, (x,y), 4, (255,80,30), -1)
    return _rgb(out)

def arithmetic(image, other, operation='add', weight=0.5, coefficient=1.0):
    if other is None: raise ValueError('A second image is required.')
    if image.ndim == 2:
        a = np.ascontiguousarray(image); b = _gray(other)
    else:
        a = np.ascontiguousarray(image[:, :, :3]) if image.ndim == 3 else np.ascontiguousarray(image)
        b = np.ascontiguousarray(other[:, :, :3]) if other.ndim == 3 else cv2.cvtColor(other, cv2.COLOR_GRAY2RGB)
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    if a.ndim != b.ndim or (a.ndim == 3 and a.shape[2] != b.shape[2]):
        if a.ndim == 2: b = _gray(b)
        else: b = cv2.cvtColor(_gray(b), cv2.COLOR_GRAY2RGB)
    coefficient = max(0.0, float(coefficient))
    if operation == 'add': return np.clip(a.astype(np.float32)+b.astype(np.float32)*coefficient, 0, 255).astype(np.uint8)
    if operation == 'subtract': return np.clip(a.astype(np.float32)-b.astype(np.float32)*coefficient, 0, 255).astype(np.uint8)
    if operation == 'difference': return cv2.absdiff(a, b)
    if operation == 'multiply': return np.clip(a.astype(np.float32)*b.astype(np.float32)*(coefficient/255.0), 0, 255).astype(np.uint8)
    if operation == 'divide': return cv2.divide(a, b)
    if operation == 'blend': return cv2.addWeighted(a, float(weight), b, 1-float(weight), 0)
    raise ValueError(f'Unsupported arithmetic operation: {operation}')

def detect_haar(image, model_path, label='Object', scale_factor=1.1, min_neighbors=5):
    from pathlib import Path
    p = Path(model_path)
    if not p.is_file(): raise FileNotFoundError(f'Model not found: {p}')
    cascade = cv2.CascadeClassifier(str(p))
    if cascade.empty(): raise RuntimeError(f'Cannot load cascade model: {p}')
    gray = _gray(image); detections = cascade.detectMultiScale(gray, scaleFactor=max(1.01, float(scale_factor)), minNeighbors=max(0, int(min_neighbors)))
    out = _bgr(image)
    if len(detections) == 0:
        cv2.putText(out, "No objects detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for x,y,w,h in detections:
            cv2.rectangle(out, (x,y), (x+w,y+h), (0,220,255), 2); cv2.putText(out, label, (x,max(18,y-6)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,220,255), 2, cv2.LINE_AA)
    return _rgb(out)

def color_space_view(image, space='HSV', channel=0):
    if image.ndim == 2: rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4: rgb = np.ascontiguousarray(image[:, :, :3])
    else: rgb = np.ascontiguousarray(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    if channel == 0: return np.ascontiguousarray(hsv[:, :, 0])
    elif channel == 1: return np.ascontiguousarray(hsv[:, :, 1])
    elif channel == 2: return np.ascontiguousarray(hsv[:, :, 2])
    return hsv

def color_mask(image, target_color=(255, 255, 0), tolerance=30):
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else np.ascontiguousarray(image[:, :, :3])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    target_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = target_hsv
    lower = np.array([max(0, h - tolerance), max(0, s - tolerance), max(0, v - tolerance)])
    upper = np.array([min(179, h + tolerance), min(255, s + tolerance), min(255, v + tolerance)])
    mask = cv2.inRange(hsv, lower, upper)
    out = cv2.bitwise_and(rgb, rgb, mask=mask)
    return np.dstack((out, image[:, :, 3])) if image.ndim == 3 and image.shape[2] == 4 else out

def extract_object(image, threshold_value=127, invert=False):
    gray = _gray(image)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, int(threshold_value), 255, mode)
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image[:, :, :3]
    return cv2.bitwise_and(rgb, rgb, mask=mask)

def logical_operation(image, other=None, operation='not', threshold_value=127):
    a = _gray(image)
    if operation == 'not': return cv2.bitwise_not(a)
    if other is None:
        _, a = cv2.threshold(a, int(threshold_value), 255, cv2.THRESH_BINARY)
        b = a.copy()
    else:
        b = _gray(other)
        if b.shape != a.shape: b = cv2.resize(b, (a.shape[1], a.shape[0]))
        _, a = cv2.threshold(a, int(threshold_value), 255, cv2.THRESH_BINARY)
        _, b = cv2.threshold(b, int(threshold_value), 255, cv2.THRESH_BINARY)
    op = str(operation).lower()
    if op == 'and': return cv2.bitwise_and(a, b)
    if op == 'or': return cv2.bitwise_or(a, b)
    if op == 'xor': return cv2.bitwise_xor(a, b)
    raise ValueError(f'Unsupported logical operation: {operation}')

def draw_primitive(image, primitive='rectangle', x1=None, y1=None, x2=None, y2=None, color=(0,220,255), thickness=3):
    out = _bgr(image); h, w = out.shape[:2]
    x1 = int(np.clip(w*.2 if x1 is None else x1, 0, w-1)); y1 = int(np.clip(h*.2 if y1 is None else y1, 0, h-1))
    x2 = int(np.clip(w*.8 if x2 is None else x2, 0, w-1)); y2 = int(np.clip(h*.8 if y2 is None else y2, 0, h-1))
    color = tuple(int(np.clip(c, 0, 255)) for c in color); thickness = max(1, int(thickness))
    if primitive == 'line': cv2.line(out, (x1,y1), (x2,y2), color, thickness, cv2.LINE_AA)
    elif primitive == 'circle':
        radius = max(1, int(np.hypot(x2-x1, y2-y1))); cv2.circle(out, (x1,y1), radius, color, thickness, cv2.LINE_AA)
    elif primitive == 'rectangle': cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness, cv2.LINE_AA)
    else: raise ValueError(f'Unsupported drawing primitive: {primitive}')
    return _rgb(out)

def histogram_channels(image):
    if image.ndim == 2: return {'Gray': cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()}
    names = ('Red', 'Green', 'Blue'); rgb = image[:, :, :3]
    return {name: cv2.calcHist([rgb], [idx], None, [256], [0, 256]).flatten() for idx, name in enumerate(names)}

def image_properties(image):
    if image is None: raise ValueError('Import an image first.')
    h, w = image.shape[:2]
    return {'width': int(w), 'height': int(h), 'shape': tuple(image.shape), 'dtype': str(image.dtype), 'channels': 1 if image.ndim == 2 else int(image.shape[2]), 'min': int(image.min()), 'max': int(image.max())}
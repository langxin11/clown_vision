from __future__ import annotations

import cv2
import numpy as np


def read_image_any_path(path: str) -> np.ndarray | None:
    """Read image from path supporting non-ASCII characters.

    Returns BGR np.ndarray or None if failed.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def to_uint8_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert input image to uint8 grayscale consistently.

    - Accepts BGR/GRAY, float or integer arrays
    - Scales/normalizes to 0-255 uint8
    """
    if image is None:
        raise ValueError("image is None")

    arr = image
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v <= min_v:
        return np.zeros_like(arr, dtype=np.uint8)
    arr = (arr - min_v) / (max_v - min_v) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_image_any_path(path: str, image: np.ndarray) -> bool:
    """Write image to path supporting non-ASCII characters."""
    try:
        ext = path.split(".")[-1].lower()
        result, encoded = cv2.imencode(f".{ext}", image)
        if not result:
            return False
        encoded.tofile(path)
        return True
    except Exception:
        return False



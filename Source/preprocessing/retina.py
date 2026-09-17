from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]

DETECT_MAX_SIDE = 1024
DARK_LEVEL = 10  # a border pixel is darker than this on every channel
OPEN_FRACTION = 0.005  # kernel radius as a fraction of the short side

def _largest_component(mask: np.ndarray) -> Optional[Tuple[Box, float]]:
    """Bounding box of the largest blob, with how densely it fills that box."""

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    left = int(stats[index, cv2.CC_STAT_LEFT])
    top = int(stats[index, cv2.CC_STAT_TOP])
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])
    if width <= 0 or height <= 0:
        return None
    area = float(stats[index, cv2.CC_STAT_AREA])
    return (left, top, left + width, top + height), area / (width * height)


def _open(mask: np.ndarray) -> np.ndarray:
    radius = max(1, int(round(min(mask.shape) * OPEN_FRACTION)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def retina_box(rgb: np.ndarray) -> Optional[Box]:

    height, width = rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = rgb

    found = _largest_component(_open((small.max(axis=2) > DARK_LEVEL).astype(np.uint8)))
    if found is None:
        return None

    box, _ = found
    inverse = 1.0 / scale if scale < 1.0 else 1.0
    left = max(0, int(box[0] * inverse))
    top = max(0, int(box[1] * inverse))
    right = min(width, int(round(box[2] * inverse)))
    bottom = min(height, int(round(box[3] * inverse)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def crop_to_retina(rgb: np.ndarray) -> np.ndarray:
    box = retina_box(rgb)
    if box is None:
        return rgb
    left, top, right, bottom = box
    return rgb[top:bottom, left:right]


def pad_to_square(rgb: np.ndarray) -> np.ndarray:
    height, width = rgb.shape[:2]
    side = max(height, width)
    if height == width:
        return rgb
    top = (side - height) // 2
    left = (side - width) // 2
    canvas = np.zeros((side, side, rgb.shape[2]), dtype=rgb.dtype)
    canvas[top : top + height, left : left + width] = rgb
    return canvas


def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    interpolation = cv2.INTER_AREA if rgb.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interpolation)


def standardise(rgb: np.ndarray, size: int) -> np.ndarray:
    return resize_square(pad_to_square(crop_to_retina(rgb)), size)


def retina_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(crop_to_retina(rgb), cv2.COLOR_RGB2GRAY)

"""Retina localisation, shared by hashing and by dataset building.

Both stages must agree on where the retina is: if the deduplicator hashes a
different crop than the trainer sees, a pair judged identical here is not the
pair the model gets.  So the geometry lives in exactly one place.

The detector is the one measured on this data (FINDINGS.md #5):

* detection runs on a copy scaled to <= 1024 px, then coordinates map back, so
  a 4288x2848 frame costs the same as a small one;
* the mask is the **largest connected component**, not the bounding box of
  every non-dark pixel -- a burnt-in white caption in a corner would otherwise
  stretch the crop to the whole frame;
* a morphological open with a kernel scaled to 0.5 % of the short side removes
  speckle without eating the disc edge;
**No Otsu fallback.**  FINDINGS.md #5 lists one as a planned improvement for
frames whose border is grey rather than black; measuring it on 1,200 images of
this data killed the idea.  The fallback path was reached by 48 of them and
*every* acceptance cut real retina away -- 9.9 % of the bright pixels on
average, 59.6 % in the worst case -- because a shape guard (disc-like fill,
near-square, sane area) cannot tell a dim retinal periphery from a border.
The primary detector needs no rescuing: on images where the two disagree it
discards 0.05 % of bright pixels on average, 0.23 % at worst.  When no border
exists the largest component is the whole frame, and keeping the whole frame is
the right answer.
"""

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
    """Bounding box of the retina in ``rgb`` (H, W, 3), or None to keep it all."""

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
    """Centre the crop on black, so resizing cannot change the aspect ratio.

    Squashing a wide crop to a square would stretch every lesion along one axis
    and make its shape depend on the camera's frame, not on the eye.
    """

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
    """INTER_AREA when shrinking is the only choice that averages, not samples.

    The median source retina is 600 px across, so 448 is a 1.34x reduction;
    point sampling there would alias the microaneurysms this resolution exists
    to preserve.
    """

    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    interpolation = cv2.INTER_AREA if rgb.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(rgb, (size, size), interpolation=interpolation)


def standardise(rgb: np.ndarray, size: int) -> np.ndarray:
    return resize_square(pad_to_square(crop_to_retina(rgb)), size)


def retina_gray(rgb: np.ndarray) -> np.ndarray:
    """Grayscale retina crop -- the input every perceptual hash is taken on."""

    return cv2.cvtColor(crop_to_retina(rgb), cv2.COLOR_RGB2GRAY)

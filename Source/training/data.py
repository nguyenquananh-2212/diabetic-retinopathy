"""Dataset, augmentation and sampling for the 448 px build.

Three decisions here are not defaults, they are measurements (FINDINGS.md):

**Geometry is dihedral only.**  76 % of these images are not a circular disc --
24.2 % of their content lies outside the inscribed circle -- so a free rotation
crops real retina away: 1.7 % of it at 10 degrees, 7.5 % at 45.  The eight
symmetries of the square (four 90-degree rotations x flip) were verified to lose
exactly 0 pixels over 60 real images x 8 draws.  Small affine jitter is offered
but off by default, and it pads by reflection so it cannot introduce black.

**Colour normalisation is a switch, not a given.**  Ben Graham subtraction cuts
between-image variance 91 % against CLAHE's 26 %, but it destroys absolute
colour, and hard exudates (yellow) are told from haemorrhages (red) by colour.
It has to be A/B tested per model, so it lives here as an option.

**The sampler and the loss are separate knobs.**  Oversampling changes what the
model sees; class weights change what it is punished for.  Turning both to full
strength double-counts the correction, so each takes a strength in [0, 1] and
the default leans on the sampler.
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CLASSES = 5


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #


def ben_graham(rgb: np.ndarray, sigma_divisor: float = 30.0) -> np.ndarray:
    """``4*img - 4*blur + 128`` -- flattens illumination, destroys absolute hue."""

    sigma = max(1.0, rgb.shape[1] / sigma_divisor)
    blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
    out = cv2.addWeighted(rgb, 4.0, blur, -4.0, 128.0)
    return out


def clahe(rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    """Local contrast in LAB, so only lightness moves and hue survives."""

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    operator = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[:, :, 0] = operator.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


COLOUR_MODES = {"none": None, "ben_graham": ben_graham, "clahe": clahe}


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def dihedral(rgb: np.ndarray, index: int) -> np.ndarray:
    """One of the 8 symmetries of the square.  Lossless by construction."""

    out = np.rot90(rgb, index % 4)
    if index >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


@dataclass
class AugmentConfig:
    """Everything that may differ between train and eval."""

    colour_mode: str = "none"
    dihedral: bool = True
    # The only transform here that can lose retina.  Measured on this data:
    # 10 degrees costs 1.7 % of retinal pixels, 45 degrees costs 7.5 %, and 90
    # costs nothing (which is why the dihedral group is free).  TRAIN_AUGMENT
    # spends the 1.7 % deliberately; the dataclass default does not, so an eval
    # or debug config built from scratch never rotates by accident.
    affine_degrees: float = 0.0
    affine_scale: Tuple[float, float] = (1.0, 1.0)
    affine_translate: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    cutout_fraction: float = 0.0

    def is_identity(self) -> bool:
        return not (
            self.dihedral
            or self.affine_degrees
            or self.affine_translate
            or self.affine_scale != (1.0, 1.0)
            or self.brightness
            or self.contrast
            or self.cutout_fraction
        )


TRAIN_AUGMENT = AugmentConfig(
    dihedral=True,
    affine_degrees=10.0,
    affine_scale=(0.92, 1.08),
    affine_translate=0.04,
    brightness=0.12,
    contrast=0.12,
    cutout_fraction=0.10,
)

EVAL_AUGMENT = AugmentConfig(dihedral=False)


def _apply_affine(
    rgb: np.ndarray, config: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    if not (config.affine_degrees or config.affine_translate or config.affine_scale != (1.0, 1.0)):
        return rgb
    height, width = rgb.shape[:2]
    angle = rng.uniform(-config.affine_degrees, config.affine_degrees)
    scale = rng.uniform(*config.affine_scale)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    if config.affine_translate:
        matrix[0, 2] += rng.uniform(-1, 1) * config.affine_translate * width
        matrix[1, 2] += rng.uniform(-1, 1) * config.affine_translate * height
    # Reflect, never constant: a black wedge would look like the camera border
    # the preprocessing stage spent its time removing.
    return cv2.warpAffine(
        rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def augment(rgb: np.ndarray, config: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if config.dihedral:
        rgb = dihedral(rgb, int(rng.integers(8)))
    rgb = _apply_affine(rgb, config, rng)

    if config.brightness or config.contrast:
        gain = 1.0 + rng.uniform(-config.contrast, config.contrast)
        bias = 255.0 * rng.uniform(-config.brightness, config.brightness)
        rgb = cv2.convertScaleAbs(rgb, alpha=gain, beta=bias)

    if config.cutout_fraction and rng.random() < 0.5:
        height, width = rgb.shape[:2]
        side = int(min(height, width) * config.cutout_fraction)
        if side > 1:
            top = int(rng.integers(0, height - side))
            left = int(rng.integers(0, width - side))
            rgb = rgb.copy()
            rgb[top : top + side, left : left + side] = 0
    return rgb


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    image_id: str
    path: str
    label: int
    group_id: str
    dataset_source: str


def relocate(path: str, root: Path) -> str:
    """Re-root ``<...>/images/<source>/<file>.png`` under a new directory.

    The manifest stores absolute paths from the machine that built it, so a
    Windows manifest is unusable on Kaggle as written.  Splitting on both
    separators rather than using ``Path`` is deliberate: ``PurePosixPath`` does
    not treat a backslash as a separator, so a Windows path read on Linux comes
    back as one long filename and the rewrite silently does nothing.
    """

    parts = re.split(r"[\\/]+", path.strip())
    if len(parts) < 2:
        raise ValueError(f"cannot re-root {path!r}: no <source>/<file> tail")
    return str(Path(root) / parts[-2] / parts[-1])


def load_split(
    manifest: Path, splits: Path, split: str, image_root: Optional[Path] = None
) -> List[Sample]:
    with Path(splits).open(newline="", encoding="utf-8-sig") as handle:
        wanted = {r["image_id"] for r in csv.DictReader(handle) if r["split"] == split}
    with Path(manifest).open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["image_id"] in wanted]
    if not wanted:
        # 0 == 0 passes the count check below, so an empty split would sail
        # through here and only surface as "num_samples should be a positive
        # integer" from deep inside the sampler.
        raise ValueError(f"no rows with split == '{split}' in {splits}")
    if len(rows) != len(wanted):
        raise ValueError(
            f"split '{split}' names {len(wanted)} images but the manifest has {len(rows)}"
        )
    samples = [
        Sample(
            image_id=r["image_id"],
            path=r["processed_path"]
            if image_root is None
            else relocate(r["processed_path"], image_root),
            label=int(r["class_label"]),
            group_id=r["group_id"],
            dataset_source=r["dataset_source"],
        )
        for r in rows
    ]
    # Fail here, with a path to look at, rather than inside a worker 200 steps in.
    missing = [s for s in samples[:64] if not Path(s.path).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the first 64 '{split}' images are missing, "
            f"e.g. {missing[0].path}"
        )
    return samples


class FundusDataset(Dataset):
    """Reads the pre-rendered PNGs; all cropping already happened on disk."""

    def __init__(
        self,
        samples: Sequence[Sample],
        config: AugmentConfig = EVAL_AUGMENT,
        seed: int = 0,
        tta_index: Optional[int] = None,
    ) -> None:
        self.samples = list(samples)
        self.config = config
        self.seed = seed
        self.tta_index = tta_index
        self.colour = COLOUR_MODES[config.colour_mode]
        self._rng: Optional[np.random.Generator] = None

    def rng(self) -> np.random.Generator:
        """One stream per worker, created on first use and never reset.

        Seeding per ``(seed, index)`` instead looks reproducible and is in fact
        broken: the same image would draw the same rotation, the same crop and
        the same brightness in every epoch, so the augmentation adds variety
        exactly once and then repeats a fixed 29k-image set forever.  A stream
        that keeps advancing gives different draws each epoch, and seeding it
        from the worker's own torch seed keeps the run reproducible as a whole,
        since DataLoader derives those seeds from its ``generator``.
        """

        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = self.seed if info is None else (self.seed + info.seed) % (2**63)
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        # np.fromfile, not cv2.imread: cv2 cannot open a non-ASCII path on Windows.
        buffer = np.fromfile(sample.path, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unreadable image: {sample.path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.colour is not None:
            rgb = self.colour(rgb)

        if self.tta_index is not None:
            rgb = dihedral(rgb, self.tta_index)
        elif not self.config.is_identity():
            rgb = augment(rgb, self.config, self.rng())

        tensor = rgb.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(tensor.transpose(2, 0, 1)), sample.label


# --------------------------------------------------------------------------- #
# Imbalance
# --------------------------------------------------------------------------- #


def class_counts(samples: Sequence[Sample]) -> np.ndarray:
    counter = Counter(s.label for s in samples)
    return np.array([counter.get(c, 0) for c in range(NUM_CLASSES)], dtype=np.float64)


def balanced_sampler(
    samples: Sequence[Sample], strength: float = 0.5, generator: Optional[torch.Generator] = None
) -> WeightedRandomSampler:
    """Sample weight proportional to ``(1 / count) ** strength``.

    ``strength=0`` is the natural distribution, ``1`` is fully balanced.  Full
    balance draws each grade equally often.  On the current train split, grade
    3 has 1,320 images: each is drawn 3.12 times per epoch in expectation at
    strength=1, versus 1.87 at the default 0.5.  Weaker sampling reduces repeats
    without removing minority-class images.
    """

    counts = class_counts(samples)
    counts[counts == 0] = 1.0
    per_class = (1.0 / counts) ** float(strength)
    per_class = per_class / per_class.sum()
    weights = torch.as_tensor(
        [per_class[s.label] for s in samples], dtype=torch.double
    )
    return WeightedRandomSampler(
        weights, num_samples=len(samples), replacement=True, generator=generator
    )


def class_weights(samples: Sequence[Sample], strength: float = 0.5) -> torch.Tensor:
    """Loss weights, normalised to mean 1 so the learning rate keeps its meaning."""

    counts = class_counts(samples)
    counts[counts == 0] = 1.0
    weights = (counts.sum() / (NUM_CLASSES * counts)) ** float(strength)
    weights = weights / weights.mean()
    return torch.as_tensor(weights, dtype=torch.float32)


def describe(samples: Sequence[Sample]) -> str:
    counts = class_counts(samples).astype(int)
    total = int(counts.sum())
    per_source = Counter(s.dataset_source for s in samples)
    shares = " ".join(
        f"{c}:{n:,}({100 * n / max(1, total):.1f}%)" for c, n in enumerate(counts)
    )
    return f"{total:,} images | {shares} | sources {dict(sorted(per_source.items()))}"

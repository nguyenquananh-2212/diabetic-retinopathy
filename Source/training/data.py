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
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NUM_CLASSES = 5

def ben_graham(rgb: np.ndarray, sigma_divisor: float = 30.0) -> np.ndarray:
    sigma = max(1.0, rgb.shape[1] / sigma_divisor)
    blur = cv2.GaussianBlur(rgb, (0, 0), sigma)
    out = cv2.addWeighted(rgb, 4.0, blur, -4.0, 128.0)
    return out


def clahe(rgb: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    operator = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[:, :, 0] = operator.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


COLOUR_MODES = {"none": None, "ben_graham": ben_graham, "clahe": clahe}

def dihedral(rgb: np.ndarray, index: int) -> np.ndarray:

    out = np.rot90(rgb, index % 4)
    if index >= 4:
        out = np.fliplr(out)
    return np.ascontiguousarray(out)


@dataclass
class AugmentConfig:

    colour_mode: str = "none"
    dihedral: bool = True
    affine_degrees: float = 0.0
    affine_scale: Tuple[float, float] = (1.0, 1.0)
    affine_translate: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0

    saturation: float = 0.0
    cutout_fraction: float = 0.0

    def is_identity(self) -> bool:
        return not (
            self.dihedral
            or self.affine_degrees
            or self.affine_translate
            or self.affine_scale != (1.0, 1.0)
            or self.photometric()
            or self.cutout_fraction
        )

    def photometric(self) -> bool:
        return bool(self.brightness or self.contrast or self.gamma or self.saturation)


TRAIN_AUGMENT = AugmentConfig(
    dihedral=True,
    affine_degrees=10.0,
    affine_scale=(0.92, 1.08),
    affine_translate=0.04,
    brightness=0.12,
    contrast=0.12,
    cutout_fraction=0.10,
)

STRONG_AUGMENT = AugmentConfig(
    dihedral=True,
    affine_degrees=10.0,
    affine_scale=(0.90, 1.10),
    affine_translate=0.05,
    brightness=0.05,
    contrast=0.25,
    gamma=0.20,
    saturation=0.20,
    cutout_fraction=0.10,
)

AUGMENT_PRESETS = {"standard": TRAIN_AUGMENT, "strong": STRONG_AUGMENT}

EVAL_AUGMENT = AugmentConfig(dihedral=False)

DARK = 10


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

    return cv2.warpAffine(
        rgb, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def _apply_photometric(
    rgb: np.ndarray, config: AugmentConfig, rng: np.random.Generator
) -> np.ndarray:
    if not config.photometric():
        return rgb
    background = rgb.max(axis=2) <= DARK
    out = rgb.astype(np.float32)
    if config.brightness or config.contrast:
        gain = 1.0 + rng.uniform(-config.contrast, config.contrast)
        bias = 255.0 * rng.uniform(-config.brightness, config.brightness)
        out = out * gain + bias
    if config.saturation:
        grey = out @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        factor = 1.0 + rng.uniform(-config.saturation, config.saturation)
        out = grey[..., None] + factor * (out - grey[..., None])
    np.clip(out, 0.0, 255.0, out=out)
    out = np.rint(out).astype(np.uint8)
    if config.gamma:

        exponent = float(np.exp(rng.uniform(-config.gamma, config.gamma)))
        table = np.rint(255.0 * (np.arange(256) / 255.0) ** exponent).astype(np.uint8)
        out = cv2.LUT(out, table)
    out[background] = rgb[background]
    return out


def augment(rgb: np.ndarray, config: AugmentConfig, rng: np.random.Generator) -> np.ndarray:
    if config.dihedral:
        rgb = dihedral(rgb, int(rng.integers(8)))
    rgb = _apply_affine(rgb, config, rng)
    rgb = _apply_photometric(rgb, config, rng)

    if config.cutout_fraction and rng.random() < 0.5:
        height, width = rgb.shape[:2]
        side = int(min(height, width) * config.cutout_fraction)
        if side > 1:
            top = int(rng.integers(0, height - side))
            left = int(rng.integers(0, width - side))
            rgb = rgb.copy()
            rgb[top : top + side, left : left + side] = 0
    return rgb

@dataclass
class Sample:
    image_id: str
    path: str
    label: int
    group_id: str
    dataset_source: str
    eye_id: str = ""
    patient_id: str = ""
    side: str = ""  

def eye_key(image_id: str, dataset_source: str) -> str:

    if dataset_source in {"mfiddr", "drtid"}:
        head, _, tail = image_id.rpartition("_")
        return head if head and tail.isdigit() else image_id
    if dataset_source == "deepdrid":
        return image_id[:-1] if image_id[-1:].isdigit() else image_id
    return image_id


def relocate(path: str, root: Path) -> str:

    parts = re.split(r"[\\/]+", path.strip())
    if len(parts) < 2:
        raise ValueError(f"cannot re-root {path!r}: no <source>/<file> tail")
    return str(Path(root) / parts[-2] / parts[-1])


def load_split(
    manifest: Path,
    splits: Path,
    split: str,
    image_root: Optional[Path] = None,
    check_files: bool = True,
) -> List[Sample]:
    with Path(splits).open(newline="", encoding="utf-8-sig") as handle:
        wanted = {r["image_id"] for r in csv.DictReader(handle) if r["split"] == split}
    with Path(manifest).open(newline="", encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle) if r["image_id"] in wanted]
    if not wanted:
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
            eye_id=eye_key(r["image_id"], r["dataset_source"]),
            patient_id=r.get("patient_id") or "",
            side=(r.get("eye") or "").strip().upper(),
        )
        for r in rows
    ]

    missing = [s for s in samples[:64] if not Path(s.path).is_file()] if check_files else []
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of the first 64 '{split}' images are missing, "
            f"e.g. {missing[0].path}"
        )
    return samples


class FundusDataset(Dataset):

    def __init__(
        self,
        samples: Sequence[Sample],
        config: AugmentConfig = EVAL_AUGMENT,
        seed: int = 0,
        tta_index: Optional[int] = None,
        return_bag: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.return_bag = return_bag
        order: Dict[str, int] = {}
        for sample in self.samples:
            order.setdefault(sample.eye_id or sample.image_id, len(order))
        self.bag_of = [order[s.eye_id or s.image_id] for s in self.samples]
        self.config = config
        self.seed = seed
        self.tta_index = tta_index
        self.colour = COLOUR_MODES[config.colour_mode]
        self._rng: Optional[np.random.Generator] = None

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            seed = self.seed if info is None else (self.seed + info.seed) % (2**63)
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
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
        image = torch.from_numpy(tensor.transpose(2, 0, 1))
        if self.return_bag:
            return image, sample.label, self.bag_of[index]
        return image, sample.label

def class_counts(samples: Sequence[Sample]) -> np.ndarray:
    counter = Counter(s.label for s in samples)
    return np.array([counter.get(c, 0) for c in range(NUM_CLASSES)], dtype=np.float64)


def balanced_sampler(
    samples: Sequence[Sample],
    strength: float = 0.5,
    generator: Optional[torch.Generator] = None,
    source_balance: float = 0.0,
) -> WeightedRandomSampler:

    weights = torch.as_tensor(image_weights(samples, strength, source_balance), dtype=torch.double)
    return WeightedRandomSampler(
        weights, num_samples=len(samples), replacement=True, generator=generator
    )


def image_weights(
    samples: Sequence[Sample], strength: float = 0.5, source_balance: float = 0.0
) -> np.ndarray:

    counts = class_counts(samples)
    counts[counts == 0] = 1.0
    per_class = (1.0 / counts) ** float(strength)
    per_class = per_class / per_class.sum()
    values = np.array([per_class[s.label] for s in samples], dtype=np.float64)
    if source_balance:
        cells = Counter((s.label, s.dataset_source) for s in samples)
        totals: Dict[int, float] = {}
        for (label, _), n in cells.items():
            totals[label] = totals.get(label, 0.0) + n ** (1.0 - source_balance)

        values *= np.array([
            (cells[(s.label, s.dataset_source)] ** (1.0 - source_balance) / totals[s.label])
            * (counts[s.label] / cells[(s.label, s.dataset_source)])
            for s in samples
        ])
    return values


def eye_bags(samples: Sequence[Sample]) -> List[List[int]]:


    order: Dict[str, int] = {}
    bags: List[List[int]] = []
    for index, sample in enumerate(samples):
        key = sample.eye_id or sample.image_id
        if key not in order:
            order[key] = len(bags)
            bags.append([])
        bags[order[key]].append(index)
    return bags


def eye_bag_weights(
    samples: Sequence[Sample],
    bags: Sequence[Sequence[int]],
    strength: float = 0.5,
    source_balance: float = 0.0,
) -> np.ndarray:

    weights = image_weights(samples, strength, source_balance)
    return np.array([weights[list(bag)].sum() for bag in bags])


class EyeBatchSampler(Sampler):

    def __init__(self, bags, weights, batch_size, images_per_epoch, generator=None):
        self.bags = [list(bag) for bag in bags]
        weights = np.asarray(weights, dtype=np.float64)
        self.probabilities = weights / weights.sum()
        self.batch_size = int(batch_size)
        self.images_per_epoch = int(images_per_epoch)
        self.generator = generator
        self.average_views = float(np.mean([len(bag) for bag in self.bags]))
        self.expected_views = float(np.dot(self.probabilities, [len(bag) for bag in self.bags]))
        drawn, batches = 0, 0
        for batch in self._batches(np.random.default_rng(0)):
            drawn += len(batch)
            batches += 1
            if drawn >= self.images_per_epoch:
                break
        self.length = max(1, batches)

    def _batches(self, rng):

        block = max(256, self.batch_size * 8)
        batch, pending, taken = [], [], set()
        while True:
            if not pending:
                pending = rng.choice(len(self.bags), size=block, p=self.probabilities).tolist()
            index = pending.pop()

            if index in taken:
                if len(taken) < len(self.bags):
                    continue
                yield batch
                batch, taken = [], set()
                continue
            taken.add(index)
            batch.extend(self.bags[index])
            if len(batch) >= self.batch_size:
                yield batch
                batch, taken = [], set()

    def __iter__(self):
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=self.generator).item())
        batches = self._batches(np.random.default_rng(seed))
        for _ in range(self.length):
            yield next(batches)

    def __len__(self):
        return self.length


def fellow_eyes(samples: Sequence[Sample], bags: Sequence[Sequence[int]]) -> np.ndarray:

    where: Dict[Tuple[str, str], List[int]] = {}
    for index, bag in enumerate(bags):
        first = samples[bag[0]]
        patient, side = getattr(first, "patient_id", ""), getattr(first, "side", "")
        if patient and side in ("L", "R"):
            where.setdefault((patient, side), []).append(index)
    fellow = np.full(len(bags), -1, dtype=np.int64)
    for (patient, side), mine in where.items():
        other = where.get((patient, "R" if side == "L" else "L"), [])
        if len(mine) == 1 and len(other) == 1:
            fellow[mine[0]] = other[0]
    return fellow


def class_weights(samples: Sequence[Sample], strength: float = 0.5) -> torch.Tensor:

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

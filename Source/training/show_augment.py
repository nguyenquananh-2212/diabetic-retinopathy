from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "preprocessing"))

from data import AUGMENT_PRESETS, AugmentConfig, augment, dihedral  # noqa: E402

PROJECT_ROOT = _HERE.parents[1]
DARK = 10


def read(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


_FONT_CANDIDATES = (
    r"C:/Windows/Fonts/segoeui.ttf", r"C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int = 18):
    from PIL import ImageFont

    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def label(rgb: np.ndarray, text: str) -> np.ndarray:

    from PIL import Image, ImageDraw

    image = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, image.width, 28], fill=(0, 0, 0))
    draw.text((6, 4), text, font=_font(), fill=(255, 255, 255))
    return np.asarray(image)


def grid(rows: List[List[np.ndarray]], gap: int = 6) -> np.ndarray:
    height = max(t.shape[0] for row in rows for t in row)
    width = max(t.shape[1] for row in rows for t in row)
    columns = max(len(row) for row in rows)
    canvas = np.full(
        (len(rows) * (height + gap) - gap, columns * (width + gap) - gap, 3),
        32, dtype=np.uint8,
    )
    for r, row in enumerate(rows):
        for c, tile in enumerate(row):
            y, x = r * (height + gap), c * (width + gap)
            canvas[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
    return canvas
def components(base: AugmentConfig) -> Dict[str, AugmentConfig]:

    off = dict(dihedral=False, affine_degrees=0.0, affine_scale=(1.0, 1.0),
               affine_translate=0.0, brightness=0.0, contrast=0.0,
               gamma=0.0, saturation=0.0, cutout_fraction=0.0)
    parts = {
        "original": AugmentConfig(**off),
        "dihedral": AugmentConfig(**{**off, "dihedral": True}),
        f"affine +/-{base.affine_degrees:.0f} deg": AugmentConfig(**{
            **off, "affine_degrees": base.affine_degrees,
            "affine_scale": base.affine_scale, "affine_translate": base.affine_translate,
        }),
        f"brightness/contrast +/-{100*base.brightness:.0f}%": AugmentConfig(**{
            **off, "brightness": base.brightness, "contrast": base.contrast,
        }),
        f"cutout {100*base.cutout_fraction:.0f}%": AugmentConfig(**{
            **off, "cutout_fraction": base.cutout_fraction,
        }),
    }
    if base.gamma or base.saturation:
        parts[f"gamma/saturation +/-{base.gamma:.2f}/{100*base.saturation:.0f}%"] = AugmentConfig(**{
            **off, "gamma": base.gamma, "saturation": base.saturation,
        })
    parts["all combined"] = base
    return parts

def retina_loss(rgb: np.ndarray, degrees: float, trials: int = 24) -> float:
    mask = (rgb.max(axis=2) > DARK).astype(np.float32)
    before = mask.sum()
    if before == 0:
        return float("nan")
    height, width = mask.shape
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(trials):
        angle = degrees if trials == 1 else rng.uniform(-degrees, degrees)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        moved = cv2.warpAffine(mask, matrix, (width, height),
                               flags=cv2.INTER_NEAREST, borderValue=0)
        losses.append(1.0 - moved.sum() / before)
    return float(np.mean(losses))


def measure(images: List[np.ndarray], config: AugmentConfig) -> None:
    print(f"\nRETINA LOST TO EACH GEOMETRIC TRANSFORM ({len(images)} images)")
    print(f"  {'transform':28}{'pixels lost':>12}")
    for name, degrees in ((f"rotate {config.affine_degrees:.0f} deg (preset)", config.affine_degrees),
                          ("rotate 45 deg", 45.0)):
        values = [retina_loss(im, degrees) for im in images]
        print(f"  {name:28}{100 * float(np.mean(values)):11.2f}%")
    worst = 0.0
    for im in images:
        mask = (im.max(axis=2) > DARK)
        for k in range(8):
            worst = max(worst, abs(int(dihedral(mask.astype(np.uint8), k).sum())
                                   - int(mask.sum())))
    print(f"  {'dihedral group (8 ops)':28}{int(worst):11d} px  (must be 0)")
    rng = np.random.default_rng(0)
    lit = 0
    for im in images:
        border = im.max(axis=2) <= 10
        for _ in range(8):
            lit += int(augment(im, AugmentConfig(**{**config.__dict__, "dihedral": False,
                                                    "affine_degrees": 0.0, "affine_scale": (1.0, 1.0),
                                                    "affine_translate": 0.0, "cutout_fraction": 0.0}),
                               rng)[border].max(initial=0) > 10)
    print(f"  {'border lit by photometrics':28}{lit:11d} draws  (must be 0)")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show what online augmentation does, and measure what it costs."
    )
    parser.add_argument("images", nargs="*", type=Path,
                        help="specific images; empty means sample from the manifest")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "Data" / "processed" / "img448" / "manifest.csv")
    parser.add_argument("--count", type=int, default=4, help="how many images to sample")
    parser.add_argument("--draws", type=int, default=5, help="random draws per image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo_images")
    parser.add_argument("--preset", default="standard", choices=sorted(AUGMENT_PRESETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    if args.images:
        paths = list(args.images)
    else:
        with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        by_class: Dict[str, List[dict]] = {}
        for row in rows:
            by_class.setdefault(row["class_label"], []).append(row)
        paths = [
            Path(random.choice(by_class[c])["processed_path"])
            for c in sorted(by_class)[: args.count]
        ]
    images = [read(p) for p in paths]
    print(f"{len(images)} images: " + ", ".join(p.name for p in paths))

    config = AUGMENT_PRESETS[args.preset]
    suffix = "" if args.preset == "standard" else f"_{args.preset}"
    print(f"preset: {args.preset}")
    rng = np.random.default_rng(args.seed)
    rows_out = []
    for path, image in zip(paths, images):
        tiles = [label(image, f"original · {path.stem[:22]}")]
        for draw in range(args.draws):
            tiles.append(label(augment(image, config, rng), f"draw #{draw + 1}"))
        rows_out.append(tiles)
    args.out.mkdir(parents=True, exist_ok=True)
    first = args.out / f"augment_grid{suffix}.png"
    cv2.imwrite(str(first), cv2.cvtColor(grid(rows_out), cv2.COLOR_RGB2BGR))
    print(f"random draws  -> {first}")

    parts = components(config)
    rows_out = []
    for name, config in parts.items():
        seeded = np.random.default_rng(args.seed)
        rows_out.append([
            label(augment(image, config, seeded), name) for image in images[:4]
        ])
    second = args.out / f"augment_parts{suffix}.png"
    cv2.imwrite(str(second), cv2.cvtColor(grid(rows_out), cv2.COLOR_RGB2BGR))
    print(f"each transform -> {second}")

    measure(images, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

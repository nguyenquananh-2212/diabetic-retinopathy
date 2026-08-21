from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Create a deterministic, normalized fundus image dataset."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "Data" / "split_dataset",
        help="Original dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "Data" / "processed" / "fundus_224_pad_v1",
        help="Output root for normalized images and metadata.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output image width and height. Default: 224.",
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        default=10,
        help="Pixels with max RGB <= this value are treated as background.",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.02,
        help="Extra margin added around the non-dark bounding box. Default: 0.02.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess output files that already exist.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:

    return path.expanduser().resolve()

def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

def image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path

def crop_dark_border(
    image: Image.Image, dark_threshold: int, margin_fraction: float
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = np.asarray(rgb, dtype=np.uint8)
    foreground = pixels.max(axis=2) > dark_threshold
    ys, xs = np.where(foreground)

    if len(xs) == 0 or len(ys) == 0:
        return rgb, (0, 0, width, height)

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    if max_x < min_x or max_y < min_y:
        return rgb, (0, 0, width, height)

    box_width = max_x - min_x + 1
    box_height = max_y - min_y + 1

    if box_width >= width * 0.98 and box_height >= height * 0.98:
        return rgb, (0, 0, width, height)

    margin_x = max(1, int(box_width * margin_fraction))
    margin_y = max(1, int(box_height * margin_fraction))

    left = max(0, min_x - margin_x)
    top = max(0, min_y - margin_y)
    right = min(width, max_x + margin_x + 1)
    bottom = min(height, max_y + margin_y + 1)

    return rgb.crop((left, top, right, bottom)), (left, top, right, bottom)


def pad_and_resize(image: Image.Image, size: int) -> Image.Image:
    """Fit an image into a square without changing its aspect ratio."""

    return ImageOps.pad(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )

def output_name(source_path: Path, output_dir: Path) -> Path:

    candidate = output_dir / f"{source_path.stem}.png"
    if not candidate.exists():
        return candidate

    return output_dir / f"{source_path.stem}__{source_path.suffix[1:].lower()}.png"

def process_one(
    source_path: Path,
    output_dir: Path,
    size: int,
    dark_threshold: int,
    crop_margin: float,
    overwrite: bool,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_name(source_path, output_dir)

    row: Dict[str, object] = {
        "source_path": str(source_path),
        "processed_path": str(destination),
        "status": "error",
        "original_width": "",
        "original_height": "",
        "crop_left": "",
        "crop_top": "",
        "crop_right": "",
        "crop_bottom": "",
        "error": "",
    }

    if destination.exists() and not overwrite:
        row["status"] = "skipped_exists"
        return row

    try:
        # The context manager closes the source file before the output is saved.
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            row["original_width"], row["original_height"] = image.size

        cropped, crop_box = crop_dark_border(image, dark_threshold, crop_margin)
        normalized = pad_and_resize(cropped, size)
        normalized.save(destination, format="PNG", optimize=True)

        row.update(
            {
                "status": "processed",
                "crop_left": crop_box[0],
                "crop_top": crop_box[1],
                "crop_right": crop_box[2],
                "crop_bottom": crop_box[3],
            }
        )
    except Exception as exc:  # noqa: BLE001 - recorded per file in manifest
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()

    if args.size <= 0:
        raise SystemExit("--size must be a positive integer")
    if not 0 <= args.dark_threshold <= 255:
        raise SystemExit("--dark-threshold must be between 0 and 255")
    if not 0 <= args.crop_margin <= 1:
        raise SystemExit("--crop-margin must be between 0 and 1")

    input_root = resolve_path(args.input_root)
    output_root = resolve_path(args.output_root)

    if not input_root.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_root}")
    if input_root == output_root or is_inside(output_root, input_root):
        raise SystemExit(
            "Output folder must be outside the input folder to avoid recursive processing."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "output_size": [args.size, args.size],
        "output_format": "PNG",
        "color_mode": "RGB",
        "dark_threshold": args.dark_threshold,
        "crop_margin": args.crop_margin,
        "aspect_ratio_preserved": True,
        "random_augmentation": False,
        "normalization": "Apply ImageNet mean/std later in the DataLoader, not to saved pixels.",
    }
    write_json(output_root / "preprocessing_v1.json", config)

    manifest_path = output_root / "manifest.csv"
    fieldnames = [
        "split",
        "class_label",
        "source_path",
        "processed_path",
        "status",
        "original_width",
        "original_height",
        "crop_left",
        "crop_top",
        "crop_right",
        "crop_bottom",
        "error",
    ]

    counts = Counter()
    total = 0
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()

        for split in EXPECTED_SPLITS:
            split_dir = input_root / split
            if not split_dir.is_dir():
                print(f"[WARNING] Missing split folder: {split_dir}")
                continue

            class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
            for class_dir in class_dirs:
                files = list(image_files(class_dir))
                for source_path in files:
                    relative_class_dir = output_root / split / class_dir.name
                    row = process_one(
                        source_path=source_path,
                        output_dir=relative_class_dir,
                        size=args.size,
                        dark_threshold=args.dark_threshold,
                        crop_margin=args.crop_margin,
                        overwrite=args.overwrite,
                    )
                    row.update({"split": split, "class_label": class_dir.name})
                    writer.writerow(row)
                    manifest_file.flush()

                    counts[f"{split}_{row['status']}"] += 1
                    total += 1
                    if total % 500 == 0:
                        print(f"Processed entries: {total}")

    summary = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_entries": total,
        "counts": dict(sorted(counts.items())),
        "manifest": str(manifest_path),
    }
    write_json(output_root / "summary.json", summary)

    print("\nPreprocessing completed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

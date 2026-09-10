"""Render the selected manifest to fixed-size images the trainer can load flat.

One image in, one image out, no filtering: every decision about *which* eyes
survive was already made in ``dedup_and_select.py``.  This stage only changes
geometry, and it does so through the same ``retina`` module the hashes were
taken through, so the picture judged unique is the picture trained on.

Colour is deliberately left alone.  Ben Graham normalisation measures far
better than CLAHE on this data (-91 % between-image variance against -26 %),
but it destroys absolute colour, and hard exudates are told from haemorrhages
by colour.  Baking it in here would make that untestable; it belongs in the
training transform where it can be switched off.

Output is PNG.  At 448 that is ~192 KB an image against ~55 KB for JPEG q95,
and the difference buys the freedom to re-crop or re-scale later without
stacking generation loss on data that took hours to assemble.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retina import pad_to_square, resize_square, retina_box  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# A retina narrower than this was measured once in 25,601 images.  It is not a
# quality filter -- it is a tripwire that says the detector failed.
MIN_RETINA_PX = 200

# Every image_id starts "<source>:", and on NTFS a colon opens an alternate data
# stream instead of naming a file: "deepdrid\deepdrid:100_l1.png.part" silently
# becomes a stream on a file called "deepdrid", and the rename then fails with
# WinError 87.  A first run lost all 15,355 images it attempted this way.
ILLEGAL_IN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_stem(image_id: str) -> str:
    return ILLEGAL_IN_FILENAME.sub("_", image_id).rstrip(" .") or "_"


try:  # progress bars are a convenience, never a dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - environment dependent
    tqdm = None


def progress(iterable, desc, total=None):
    """A bar when tqdm is installed, the bare iterable when it is not."""

    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, mininterval=1.0, dynamic_ncols=True)


RECORD_FIELDS = [
    "image_id",
    "processed_path",
    "source_width",
    "source_height",
    "retina_width",
    "retina_height",
    "scale",
    "degenerate",
    "error",
]


def render_one(task: Tuple[str, str, str, int]) -> Dict[str, object]:
    image_id, source_path, out_path, size = task
    record: Dict[str, object] = {field: "" for field in RECORD_FIELDS}
    record["image_id"] = image_id
    record["processed_path"] = out_path
    try:
        with Image.open(source_path) as opened:
            rgb = np.asarray(
                ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8
            )
        record["source_height"], record["source_width"] = rgb.shape[:2]

        box = retina_box(rgb)
        if box is not None:
            left, top, right, bottom = box
            rgb = rgb[top:bottom, left:right]
        record["retina_height"], record["retina_width"] = rgb.shape[:2]
        record["degenerate"] = int(min(rgb.shape[:2]) < MIN_RETINA_PX)
        record["scale"] = round(max(rgb.shape[:2]) / size, 4)

        out = resize_square(pad_to_square(rgb), size)
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite mangles non-ASCII paths on Windows; encode, then write.
        ok, buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 6],
        )
        if not ok:
            raise RuntimeError("PNG encoding failed")
        partial = target.with_suffix(".png.part")
        partial.write_bytes(buffer.tobytes())
        partial.replace(target)  # a reader never sees a half-written image
    except Exception as exc:  # noqa: BLE001 - recorded per image
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop, pad and resize the selection.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "dedup" / "selected_manifest.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="re-render images that already exist"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    size = int(args.size)
    output_dir = (
        args.output_dir or PROJECT_ROOT / "Data" / "processed" / f"img{size}"
    )
    output_dir = output_dir.expanduser().resolve()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.expanduser().resolve().open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    print(f"selection: {len(rows):,} images -> {size}x{size}")

    for row in rows:
        row["processed_path"] = str(
            image_dir / row["dataset_source"] / f"{safe_stem(row['image_id'])}.png"
        )
    collisions = Counter(r["processed_path"] for r in rows)
    clashing = [path for path, count in collisions.items() if count > 1]
    if clashing:
        raise SystemExit(
            f"{len(clashing)} image_ids collide once made filename-safe, "
            f"e.g. {clashing[0]}"
        )

    # Resume from whatever a previous run got through, so an interrupted build
    # costs the current image rather than the whole hour.
    journal_path = output_dir / "render_log.csv"
    done: Dict[str, Dict[str, str]] = {}
    if journal_path.is_file() and not args.force:
        with journal_path.open(newline="", encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                if not record.get("error") and Path(record["processed_path"]).is_file():
                    done[record["image_id"]] = record

    pending = [
        (r["image_id"], r["source_path"], r["processed_path"], size)
        for r in rows
        if r["image_id"] not in done
    ]
    print(f"rendering {len(pending):,} ({len(rows) - len(pending):,} already on disk)")

    workers = int(args.workers) or max(1, (os.cpu_count() or 2) - 1)
    fresh = args.force or not journal_path.is_file()
    if pending:
        with journal_path.open(
            "w" if fresh else "a", newline="", encoding="utf-8-sig"
        ) as journal:
            writer = csv.DictWriter(
                journal, fieldnames=RECORD_FIELDS, extrasaction="ignore"
            )
            if fresh:
                writer.writeheader()
            pool = None
            if workers <= 1:
                results: Iterable[Dict[str, object]] = (render_one(t) for t in pending)
            else:
                pool = ProcessPoolExecutor(max_workers=workers)
                results = pool.map(render_one, pending, chunksize=16)
            count = 0
            try:
                for record in progress(results, f"rendering @{size}", len(pending)):
                    writer.writerow(record)
                    count += 1
                    if not record["error"]:
                        done[str(record["image_id"])] = {
                            key: str(value) for key, value in record.items()
                        }
                    if count % 1000 == 0:
                        journal.flush()
            finally:
                if pool is not None:
                    pool.shutdown()

    ok_rows = [r for r in rows if r["image_id"] in done]
    failed = len(rows) - len(ok_rows)
    if failed:
        print(f"[WARNING] {failed} images failed; see {journal_path}")

    carried = [f for f in RECORD_FIELDS if f not in rows[0] and f != "error"]
    out_fields = list(rows[0].keys()) + carried
    with (output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in ok_rows:
            merged = dict(row)
            merged.update({key: done[row["image_id"]][key] for key in carried})
            writer.writerow(merged)

    scales = np.array([float(done[r["image_id"]]["scale"] or 0) for r in ok_rows])
    degenerate = [
        r["image_id"] for r in ok_rows if done[r["image_id"]]["degenerate"] == "1"
    ]
    sampled = ok_rows[:: max(1, len(ok_rows) // 2000)]
    mean_bytes = sum(
        Path(done[r["image_id"]]["processed_path"]).stat().st_size for r in sampled
    ) / max(1, len(sampled))

    by_source_class: Dict[str, Counter] = defaultdict(Counter)
    for row in ok_rows:
        by_source_class[row["dataset_source"]][row["class_label"]] += 1
    total: Counter = Counter()
    for counts in by_source_class.values():
        total += counts

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_in": str(args.manifest),
        "size": size,
        "images_in": len(rows),
        "images_out": len(ok_rows),
        "failed": failed,
        "degenerate_retina": len(degenerate),
        "degenerate_ids": degenerate[:50],
        "downscale_factor": {
            "p05": round(float(np.percentile(scales, 5)), 3),
            "p50": round(float(np.percentile(scales, 50)), 3),
            "p95": round(float(np.percentile(scales, 95)), 3),
            "upscaled_images": int((scales < 1.0).sum()),
        },
        "estimated_gb_on_disk": round(mean_bytes * len(ok_rows) / 2**30, 2),
        "class_counts": dict(sorted(total.items())),
        "class_counts_by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source_class.items())
        },
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

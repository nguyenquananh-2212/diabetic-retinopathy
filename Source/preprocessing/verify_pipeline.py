"""Audit the finished pipeline the way a sceptical reviewer would.

Every check here exists because its absence already cost this project once: the
previous test set turned out to be 49.9 % memorisable from its own train set,
and nothing in the code said so.  A green run of this file is the claim that
the new split does not repeat that.

Checks, in order of how badly a failure would invalidate results:

1. **Content leak** -- no byte-identical, pixel-identical, or perceptually
   identical (pHash <= 1 with the dHash guard) pair may straddle a split.
2. **Group leak** -- no ``group_id`` may straddle a split.
3. **Referential integrity** -- every split row has a manifest row, every
   manifest row has a file on disk, ids are unique at every stage.
4. **Geometry** -- rendered images really are ``size x size`` RGB, and the
   retina really does fill them.
5. **Distribution** -- class shares per split, per source, and the pooled
   baseline they should track.

Exit code is non-zero if any check in 1-4 fails, so this can gate a run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dedup_and_select import (  # noqa: E402
    DHASH_GUARD,
    PHASH_THRESHOLD,
    perceptual_pairs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> List[Dict[str, str]]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the preprocessing pipeline.")
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "Data" / "processed")
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    built = root / f"img{args.size}"

    hashes = {r["image_id"]: r for r in load(root / "dedup" / "hashes.csv")}
    manifest = load(built / "manifest.csv")
    splits = {r["image_id"]: r for r in load(built / "splits.csv")}
    print(f"manifest {len(manifest):,} images | splits {len(splits):,} | hashes {len(hashes):,}")

    failures: List[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not ok:
            failures.append(f"{name}: {detail}")

    # ---- 3. referential integrity ---------------------------------------- #
    print("referential integrity")
    ids = [r["image_id"] for r in manifest]
    duplicated = [i for i, c in Counter(ids).items() if c > 1]
    check("manifest ids unique", not duplicated, f"{len(duplicated)} repeated")
    missing_split = [i for i in ids if i not in splits]
    check("every image has a split", not missing_split, f"{len(missing_split)} missing")
    orphan_split = [i for i in splits if i not in set(ids)]
    check("no split row without an image", not orphan_split, f"{len(orphan_split)} orphans")
    missing_hash = [i for i in ids if i not in hashes]
    check("every image has a hash", not missing_hash, f"{len(missing_hash)} missing")

    absent = [r["processed_path"] for r in manifest if not Path(r["processed_path"]).is_file()]
    check("every rendered file exists", not absent, f"{len(absent)} missing")

    bases = Counter((r["dataset_source"], r["variant_base"]) for r in manifest)
    repeated_base = [b for b, c in bases.items() if c > 1]
    check(
        "one image per variant_base",
        not repeated_base,
        f"{len(repeated_base)} bases kept twice",
    )

    # ---- 2. group leak ---------------------------------------------------- #
    print("group isolation")
    split_of = {i: r["split"] for i, r in splits.items()}
    group_splits: Dict[str, set] = defaultdict(set)
    for row in manifest:
        group_splits[row["group_id"]].add(split_of[row["image_id"]])
    straddling = [g for g, s in group_splits.items() if len(s) > 1]
    check("no group straddles a split", not straddling, f"{len(straddling)} groups")

    # ---- 1. content leak -------------------------------------------------- #
    print("content isolation")
    for field in ("byte_hash", "pixel_hash"):
        buckets: Dict[str, set] = defaultdict(set)
        for row in manifest:
            value = hashes[row["image_id"]].get(field) or ""
            if value:
                buckets[value].add(split_of[row["image_id"]])
        crossing = [v for v, s in buckets.items() if len(s) > 1]
        check(f"no {field} crosses a split", not crossing, f"{len(crossing)} values")

    order = [r["image_id"] for r in manifest]
    phash = np.array([int(hashes[i]["phash"], 16) for i in order], dtype=np.uint64)
    dhash = np.array([int(hashes[i]["dhash"], 16) for i in order], dtype=np.uint64)
    pairs = perceptual_pairs(phash, dhash, PHASH_THRESHOLD, DHASH_GUARD)
    crossing_pairs = [
        (order[a], order[b])
        for a, b in pairs
        if split_of[order[a]] != split_of[order[b]]
    ]
    check(
        f"no pHash<={PHASH_THRESHOLD} pair crosses a split",
        not crossing_pairs,
        f"{len(crossing_pairs)} of {len(pairs)} pairs",
    )

    # Residual risk, reported rather than fixed.  pHash <= 2 is deliberately NOT
    # merged: at that radius the pairs were measured to be mostly lookalikes,
    # not copies, and unioning them chains 260 images spanning all five grades
    # into one group.  The label-agreement rate is the diagnostic -- true
    # duplicates of one eye agree on the grade, lookalikes agree at chance.
    label_of = {r["image_id"]: r["class_label"] for r in manifest}
    wide = perceptual_pairs(phash, dhash, 2, 8)
    wide_crossing = [
        (order[a], order[b]) for a, b in wide if split_of[order[a]] != split_of[order[b]]
    ]
    shares = sum(label_of[a] == label_of[b] for a, b in wide_crossing)
    counts = Counter(label_of.values())
    chance = sum((n / len(manifest)) ** 2 for n in counts.values())
    print(
        f"  [INFO] pHash<=2 (not merged): {len(wide):,} pairs, "
        f"{len(wide_crossing):,} cross a split, "
        f"{100 * shares / max(1, len(wide_crossing)):.1f}% share a grade "
        f"(chance {100 * chance:.1f}%)"
    )

    # ---- 4. geometry ------------------------------------------------------ #
    print("rendered geometry")
    random.seed(args.seed)
    sample = random.sample(manifest, min(args.sample, len(manifest)))
    wrong_shape: List[str] = []
    fills: List[float] = []
    for row in sample:
        data = np.fromfile(row["processed_path"], dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None or image.shape != (args.size, args.size, 3):
            wrong_shape.append(row["image_id"])
            continue
        fills.append(float((image.max(axis=2) > 10).mean()))
    check(f"all sampled images are {args.size}x{args.size}x3", not wrong_shape,
          f"{len(wrong_shape)} bad")
    if fills:
        fill = np.array(fills)
        low = int((fill < 0.30).sum())
        check(
            "retina fills the frame",
            low == 0,
            f"{low} of {len(fill)} below 30% (median {np.median(fill):.3f})",
        )
        print(
            f"  [INFO] frame coverage: p05 {np.percentile(fill, 5):.3f} "
            f"p50 {np.median(fill):.3f} p95 {np.percentile(fill, 95):.3f}"
        )

    # ---- 5. distribution -------------------------------------------------- #
    print("distribution")
    per_split: Dict[str, Counter] = defaultdict(Counter)
    per_source: Dict[str, Counter] = defaultdict(Counter)
    for row in manifest:
        per_split[split_of[row["image_id"]]][row["class_label"]] += 1
        per_source[row["dataset_source"]][row["class_label"]] += 1
    pooled = Counter(r["class_label"] for r in manifest)
    classes = sorted(pooled)
    header = " ".join(f"{c:>8}" for c in classes)
    print(f"  {'split':8} {'images':>7}  {header}")
    for name in ("train", "val", "test"):
        counts = per_split[name]
        total = max(1, sum(counts.values()))
        cells = " ".join(f"{counts[c]:5,}{100 * counts[c] / total:5.1f}%"[-8:] for c in classes)
        print(f"  {name:8} {sum(counts.values()):7,}  " + " ".join(
            f"{counts[c]:>4,}/{100 * counts[c] / total:4.1f}%" for c in classes
        ))
    print(f"  {'pooled':8} {len(manifest):7,}  " + " ".join(
        f"{pooled[c]:>4,}/{100 * pooled[c] / len(manifest):4.1f}%" for c in classes
    ))
    print("  by source: " + json.dumps(
        {s: dict(sorted(c.items())) for s, c in sorted(per_source.items())},
        ensure_ascii=False,
    ))

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        for line in failures:
            print(f"  - {line}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

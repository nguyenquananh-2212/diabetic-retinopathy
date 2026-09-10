"""Collapse the manifest to one image per eye, then group what is left.

Two separate jobs, in order:

**Selection** -- ``legacy`` contains offline augmentation variants, and for
16,655 of its 29,737 originals *only* variants survive, so dropping every
augmented file would delete those cases outright (70% of class 1).  Instead one
image is kept per ``variant_base``, preferring an un-augmented file when the
source kept one.

**Grouping** -- whatever must never straddle a train/val/test boundary is
unioned together: the published patient/eye key, plus any images proven
identical by content.

Hashing follows the thresholds measured on this data (see FINDINGS.md):

* byte and pixel hashes are exact, so they merge unconditionally;
* pHash at Hamming distance <= 1 constrains the *split* but never deletes.
  At radius 5 an earlier run collapsed 8,907 images into two groups spanning all
  five classes, because "looks similar" is not transitive -- and even at radius
  1 it merges different patients whenever a source photographs standardised
  fields, which MFIDDR does.  See the measurement in ``main``;
* anything looser is written to a review file and never auto-merged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retina import retina_gray  # noqa: E402  - same crop the trainer sees

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHASH_THRESHOLD = 1  # measured: the only radius with zero false matches here
DHASH_GUARD = 6  # second opinion, independent of overall brightness
MAX_PERCEPTUAL_GROUP = 20  # blast radius if a perceptual pair is still wrong

HASH_FIELDS = [
    "image_id",
    "source_path",
    "file_size",
    "width",
    "height",
    "byte_hash",
    "pixel_hash",
    "phash",
    "dhash",
    "error",
]

_POPCOUNT = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)

try:  # progress bars are a convenience, never a dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - environment dependent
    tqdm = None


def progress(iterable, desc, total=None):
    """A bar when tqdm is installed, the bare iterable when it is not."""

    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, mininterval=1.0, dynamic_ncols=True)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def _bits_to_uint64(bits: np.ndarray) -> int:
    packed = np.packbits(bits.astype(np.uint8), bitorder="little")
    buffer = np.zeros(8, dtype=np.uint8)
    buffer[: packed.size] = packed[:8]
    return int(buffer.view(np.uint64)[0])


def hash_one(task: Tuple[str, str]) -> Dict[str, object]:
    image_id, source_path = task
    row: Dict[str, object] = {field: "" for field in HASH_FIELDS}
    row["image_id"] = image_id
    row["source_path"] = source_path
    path = Path(source_path)
    try:
        row["file_size"] = path.stat().st_size
        digest = hashlib.blake2b(digest_size=16)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        row["byte_hash"] = digest.hexdigest()

        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        row["width"], row["height"] = width, height

        pixel = hashlib.blake2b(digest_size=16)
        pixel.update(f"{width}x{height}|".encode("ascii"))
        pixel.update(np.asarray(image, dtype=np.uint8).tobytes())
        row["pixel_hash"] = pixel.hexdigest()

        gray = retina_gray(np.asarray(image, dtype=np.uint8))
        thumb32 = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        low = cv2.dct(thumb32)[:8, :8].ravel()[1:]  # drop DC
        row["phash"] = f"{_bits_to_uint64(low > np.median(low)):016x}"
        thumb98 = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        row["dhash"] = f"{_bits_to_uint64((thumb98[:, 1:] > thumb98[:, :-1]).ravel()):016x}"
    except Exception as exc:  # noqa: BLE001 - recorded per image
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def compute_hashes(
    rows: Sequence[Dict[str, str]], cache_path: Path, workers: int
) -> List[Dict[str, object]]:
    """Hash every image, reusing a previous run's cache where the path matches."""

    cache: Dict[str, Dict[str, object]] = {}
    if cache_path.is_file():
        with cache_path.open(newline="", encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                if record.get("phash") and not record.get("error"):
                    cache[record["image_id"]] = record
    pending = [
        (r["image_id"], r["source_path"]) for r in rows if r["image_id"] not in cache
    ]
    print(f"hashing {len(pending):,} images ({len(rows) - len(pending):,} reused)")

    computed: Dict[str, Dict[str, object]] = {}
    if pending:
        # Hashing is the only expensive stage, so every result is appended the
        # moment it lands. A crash then costs the current image, not the hour.
        fresh = not cache_path.is_file()
        with cache_path.open("a", newline="", encoding="utf-8-sig") as journal:
            writer = csv.DictWriter(
                journal, fieldnames=HASH_FIELDS, extrasaction="ignore"
            )
            if fresh:
                writer.writeheader()
            done = 0
            if workers <= 1:
                results: Iterable[Dict[str, object]] = (hash_one(t) for t in pending)
            else:
                pool = ProcessPoolExecutor(max_workers=workers)
                results = pool.map(hash_one, pending, chunksize=32)
            try:
                for row in progress(results, "hashing", len(pending)):
                    computed[str(row["image_id"])] = row
                    writer.writerow(row)
                    done += 1
                    if done % 2000 == 0:
                        journal.flush()
            finally:
                if workers > 1:
                    pool.shutdown()

    merged = [dict(computed.get(r["image_id"]) or cache[r["image_id"]]) for r in rows]
    # Rewrite in manifest order so the cache stays readable, not just replayable.
    with cache_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=HASH_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
    return merged


# --------------------------------------------------------------------------- #
# Union-find
# --------------------------------------------------------------------------- #


class UnionFind:
    """Disjoint sets with an optional size cap for non-transitive relations."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size
        self.size = [1] * size
        self.rejected = 0

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int, cap: int = 0) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if cap and self.size[ra] + self.size[rb] > cap:
            # Only meaningful for perceptual matches. Exact and published keys
            # are equivalence classes: capping those would split a real family
            # and let one copy survive in each split, which is the leak this
            # whole script exists to prevent.
            self.rejected += 1
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def _band_edges(bits: int, bands: int) -> List[Tuple[int, int]]:
    edges, start = [], 0
    for index in range(bands):
        width = bits // bands + (1 if index < bits % bands else 0)
        edges.append((start, width))
        start += width
    return edges


def perceptual_pairs(
    phash: np.ndarray, dhash: np.ndarray, threshold: int, guard: int
) -> List[Tuple[int, int]]:
    """Exact-recall candidate search via banded LSH.

    With ``threshold + 1`` bands, the pigeonhole principle guarantees any true
    pair within ``threshold`` bits agrees exactly on at least one band, so no
    real match is missed without ever building the N x N distance matrix.
    """

    values, first_index, inverse = np.unique(phash, return_index=True, return_inverse=True)
    guard_hash = dhash[first_index]
    members: Dict[int, List[int]] = defaultdict(list)
    for image_index, value_index in enumerate(inverse.tolist()):
        members[value_index].append(image_index)

    pairs: List[Tuple[int, int]] = []
    for group in members.values():  # identical pHash: distance 0
        for other in group[1:]:
            pairs.append((group[0], other))

    seen: set = set()
    for start, width in _band_edges(64, max(1, threshold + 1)):
        if width == 0:
            continue
        mask = np.uint64(((1 << width) - 1) << start)
        keys = (values & mask) >> np.uint64(start)
        buckets: Dict[int, List[int]] = defaultdict(list)
        for position, key in enumerate(keys.tolist()):
            buckets[key].append(position)
        for bucket in buckets.values():
            if len(bucket) < 2 or len(bucket) > 2000:
                continue
            index = np.asarray(bucket, dtype=np.int64)
            block = values[index]
            xor = np.bitwise_xor(block[:, None], block[None, :])
            distance = _POPCOUNT[xor.view(np.uint8).reshape(len(index), len(index), 8)].sum(2)
            rows, cols = np.where(np.triu(distance <= threshold, k=1))
            for left, right in zip(index[rows].tolist(), index[cols].tolist()):
                key_pair = (left, right) if left < right else (right, left)
                if key_pair in seen:
                    continue
                seen.add(key_pair)
                delta = np.bitwise_xor(guard_hash[left], guard_hash[right])
                if int(_POPCOUNT[np.array([delta], dtype=np.uint64).view(np.uint8)].sum()) <= guard:
                    pairs.append((members[left][0], members[right][0]))
    return pairs


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep one image per eye and group what must not be split."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "manifest" / "image_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "dedup",
    )
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.expanduser().resolve().open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    print(f"manifest: {len(rows):,} images")

    workers = int(args.workers) or max(1, (os.cpu_count() or 2) - 1)
    hashes = compute_hashes(rows, output_dir / "hashes.csv", workers)
    for row, digest in zip(rows, hashes):
        row.update(digest)

    usable = [r for r in rows if not r.get("error")]
    unreadable = len(rows) - len(usable)
    if unreadable:
        print(f"[WARNING] {unreadable} images could not be hashed; see hashes.csv")

    # Two different questions need two different unions, and conflating them is
    # a real trap: a DeepDRiD patient contributes four distinct images and a
    # DRTiD eye two, so a patient/eye key must keep them in the *same split*
    # without ever implying they are copies of each other.
    #
    #   identity  -- proven to be the same picture; only one may survive
    #   grouping  -- must not straddle a split; all may survive
    identity = UnionFind(len(usable))
    grouping = UnionFind(len(usable))

    stats: Dict[str, int] = {}
    for field in ("byte_hash", "pixel_hash"):
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, row in enumerate(usable):
            value = str(row.get(field) or "")
            if value:
                buckets[value].append(i)
        merged = 0
        for members in buckets.values():
            for other in members[1:]:
                identity.union(members[0], other)
                merged += int(grouping.union(members[0], other))
        stats[f"merges_{field}"] = merged
        print(f"  {field}: {merged:,} merges")

    # Perceptual evidence goes to *grouping only*, never to identity.
    #
    # It used to do both, on the strength of pHash <= 1 showing no false match
    # on the original corpus.  MFIDDR broke that: it photographs four
    # standardised fields per eye, so different patients' same-numbered field
    # share a layout closely enough to collide.  Measured on the merged corpus:
    #
    #   4,128 pairs at pHash <= 1
    #     3,268 already share a group_key   -- merging them changes nothing
    #       860 cross-group, and of those:
    #           149 byte-identical          -- byte_hash already merges these
    #            49 cross-source            -- dHash median 6, p10 3
    #           662 same source, DIFFERENT patient  -- dHash median 4, p10 0
    #
    # The last two distributions overlap completely, so no dHash threshold
    # separates a real cross-source duplicate from a field-protocol collision.
    # And the one group perceptual matching could delete safely -- the
    # byte-identical 149 -- is already handled exactly by byte_hash.
    #
    # So the merge earns nothing in identity and costs 662 wrong deletions.
    # In grouping it stays valuable and cannot do harm: forcing two images that
    # *might* be the same into one split over-constrains at worst, and that is
    # exactly the leak this stage exists to prevent.  Dropping it from identity
    # recovered 816 images and cut label conflicts from 294 classes to 31.
    phash = np.array([int(r["phash"], 16) for r in usable], dtype=np.uint64)
    dhash = np.array([int(r["dhash"], 16) for r in usable], dtype=np.uint64)
    pairs = sorted(perceptual_pairs(phash, dhash, PHASH_THRESHOLD, DHASH_GUARD))
    merged = sum(
        int(grouping.union(a, b, cap=MAX_PERCEPTUAL_GROUP)) for a, b in pairs
    )
    stats["perceptual_pairs"] = len(pairs)
    stats["merges_perceptual"] = merged
    stats["perceptual_merges_refused_by_cap"] = grouping.rejected
    print(f"  perceptual (pHash<={PHASH_THRESHOLD}): {len(pairs):,} pairs, {merged:,} merges")

    # The published patient/eye key affects grouping only.
    buckets = defaultdict(list)
    for i, row in enumerate(usable):
        value = str(row.get("group_key") or "")
        if value:
            buckets[value].append(i)
    merged = 0
    for members in buckets.values():
        for other in members[1:]:
            merged += int(grouping.union(members[0], other))
    stats["merges_group_key"] = merged
    print(f"  group_key (grouping only): {merged:,} merges")

    # 3. One image per variant_base, preferring a file the source did not
    #    augment. Ties break on size then path so the choice is reproducible.
    by_base: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, row in enumerate(usable):
        by_base[(row["dataset_source"], row["variant_base"])].append(i)

    def rank(i: int) -> Tuple[int, int, str]:
        row = usable[i]
        return (
            int(row["is_augmented"]),
            -int(row["file_size"] or 0),
            str(row["source_path"]),
        )

    chosen = {min(members, key=rank) for members in by_base.values()}
    print(f"  one per variant_base: {len(chosen):,} of {len(usable):,} kept")

    # 4. An identity class that survived twice means the same picture reached us
    #    from two sources. Disagreeing labels on one picture cannot both be
    #    right, so the whole class goes; otherwise keep the copy whose grouping
    #    is strongest, since that copy carries the safest split constraint.
    level_rank = {"patient": 0, "eye": 1, "image": 2}
    by_identity: Dict[int, List[int]] = defaultdict(list)
    for i in sorted(chosen):
        by_identity[identity.find(i)].append(i)

    def merge_key(i: int) -> Tuple[int, Tuple[int, int, str]]:
        return (level_rank[usable[i]["group_level"]], rank(i))

    final: List[int] = []
    conflicted = 0
    dropped_conflict = 0
    for members in by_identity.values():
        if len({usable[i]["class_label"] for i in members}) > 1:
            conflicted += 1
            dropped_conflict += len(members)
            continue
        final.append(min(members, key=merge_key))
    final.sort()

    stats["dropped_cross_source_duplicate"] = (
        len(chosen) - dropped_conflict - len(final)
    )
    stats["identity_classes_with_label_conflict"] = conflicted
    stats["dropped_label_conflict"] = dropped_conflict
    print(f"  after cross-source dedup: {len(final):,}")

    out_fields = list(rows[0].keys() - {"error"})
    out_fields = [f for f in rows[0] if f != "error"] + ["group_id"]
    with (output_dir / "selected_manifest.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for i in final:
            record = dict(usable[i])
            record["group_id"] = f"g{grouping.find(i):07d}"
            writer.writerow(record)

    by_source_class: Dict[str, Counter] = defaultdict(Counter)
    for i in final:
        by_source_class[usable[i]["dataset_source"]][usable[i]["class_label"]] += 1
    total = Counter()
    for counts in by_source_class.values():
        total += counts

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_in": str(args.manifest),
        "images_in": len(rows),
        "unreadable": unreadable,
        "images_out": len(final),
        "groups_out": len({grouping.find(i) for i in final}),
        "thresholds": {
            "phash": PHASH_THRESHOLD,
            "dhash_guard": DHASH_GUARD,
            "max_perceptual_group": MAX_PERCEPTUAL_GROUP,
        },
        "stats": stats,
        "class_counts": dict(sorted(total.items())),
        "class_counts_by_source": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source_class.items())
        },
    }
    (output_dir / "dedup_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

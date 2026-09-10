"""Normalise every image source into one manifest.

Each dataset gets a small adapter that knows only its own quirks; everything
downstream (dedup, grouping, splitting, preprocessing) reads the unified schema
and never needs to know where a row came from.  Adding a sixth source means
writing one adapter, not touching the core.

Why this exists
---------------
The previous pipeline inferred patient identity from filenames.  That worked
for one dataset and broke silently on others: measured on the old split,
**1,477 patients had images in more than one split** (2,954 images, 11.5% of
the data), and 79% of multi-image patients carry the same grade in both eyes,
so that leakage was exploitable.  Sources that publish a patient id are now
asked for it directly, and the ones that do not are marked as such instead of
being guessed at.

Output columns
--------------
``dataset_source``  which adapter produced the row
``image_id``        unique across all sources (prefixed with the source name)
``patient_id``      empty when the source does not publish one
``eye``             ``L`` / ``R`` / empty
``class_label``     0-4
``group_key``       what must never straddle a split
``group_level``     ``patient`` | ``eye`` | ``image`` -- how strong that key is
``is_augmented``    1 when the file is an offline augmentation of another
``variant_base``    the original this row is a variant of
``source_path``     absolute path on disk
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FIELDS = [
    "dataset_source",
    "image_id",
    "patient_id",
    "eye",
    "class_label",
    "group_key",
    "group_level",
    "is_augmented",
    "variant_base",
    "source_path",
]

NUM_CLASSES = 5


# --------------------------------------------------------------------------- #
# Adapter: the existing merged folder (EyePACS + APTOS + IDRiD + Messidor + DDR)
# --------------------------------------------------------------------------- #

# The minority classes were oversampled offline and the augmenter wrote what it
# did into the filename:  <base>[-GF]-600[-CODE]
# Anchoring on the literal "-600" keeps unrelated schemes such as 007-0004-000
# out of the match.
AUGMENT = re.compile(
    r"^(?P<base>.+?)(?:-GF)?-600(?:-(?:ALL|BF|FA|FF|FS|HB|HBF|HF|HFF|SF))?$"
)

# Only two of the five sources in this folder publish anything patient-like.
EYEPACS = re.compile(r"^(?P<patient>\d+)_(?P<eye>left|right)$")
MESSIDOR = re.compile(r"^\d{8}_(?P<patient>\d+)_\d+_PP$")

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _split_dataset_identity(stem: str) -> Tuple[str, str, str, str]:
    """Return ``(base, patient_id, eye, group_level)`` for one filename.

    APTOS, DDR and IDRiD publish no patient id at all, so each of their images
    becomes its own group.  Saying so explicitly is the point: a guessed id
    would look like protection while providing none.
    """

    match = AUGMENT.match(stem)
    base = match.group("base") if match else stem

    eyepacs = EYEPACS.match(base)
    if eyepacs:
        eye = "L" if eyepacs.group("eye") == "left" else "R"
        return base, f"eyepacs:{eyepacs.group('patient')}", eye, "patient"

    messidor = MESSIDOR.match(base)
    if messidor:
        return base, f"messidor:{messidor.group('patient')}", "", "patient"

    return base, "", "", "image"


def adapt_split_dataset(root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Read ``train/val/test`` x ``0..4`` folders, ignoring the old split.

    The old train/val/test division leaked badly and is being rebuilt from
    scratch, so the folder a file sits in is used only for its class label.
    """

    rows: List[Dict[str, object]] = []
    label_of_base: Dict[str, set] = defaultdict(set)
    seen: Dict[str, Path] = {}

    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_dir.name
            if label not in {str(i) for i in range(NUM_CLASSES)}:
                raise ValueError(f"Unexpected class folder: {class_dir}")
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                base, patient, eye, level = _split_dataset_identity(path.stem)
                label_of_base[base].add(label)
                # The same stem can appear in two class folders -- IDRiD_012 is
                # in both test/3 and train/2. Those are the label conflicts this
                # adapter quarantines below, so the id has to stay unique long
                # enough for the filter to see both copies.
                relative = path.relative_to(root).as_posix()
                digest = hashlib.blake2b(relative.encode("utf-8"), digest_size=4).hexdigest()
                image_id = f"legacy:{path.stem}_{digest}"
                if image_id in seen:
                    raise ValueError(
                        f"Duplicate image_id {image_id}: {seen[image_id]} and {path}"
                    )
                seen[image_id] = path
                rows.append(
                    {
                        "dataset_source": "legacy",
                        "image_id": image_id,
                        "patient_id": patient,
                        "eye": eye,
                        "class_label": label,
                        "group_key": patient or f"legacy_base:{base}",
                        "group_level": level if patient else "image",
                        "is_augmented": int(base != path.stem),
                        "variant_base": base,
                        "source_path": str(path),
                    }
                )

    # A base name carrying two different labels cannot be trusted either way.
    conflicted = {b for b, labels in label_of_base.items() if len(labels) > 1}
    kept = [r for r in rows if r["variant_base"] not in conflicted]
    stats = {
        "rows_read": len(rows),
        "rows_kept": len(kept),
        "conflicting_bases": len(conflicted),
        "dropped_label_conflict": len(rows) - len(kept),
    }
    return kept, stats


# --------------------------------------------------------------------------- #
# Adapter: DeepDRiD
# --------------------------------------------------------------------------- #

DEEPDRID_IMAGE = re.compile(r"^(?P<patient>\d+)_(?P<eye>[lr])(?P<field>\d+)$")


def adapt_deepdrid(root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Read the regular fundus CSVs; the evaluation split has no labels.

    The label to use is the one for *this image's own eye*.  Taking
    ``patient_DR_Level`` instead would mislabel 22.3% of the images, because a
    patient's grade is the worse of their two eyes.
    """

    csv_paths = [
        root / "regular-fundus-training" / "regular-fundus-training.csv",
        root / "regular-fundus-validation" / "regular-fundus-validation.csv",
    ]
    raw: List[Dict[str, str]] = []
    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            raw.extend(dict(row, _dir=str(path.parent)) for row in csv.DictReader(handle))

    rows: List[Dict[str, object]] = []
    eye_labels: Dict[Tuple[str, str], set] = defaultdict(set)
    missing_label = 0
    missing_file = 0

    for record in raw:
        match = DEEPDRID_IMAGE.match(record["image_id"])
        if not match:
            raise ValueError(f"Unexpected DeepDRiD image_id: {record['image_id']}")
        patient, side = match.group("patient"), match.group("eye")
        column = "left_eye_DR_Level" if side == "l" else "right_eye_DR_Level"
        label = (record.get(column) or "").strip()
        if not label:
            # The row filled in the *other* eye's column. Borrowing that value
            # would be exactly the per-eye/per-patient mistake this adapter
            # exists to avoid, so the image is dropped instead.
            missing_label += 1
            continue

        path = Path(record["_dir"]) / "Images" / patient / f"{record['image_id']}.jpg"
        if not path.is_file():
            missing_file += 1
            continue

        eye = "L" if side == "l" else "R"
        eye_labels[(patient, side)].add(label)
        rows.append(
            {
                "dataset_source": "deepdrid",
                "image_id": f"deepdrid:{record['image_id']}",
                "patient_id": f"deepdrid:{patient}",
                "eye": eye,
                "class_label": label,
                "group_key": f"deepdrid:{patient}",
                "group_level": "patient",
                "is_augmented": 0,
                "variant_base": record["image_id"],
                "source_path": str(path),
            }
        )

    # Two fields of one eye disagreeing means one of them is wrong; drop the eye.
    conflicted = {key for key, labels in eye_labels.items() if len(labels) > 1}
    kept = [
        r
        for r in rows
        if (
            r["patient_id"].split(":", 1)[1],
            "l" if r["eye"] == "L" else "r",
        )
        not in conflicted
    ]
    stats = {
        "csv_rows": len(raw),
        "rows_kept": len(kept),
        "dropped_missing_label": missing_label,
        "dropped_missing_file": missing_file,
        "conflicting_eyes": len(conflicted),
        "dropped_eye_conflict": len(rows) - len(kept),
    }
    return kept, stats


# --------------------------------------------------------------------------- #
# Adapter: DRTiD
# --------------------------------------------------------------------------- #


def adapt_drtid(root: Path) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """One CSV row is one *eye*, carrying its two views.

    DRTiD publishes an eye id and a left/right flag but no patient id, so the
    two eyes of one person cannot be tied together.  Grouping stops at the eye,
    and ``group_level`` records that.
    """

    csv_paths = [
        root / "Ground Truths" / "DR_grade" / "a. DR_grade_Training.csv",
        root / "Ground Truths" / "DR_grade" / "b. DR_grade_Testing.csv",
    ]
    images_dir = root / "Original Images"
    rows: List[Dict[str, object]] = []
    read = 0
    missing_file = 0

    for path in csv_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                eye_id = (record.get("ID") or "").strip()
                if not eye_id:
                    continue
                read += 1
                label = (record.get("Grade") or "").strip()
                side = (record.get("LR") or "").strip().upper()
                for column, view in (("Macula", "macula"), ("Optic disc", "disc")):
                    stem = (record.get(column) or "").strip()
                    image_path = images_dir / f"{stem}.jpg"
                    if not image_path.is_file():
                        missing_file += 1
                        continue
                    rows.append(
                        {
                            "dataset_source": "drtid",
                            "image_id": f"drtid:{stem}",
                            "patient_id": "",
                            "eye": side if side in ("L", "R") else "",
                            "class_label": label,
                            "group_key": f"drtid:eye{eye_id}",
                            "group_level": "eye",
                            "is_augmented": 0,
                            "variant_base": stem,
                            "source_path": str(image_path),
                            "_view": view,
                        }
                    )
    for row in rows:
        row.pop("_view", None)
    return rows, {"csv_eyes": read, "rows_kept": len(rows), "dropped_missing_file": missing_file}


# --------------------------------------------------------------------------- #
# Adapter: MFIDDR
# --------------------------------------------------------------------------- #

MFIDDR_EYE = re.compile(r"^(?P<patient>\d+_\d+)_(?P<eye>left|right)$")

# How many of an eye's four fields to keep, per grade.  Grade 0 is thinned
# because it is the one class where MFIDDR would otherwise dominate for no gain:
# measured on 400 samples, 84 % of MFIDDR's normals already have a close
# neighbour among the normals this project holds (15.8 % outside the covered
# region, against a 5.0 % baseline for held-out existing normals).  Thinning it
# to one field an eye cuts grade 0 by 48 % and drops MFIDDR's share of it from
# 63 % to 30 %, while keeping every one of its 5,248 normal eyes -- for grade 0
# the useful thing is the spread of *normal variation* across eyes, not four
# angles of the same one.
#
# Grades 1-4 keep all four fields: those eyes are the reason this source was
# added, and grade 3 in particular gains 621 eyes the project did not have.
MFIDDR_FIELDS_PER_CLASS = {"0": 1, "1": 4, "2": 4, "3": 4, "4": 4}


def _fields_for(eye_id: str, keep: int) -> List[int]:
    """Which of the four field slots to keep, rotated per eye.

    Always keeping slot 1 would make every thinned grade-0 image the *same*
    retinal region, which is the opposite of the coverage the thinning is meant
    to preserve.  Rotating by a hash of the eye id spreads the kept slots evenly
    across the population and stays reproducible.
    """

    if keep >= 4:
        return [0, 1, 2, 3]
    offset = hashlib.blake2b(eye_id.encode("utf-8"), digest_size=2).digest()[0] % 4
    step = 4 // keep
    return sorted((offset + i * step) % 4 for i in range(keep))


def adapt_mfiddr(
    root: Path, fields_per_class: Optional[Dict[str, int]] = None
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """One CSV row is one *eye* and names its four fields explicitly.

    The eye id is ``<patient>_<eye>``, so unlike DRTiD both eyes of one person
    can be tied together and ``group_level`` reaches ``patient`` -- the
    strongest constraint the splitter can be given.  It matters here: the two
    eyes carry the same grade 83.5 % of the time, so letting them straddle a
    split would leak most of the answer.

    The source's own train/test folders are read but *not* honoured as a split.
    They are clean (no patient appears in both), yet this project cuts its own
    group-stratified split across every source at once; keeping a second,
    inconsistent split would only make the numbers harder to compare.
    """

    fields_per_class = fields_per_class or MFIDDR_FIELDS_PER_CLASS
    rows: List[Dict[str, object]] = []
    read = 0
    missing_file = 0
    missing_label = 0
    thinned = 0

    for split in ("train", "test"):
        table = root / f"{split}_fourpic_label.csv"
        images_dir = root / split
        if not table.is_file():
            raise FileNotFoundError(table)
        with table.open(newline="", encoding="utf-8-sig") as handle:
            for record in csv.DictReader(handle):
                eye_id = (record.get("id") or "").strip()
                label = (record.get("level") or "").strip()
                if not eye_id:
                    continue
                read += 1
                if label not in {"0", "1", "2", "3", "4"}:
                    missing_label += 1
                    continue
                match = MFIDDR_EYE.match(eye_id)
                if match is None:
                    missing_label += 1
                    continue
                patient = match.group("patient")
                side = match.group("eye")[0].upper()
                keep = int(fields_per_class.get(label, 4))
                slots = _fields_for(eye_id, keep)
                thinned += 4 - len(slots)
                for slot in slots:
                    stem = (record.get(f"id{slot + 1}") or "").strip()
                    if not stem:
                        continue
                    image_path = images_dir / f"{stem}.jpg"
                    if not image_path.is_file():
                        missing_file += 1
                        continue
                    rows.append(
                        {
                            "dataset_source": "mfiddr",
                            "image_id": f"mfiddr:{stem}",
                            "patient_id": patient,
                            "eye": side,
                            "class_label": label,
                            "group_key": f"mfiddr:pat{patient}",
                            "group_level": "patient",
                            "is_augmented": 0,
                            "variant_base": stem,
                            "source_path": str(image_path),
                        }
                    )

    return rows, {
        "csv_eyes": read,
        "rows_kept": len(rows),
        "fields_per_class": dict(fields_per_class),
        "dropped_thinned_fields": thinned,
        "dropped_missing_label": missing_label,
        "dropped_missing_file": missing_file,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

ADAPTERS = {
    "legacy": (adapt_split_dataset, Path("Data") / "split_dataset"),
    "deepdrid": (
        adapt_deepdrid,
        Path("Data") / "deepdrid" / "regular_fundus_images",
    ),
    "drtid": (adapt_drtid, Path("Data") / "DRTiD" / "DRTiD"),
    "mfiddr": (adapt_mfiddr, Path("Data") / "MFIDDR" / "MFIDDR"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalise every fundus source into one manifest."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "manifest",
        help="Where image_manifest.csv and its summary are written.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        choices=sorted(ADAPTERS),
        default=sorted(ADAPTERS),
        help="Subset of adapters to run. Defaults to all available.",
    )
    for name, (_, default) in ADAPTERS.items():
        parser.add_argument(
            f"--{name}-root",
            type=Path,
            default=PROJECT_ROOT / default,
            help=f"Root folder for the {name} source.",
        )
    return parser.parse_args()


def summarise(rows: List[Dict[str, object]]) -> Dict[str, object]:
    by_source_class: Dict[str, Counter] = defaultdict(Counter)
    groups_by_source: Dict[str, set] = defaultdict(set)
    level_counts: Counter = Counter()
    for row in rows:
        by_source_class[str(row["dataset_source"])][str(row["class_label"])] += 1
        groups_by_source[str(row["dataset_source"])].add(str(row["group_key"]))
        level_counts[str(row["group_level"])] += 1
    return {
        "images_total": len(rows),
        "images_by_source_class": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(by_source_class.items())
        },
        "groups_by_source": {k: len(v) for k, v in sorted(groups_by_source.items())},
        "images_by_group_level": dict(sorted(level_counts.items())),
        "augmented_images": sum(int(r["is_augmented"]) for r in rows),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    stats: Dict[str, object] = {}
    for name in args.sources:
        adapter, _ = ADAPTERS[name]
        root = getattr(args, f"{name}_root").expanduser().resolve()
        if not root.is_dir():
            print(f"[SKIP] {name}: {root} does not exist")
            continue
        print(f"[{name}] reading {root}")
        source_rows, source_stats = adapter(root)
        stats[name] = source_stats
        rows.extend(source_rows)
        print(f"[{name}] {source_stats}")

    if not rows:
        raise SystemExit("No source produced any row.")

    seen: set = set()
    for row in rows:
        if row["image_id"] in seen:
            raise ValueError(f"image_id collides across sources: {row['image_id']}")
        seen.add(str(row["image_id"]))
        if str(row["class_label"]) not in {str(i) for i in range(NUM_CLASSES)}:
            raise ValueError(f"Bad class_label: {row}")

    rows.sort(key=lambda r: (r["dataset_source"], r["image_id"]))
    manifest_path = output_dir / "image_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": stats,
        **summarise(rows),
        "manifest": str(manifest_path),
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote {len(rows):,} rows to {manifest_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

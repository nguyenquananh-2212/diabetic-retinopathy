"""Assign train/val/test so that no group is ever split across the boundary.

The previous dataset's test set was 49.9 % memorisable from its own train set:
44.6 % of it was duplicated content and 37.0 % came from a patient the model
had already seen.  Every number measured on it was therefore meaningless, which
is why this stage exists as its own file with its own verification.

``group_id`` from the dedup stage already carries both constraints -- the
published patient/eye key *and* every pair proven identical by content -- so
splitting on groups closes both holes at once.  Stratification runs on the
class label so the rare grades stay proportional; ``StratifiedGroupKFold``
balances the two, since a patient group can hold two different grades and no
assignment can satisfy both perfectly.

The split is a fold merge rather than a single call: K folds are cut once, then
contiguous blocks of them become test and val.  Each fold is individually
stratified, so any block of them is too, and the ratio is exact in folds
instead of approximate in samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-stratified train/val/test.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "img448" / "manifest.csv",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=20)
    parser.add_argument("--val-folds", type=int, default=3)
    parser.add_argument("--test-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    output_path = (args.output or manifest_path.parent / "splits.csv").expanduser()

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty manifest: {manifest_path}")
    print(f"manifest: {len(rows):,} images")

    labels = np.array([int(r["class_label"]) for r in rows])
    groups = np.array([r["group_id"] for r in rows])

    if args.val_folds + args.test_folds >= args.folds:
        raise SystemExit("val + test folds must leave something for train")

    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    fold_of = np.full(len(rows), -1, dtype=np.int64)
    for index, (_, held_out) in enumerate(splitter.split(labels, labels, groups)):
        fold_of[held_out] = index
    if (fold_of < 0).any():
        raise SystemExit("StratifiedGroupKFold left samples unassigned")

    test_folds = set(range(args.test_folds))
    val_folds = set(range(args.test_folds, args.test_folds + args.val_folds))
    split_of = np.where(
        np.isin(fold_of, list(test_folds)),
        "test",
        np.where(np.isin(fold_of, list(val_folds)), "val", "train"),
    )

    # --- verification, not decoration ------------------------------------- #
    group_splits: Dict[str, set] = defaultdict(set)
    for group, split in zip(groups, split_of):
        group_splits[group].add(split)
    straddling = [g for g, s in group_splits.items() if len(s) > 1]
    if straddling:
        raise SystemExit(f"{len(straddling)} groups straddle a split boundary")

    seen_ids = Counter(r["image_id"] for r in rows)
    duplicated = [i for i, c in seen_ids.items() if c > 1]
    if duplicated:
        raise SystemExit(f"{len(duplicated)} image_ids appear more than once")

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "split", "fold", "group_id", "class_label"])
        for row, split, fold in zip(rows, split_of, fold_of):
            writer.writerow(
                [row["image_id"], split, int(fold), row["group_id"], row["class_label"]]
            )

    per_split: Dict[str, Counter] = defaultdict(Counter)
    per_split_source: Dict[str, Counter] = defaultdict(Counter)
    groups_per_split: Dict[str, set] = defaultdict(set)
    for row, split in zip(rows, split_of):
        per_split[split][row["class_label"]] += 1
        per_split_source[split][row["dataset_source"]] += 1
        groups_per_split[split].add(row["group_id"])

    overall = Counter(r["class_label"] for r in rows)
    order = ["train", "val", "test"]
    print(f"{'split':6} {'images':>7} {'groups':>7} " + " ".join(f"{c:>7}" for c in sorted(overall)))
    for split in order:
        counts = per_split[split]
        shares = " ".join(
            f"{100 * counts[c] / max(1, sum(counts.values())):6.2f}%" for c in sorted(overall)
        )
        print(f"{split:6} {sum(counts.values()):7,} {len(groups_per_split[split]):7,} {shares}")
    baseline = " ".join(
        f"{100 * overall[c] / len(rows):6.2f}%" for c in sorted(overall)
    )
    print(f"{'all':6} {len(rows):7,} {len(group_splits):7,} {baseline}")

    # Largest deviation of any class share from the pooled share, in points.
    drift = max(
        abs(
            100 * per_split[s][c] / max(1, sum(per_split[s].values()))
            - 100 * overall[c] / len(rows)
        )
        for s in order
        for c in overall
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_in": str(manifest_path),
        "seed": args.seed,
        "folds": args.folds,
        "val_folds": args.val_folds,
        "test_folds": args.test_folds,
        "images": len(rows),
        "groups": len(group_splits),
        "groups_straddling_splits": 0,
        "max_class_share_drift_points": round(float(drift), 3),
        "counts": {
            s: {
                "images": sum(per_split[s].values()),
                "groups": len(groups_per_split[s]),
                "by_class": dict(sorted(per_split[s].items())),
                "by_source": dict(sorted(per_split_source[s].items())),
            }
            for s in order
        },
    }
    summary_path = output_path.with_name("split_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"max class-share drift: {drift:.3f} points")
    print(f"wrote {output_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

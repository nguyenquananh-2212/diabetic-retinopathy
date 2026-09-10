"""Combine the saved logits of several runs into one prediction.

Averaging happens in **probability** space, not logit space.  Two heads can be
confident to different degrees -- and CORAL's 4 cumulative logits are not even
the same object as a softmax head's 5 scores -- so averaging raw logits would
weight whichever model happens to output larger numbers.  Probabilities are
comparable by construction, and each model's own calibration is applied first so
"0.7" means the same thing coming from either.

Everything is fitted on validation and frozen before test is touched: the blend
weights, each model's temperature, and the ordinal thresholds.  Fitting any of
them on test would put the leak this project spent its effort removing straight
back into the numbers.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from models import coral_to_probabilities  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def decoder_for(head: str):
    if head == "coral":
        def decode(logits: np.ndarray) -> np.ndarray:
            return coral_to_probabilities(
                torch.as_tensor(logits, dtype=torch.float32)
            ).numpy()

        def nll(logits: np.ndarray, labels: np.ndarray) -> float:
            from models import coral_loss

            return float(
                coral_loss(
                    torch.as_tensor(logits, dtype=torch.float32),
                    torch.as_tensor(labels, dtype=torch.long),
                )
            )

        return decode, nll
    return M.softmax_probabilities, M.softmax_nll


def load_run(directory: Path, name: str, tta: bool) -> Dict:
    """One run's val/test probabilities, already temperature-calibrated."""

    backbone, head = name.rsplit("_", 1)
    suffix = "_tta" if tta else ""
    val = np.load(directory / f"logits_{backbone}_{head}_val{suffix}.npz")
    test = np.load(directory / f"logits_{backbone}_{head}_test{suffix}.npz")
    decode, nll = decoder_for(head)

    # Temperature only.  Thresholds are tuned once, on the blend, not per model.
    calibration = M.Calibration.fit(
        val["logits"], val["labels"], ordinal=False, decoder=decode, nll=nll
    )
    return {
        "name": name,
        "temperature": calibration.temperature,
        "val": calibration.probabilities(val["logits"]),
        "test": calibration.probabilities(test["logits"]),
        "val_labels": val["labels"],
        "test_labels": test["labels"],
    }


def search_weights(
    runs: Sequence[Dict], step: float = 0.1
) -> Tuple[np.ndarray, float]:
    """Grid over the simplex.

    With two or three models a grid is exhaustive enough to be the right answer
    rather than an approximation of it, and it cannot land in a local optimum
    the way a gradient step on a non-smooth metric can.
    """

    labels = runs[0]["val_labels"]
    ticks = int(round(1.0 / step))
    best_weights = np.full(len(runs), 1.0 / len(runs))
    best_score = -np.inf

    for combination in itertools.product(range(ticks + 1), repeat=len(runs)):
        if sum(combination) != ticks:
            continue
        weights = np.array(combination, dtype=np.float64) / ticks
        blended = sum(w * r["val"] for w, r in zip(weights, runs))
        score = M.composite_score(labels, blended.argmax(axis=1))
        if score > best_score:
            best_score, best_weights = score, weights
    return best_weights, float(best_score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend saved run logits.")
    parser.add_argument(
        "--checkpoints", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["convnext_tiny_softmax", "swin_t_softmax"],
        help="<backbone>_<head> names, matching the saved logit files",
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runs = [load_run(args.checkpoints, name, args.tta) for name in args.runs]

    labels = runs[0]["val_labels"]
    test_labels = runs[0]["test_labels"]
    for run in runs[1:]:
        if not np.array_equal(run["val_labels"], labels) or not np.array_equal(
            run["test_labels"], test_labels
        ):
            raise SystemExit(
                f"{run['name']} was evaluated on a different split ordering; "
                "the runs must share one splits.csv"
            )

    print(f"{'run':28} {'T':>6}  val (argmax)")
    for run in runs:
        print(
            f"{run['name']:28} {run['temperature']:6.3f}  "
            f"{M.format_metrics(M.summarise(labels, run['val'].argmax(axis=1)))}"
        )

    weights, val_score = search_weights(runs, args.step)
    print("\nweights " + ", ".join(
        f"{r['name']} {w:.2f}" for w, r in zip(weights, runs)
    ) + f"  -> val composite {val_score:.4f}")

    val_blend = sum(w * r["val"] for w, r in zip(weights, runs))
    test_blend = sum(w * r["test"] for w, r in zip(weights, runs))

    # Thresholds fitted on the *blend*, on validation, then frozen.
    thresholds, _ = M.tune_thresholds(M.expected_grade(val_blend), labels)

    results = {
        "val_argmax": M.summarise(labels, val_blend.argmax(axis=1)),
        "test_argmax": M.summarise(test_labels, test_blend.argmax(axis=1)),
        "test_ordinal": M.summarise(
            test_labels, M.apply_thresholds(M.expected_grade(test_blend), thresholds)
        ),
    }
    for run in runs:
        results[f"test_{run['name']}"] = M.summarise(
            test_labels, run["test"].argmax(axis=1)
        )

    print()
    for key in sorted(results):
        print(f"{key:32} {M.format_metrics(results[key])}")

    summary = {
        "runs": [r["name"] for r in runs],
        "tta": args.tta,
        "temperatures": {r["name"]: r["temperature"] for r in runs},
        "weights": dict(zip([r["name"] for r in runs], weights.tolist())),
        "val_composite": val_score,
        "thresholds": thresholds.tolist(),
        "results": results,
    }
    name = "ensemble_tta.json" if args.tta else "ensemble.json"
    (args.checkpoints / name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print(f"\nwrote {args.checkpoints / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

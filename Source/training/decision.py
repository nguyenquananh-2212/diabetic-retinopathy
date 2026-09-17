from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOUNDARIES = (M.REFERABLE_FROM, M.URGENT_FROM)


def decode(probabilities: np.ndarray, cuts: Sequence[float]) -> np.ndarray:
    """Grade = the highest k whose ``P(grade >= k)`` clears ``cuts[k - 1]``.

    The cuts need not increase: a low urgent cut on a case with a middling
    referable probability yields grade 3, which is the safe reading -- the
    urgent flag has priority over the boundary below it.
    """

    cumulative = M.cumulative_probabilities(np.atleast_2d(np.asarray(probabilities, dtype=np.float64)))
    grade = np.zeros(len(cumulative), dtype=np.int64)
    for k in range(1, M.NUM_CLASSES):
        grade[cumulative[:, k - 1] >= cuts[k - 1]] = k
    return grade


def sensitivity_cut(scores: np.ndarray, positive: np.ndarray, target: float) -> float:
    return float(np.quantile(np.asarray(scores)[np.asarray(positive, bool)], 1.0 - target, method="lower"))


def ppv_cut(scores: np.ndarray, positive: np.ndarray, target: float, min_called: int = 20) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.asarray(positive, bool)
    order = np.argsort(-scores, kind="stable")
    ranked, hits = scores[order], positive[order]
    called = np.arange(1, len(ranked) + 1)
    ppv = np.cumsum(hits) / called
    last_of_tie = np.r_[ranked[1:] != ranked[:-1], True]
    cut = float(ranked[0]) + 1e-12
    for index in np.flatnonzero(last_of_tie):
        if called[index] < min_called:
            cut = float(ranked[index])
            continue
        if ppv[index] < target:
            break
        cut = float(ranked[index])
    return cut


def fit_rule(
    probabilities: np.ndarray,
    labels: np.ndarray,
    referable_sensitivity: float = 0.90,
    urgent_sensitivity: float = 0.90,
    review_sensitivity: float = 0.98,
    review_ppv: float = 0.90,
    grid_step: float = 0.02,
) -> Dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    cumulative = M.cumulative_probabilities(probabilities)
    floors = {M.REFERABLE_FROM: referable_sensitivity, M.URGENT_FROM: urgent_sensitivity}
    fixed = {k: sensitivity_cut(cumulative[:, k - 1], labels >= k, floors[k]) for k in BOUNDARIES}

    grid = np.round(np.arange(grid_step, 1.0, grid_step), 6)
    best, cuts = -np.inf, None
    for low in grid:
        for high in grid:
            candidate = [float(low), fixed[2], fixed[3], float(high)]
            score = M.composite_score(labels, decode(probabilities, candidate))
            if score > best:
                best, cuts = score, candidate

    review = {}
    for k in BOUNDARIES:
        scores, positive = cumulative[:, k - 1], labels >= k
        low = min(sensitivity_cut(scores, positive, review_sensitivity), cuts[k - 1])
        high = max(ppv_cut(scores, positive, review_ppv), cuts[k - 1])
        review[str(k)] = [float(low), float(high)]
    return {
        "cuts": [float(c) for c in cuts],
        "review": review,
        "targets": {"referable_sensitivity": referable_sensitivity, "urgent_sensitivity": urgent_sensitivity,
                    "review_sensitivity": review_sensitivity, "review_ppv": review_ppv},
        "val_composite": float(best),
    }


def needs_review(probabilities: np.ndarray, rule: Dict) -> np.ndarray:
    cumulative = M.cumulative_probabilities(np.atleast_2d(np.asarray(probabilities, dtype=np.float64)))
    flag = np.zeros(len(cumulative), dtype=bool)
    for k in BOUNDARIES:
        low, high = rule["review"][str(k)]
        flag |= (cumulative[:, k - 1] >= low) & (cumulative[:, k - 1] < high)
    return flag


def _endpoint(labels: np.ndarray, grades: np.ndarray, k: int, mask: Optional[np.ndarray] = None) -> Dict:
    mask = np.ones(len(labels), bool) if mask is None else mask
    actual, called = labels >= k, grades >= k
    return {"missed": int((actual & ~called & mask).sum()), "positives": int((actual & mask).sum()),
            "false_alarms": int((~actual & called & mask).sum()), "negatives": int((~actual & mask).sum())}


def evaluate(probabilities: np.ndarray, labels: np.ndarray, rule: Optional[Dict]) -> Dict:
    labels = np.asarray(labels)
    grades = probabilities.argmax(1) if rule is None else decode(probabilities, rule["cuts"])
    out = {"eyes": int(len(labels)), "grades": M.summarise(labels, grades),
           "referable": _endpoint(labels, grades, M.REFERABLE_FROM),
           "urgent": _endpoint(labels, grades, M.URGENT_FROM)}
    if rule is not None:
        review = needs_review(probabilities, rule)
        auto = ~review
        out["review"] = {
            "flagged": int(review.sum()), "rate": float(review.mean()),
            "auto_accuracy": float(np.mean(grades[auto] == labels[auto])) if auto.any() else float("nan"),
            "review_accuracy": float(np.mean(grades[review] == labels[review])) if review.any() else float("nan"),
            "auto_referable": _endpoint(labels, grades, M.REFERABLE_FROM, auto),
            "auto_urgent": _endpoint(labels, grades, M.URGENT_FROM, auto),
        }
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def blend_eyes(checkpoints: Path, blend: Dict, samples: Dict, bags: Dict) -> Dict[str, np.ndarray]:
    out = {}
    for split in ("val", "test"):
        total = 0.0
        for run, weight in blend["weights"].items():
            if weight <= 0:
                continue
            saved = np.load(checkpoints / f"logits_{run}_{split}_tta.npz")
            if not np.array_equal(saved["labels"], [s.label for s in samples[split]]):
                raise SystemExit(f"{run}/{split} is not in splits.csv order")
            pooled = M.pool_views(M.softmax_probabilities(saved["logits"]), bags[split], blend["pooling"][run])
            total = total + weight * pooled
        out[split] = total
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit the demo's thresholds and write its bundle.")
    parser.add_argument("--checkpoints", type=Path, default=PROJECT_ROOT / "results" / "full_data_v5" / "checkpoints",
                        help="folder holding ensemble_eye_tta.json and every run's TTA logits")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "Data" / "processed" / "img448")
    parser.add_argument("--model", action="append", default=[], metavar="RUN=NAME=CHECKPOINT",
                        help="run name in the blend, the name the demo shows, and its checkpoint file")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "checkpoints" / "demo_v345" / "decision.json")
    parser.add_argument("--referable-sensitivity", type=float, default=0.90)
    parser.add_argument("--urgent-sensitivity", type=float, default=0.90)
    parser.add_argument("--review-sensitivity", type=float, default=0.98)
    parser.add_argument("--review-ppv", type=float, default=0.90)
    return parser.parse_args()


def main() -> int:
    from data import eye_bags, load_split

    args = parse_args()
    blend = json.loads((args.checkpoints / "ensemble_eye_tta.json").read_text(encoding="utf-8"))
    samples = {s: load_split(args.data / "manifest.csv", args.data / "splits.csv", s, check_files=False)
               for s in ("val", "test")}
    bags = {s: eye_bags(samples[s]) for s in samples}
    labels = {s: np.array([samples[s][b[0]].label for b in bags[s]]) for s in samples}
    single = np.array([len(b) == 1 for b in bags["test"]])

    models: List[Dict] = []
    given = {}
    for spec in args.model:
        run, name, checkpoint = spec.split("=", 2)
        given[run] = (name, Path(checkpoint))
    for run, weight in blend["weights"].items():
        if weight <= 0:
            continue
        if run not in given:
            raise SystemExit(f"--model missing for {run} (weight {weight})")
        name, checkpoint = given[run]
        path = checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint
        models.append({"name": name, "run": run, "checkpoint": checkpoint.as_posix(),
                       "sha256": sha256(path), "weight": float(weight), "pooling": blend["pooling"][run]})

    probabilities = blend_eyes(args.checkpoints, blend, samples, bags)
    check = M.composite_score(labels["val"], probabilities["val"].argmax(1))
    if abs(check - blend["val_composite"]) > 1e-9:
        raise SystemExit(f"blend rebuilt to val composite {check:.6f}, the JSON says {blend['val_composite']:.6f}")

    rule = fit_rule(probabilities["val"], labels["val"], args.referable_sensitivity, args.urgent_sensitivity,
                    args.review_sensitivity, args.review_ppv)
    results = {
        "val": evaluate(probabilities["val"], labels["val"], rule),
        "test": evaluate(probabilities["test"], labels["test"], rule),
        "test_single_view": evaluate(probabilities["test"][single], labels["test"][single], rule),
        "test_argmax": evaluate(probabilities["test"], labels["test"], None),
    }
    bundle = {"kind": "eye_decision", "version": 1, "tta": True, "temperature": 1.0,
              "source": (args.checkpoints.resolve().relative_to(PROJECT_ROOT).as_posix()
                         if args.checkpoints.resolve().is_relative_to(PROJECT_ROOT) else str(args.checkpoints)),
              "models": models, "rule": rule, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    print("models: " + ", ".join(f"{m['name']} w={m['weight']:.2f} {m['pooling']}" for m in models))
    print(f"cuts P(>=1..4) {[round(c, 3) for c in rule['cuts']]} | review {rule['review']}")
    for key in ("test_argmax", "test", "test_single_view"):
        r = results[key]
        line = (f"{key:17} {M.format_metrics(r['grades'])} | refer missed {r['referable']['missed']}/"
                f"{r['referable']['positives']} fa {r['referable']['false_alarms']} | urgent missed "
                f"{r['urgent']['missed']}/{r['urgent']['positives']} fa {r['urgent']['false_alarms']}")
        if "review" in r:
            v = r["review"]
            line += (f" | review {v['rate']:.1%}, auto acc {v['auto_accuracy']:.3f}, auto refer missed "
                     f"{v['auto_referable']['missed']}, auto urgent missed {v['auto_urgent']['missed']}")
        print(line)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def default_step(runs: int) -> float:
    return 0.05 if runs <= 4 else 0.1


STACK_C = (0.01, 0.1, 1.0)
STACK_CLASS_WEIGHT = (None, "balanced")


def stack_features(probabilities: np.ndarray, fellow: Optional[np.ndarray]) -> np.ndarray:
    own = np.log(np.clip(probabilities, 1e-6, 1.0))
    if fellow is None:
        return own
    has = fellow >= 0
    other = np.where(has[:, None], own[np.clip(fellow, 0, None)], 0.0)
    return np.hstack([own, other, has[:, None].astype(np.float64)])


def fit_fellow_stack(
    val_blend: np.ndarray,
    test_blend: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    val_fellow: np.ndarray,
    test_fellow: np.ndarray,
    val_patient_groups: Sequence[str],
    repeats: int = 3,
    folds: int = 5,
    seed: int = 0,
) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from sklearn.linear_model import LogisticRegression

    groups = np.asarray(val_patient_groups)
    unique = np.unique(groups)
    fold_sets = []
    for repeat in range(repeats):
        order = np.random.default_rng(seed + repeat).permutation(len(unique))
        fold_of = dict(zip(unique.tolist(), (order % folds).tolist()))
        fold_sets.append(np.array([fold_of[g] for g in groups.tolist()]))

    def model(C, class_weight):
        return LogisticRegression(C=C, class_weight=class_weight, max_iter=5000)

    summary, predictions = {}, {}
    for name, use in (("own_stack", False), ("fellow_stack", True)):
        xv = stack_features(val_blend, val_fellow if use else None)
        xt = stack_features(test_blend, test_fellow if use else None)
        scores = {}
        for C in STACK_C:
            for class_weight in STACK_CLASS_WEIGHT:
                cv = []
                for fold_of in fold_sets:
                    oof = np.zeros(len(val_labels), dtype=np.int64)
                    for fold in range(folds):
                        held = fold_of == fold
                        oof[held] = model(C, class_weight).fit(xv[~held], val_labels[~held]).predict(xv[held])
                    cv.append(M.composite_score(val_labels, oof))
                scores[(C, class_weight)] = float(np.mean(cv))
        C, class_weight = max(scores, key=scores.get)
        fitted = model(C, class_weight).fit(xv, val_labels)
        predictions[name] = fitted.predict(xt)
        summary[name] = {
            "C": C, "class_weight": class_weight, "val_cv_composite": scores[(C, class_weight)],
            "val_cv_grid": {f"C={c},class_weight={w}": s for (c, w), s in scores.items()},
            "test": M.summarise(test_labels, predictions[name]),
        }
    paired = test_fellow >= 0
    summary["coverage"] = {"val": float(np.mean(val_fellow >= 0)), "test": float(np.mean(paired))}
    if paired.any():
        summary["test_paired_eyes"] = {
            "eyes": int(paired.sum()),
            **{name: M.summarise(test_labels[paired], predictions[name][paired])
               for name in ("own_stack", "fellow_stack")},
        }
    summary["repeats"], summary["folds"] = repeats, folds
    return summary, predictions


def fit_eye_ensemble(
    val_probabilities: Dict[str, np.ndarray],
    test_probabilities: Dict[str, np.ndarray],
    val_groups: Sequence[Sequence[int]],
    test_groups: Sequence[Sequence[int]],
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    step: float = None,
    urgent_target: float = 0.92,
    val_fellow: Optional[np.ndarray] = None,
    test_fellow: Optional[np.ndarray] = None,
    val_patient_groups: Optional[Sequence[str]] = None,
) -> Dict:
    names = list(val_probabilities)

    def eye_labels(labels, groups):
        labels = np.asarray(labels)
        for group in groups:
            if len(set(labels[list(group)].tolist())) != 1:
                raise ValueError("an eye carries more than one grade")
        return np.array([labels[group[0]] for group in groups])

    yv, yt = eye_labels(val_labels, val_groups), eye_labels(test_labels, test_groups)

    rules, rule_scores = {}, {}
    for name in names:
        scores = {rule: M.composite_score(yv, M.pool_views(val_probabilities[name], val_groups, rule).argmax(1))
                  for rule in ("mean", "max")}
        rules[name] = max(scores, key=scores.get)
        rule_scores[name] = scores
    pooled_val = [M.pool_views(val_probabilities[n], val_groups, rules[n]) for n in names]
    pooled_test = [M.pool_views(test_probabilities[n], test_groups, rules[n]) for n in names]
    if step is None:
        step = default_step(len(names))
    ticks = int(round(1.0 / step))
    best, weights = -np.inf, None
    for combination in itertools.product(range(ticks + 1), repeat=len(names)):
        if sum(combination) != ticks:
            continue
        candidate = np.array(combination, dtype=np.float64) / ticks
        score = M.composite_score(yv, sum(w * p for w, p in zip(candidate, pooled_val)).argmax(1))
        if score > best:
            best, weights = score, candidate

    val_blend = sum(w * p for w, p in zip(weights, pooled_val))
    test_blend = sum(w * p for w, p in zip(weights, pooled_test))
    thresholds, _ = M.tune_thresholds(M.expected_grade(val_blend), yv)

    urgent_val = val_blend[:, M.URGENT_FROM:].sum(axis=1)
    urgent_test = test_blend[:, M.URGENT_FROM:].sum(axis=1)
    positive = yv >= M.URGENT_FROM
    cut = float(np.quantile(urgent_val[positive], 1 - urgent_target, method="lower")) if positive.any() else 1.0
    called, actual = urgent_test >= cut, yt >= M.URGENT_FROM

    results = {
        "val_argmax": M.summarise(yv, val_blend.argmax(axis=1)),
        "test_argmax": M.summarise(yt, test_blend.argmax(axis=1)),
        "test_ordinal": M.summarise(yt, M.apply_thresholds(M.expected_grade(test_blend), thresholds)),
        "test_urgent_flag": {
            "threshold": cut,
            "sensitivity": float((called & actual).sum() / max(actual.sum(), 1)),
            "specificity": float((~called & ~actual).sum() / max((~actual).sum(), 1)),
            "ppv": float((called & actual).sum() / max(called.sum(), 1)),
            "missed": int((~called & actual).sum()),
            "positives": int(actual.sum()),
            "false_alarms": int((called & ~actual).sum()),
        },
    }
    for name, pooled in zip(names, pooled_test):
        results[f"test_{name}"] = M.summarise(yt, pooled.argmax(axis=1))

    summary = {
        "level": "eye", "runs": names, "pooling": rules, "pooling_val_composite": rule_scores,
        "weights": dict(zip(names, weights.tolist())), "step": step, "val_composite": float(best),
        "thresholds": thresholds.tolist(), "urgent_threshold": cut,
        "urgent_target_sensitivity": urgent_target,
        "eyes": {"val": int(len(yv)), "test": int(len(yt))}, "results": results,
        "primary": "argmax",
    }
    if val_fellow is not None and test_fellow is not None and val_patient_groups is not None:
        fellow, _ = fit_fellow_stack(val_blend, test_blend, yv, yt, np.asarray(val_fellow),
                                     np.asarray(test_fellow), val_patient_groups)
        summary["fellow"] = fellow
        candidates = {"argmax": float(best)}
        for name in ("own_stack", "fellow_stack"):
            results[f"test_{name}"] = fellow[name]["test"]
            candidates[name] = fellow[name]["val_cv_composite"]
        summary["primary"] = max(candidates, key=candidates.get)
        summary["primary_val_composite"] = candidates
    results["test_primary"] = results[f"test_{summary['primary']}"]
    return summary


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
    parser.add_argument("--step", type=float, default=None,
                        help="weight grid; default 0.05 up to 4 runs, else 0.1 (see default_step)")
    parser.add_argument("--level", choices=("image", "eye"), default="image")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "Data" / "processed" / "img448",
                        help="manifest.csv + splits.csv, for the eye grouping (--level eye)")
    parser.add_argument("--urgent-target", type=float, default=0.92)
    return parser.parse_args()


def eye_context(samples: Dict[str, Sequence], bags: Dict[str, Sequence[Sequence[int]]]):
    from data import fellow_eyes

    return (fellow_eyes(samples["val"], bags["val"]), fellow_eyes(samples["test"], bags["test"]),
            [samples["val"][bag[0]].group_id for bag in bags["val"]])


def print_fellow(summary: Dict) -> None:
    fellow = summary.get("fellow")
    if not fellow:
        return
    print(f"\nother eye known for {fellow['coverage']['test']:.1%} of test eyes")
    for name in ("own_stack", "fellow_stack"):
        part = fellow[name]
        print(f"{name:14} val-cv {part['val_cv_composite']:.4f} (C={part['C']}, "
              f"class_weight={part['class_weight']}) | test {M.format_metrics(part['test'])}")
    print(f"primary        {summary['primary']} -> test {M.format_metrics(summary['results']['test_primary'])}")


def eye_main(args: argparse.Namespace) -> int:
    from data import eye_bags, load_split

    suffix = "_tta" if args.tta else ""
    samples = {split: load_split(args.data / "manifest.csv", args.data / "splits.csv", split,
                                 check_files=False) for split in ("val", "test")}
    probabilities = {"val": {}, "test": {}}
    for name in args.runs:
        backbone, head = name.rsplit("_", 1)
        decode, _ = decoder_for(head)
        for split in ("val", "test"):
            saved = np.load(args.checkpoints / f"logits_{backbone}_{head}_{split}{suffix}.npz")
            if not np.array_equal(saved["labels"], [s.label for s in samples[split]]):
                raise SystemExit(f"{name}/{split} is not in the order of {args.data / 'splits.csv'}")
            probabilities[split][name] = decode(saved["logits"])

    bags = {split: eye_bags(samples[split]) for split in ("val", "test")}
    summary = fit_eye_ensemble(
        probabilities["val"], probabilities["test"], bags["val"], bags["test"],
        np.array([s.label for s in samples["val"]]), np.array([s.label for s in samples["test"]]),
        args.step, args.urgent_target,
        *eye_context(samples, bags),
    )
    summary["tta"] = args.tta

    print(f"{'run':30}{'gộp':>6}{'val mean':>10}{'val max':>10}{'trọng số':>10}")
    for name in summary["runs"]:
        scores = summary["pooling_val_composite"][name]
        print(f"{name:30}{summary['pooling'][name]:>6}{scores['mean']:>10.4f}{scores['max']:>10.4f}"
              f"{summary['weights'][name]:>10.2f}")
    print(f"\nval composite (eye) {summary['val_composite']:.4f}   "
          f"eyes: val {summary['eyes']['val']:,}, test {summary['eyes']['test']:,}")
    for key in ("val_argmax", "test_argmax", "test_ordinal"):
        print(f"{key:14} {M.format_metrics(summary['results'][key])}")
    flag = summary["results"]["test_urgent_flag"]
    print(f"urgent flag    P(3)+P(4) >= {flag['threshold']:.3f}: sens {flag['sensitivity']:.3f} "
          f"| missed {flag['missed']}/{flag['positives']} | false alarms {flag['false_alarms']}")
    print_fellow(summary)

    target = args.checkpoints / f"ensemble_eye{suffix}.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


def main() -> int:
    args = parse_args()
    if args.level == "eye":
        return eye_main(args)
    args.step = default_step(len(args.runs)) if args.step is None else args.step
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
    val_argmax = val_blend.argmax(axis=1)
    test_argmax = test_blend.argmax(axis=1)

    thresholds, _ = M.tune_thresholds(M.expected_grade(val_blend), labels)
    test_ordinal = M.apply_thresholds(M.expected_grade(test_blend), thresholds)

    results = {
        "val_argmax": M.summarise(labels, val_argmax),
        "test_argmax": M.summarise(test_labels, test_argmax),
        "test_ordinal": M.summarise(test_labels, test_ordinal),
    }
    for run in runs:
        results[f"test_{run['name']}"] = M.summarise(
            test_labels, run["test"].argmax(axis=1)
        )

    print()
    for key in sorted(results):
        print(f"{key:32} {M.format_metrics(results[key])}")

    stem = "ensemble_tta" if args.tta else "ensemble"
    matrix_artifacts = {
        "val": f"confusion_matrix_{stem}_val.png",
        "test": f"confusion_matrix_{stem}_test.png",
    }
    M.save_confusion_matrix(
        labels, val_argmax, args.checkpoints / matrix_artifacts["val"],
        f"{stem} — val argmax",
    )
    M.save_confusion_matrix(
        test_labels, test_argmax, args.checkpoints / matrix_artifacts["test"],
        f"{stem} — test argmax",
    )

    summary = {
        "runs": [r["name"] for r in runs],
        "tta": args.tta,
        "temperatures": {r["name"]: r["temperature"] for r in runs},
        "weights": dict(zip([r["name"] for r in runs], weights.tolist())),
        "val_composite": val_score,
        "thresholds": thresholds.tolist(),
        "results": results,
        "artifacts": {"confusion_matrices": matrix_artifacts},
    }
    name = f"{stem}.json"
    (args.checkpoints / name).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8",
    )
    print(f"\nwrote {args.checkpoints / name}")
    print(f"wrote {args.checkpoints / matrix_artifacts['val']}")
    print(f"wrote {args.checkpoints / matrix_artifacts['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

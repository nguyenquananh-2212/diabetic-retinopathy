"""Scoring, threshold tuning and calibration for an ordinal 5-grade problem.

DR grades are ordered, so predicting 4 when the truth is 0 is not the same
mistake as predicting 1.  Quadratic weighted kappa says so and plain accuracy
does not, which is why QWK is the headline number here.

But QWK is not a safe *model-selection* criterion on its own, and this project
has the receipts.  Two consecutive epochs of the same run:

```
epoch  9   QWK 0.8516   recall(grade 3) 0.64
epoch 10   QWK 0.8584   recall(grade 3) 0.47
```

Selecting on QWK picks epoch 10 and throws away the entire point of the
balanced sampler.  ``composite_score`` therefore blends QWK with balanced
accuracy, and the blend weight is explicit rather than hidden inside a
checkpoint callback.  Reproducing those two epochs against it: QWK ranks them
0.9570 > 0.9607 the wrong way round, the composite ranks them 0.9072 > 0.8929
the right way round.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

NUM_CLASSES = 5
RARE_CLASSES = (1, 3)  # mild and severe: the two the models actually miss (reporting only)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def confusion(true: np.ndarray, pred: np.ndarray, classes: int = NUM_CLASSES) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (true.astype(int), pred.astype(int)), 1)
    return matrix


def quadratic_weighted_kappa(
    true: np.ndarray, pred: np.ndarray, classes: int = NUM_CLASSES
) -> float:
    observed = confusion(true, pred, classes).astype(np.float64)
    index = np.arange(classes)
    weight = (index[:, None] - index[None, :]) ** 2 / (classes - 1) ** 2

    actual = observed.sum(axis=1)
    predicted = observed.sum(axis=0)
    expected = np.outer(actual, predicted) / max(1.0, observed.sum())

    denominator = float((weight * expected).sum())
    if denominator == 0.0:
        return 0.0
    return 1.0 - float((weight * observed).sum()) / denominator


def per_class_recall(
    true: np.ndarray, pred: np.ndarray, classes: int = NUM_CLASSES
) -> np.ndarray:
    matrix = confusion(true, pred, classes)
    support = matrix.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.diag(matrix) / support
    return np.where(support > 0, recall, np.nan)


def macro_f1(true: np.ndarray, pred: np.ndarray, classes: int = NUM_CLASSES) -> float:
    matrix = confusion(true, pred, classes)
    true_positive = np.diag(matrix).astype(np.float64)
    precision_denominator = matrix.sum(axis=0)
    recall_denominator = matrix.sum(axis=1)
    scores = []
    for c in range(classes):
        if recall_denominator[c] == 0:
            continue
        precision = true_positive[c] / precision_denominator[c] if precision_denominator[c] else 0.0
        recall = true_positive[c] / recall_denominator[c]
        scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return float(np.mean(scores)) if scores else 0.0


def balanced_accuracy(true: np.ndarray, pred: np.ndarray, classes: int = NUM_CLASSES) -> float:
    recall = per_class_recall(true, pred, classes)
    return float(np.nanmean(recall))


def composite_score(
    true: np.ndarray, pred: np.ndarray, balance_weight: float = 0.35
) -> float:
    """QWK, pulled back toward the grades a QWK-only run is happy to abandon.

    The second term is balanced accuracy, not the mean recall of a hand-picked
    rare pair.  Rewarding only grades 1 and 3 was tried and it simply moves the
    hole: on a 4,000-sample check the threshold tuner raised the composite from
    0.720 to 0.750 by pushing grade 2 -- 29 % of the data -- from 0.78 recall to
    0.44.  Balanced accuracy weights every grade equally, so nothing can be
    traded away, while QWK still supplies the ordinal signal that balanced
    accuracy lacks.
    """

    kappa = quadratic_weighted_kappa(true, pred)
    balanced = balanced_accuracy(true, pred)
    if np.isnan(balanced):
        return kappa
    return (1.0 - balance_weight) * kappa + balance_weight * balanced


def summarise(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    recall = per_class_recall(true, pred)
    return {
        "qwk": quadratic_weighted_kappa(true, pred),
        "balanced_accuracy": balanced_accuracy(true, pred),
        "macro_f1": macro_f1(true, pred),
        "accuracy": float((true == pred).mean()),
        "composite": composite_score(true, pred),
        **{f"recall_{c}": float(recall[c]) for c in range(NUM_CLASSES)},
    }


def format_metrics(values: Dict[str, float]) -> str:
    """One line per epoch.

    ``acc`` sits next to ``bal`` on purpose: plain accuracy is dominated by
    grade 0 (41 % of the data), so a run can gain accuracy while losing the rare
    grades entirely.  Reading the two side by side makes that visible -- a
    widening gap between them is the model retreating to the majority class.
    """

    recalls = "/".join(f"{values[f'recall_{c}']:.2f}" for c in range(NUM_CLASSES))
    return (
        f"qwk {values['qwk']:.4f} | acc {values['accuracy']:.4f} "
        f"| bal {values['balanced_accuracy']:.4f} | f1 {values['macro_f1']:.4f} "
        f"| comp {values['composite']:.4f} | rec {recalls}"
    )


# --------------------------------------------------------------------------- #
# Clinical reading of the five grades
# --------------------------------------------------------------------------- #

GRADE_NAMES = {
    0: "không có DR",
    1: "NPDR nhẹ",
    2: "NPDR trung bình",
    3: "NPDR nặng",
    4: "PDR (tăng sinh)",
}

# Two boundaries decide what happens to the patient; the other three do not.
# Crossing one of these is a different kind of error from moving one grade
# inside a band, and every warning in the demo is built on that distinction.
BANDS = (
    (range(0, 2), "theo dõi", "chụp lại sau 12-24 tháng"),
    (range(2, 3), "chuyển tuyến", "khám chuyên khoa mắt"),
    (range(3, 5), "chuyển tuyến gấp", "nguy cơ mất thị lực"),
)
REFERABLE_FROM = 2  # referable DR: grade >= 2
URGENT_FROM = 3  # sight-threatening DR: grade >= 3


def band_of(grade: int) -> Tuple[str, str]:
    for grades, name, action in BANDS:
        if int(grade) in grades:
            return name, action
    raise ValueError(f"grade out of range: {grade}")


def adjacent_accuracy(true: np.ndarray, pred: np.ndarray, tolerance: int = 1) -> float:
    """Fraction predicted within ``tolerance`` grades.

    Reported because inter-grader agreement on DR is itself only ~85 % exact:
    an exact-match score punishes the model for disagreements two humans would
    also have.
    """

    return float((np.abs(true.astype(int) - pred.astype(int)) <= tolerance).mean())


def binary_endpoint(
    true: np.ndarray, pred: np.ndarray, threshold: int, score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Sensitivity / specificity / PPV / NPV for "grade >= threshold".

    This is the number a screening programme is actually judged on.  A model can
    hold a fine QWK while missing referable disease, because QWK spreads its
    weight over all five grades and the referral decision only cares about one
    cut.
    """

    actual = true.astype(int) >= threshold
    called = pred.astype(int) >= threshold
    tp = int((actual & called).sum())
    fp = int((~actual & called).sum())
    fn = int((actual & ~called).sum())
    tn = int((~actual & ~called).sum())

    out = {
        "threshold": threshold,
        "positives": int(actual.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / max(1, tp + fn),
        "specificity": tn / max(1, tn + fp),
        "ppv": tp / max(1, tp + fp),
        "npv": tn / max(1, tn + fn),
    }
    if score is not None and 0 < actual.sum() < len(actual):
        out["auc"] = roc_auc(actual, score)
    return out


def roc_auc(positive: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged."""

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    sorted_score = score[order]
    i = 0
    while i < len(sorted_score):
        j = i
        while j + 1 < len(sorted_score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_class_report(true: np.ndarray, pred: np.ndarray) -> List[Dict[str, float]]:
    matrix = confusion(true, pred)
    out = []
    for c in range(NUM_CLASSES):
        tp = int(matrix[c, c])
        support = int(matrix[c].sum())
        called = int(matrix[:, c].sum())
        precision = tp / called if called else 0.0
        recall = tp / support if support else float("nan")
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        out.append(
            {"grade": c, "name": GRADE_NAMES[c], "support": support,
             "precision": precision, "recall": recall, "f1": f1}
        )
    return out


def cumulative_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """``P(grade >= k)`` for k = 1..4, the form the decision boundaries need."""

    return np.cumsum(probabilities[..., ::-1], axis=-1)[..., ::-1][..., 1:]


# --------------------------------------------------------------------------- #
# Ordinal decoding
# --------------------------------------------------------------------------- #


def expected_grade(probabilities: np.ndarray) -> np.ndarray:
    """``sum(p_i * i)`` -- a continuous grade that respects the ordering.

    Argmax throws the ordering away: a distribution split evenly between 0 and 4
    and one concentrated on 2 both decode to 2 under argmax, but only the second
    means it.  The expected value keeps the difference, and thresholds turn it
    back into a class.
    """

    index = np.arange(probabilities.shape[1], dtype=np.float64)
    return probabilities @ index


def apply_thresholds(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    return np.searchsorted(np.asarray(thresholds, dtype=np.float64), scores).astype(np.int64)


def tune_thresholds(
    scores: np.ndarray,
    true: np.ndarray,
    objective=composite_score,
    rounds: int = 3,
    grid: int = 60,
) -> Tuple[np.ndarray, float]:
    """Coordinate ascent on the four cut points, from the naive [0.5, 1.5, ...].

    Tuned on validation and then frozen: fitting them on the test set would be
    the same class of mistake as the leakage this project already paid for.
    """

    thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
    best = objective(true, apply_thresholds(scores, thresholds))
    low, high = float(scores.min()) - 0.5, float(scores.max()) + 0.5

    for round_index in range(rounds):
        span = (high - low) / (2 ** (round_index + 1))
        for position in range(len(thresholds)):
            centre = thresholds[position]
            floor = thresholds[position - 1] if position else low
            ceiling = thresholds[position + 1] if position + 1 < len(thresholds) else high
            candidates = np.linspace(
                max(floor + 1e-6, centre - span), min(ceiling - 1e-6, centre + span), grid
            )
            for candidate in candidates:
                trial = thresholds.copy()
                trial[position] = candidate
                score = objective(true, apply_thresholds(scores, trial))
                if score > best:
                    best, thresholds = score, trial
    return thresholds, float(best)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def expected_calibration_error(
    probabilities: np.ndarray, true: np.ndarray, bins: int = 15
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if not inside.any():
            continue
        error += inside.mean() * abs(correct[inside].mean() - confidence[inside].mean())
    return float(error)


def softmax_probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()


def softmax_nll(logits: np.ndarray, true: np.ndarray) -> float:
    return float(
        F.cross_entropy(
            torch.as_tensor(logits, dtype=torch.float32),
            torch.as_tensor(true, dtype=torch.long),
        )
    )


def fit_temperature(
    logits: np.ndarray,
    true: np.ndarray,
    nll=softmax_nll,
    low: float = 0.05,
    high: float = 10.0,
) -> float:
    """Grid search, not LBFGS.

    The obvious implementation optimises T by autograd on the stored logits, and
    it fails with "Inference tensors cannot be saved for backward" whenever the
    logits were collected under ``torch.inference_mode``.  A coarse-to-fine grid
    on a one-dimensional convex objective costs milliseconds and cannot fail
    that way.

    ``nll`` is supplied by the caller because the head decides what the logits
    mean: 5 softmax scores and 4 cumulative CORAL scores need different losses,
    and scoring one with the other's loss is not a small error -- it indexes out
    of bounds on the first grade the 4-column tensor cannot represent.

    Temperature never changes the argmax of a softmax head -- dividing every
    logit by the same positive number preserves their order.  It changes the
    *probabilities*, so it moves ECE, expected-grade decoding and any threshold
    tuned on top.
    """

    best_temperature, best_loss = 1.0, float("inf")
    for _ in range(4):
        for temperature in np.linspace(low, high, 60):
            loss = nll(logits / float(temperature), true)
            if loss < best_loss:
                best_loss, best_temperature = loss, float(temperature)
        span = (high - low) / 8.0
        low, high = max(1e-3, best_temperature - span), best_temperature + span
    return best_temperature


@dataclass
class Calibration:
    """What validation decided, applied unchanged to test.

    ``decoder`` turns raw logits into a 5-class distribution and is what makes
    this work for both heads: softmax for the 5-way head, the cumulative
    difference for CORAL's 4 logits.
    """

    temperature: float = 1.0
    thresholds: Optional[np.ndarray] = None
    decoder: Callable[[np.ndarray], np.ndarray] = softmax_probabilities

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        return self.decoder(np.asarray(logits, dtype=np.float32) / self.temperature)

    def predict(self, logits: np.ndarray) -> np.ndarray:
        probabilities = self.probabilities(logits)
        if self.thresholds is None:
            return probabilities.argmax(axis=1)
        return apply_thresholds(expected_grade(probabilities), self.thresholds)

    @classmethod
    def fit(
        cls,
        logits: np.ndarray,
        true: np.ndarray,
        ordinal: bool = True,
        decoder: Callable[[np.ndarray], np.ndarray] = softmax_probabilities,
        nll=softmax_nll,
    ) -> "Calibration":
        calibration = cls(
            temperature=fit_temperature(logits, true, nll), decoder=decoder
        )
        if ordinal:
            scores = expected_grade(calibration.probabilities(logits))
            calibration.thresholds, _ = tune_thresholds(scores, true)
        return calibration

    def report(self, logits: np.ndarray, true: np.ndarray) -> Dict[str, float]:
        raw = self.decoder(np.asarray(logits, dtype=np.float32))
        scaled = self.probabilities(logits)
        return {
            "temperature": self.temperature,
            "ece_before": expected_calibration_error(raw, true),
            "ece_after": expected_calibration_error(scaled, true),
            "thresholds": None if self.thresholds is None else self.thresholds.tolist(),
        }

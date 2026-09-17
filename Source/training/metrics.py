from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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


def per_class_precision(
    true: np.ndarray, pred: np.ndarray, classes: int = NUM_CLASSES
) -> np.ndarray:

    matrix = confusion(true, pred, classes)
    called = matrix.sum(axis=0)
    return np.divide(
        np.diag(matrix).astype(np.float64),
        called,
        out=np.zeros(classes, dtype=np.float64),
        where=called > 0,
    )


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
    kappa = quadratic_weighted_kappa(true, pred)
    balanced = balanced_accuracy(true, pred)
    if np.isnan(balanced):
        return kappa
    return (1.0 - balance_weight) * kappa + balance_weight * balanced


def summarise(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    recall = per_class_recall(true, pred)
    precision = per_class_precision(true, pred)
    return {
        "qwk": quadratic_weighted_kappa(true, pred),
        "balanced_accuracy": balanced_accuracy(true, pred),
        "macro_precision": float(precision.mean()),
        "macro_f1": macro_f1(true, pred),
        "accuracy": float((true == pred).mean()),
        "composite": composite_score(true, pred),
        **{f"precision_{c}": float(precision[c]) for c in range(NUM_CLASSES)},
        **{f"recall_{c}": float(recall[c]) for c in range(NUM_CLASSES)},
    }


def format_metrics(values: Dict[str, float]) -> str:
    recalls = "/".join(f"{values[f'recall_{c}']:.2f}" for c in range(NUM_CLASSES))
    return (
        f"qwk {values['qwk']:.4f} | acc {values['accuracy']:.4f} "
        f"| bal {values['balanced_accuracy']:.4f} | prec {values['macro_precision']:.4f} "
        f"| f1 {values['macro_f1']:.4f} "
        f"| comp {values['composite']:.4f} | rec {recalls}"
    )


def _save_figure(fig, path: Path) -> None:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        fig.savefig(partial, format="png", dpi=160, bbox_inches="tight")
        partial.replace(path)
    finally:
        plt.close(fig)
        if partial.exists():
            partial.unlink()


def save_precision_history(history: Sequence[Dict], path: Path, title: str) -> None:
    if not history:
        return
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [int(row["epoch"]) for row in history]
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.plot(epochs, [row["macro_precision"] for row in history],
              color="black", linewidth=2.6, marker="o", label="Macro precision")
    colours = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2")
    for grade, colour in enumerate(colours):
        axis.plot(epochs, [row[f"precision_{grade}"] for row in history],
                  color=colour, linewidth=1.2, alpha=0.85, label=f"Grade {grade}")
    if all("composite" in row for row in history):
        best = max(history, key=lambda row: row["composite"])["epoch"]
        axis.axvline(best, color="grey", linestyle=":", linewidth=1.2,
                     label=f"Best composite: epoch {best}")
    axis.set(title=title, xlabel="Epoch", ylabel="Precision")
    axis.set_ylim(0.0, 1.02)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, path)


def save_confusion_matrix(true: np.ndarray, pred: np.ndarray, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    true, pred = np.asarray(true), np.asarray(pred)
    if true.shape != pred.shape or true.ndim != 1:
        raise ValueError(f"confusion inputs must be equal 1-D arrays, got {true.shape} and {pred.shape}")
    matrix = confusion(true, pred)
    support = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, support, out=np.zeros_like(matrix, dtype=float), where=support > 0)

    fig, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Fraction within true grade")
    axis.set(title=title, xlabel="Predicted grade", ylabel="True grade",
             xticks=range(NUM_CLASSES), yticks=range(NUM_CLASSES))
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            percentage = normalized[row, column]
            axis.text(column, row, f"{matrix[row, column]:,}\n{percentage:.1%}",
                      ha="center", va="center", fontsize=8,
                      color="white" if percentage >= 0.5 else "black")
    fig.tight_layout()
    _save_figure(fig, path)

GRADE_NAMES = {
    0: "không có DR",
    1: "NPDR nhẹ",
    2: "NPDR trung bình",
    3: "NPDR nặng",
    4: "PDR (tăng sinh)",
}

BANDS = (
    (range(0, 2), "theo dõi", "chụp lại sau 12-24 tháng"),
    (range(2, 3), "chuyển tuyến", "khám chuyên khoa mắt"),
    (range(3, 5), "chuyển tuyến gấp", "nguy cơ mất thị lực"),
)
REFERABLE_FROM = 2
URGENT_FROM = 3


def band_of(grade: int) -> Tuple[str, str]:
    for grades, name, action in BANDS:
        if int(grade) in grades:
            return name, action
    raise ValueError(f"grade out of range: {grade}")


def adjacent_accuracy(true: np.ndarray, pred: np.ndarray, tolerance: int = 1) -> float:
    return float((np.abs(true.astype(int) - pred.astype(int)) <= tolerance).mean())


def binary_endpoint(
    true: np.ndarray, pred: np.ndarray, threshold: int, score: Optional[np.ndarray] = None
) -> Dict[str, float]:
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

def pool_views(
    probabilities: np.ndarray, groups: Sequence[Sequence[int]], rule: str = "mean"
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if rule == "mean":
        return np.array([probabilities[list(group)].mean(axis=0) for group in groups])
    if rule != "max":
        raise ValueError(f"rule must be 'mean' or 'max', got {rule!r}")
    cumulative = cumulative_probabilities(probabilities)
    top = np.array([cumulative[list(group)].max(axis=0) for group in groups])
    ones = np.ones((len(groups), 1))
    edges = np.concatenate([ones, top, np.zeros_like(ones)], axis=1)
    return np.clip(edges[:, :-1] - edges[:, 1:], 0.0, None)


def expected_grade(probabilities: np.ndarray) -> np.ndarray:
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

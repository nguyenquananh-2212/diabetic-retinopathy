from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "preprocessing"))
sys.path.insert(0, str(_HERE.parent / "training"))
from retina import DARK_LEVEL, retina_box  # noqa: E402

STAT_FEATURES = (
    "fill", "aspect", "box_frac", "outside_bright", "disc_iou", "rg", "bg", "sat", "texture"
)
# Which of the above are ratios and want a log before any Gaussian is assumed.
_LOG_INDEX = (1, 5, 6, 8)
_SAT_INDEX = 7


def image_statistics(rgb: np.ndarray) -> Dict[str, float]:
    """The tier-1 feature vector, computed on the raw input image."""

    height, width = rgb.shape[:2]
    frame = float(height * width)
    bright = rgb.max(axis=2) > DARK_LEVEL

    box = retina_box(rgb) or (0, 0, width, height)
    left, top, right, bottom = box
    box_w, box_h = right - left, bottom - top
    inside = bright[top:bottom, left:right]

    yy, xx = np.mgrid[0:box_h, 0:box_w]
    radius = max(1e-6, min(box_h, box_w) / 2)
    circle = ((yy - (box_h - 1) / 2) ** 2 + (xx - (box_w - 1) / 2) ** 2) <= radius**2
    disc_iou = float((inside & circle).sum()) / max(1.0, float((inside | circle).sum()))

    crop = rgb[top:bottom, left:right].astype(np.float32)
    mask = inside if inside.sum() >= 64 else np.ones_like(inside, dtype=bool)
    red, green, blue = crop[..., 0][mask], crop[..., 1][mask], crop[..., 2][mask]
    hsv = cv2.cvtColor(rgb[top:bottom, left:right], cv2.COLOR_RGB2HSV)

    small = cv2.resize(crop[..., 1], (256, 256), interpolation=cv2.INTER_AREA)
    high_pass = small - cv2.GaussianBlur(small, (0, 0), 6)

    return {
        "fill": float(inside.mean()) if inside.size else 0.0,
        "aspect": box_w / max(1, box_h),
        "box_frac": (box_w * box_h) / frame,
        "outside_bright": (bright.sum() - inside.sum()) / max(1.0, frame - box_w * box_h),
        "disc_iou": disc_iou,
        "rg": float(np.median(red / (green + 1.0))),
        "bg": float(np.median(blue / (green + 1.0))),
        "sat": float(np.median(hsv[..., 1][mask])),
        "texture": float(high_pass.std() / (small.mean() + 1.0)),
    }


def _prepare(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    for index in _LOG_INDEX:
        out[:, index] = np.log(np.clip(out[:, index], 1e-3, None))
    out[:, _SAT_INDEX] = out[:, _SAT_INDEX] / 255.0
    return out


@dataclass
class Mahalanobis:
    mean: np.ndarray
    precision: np.ndarray
    threshold: float

    @classmethod
    def fit(
        cls, train: np.ndarray, calibration: np.ndarray, percentile: float = 99.0,
        shrinkage: float = 0.10,
    ) -> "Mahalanobis":
        mean = train.mean(axis=0)
        covariance = np.cov(train.T)
        covariance = (1 - shrinkage) * covariance + shrinkage * np.eye(len(mean)) * (
            np.trace(covariance) / len(mean)
        )
        model = cls(mean, np.linalg.inv(covariance), threshold=float("inf"))
        model.threshold = float(np.percentile(model.score(calibration), percentile))
        return model

    def score(self, x: np.ndarray) -> np.ndarray:
        delta = np.atleast_2d(x) - self.mean
        return np.sqrt(
            np.maximum(0.0, np.einsum("ij,jk,ik->i", delta, self.precision, delta))
        )

    def to_dict(self) -> Dict:
        return {
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, blob: Dict) -> "Mahalanobis":
        return cls(
            np.asarray(blob["mean"], dtype=np.float64),
            np.asarray(blob["precision"], dtype=np.float64),
            float(blob["threshold"]),
        )


@dataclass
class Verdict:
    is_fundus: bool
    stat_score: float
    stat_threshold: float
    feature_scores: Dict[str, float] = field(default_factory=dict)
    feature_thresholds: Dict[str, float] = field(default_factory=dict)
    max_probability: Optional[float] = None
    entropy: Optional[float] = None
    reasons: List[str] = field(default_factory=list)

    @property
    def stat_flag(self) -> bool:
        return self.stat_score > self.stat_threshold

    @property
    def flagged_models(self) -> List[str]:
        return [
            name
            for name, score in self.feature_scores.items()
            if score > self.feature_thresholds.get(name, float("inf"))
        ]

    @property
    def feature_flag(self) -> bool:
        return bool(self.flagged_models)


class FundusGate:
    VERSION = 2

    def __init__(
        self, stats: Mahalanobis, features: Optional[Dict[str, Mahalanobis]] = None
    ) -> None:
        self.stats = stats
        self.features: Dict[str, Mahalanobis] = dict(features or {})
    def save(self, path: Path) -> None:
        blob = {
            "version": self.VERSION,
            "stat_features": list(STAT_FEATURES),
            "stats": self.stats.to_dict(),
            "features": {name: m.to_dict() for name, m in self.features.items()},
        }
        Path(path).write_text(json.dumps(blob), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FundusGate":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        version = blob.get("version", 1)
        if version != cls.VERSION:
            raise SystemExit(
                f"{path} là cổng lọc phiên bản {version}, code cần phiên bản "
                f"{cls.VERSION} (một tầng đặc trưng cho MỖI model). "
                "Chạy lại với --fit-gate kèm đủ các --checkpoint."
            )
        return cls(
            Mahalanobis.from_dict(blob["stats"]),
            {name: Mahalanobis.from_dict(m) for name, m in (blob.get("features") or {}).items()},
        )
    @classmethod
    def fit(
        cls,
        train_stats: np.ndarray,
        val_stats: np.ndarray,
        train_features: Optional[Dict[str, np.ndarray]] = None,
        val_features: Optional[Dict[str, np.ndarray]] = None,
        percentile: float = 99.0,
    ) -> "FundusGate":
        stats = Mahalanobis.fit(_prepare(train_stats), _prepare(val_stats), percentile)
        features: Dict[str, Mahalanobis] = {}
        for name in sorted(train_features or {}):
            if name not in (val_features or {}):
                continue
            features[name] = Mahalanobis.fit(
                np.asarray(train_features[name], dtype=np.float64),
                np.asarray(val_features[name], dtype=np.float64),
                percentile,
            )
        return cls(stats, features)

    def judge(
        self,
        statistics: Dict[str, float],
        feature_vectors: Optional[Dict[str, np.ndarray]] = None,
        probabilities: Optional[np.ndarray] = None,
    ) -> Verdict:
        vector = np.array([[statistics[k] for k in STAT_FEATURES]], dtype=np.float64)
        verdict = Verdict(
            is_fundus=True,
            stat_score=float(self.stats.score(_prepare(vector))[0]),
            stat_threshold=self.stats.threshold,
        )
        for name, feature in (feature_vectors or {}).items():
            model = self.features.get(name)
            if model is None:
                verdict.reasons.append(
                    f"chưa hiệu chuẩn tầng đặc trưng cho '{name}' -- model này không được gác"
                )
                continue
            verdict.feature_scores[name] = float(model.score(feature.reshape(1, -1))[0])
            verdict.feature_thresholds[name] = model.threshold

        if probabilities is not None:
            probability = np.asarray(probabilities, dtype=np.float64).ravel()
            verdict.max_probability = float(probability.max())
            verdict.entropy = float(
                -(probability * np.log(np.clip(probability, 1e-12, None))).sum()
            )

        for name in verdict.flagged_models:
            verdict.is_fundus = False
            verdict.reasons.append(
                f"đặc trưng '{name}' lệch xa phân bố ảnh đáy mắt "
                f"({verdict.feature_scores[name]:.1f} > "
                f"{verdict.feature_thresholds[name]:.1f})"
            )
        if verdict.stat_flag:
            verdict.reasons.append(
                f"thống kê ảnh bất thường "
                f"({verdict.stat_score:.2f} > {verdict.stat_threshold:.2f})"
            )
            if not verdict.feature_scores:
                verdict.is_fundus = False
        if not verdict.feature_scores:
            verdict.reasons.append(
                "không có tầng đặc trưng nào chấm được -- chỉ còn tầng thống kê, "
                "vốn chỉ bắt được 6/16 ảnh lạ khi đo"
            )
        return verdict


def worst_statistics(statistics: Dict[str, float], gate: FundusGate, top: int = 3) -> List[Tuple[str, float, float]]:
    vector = _prepare(np.array([[statistics[k] for k in STAT_FEATURES]], dtype=np.float64))[0]
    sigma = np.sqrt(np.diag(np.linalg.inv(gate.stats.precision)))
    deviation = (vector - gate.stats.mean) / np.maximum(sigma, 1e-9)
    order = np.argsort(-np.abs(deviation))[:top]
    return [
        (STAT_FEATURES[i], float(statistics[STAT_FEATURES[i]]), float(deviation[i]))
        for i in order
    ]

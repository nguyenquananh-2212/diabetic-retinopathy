"""Demo: grade a fundus image, or report the full test-set evaluation.

Three questions this file exists to answer.

**1. What are the numbers that matter after training?**  ``--evaluate`` prints
them.  QWK is the headline, but the number a screening programme is judged on is
the sensitivity of the *referral decision*, and that is a different cut of the
same predictions.  A model can hold QWK 0.86 while missing referable disease,
because QWK spreads its weight over five grades and the referral question has
only one boundary.  So the report gives both, plus the confusion matrix that
shows where the errors actually land.

**2. What happens when the input is not an eye?**  ``ood.FundusGate`` answers
before the classifier is allowed to speak, and the demo prints its score, its
threshold, and which image statistics are furthest out.  On measurement the
feature tier caught 16/16 real photographs while accepting an out-of-corpus
fundus image; the statistics tier alone caught 6/16, which is why it only warns.

**3. How should a grade-3 predicted as grade-4 be flagged?**  Not as a plain
error, because it is not one in any way that matters: 3 and 4 sit in the same
management band -- both are urgent referrals -- so the patient's path is
identical.  What deserves a warning is an error that *crosses* a boundary:
1 vs 2 changes rescreen into refer, 2 vs 3 changes routine into urgent.  The
demo therefore reports the probability mass on each side of those two cuts and
grades every warning by which boundary, if any, is at stake.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "preprocessing"))
sys.path.insert(0, str(_HERE.parent / "training"))

import metrics as M  # noqa: E402
from data import IMAGENET_MEAN, IMAGENET_STD, load_split  # noqa: E402
from models import ModelConfig, build_model, coral_to_probabilities  # noqa: E402
from ood import STAT_FEATURES, FundusGate, image_statistics, worst_statistics  # noqa: E402
from retina import standardise  # noqa: E402

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def progress(iterable, desc, total=None):
    return iterable if tqdm is None else tqdm(
        iterable, desc=desc, total=total, mininterval=1.0, dynamic_ncols=True
    )


RULE = "=" * 78


def read_image(path: Path) -> np.ndarray:
    """cv2.imread cannot open a non-ASCII path on Windows; decode from bytes."""

    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"không đọc được ảnh: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_tensor(rgb: np.ndarray, size: int) -> torch.Tensor:
    out = standardise(rgb, size).astype(np.float32) / 255.0
    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(out.transpose(2, 0, 1))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class Grader:
    def __init__(self, checkpoint: Path, device: Optional[torch.device] = None) -> None:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            # An unset shell variable expands to nothing, so a path like
            # "$CK\model.pt" arrives as "\model.pt".  Say that plainly instead
            # of letting torch.load throw twelve frames of traceback.
            hint = ""
            if str(checkpoint).lstrip("\/") == checkpoint.name:
                hint = ("  (đường dẫn chỉ còn tên file -- biến shell rỗng? "
                        "trong PowerShell $VAR chỉ sống trong phiên đã gán nó)")
            raise SystemExit(f"không thấy checkpoint: {checkpoint}{hint}")
        blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = blob.get("config", {})
        self.head = config.get("head", "softmax")
        self.size = int(config.get("image_size", 448))
        self.backbone_name = config.get("backbone", "convnext_tiny")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(
            ModelConfig(
                backbone=self.backbone_name, image_size=self.size,
                head=self.head, pretrained=False,
            )
        )
        self.model.load_state_dict(blob["model"])
        self.model.to(self.device).eval()
        self.trained_epoch = blob.get("epoch")
        self.val_at_best = blob.get("val", {})
        # Matches the run names ensemble.py writes, so a saved blend can be
        # paired back up with the checkpoints it was fitted for.
        self.name = f"{self.backbone_name}_{self.head}"

    @torch.no_grad()
    def forward(self, batch: torch.Tensor, tta: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Return (logits, penultimate features).

        Logits are averaged across TTA views. OOD uses the untransformed view,
        matching fit_gate(), which calibrates features without TTA.
        """

        batch = batch.to(self.device)
        views = range(8) if tta else range(1)
        logits_sum, base_features = None, None
        for k in views:
            view = torch.rot90(batch, k % 4, dims=(2, 3))
            if k >= 4:
                view = torch.flip(view, dims=(3,))
            features = self.model.backbone(view).float()
            logits = self.model.head(features).float()
            logits_sum = logits if logits_sum is None else logits_sum + logits
            if k == 0:
                base_features = features
        count = len(views)
        return (
            (logits_sum / count).cpu().numpy(),
            base_features.cpu().numpy(),
        )

    def probabilities(self, logits: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(logits, dtype=torch.float32)
        if self.head == "coral":
            return coral_to_probabilities(tensor).numpy()
        return torch.softmax(tensor, dim=1).numpy()

    def nll(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """The head's own loss -- softmax and CORAL logits are different objects,
        and scoring one with the other's loss indexes out of bounds."""

        if self.head == "coral":
            from models import coral_loss

            return float(
                coral_loss(
                    torch.as_tensor(logits, dtype=torch.float32),
                    torch.as_tensor(labels, dtype=torch.long),
                )
            )
        return M.softmax_nll(logits, labels)


class Ensemble:
    """Saved TTA setting -> per-model temperature -> probability blend -> decoder.

    Calibration settings belong to the exact runs used to fit the JSON on val.
    Replaying saved logits checks this decoding pipeline, not image preprocessing
    or whether a checkpoint/dataset is the same as the original experiment.
    """

    def __init__(
        self,
        graders: Sequence[Grader],
        temperatures: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Sequence[float]] = None,
        tta: Optional[bool] = None,
    ) -> None:
        self.graders = list(graders)
        self.names = [g.name for g in self.graders]
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("cần checkpoint có tên run không trùng nhau")
        self.temperatures = {n: float((temperatures or {}).get(n, 1.0)) for n in self.names}
        if any(not np.isfinite(t) or t <= 0 for t in self.temperatures.values()):
            raise ValueError("temperature phải hữu hạn và > 0")
        raw = {n: float((weights or {}).get(n, 1.0)) for n in self.names}
        total = sum(raw.values())
        if (any(not np.isfinite(w) or w < 0 for w in raw.values())
                or not np.isfinite(total) or total <= 0):
            raise ValueError("weights phải hữu hạn, không âm và có tổng > 0")
        self.weights = {n: w / total for n, w in raw.items()}
        self.thresholds = None if thresholds is None else np.asarray(thresholds, dtype=np.float64)
        if self.thresholds is not None and (
            self.thresholds.shape != (M.NUM_CLASSES - 1,)
            or not np.isfinite(self.thresholds).all()
            or not np.all(np.diff(self.thresholds) > 0)
        ):
            raise ValueError("thresholds phải gồm 4 ngưỡng hữu hạn, tăng nghiêm ngặt")
        self.tta = tta

    @classmethod
    def from_json(cls, graders: Sequence[Grader], path: Path) -> "Ensemble":
        try:
            blob = json.loads(Path(path).read_text(encoding="utf-8"))
            names = [g.name for g in graders]
            runs = blob["runs"]
            if (not isinstance(runs, list) or not runs
                    or not all(isinstance(n, str) for n in runs)
                    or len(set(runs)) != len(runs)
                    or len(set(names)) != len(names) or set(names) != set(runs)):
                raise ValueError(f"checkpoints {names} phải khớp chính xác runs {runs}")
            for key in ("temperatures", "weights"):
                if not isinstance(blob[key], dict) or set(blob[key]) != set(runs):
                    raise ValueError(f"{key} phải có đúng các run trong JSON")
            if not isinstance(blob["tta"], bool):
                raise ValueError("tta phải là true hoặc false")
            by_name = {g.name: g for g in graders}
            return cls([by_name[n] for n in runs], blob["temperatures"], blob["weights"],
                       blob["thresholds"], tta=blob["tta"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit(f"ensemble JSON không hợp lệ: {path}: {exc}") from None

    def resolve_tta(self, requested: Optional[bool] = None) -> bool:
        if self.tta is not None and requested is not None and requested != self.tta:
            raise ValueError("--tta/--no-tta mâu thuẫn với tta trong ensemble JSON")
        return bool(self.tta if self.tta is not None else requested)

    def describe(self) -> str:
        parts = [
            f"{n} (w={self.weights[n]:.2f}, T={self.temperatures[n]:.3f})"
            for n in self.names
        ]
        tail = "" if self.thresholds is None else \
            f"  ngưỡng ordinal {[round(float(t), 3) for t in self.thresholds]}"
        return " + ".join(parts) + tail

    def run(self, rgb: np.ndarray, tta: Optional[bool] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray],
                                                              Dict[str, np.ndarray]]:
        """Blended probabilities, per-model probabilities, per-model features."""

        tta = self.resolve_tta(tta)
        blended = None
        per_model: Dict[str, np.ndarray] = {}
        features: Dict[str, np.ndarray] = {}
        for grader in self.graders:
            logits, feats = grader.forward(
                to_tensor(rgb, grader.size).unsqueeze(0), tta=tta
            )
            features[grader.name] = feats[0]
            scaled = logits / self.temperatures[grader.name]
            probability = grader.probabilities(scaled)[0]
            per_model[grader.name] = probability
            weighted = self.weights[grader.name] * probability
            blended = weighted if blended is None else blended + weighted
        return blended, per_model, features

    def run_batch(
        self, images: Sequence[np.ndarray], tta: Optional[bool] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Same maths as ``run``, several images at once.

        Worth the extra path: at 448 a batch of one leaves the GPU idling
        between kernels -- measured 11-16 img/s at batch 1 against 32 img/s at
        batch 8, so a folder scan finishes in roughly a third of the time.  A
        single image still goes through ``run``; nothing about the arithmetic
        differs, only how many rows the tensors have.
        """

        tta = self.resolve_tta(tta)
        blended = None
        per_model: Dict[str, np.ndarray] = {}
        features: Dict[str, np.ndarray] = {}
        for grader in self.graders:
            batch = torch.stack([to_tensor(rgb, grader.size) for rgb in images])
            logits, feats = grader.forward(batch, tta=tta)
            features[grader.name] = feats
            probability = grader.probabilities(logits / self.temperatures[grader.name])
            per_model[grader.name] = probability
            weighted = self.weights[grader.name] * probability
            blended = weighted if blended is None else blended + weighted
        return blended, per_model, features

    def decode(self, probability: np.ndarray) -> int:
        if self.thresholds is None:
            return int(probability.argmax())
        score = M.expected_grade(probability.reshape(1, -1))
        return int(M.apply_thresholds(score, self.thresholds)[0])


# --------------------------------------------------------------------------- #
# 1. Evaluation report
# --------------------------------------------------------------------------- #


def evaluate(
    grader: Grader, manifest: Path, splits: Path, image_root: Optional[Path],
    split: str, batch_size: int, tta: bool, calibration: Optional[M.Calibration],
) -> Dict:
    samples = load_split(manifest, splits, split, image_root)
    logits, features, labels = [], [], []
    for start in progress(range(0, len(samples), batch_size),
                          f"{split}{' +TTA' if tta else ''}",
                          (len(samples) + batch_size - 1) // batch_size):
        chunk = samples[start : start + batch_size]
        batch = torch.stack([to_tensor(read_image(Path(s.path)), grader.size) for s in chunk])
        z, f = grader.forward(batch, tta=tta)
        logits.append(z)
        features.append(f)
        labels.extend(s.label for s in chunk)

    logits = np.concatenate(logits)
    features = np.concatenate(features)
    labels = np.array(labels, dtype=np.int64)

    raw = grader.probabilities(logits)
    predictions = {"argmax": raw.argmax(axis=1)}
    probabilities = {"argmax": raw}
    if calibration is not None:
        predictions["calibrated"] = calibration.predict(logits)
        probabilities["calibrated"] = calibration.probabilities(logits)

    report: Dict = {"split": split, "tta": tta, "n": len(labels), "variants": {}}
    for name, pred in predictions.items():
        probability = probabilities[name]
        cumulative = M.cumulative_probabilities(probability)
        report["variants"][name] = {
            "overall": M.summarise(labels, pred),
            "adjacent_accuracy": M.adjacent_accuracy(labels, pred),
            "per_class": M.per_class_report(labels, pred),
            "confusion": M.confusion(labels, pred).tolist(),
            "referable": M.binary_endpoint(
                labels, pred, M.REFERABLE_FROM, cumulative[:, M.REFERABLE_FROM - 1]
            ),
            "urgent": M.binary_endpoint(
                labels, pred, M.URGENT_FROM, cumulative[:, M.URGENT_FROM - 1]
            ),
            "ece": M.expected_calibration_error(probability, labels),
        }
    return report, logits, features, labels


def print_report(report: Dict, grader: Grader) -> None:
    print(f"\n{RULE}\nĐÁNH GIÁ — {report['split']}"
          f"{' + TTA 8 hướng' if report['tta'] else ''} — {report['n']:,} ảnh\n{RULE}")
    print(f"model: {grader.backbone_name} / {grader.head}"
          + (f", checkpoint epoch {grader.trained_epoch}" if grader.trained_epoch else ""))

    for name, block in report["variants"].items():
        v = block["overall"]
        print(f"\n--- {name} " + "-" * (72 - len(name)))
        print(f"  QWK                 {v['qwk']:.4f}   (chỉ số chính, phạt theo bình phương khoảng cách bậc)")
        print(f"  accuracy            {v['accuracy']:.4f}")
        print(f"  accuracy ±1 bậc     {block['adjacent_accuracy']:.4f}   (đồng thuận giữa hai bác sĩ cũng chỉ ~0,85)")
        print(f"  balanced accuracy   {v['balanced_accuracy']:.4f}   (mỗi lớp một phiếu ngang nhau)")
        print(f"  macro F1            {v['macro_f1']:.4f}")
        print(f"  composite           {v['composite']:.4f}   (tiêu chí đã chọn checkpoint)")
        print(f"  ECE                 {block['ece']:.4f}   (0 = xác suất nói đúng độ tin cậy thật)")

        print(f"\n  {'bậc':>3} {'tên':18} {'n':>6} {'precision':>10} {'recall':>8} {'F1':>8}")
        for row in block["per_class"]:
            print(f"  {row['grade']:>3} {row['name']:18} {row['support']:>6,} "
                  f"{row['precision']:10.3f} {row['recall']:8.3f} {row['f1']:8.3f}")

        print(f"\n  Ma trận nhầm lẫn (hàng = thật, cột = dự đoán)")
        matrix = np.array(block["confusion"])
        print("        " + "".join(f"{c:>8}" for c in range(5)))
        for c, row in enumerate(matrix):
            marks = "".join(
                f"{value:>8}" if c == j else f"{value:>8}" for j, value in enumerate(row)
            )
            print(f"    {c:>3} {marks}   {100*row[c]/max(1,row.sum()):5.1f}% đúng")

        print(f"\n  QUYẾT ĐỊNH LÂM SÀNG (thứ mà chương trình sàng lọc thật sự bị đánh giá)")
        for key, title in (("referable", "cần chuyển tuyến (bậc ≥ 2)"),
                           ("urgent", "chuyển tuyến GẤP (bậc ≥ 3)")):
            e = block[key]
            auc = f"{e['auc']:.4f}" if "auc" in e else "  n/a "
            print(f"    {title:32} sens {e['sensitivity']:.3f}  spec {e['specificity']:.3f}  "
                  f"PPV {e['ppv']:.3f}  NPV {e['npv']:.3f}  AUC {auc}")
            print(f"      {'':32} bỏ sót {e['fn']:,}/{e['positives']:,} ca dương  |  "
                  f"báo động giả {e['fp']:,}")


# --------------------------------------------------------------------------- #
# 2 + 3. Single image
# --------------------------------------------------------------------------- #


def band_report(
    probability: np.ndarray,
    grade: Optional[int] = None,
    per_model: Optional[Dict[str, np.ndarray]] = None,
) -> Dict:
    cumulative = M.cumulative_probabilities(probability.reshape(1, -1))[0]
    decoded = grade is not None and grade != int(probability.argmax())
    grade = int(probability.argmax()) if grade is None else int(grade)
    expected = float(M.expected_grade(probability.reshape(1, -1))[0])
    band, action = M.band_of(grade)

    # A boundary is "in play" when neither side of it is settled.  0.15..0.85 is
    # deliberately wide: near a cut the cost of a wrong call is asymmetric, so
    # the useful signal is "a human should look", not a sharper number.
    boundaries = []
    for cut, label in ((M.REFERABLE_FROM, "chuyển tuyến"), (M.URGENT_FROM, "chuyển tuyến GẤP")):
        p_above = float(cumulative[cut - 1])
        boundaries.append({
            "cut": cut, "label": label, "p_above": p_above,
            "uncertain": 0.15 < p_above < 0.85,
        })
    return {
        "grade": grade, "name": M.GRADE_NAMES[grade], "band": band, "action": action,
        "expected_grade": expected, "probabilities": probability.tolist(),
        "cumulative": cumulative.tolist(), "boundaries": boundaries,
        "confidence": float(probability[grade]),
        "decoded_by_thresholds": decoded,
        "per_model": None if per_model is None else
                     {k: v.tolist() for k, v in per_model.items()},
    }


def compare_to_truth(predicted: int, truth: int) -> Dict:
    """Classify an error by what it would change, not by how far it is.

    Grade 3 called 4 (or 4 called 3) is one grade wrong and changes nothing: both
    are urgent referrals.  Grade 1 called 2 is also one grade wrong and changes
    everything: rescreen becomes refer.  Distance alone cannot tell those apart,
    which is why this looks at the boundaries crossed.
    """

    crossed = [
        cut for cut in (M.REFERABLE_FROM, M.URGENT_FROM)
        if (predicted >= cut) != (truth >= cut)
    ]
    band_pred, _ = M.band_of(predicted)
    band_true, _ = M.band_of(truth)

    if predicted == truth:
        return {"severity": "ĐÚNG", "crossed": [], "message": "khớp nhãn."}
    if not crossed:
        return {
            "severity": "SAI, KHÔNG ĐỔI XỬ TRÍ",
            "crossed": [],
            "message": (
                f"lệch {abs(predicted - truth)} bậc ({truth} -> {predicted}) nhưng cả hai "
                f"đều thuộc nhóm '{band_true}'. Hướng xử trí không đổi, "
                f"nên đây là sai số ít hậu quả nhất trong các kiểu sai."
            ),
        }
    if M.REFERABLE_FROM in crossed:
        direction = "BỎ SÓT ca cần chuyển tuyến" if truth >= M.REFERABLE_FROM else "chuyển tuyến thừa"
        return {
            "severity": "NGHIÊM TRỌNG" if truth >= M.REFERABLE_FROM else "CẢNH BÁO",
            "crossed": crossed,
            "message": (
                f"vượt ranh giới chuyển tuyến (bậc ≥ {M.REFERABLE_FROM}): "
                f"'{band_true}' -> '{band_pred}'. Đây là {direction}."
            ),
        }
    direction = "hạ cấp một ca cần khám GẤP" if truth >= M.URGENT_FROM else "nâng cấp thành GẤP"
    return {
        "severity": "ĐÁNG LƯU Ý",
        "crossed": crossed,
        "message": (
            f"vượt ranh giới mức độ khẩn (bậc ≥ {M.URGENT_FROM}): "
            f"'{band_true}' -> '{band_pred}'. Vẫn được chuyển tuyến, nhưng {direction}."
        ),
    }


def print_single(
    path: Path, verdict, statistics: Dict[str, float], gate: FundusGate,
    band: Optional[Dict], truth: Optional[int],
) -> None:
    print(f"\n{RULE}\n{path.name}\n{RULE}")

    print("BƯỚC 1 — ĐÂY CÓ PHẢI ẢNH ĐÁY MẮT KHÔNG?")
    print(f"  thống kê ảnh      {verdict.stat_score:8.2f}  / ngưỡng {verdict.stat_threshold:6.2f}"
          f"   {'VƯỢT' if verdict.stat_flag else 'đạt'}")
    for name, score in verdict.feature_scores.items():
        limit = verdict.feature_thresholds[name]
        print(f"  đặc trưng {name:14} {score:8.2f}  / ngưỡng {limit:6.2f}"
              f"   {'VƯỢT' if score > limit else 'đạt'}")
    if verdict.max_probability is not None:
        print(f"  độ tin cậy tối đa {verdict.max_probability:8.3f}"
              f"   |  entropy {verdict.entropy:.3f} / {np.log(5):.3f} tối đa")

    if not verdict.is_fundus:
        print("\n  *** TỪ CHỐI — KHÔNG CHẤM ĐIỂM ***")
        for reason in verdict.reasons:
            print(f"    - {reason}")
        print("\n  Các thống kê lệch xa nhất so với ảnh đáy mắt:")
        for name, value, sigma in worst_statistics(statistics, gate):
            print(f"    {name:16} = {value:8.3f}   ({sigma:+.1f} độ lệch chuẩn)")
        print("\n  Ảnh này không nằm trong phân bố mà model được huấn luyện. Bất kỳ")
        print("  bậc nào đưa ra cũng vô nghĩa, nên không đưa ra bậc nào cả.")
        return
    for reason in verdict.reasons:
        print(f"  [lưu ý] {reason}")

    if band is None:
        return

    print(f"\nBƯỚC 2 — MỨC ĐỘ")
    print(f"  bậc {band['grade']} — {band['name']}"
          f"   (tin cậy {band['confidence']:.3f}, bậc kỳ vọng {band['expected_grade']:.2f})")
    print(f"  nhóm xử trí: {band['band'].upper()} — {band['action']}")
    print("  phân bố:  " + "  ".join(
        f"{c}:{p:.3f}" for c, p in enumerate(band["probabilities"])
    ))
    for name, probability in (band.get("per_model") or {}).items():
        argmax = int(np.argmax(probability))
        print(f"    {name:22} bậc {argmax}  " + " ".join(
            f"{c}:{v:.3f}" for c, v in enumerate(probability)
        ))
    if band.get("decoded_by_thresholds"):
        print(f"  giải mã bằng ngưỡng ordinal, không phải argmax "
              f"(argmax cho bậc {int(np.argmax(band['probabilities']))})")

    print(f"\nBƯỚC 3 — RANH GIỚI QUYẾT ĐỊNH (chỗ sai thật sự gây hậu quả)")
    for b in band["boundaries"]:
        state = "KHÔNG CHẮC — cần người đọc lại" if b["uncertain"] else "rõ ràng"
        side = "CÓ" if b["p_above"] >= 0.5 else "không"
        print(f"  P(bậc ≥ {b['cut']}) = {b['p_above']:.3f}  -> {b['label']}: {side:>6}   [{state}]")
    if not any(b["uncertain"] for b in band["boundaries"]):
        print("  Không ranh giới nào ở trạng thái lưng chừng.")

    if truth is not None:
        outcome = compare_to_truth(band["grade"], truth)
        print(f"\nBƯỚC 4 — SO VỚI NHÃN THẬT (bậc {truth} — {M.GRADE_NAMES[truth]})")
        print(f"  {outcome['severity']}: {outcome['message']}")


# --------------------------------------------------------------------------- #
# Calibration of the gate
# --------------------------------------------------------------------------- #


def fit_gate(
    graders: Sequence[Grader], manifest: Path, splits: Path, image_root: Optional[Path],
    limit: int, batch_size: int, percentile: float,
) -> FundusGate:
    """One statistics gate, plus one feature gate per model.

    Every model gets its own because the penultimate spaces are unrelated: a
    distance of 30 means nothing in common between convnext's 768 dimensions and
    swin's.  Sharing one threshold across them would be comparing millimetres to
    seconds.
    """

    def gather(split: str, cap: int):
        samples = load_split(manifest, splits, split, image_root)
        rng = np.random.default_rng(0)
        if cap and len(samples) > cap:
            samples = [samples[i] for i in rng.choice(len(samples), cap, replace=False)]
        stats: List[List[float]] = []
        feats: Dict[str, List[np.ndarray]] = {g.name: [] for g in graders}
        for start in progress(range(0, len(samples), batch_size), f"hiệu chuẩn {split}",
                              (len(samples) + batch_size - 1) // batch_size):
            chunk = samples[start : start + batch_size]
            images = [read_image(Path(s.path)) for s in chunk]
            stats.extend([image_statistics(im)[k] for k in STAT_FEATURES] for im in images)
            for grader in graders:
                batch = torch.stack([to_tensor(im, grader.size) for im in images])
                feats[grader.name].append(grader.forward(batch)[1])
        return (
            np.array(stats, dtype=np.float64),
            {name: np.concatenate(v) for name, v in feats.items() if v},
        )

    train_stats, train_features = gather("train", limit)
    val_stats, val_features = gather("val", limit)
    return FundusGate.fit(train_stats, val_stats, train_features, val_features, percentile)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

BATCH_COLUMNS = [
    "file", "is_fundus", "grade", "band", "action", "confidence",
    "expected_grade", "p_ge2", "p_ge3", "needs_review", "models_disagree",
    "stat_score", "truth", "severity", "note",
]


def list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def grade_many(
    ensemble: "Ensemble",
    gate: FundusGate,
    paths: Sequence[Path],
    tta: Optional[bool] = None,
    truths: Optional[Dict[str, int]] = None,
    batch_size: int = 8,
    on_progress=None,
) -> List[Dict[str, object]]:
    """One row per file, in the order given.

    Every failure becomes a row with a ``note`` rather than an exception: a scan
    of a few thousand clinical images will meet a corrupt file eventually, and
    losing the other 2,999 results to it would be the wrong trade.
    """

    rows: List[Dict[str, object]] = []
    truths = truths or {}
    done = 0
    for start in range(0, len(paths), batch_size):
        chunk = list(paths[start : start + batch_size])
        loaded, kept = [], []
        for path in chunk:
            try:
                loaded.append(read_image(path))
                kept.append(path)
            except SystemExit as exc:
                rows.append({"file": path.name, "is_fundus": "", "note": str(exc)})
        if kept:
            blended, per_model, features = ensemble.run_batch(loaded, tta=tta)
            for index, (path, rgb) in enumerate(zip(kept, loaded)):
                statistics = image_statistics(rgb)
                verdict = gate.judge(
                    statistics,
                    {name: f[index] for name, f in features.items()},
                    blended[index],
                )
                row: Dict[str, object] = {
                    "file": path.name,
                    "is_fundus": int(verdict.is_fundus),
                    "stat_score": round(verdict.stat_score, 3),
                    "note": "; ".join(verdict.reasons) if not verdict.is_fundus else "",
                }
                for name, score in verdict.feature_scores.items():
                    row[f"ood_{name}"] = round(score, 2)
                if verdict.is_fundus:
                    picks = {n: int(p[index].argmax()) for n, p in per_model.items()}
                    band = band_report(
                        blended[index],
                        ensemble.decode(blended[index]),
                        {n: p[index] for n, p in per_model.items()},
                    )
                    boundaries = {b["cut"]: b for b in band["boundaries"]}
                    row.update({
                        "grade": band["grade"],
                        "band": band["band"],
                        "action": band["action"],
                        "confidence": round(band["confidence"], 4),
                        "expected_grade": round(band["expected_grade"], 3),
                        "p_ge2": round(boundaries[M.REFERABLE_FROM]["p_above"], 4),
                        "p_ge3": round(boundaries[M.URGENT_FROM]["p_above"], 4),
                        "needs_review": int(any(b["uncertain"] for b in band["boundaries"])),
                        "models_disagree": int(len(set(picks.values())) > 1),
                    })
                    truth = truths.get(path.name, truths.get(path.stem))
                    if truth is not None:
                        outcome = compare_to_truth(band["grade"], int(truth))
                        row["truth"] = int(truth)
                        row["severity"] = outcome["severity"]
                rows.append(row)
        done += len(chunk)
        if on_progress is not None:
            on_progress(done, len(paths))
    return rows


def batch_summary(rows: Sequence[Dict[str, object]]) -> str:
    total = len(rows)
    rejected = sum(1 for r in rows if r.get("is_fundus") == 0)
    failed = sum(1 for r in rows if r.get("is_fundus") == "")
    graded = [r for r in rows if r.get("grade") is not None and r.get("grade") != ""]
    lines = [
        f"{total:,} ảnh  |  chấm được {len(graded):,}  |  "
        f"cổng từ chối {rejected:,}  |  không đọc được {failed:,}"
    ]
    if graded:
        counts = Counter(int(r["grade"]) for r in graded)
        lines.append("phân bố bậc: " + "  ".join(
            f"{c}:{counts.get(c, 0):,}" for c in range(5)
        ))
        refer = sum(1 for r in graded if int(r["grade"]) >= M.REFERABLE_FROM)
        urgent = sum(1 for r in graded if int(r["grade"]) >= M.URGENT_FROM)
        review = sum(1 for r in graded if r.get("needs_review"))
        disagree = sum(1 for r in graded if r.get("models_disagree"))
        lines.append(
            f"cần chuyển tuyến {refer:,} ({100*refer/len(graded):.1f}%)  |  "
            f"GẤP {urgent:,} ({100*urgent/len(graded):.1f}%)"
        )
        lines.append(
            f"cần người đọc lại {review:,} ({100*review/len(graded):.1f}%)  |  "
            f"hai model bất đồng {disagree:,} ({100*disagree/len(graded):.1f}%)"
        )
        if any(r.get("severity") for r in graded):
            sev = Counter(r["severity"] for r in graded if r.get("severity"))
            lines.append("đối chiếu nhãn: " + "  ".join(
                f"{k} {v:,}" for k, v in sev.most_common()
            ))
    return chr(10).join(lines)


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> Path:
    extra = sorted({k for r in rows for k in r} - set(BATCH_COLUMNS))
    fields = BATCH_COLUMNS[:12] + extra + BATCH_COLUMNS[12:]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_truths(path: Optional[Path]) -> Dict[str, int]:
    """Optional filename -> grade map, so a scan can be scored as it runs."""

    if path is None:
        return {}
    out: Dict[str, int] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for record in csv.DictReader(handle):
            key = record.get("file") or record.get("image_id") or record.get("id")
            value = record.get("class_label") or record.get("grade") or record.get("level")
            if key and value not in (None, ""):
                out[Path(str(key)).stem] = int(value)
                out[str(key)] = int(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chấm điểm DR cho một ảnh, hoặc in báo cáo đánh giá đầy đủ."
    )
    parser.add_argument("images", nargs="*", type=Path, help="ảnh cần chấm")
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=None,
                        help="một hoặc nhiều checkpoint; nhiều thì trộn thành ensemble")
    parser.add_argument("--ensemble", type=Path, default=None,
                        help="ensemble_*.json từ ensemble.py: temperature, trọng số, ngưỡng")
    parser.add_argument("--data", type=Path,
                        default=PROJECT_ROOT / "Data" / "processed" / "img448")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--gate", type=Path,
                        default=PROJECT_ROOT / "checkpoints" / "fundus_gate.json")
    parser.add_argument("--fit-gate", action="store_true", help="hiệu chuẩn lại cổng lọc")
    parser.add_argument("--gate-limit", type=int, default=2000)
    parser.add_argument("--gate-percentile", type=float, default=99.0)
    parser.add_argument("--evaluate", action="store_true", help="báo cáo một model trên split đã chọn")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=None,
                        help="mặc định lấy từ ensemble JSON; không có JSON thì tắt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--truth", type=int, default=None, help="nhãn thật của ảnh, để so sánh")
    parser.add_argument("--json", type=Path, default=None, help="ghi kết quả ra file JSON")
    parser.add_argument("--batch", type=Path, default=None,
                        help="chấm mọi ảnh trong thư mục này")
    parser.add_argument("--csv", type=Path, default=None, help="ghi bảng kết quả ra CSV")
    parser.add_argument("--truth-csv", type=Path, default=None,
                        help="CSV ánh xạ file -> bậc thật, để chấm luôn mức độ sai")
    args = parser.parse_args()
    if args.ensemble and not args.checkpoint:
        parser.error("--ensemble cần --checkpoint")
    if args.checkpoint and len(set(args.checkpoint)) != len(args.checkpoint):
        parser.error("--checkpoint chứa đường dẫn trùng nhau")
    if args.evaluate and (len(args.checkpoint or []) != 1 or args.ensemble):
        parser.error("--evaluate hiện chỉ hỗ trợ đúng một checkpoint, không dùng --ensemble; "
                     "không thể coi đây là đánh giá ensemble")
    return args


def main() -> int:
    args = parse_args()
    manifest, splits = args.data / "manifest.csv", args.data / "splits.csv"
    graders = [Grader(p) for p in (args.checkpoint or [])]
    grader = graders[0] if graders else None
    ensemble = None
    tta = bool(args.tta)
    if graders:
        for g in graders:
            print(f"model: {g.name} @ {g.size} trên {g.device}")
        try:
            ensemble = (Ensemble.from_json(graders, args.ensemble)
                        if args.ensemble else Ensemble(graders))
            tta = ensemble.resolve_tta(args.tta)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        if len(graders) > 1 and not args.ensemble:
            print("[chú ý] không có --ensemble: trọng số đều nhau, T=1, "
                  "không dùng ngưỡng ordinal -- KHÔNG phải pipeline đã báo cáo")
        print(f"pipeline: {ensemble.describe()} | TTA={tta}")

    if args.fit_gate:
        gate = fit_gate(graders, manifest, splits, args.image_root,
                        args.gate_limit, args.batch_size, args.gate_percentile)
        args.gate.parent.mkdir(parents=True, exist_ok=True)
        gate.save(args.gate)
        print(f"cổng lọc -> {args.gate}")
        print(f"  ngưỡng thống kê   {gate.stats.threshold:.2f}")
        for name, model in gate.features.items():
            print(f"  ngưỡng đặc trưng {name:22} {model.threshold:.2f}")
    elif args.gate.is_file():
        gate = FundusGate.load(args.gate)
    else:
        raise SystemExit(
            f"chưa có cổng lọc tại {args.gate}; chạy lại với --fit-gate "
            "(kèm --checkpoint để có cả tầng đặc trưng)"
        )

    payload: Dict = {}

    if args.evaluate:
        if grader is None:
            raise SystemExit("--evaluate cần --checkpoint")
        calibration = None
        val_report, val_logits, _, val_labels = evaluate(
            grader, manifest, splits, args.image_root, "val", args.batch_size, tta, None
        )
        calibration = M.Calibration.fit(
            val_logits, val_labels, ordinal=True,
            decoder=grader.probabilities, nll=grader.nll,
        )
        print(f"\nhiệu chuẩn trên VAL: T = {calibration.temperature:.3f}, "
              f"ngưỡng ordinal = {[round(t, 3) for t in calibration.thresholds]}")
        report, *_ = evaluate(
            grader, manifest, splits, args.image_root, args.split,
            args.batch_size, tta, calibration
        )
        print_report(report, grader)
        payload["evaluation"] = report

    for path in args.images:
        rgb = read_image(path)
        statistics = image_statistics(rgb)
        features, probability, per_model, band = None, None, None, None
        if ensemble is not None:
            probability, per_model, features = ensemble.run(rgb, tta=tta)
        verdict = gate.judge(statistics, features, probability)
        if verdict.is_fundus and probability is not None:
            band = band_report(probability, ensemble.decode(probability), per_model)
        print_single(path, verdict, statistics, gate, band, args.truth)
        payload.setdefault("images", []).append(
            {"path": str(path), "is_fundus": verdict.is_fundus,
             "stat_score": verdict.stat_score,
             "feature_scores": verdict.feature_scores,
             "grade": None if band is None else band}
        )

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float),
                             encoding="utf-8")
        print(f"\nJSON -> {args.json}")
    if args.batch:
        if ensemble is None:
            raise SystemExit("--batch cần --checkpoint")
        paths = list_images(args.batch)
        if not paths:
            raise SystemExit(f"không có ảnh nào trong {args.batch}")
        print(f"chấm {len(paths):,} ảnh trong {args.batch}")
        bar = None if tqdm is None else tqdm(
            total=len(paths), desc="chấm", mininterval=1.0, dynamic_ncols=True
        )

        def tick(done: int, total: int) -> None:
            if bar is not None:
                bar.update(done - bar.n)

        rows = grade_many(
            ensemble, gate, paths, tta=args.tta,
            truths=load_truths(args.truth_csv),
            batch_size=args.batch_size, on_progress=tick,
        )
        if bar is not None:
            bar.close()
        print(batch_summary(rows))
        target = args.csv or (args.batch / "ket_qua.csv")
        print(f"CSV -> {write_csv(rows, target)}")
        payload["batch"] = {"folder": str(args.batch), "rows": len(rows)}

    if not args.images and not args.evaluate and not args.fit_gate and not args.batch:
        print("không có gì để làm: truyền đường dẫn ảnh, hoặc --evaluate, hoặc --fit-gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

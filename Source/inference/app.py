"""Local app: drop in a fundus image, see what the model sees and what it decides.

The logic lives in ``demo.py`` and ``ood.py`` and is imported, not reimplemented.
This file is a view; if a number here disagrees with the CLI, this file is wrong.

Two display choices are deliberate.

**The detected retina box is drawn on the original, next to the 448 crop.**  A
grade is only as good as the crop it was computed on, and a bad crop is invisible
in any table of numbers.  This project nearly shipped an Otsu fallback that cut
away 59 % of the retina in the worst case; a picture of the box would have shown
that in one glance.

**Both models' distributions are shown, not just the blend.**  On a real case
from this checkpoint pair, convnext said grade 3 at 0.916 and swin said grade 4
at 0.813 -- opposite answers, both confident.  The blend hides that; the
disagreement is the useful part, because it is what turns into a low-confidence
blend that a human should look at.

The gate runs first and can stop everything.  When it refuses there is no grade
on screen at all -- not a greyed-out one, not a low-confidence one.  A model
handed a photograph of a cat returned 0.989 confidence for a DR grade, so
"confidence" is not a safety net and must not be displayed as if it were.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "preprocessing"))
sys.path.insert(0, str(_HERE.parent / "training"))

import metrics as M  # noqa: E402
from demo import (  # noqa: E402
    Ensemble, Grader, band_report, batch_summary, compare_to_truth,
    grade_many, list_images, load_truths, write_csv,
)
from ood import FundusGate, image_statistics, worst_statistics  # noqa: E402
from retina import retina_box, standardise  # noqa: E402

PROJECT_ROOT = _HERE.parents[1]
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "results" / "baseline_no_mfiddr" / "checkpoints"
DEFAULT_GATE = PROJECT_ROOT / "checkpoints" / "fundus_gate.json"

BAND_COLOUR = {
    "theo dõi": "#2e7d32",
    "chuyển tuyến": "#ef6c00",
    "chuyển tuyến gấp": "#c62828",
}


# --------------------------------------------------------------------------- #
# Loading, once
# --------------------------------------------------------------------------- #


class Backend:
    """Model and gate held in memory for the life of the process.

    Loading costs ~5 s (1.8 s of imports, 3.2 s of weights) and ~1.5 GB of VRAM.
    Doing that per request would make every click feel broken; doing it once
    makes each image take 0.39 s.
    """

    def __init__(self, checkpoints: Path, gate_path: Path) -> None:
        paths = sorted(checkpoints.glob("*_best.pt"))
        if not paths:
            raise SystemExit(f"không có checkpoint *_best.pt nào trong {checkpoints}")
        self.graders = [Grader(p) for p in paths]
        blend = checkpoints / "ensemble_tta.json"
        if blend.is_file() and len(self.graders) > 1:
            self.ensemble = Ensemble.from_json(self.graders, blend)
            self.calibrated = True
        else:
            self.ensemble = Ensemble(self.graders)
            self.calibrated = False

        if not gate_path.is_file():
            raise SystemExit(
                f"không có cổng lọc tại {gate_path}. Chạy trước:\n"
                f"  python Source/inference/demo.py --fit-gate "
                f"--checkpoint {' '.join(str(p) for p in paths)}"
            )
        self.gate = FundusGate.load(gate_path)
        missing = [g.name for g in self.graders if g.name not in self.gate.features]
        if missing:
            raise SystemExit(
                f"cổng lọc chưa hiệu chuẩn cho {missing}. Chạy lại --fit-gate "
                "với đủ các checkpoint."
            )

    def describe(self) -> str:
        line = self.ensemble.describe()
        if not self.calibrated:
            line += "  ⚠ chưa có ensemble_tta.json — trọng số đều, T=1, không ngưỡng ordinal"
        return line


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #


def with_retina_box(rgb: np.ndarray, max_side: int = 520) -> np.ndarray:
    """The original, with the box the crop will use drawn on it."""

    box = retina_box(rgb)
    canvas = rgb.copy()
    if box is not None:
        left, top, right, bottom = box
        thickness = max(2, int(round(max(canvas.shape[:2]) / 250)))
        cv2.rectangle(canvas, (left, top), (right, bottom), (0, 220, 90), thickness)
    scale = min(1.0, max_side / max(canvas.shape[:2]))
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas


def gate_html(verdict, statistics: Dict[str, float], gate: FundusGate) -> str:
    rows = [
        f"<tr><td>thống kê ảnh</td><td align=right>{verdict.stat_score:.2f}</td>"
        f"<td align=right>{verdict.stat_threshold:.2f}</td>"
        f"<td>{'VƯỢT' if verdict.stat_flag else 'đạt'}</td></tr>"
    ]
    for name, score in verdict.feature_scores.items():
        limit = verdict.feature_thresholds[name]
        rows.append(
            f"<tr><td>đặc trưng · {name}</td><td align=right>{score:.2f}</td>"
            f"<td align=right>{limit:.2f}</td>"
            f"<td>{'VƯỢT' if score > limit else 'đạt'}</td></tr>"
        )
    table = (
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th align=left>tầng</th><th align=right>điểm</th>"
        "<th align=right>ngưỡng</th><th></th></tr>" + "".join(rows) + "</table>"
    )
    if verdict.is_fundus:
        return (
            "<div style='padding:10px;border-left:5px solid #2e7d32'>"
            "<b>Là ảnh đáy mắt</b>" + table + "</div>"
        )
    worst = "".join(
        f"<li><code>{n}</code> = {v:.3f} ({s:+.1f} độ lệch chuẩn)</li>"
        for n, v, s in worst_statistics(statistics, gate)
    )
    return (
        "<div style='padding:10px;border-left:5px solid #c62828'>"
        "<b>TỪ CHỐI — không chấm điểm</b><br>"
        + "<br>".join(verdict.reasons)
        + table
        + f"<b>Thống kê lệch xa nhất:</b><ul>{worst}</ul>"
        "Ảnh không nằm trong phân bố model được huấn luyện, nên mọi bậc đưa ra "
        "đều vô nghĩa.</div>"
    )


def boundary_html(band: Dict) -> str:
    blocks = []
    for b in band["boundaries"]:
        p = b["p_above"]
        uncertain = b["uncertain"]
        colour = "#f9a825" if uncertain else ("#c62828" if p >= 0.5 else "#2e7d32")
        verdict = "CÓ" if p >= 0.5 else "không"
        note = " — <b>cần người đọc lại</b>" if uncertain else ""
        blocks.append(
            f"<div style='margin:8px 0'>"
            f"<div>P(bậc ≥ {b['cut']}) = <b>{p:.3f}</b> → {b['label']}: "
            f"<b>{verdict}</b>{note}</div>"
            f"<div style='position:relative;height:16px;background:#eee;border-radius:3px'>"
            # The 0.15-0.85 band is where a wrong call flips the patient's path.
            f"<div style='position:absolute;left:15%;width:70%;height:100%;"
            f"background:#fff3cd'></div>"
            f"<div style='position:absolute;left:0;width:{100*p:.1f}%;height:100%;"
            f"background:{colour};opacity:.75;border-radius:3px'></div>"
            f"<div style='position:absolute;left:50%;width:2px;height:100%;"
            f"background:#333'></div></div></div>"
        )
    return "".join(blocks)


def grade_html(band: Dict) -> str:
    colour = BAND_COLOUR.get(band["band"], "#555")
    extra = ""
    if band.get("decoded_by_thresholds"):
        argmax = int(np.argmax(band["probabilities"]))
        extra = (
            f"<div style='font-size:.9em;opacity:.8'>giải mã bằng ngưỡng ordinal "
            f"(argmax thô cho bậc {argmax})</div>"
        )
    return (
        f"<div style='padding:12px;border-left:6px solid {colour}'>"
        f"<div style='font-size:1.5em'><b>Bậc {band['grade']}</b> — {band['name']}</div>"
        f"<div style='color:{colour};font-weight:bold;text-transform:uppercase'>"
        f"{band['band']} — {band['action']}</div>"
        f"<div>tin cậy {band['confidence']:.3f} · bậc kỳ vọng "
        f"{band['expected_grade']:.2f}</div>{extra}</div>"
    )


def per_model_html(band: Dict) -> str:
    per = band.get("per_model") or {}
    if len(per) < 2:
        return ""
    picks = {name: int(np.argmax(p)) for name, p in per.items()}
    disagree = len(set(picks.values())) > 1
    rows = "".join(
        f"<tr><td><code>{name}</code></td><td><b>bậc {picks[name]}</b></td>"
        + "".join(f"<td align=right>{v:.3f}</td>" for v in p)
        + "</tr>"
        for name, p in per.items()
    )
    banner = (
        "<div style='color:#ef6c00'><b>Hai model bất đồng</b> — độ bất định này "
        "là thông tin, không phải nhiễu.</div>" if disagree else ""
    )
    return (
        f"<div style='padding:10px'>{banner}"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th align=left>model</th><th align=left>chọn</th>"
        + "".join(f"<th align=right>{c}</th>" for c in range(5))
        + f"</tr>{rows}</table></div>"
    )


# --------------------------------------------------------------------------- #
# The one function the UI calls
# --------------------------------------------------------------------------- #


def analyse(backend: Backend, rgb: Optional[np.ndarray], truth: str):
    if rgb is None:
        return None, None, "<i>Chưa có ảnh.</i>", {}, "", "", "", ""
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    rgb = np.ascontiguousarray(rgb[..., :3].astype(np.uint8))

    original = with_retina_box(rgb)
    crop = standardise(rgb, backend.graders[0].size)

    statistics = image_statistics(rgb)
    probability, per_model, features = backend.ensemble.run(rgb)
    verdict = backend.gate.judge(statistics, features, probability)

    gate_block = gate_html(verdict, statistics, backend.gate)
    if not verdict.is_fundus:
        # No grade on screen at all -- see the module docstring.
        return original, crop, gate_block, {}, "", "", "", ""

    band = band_report(probability, backend.ensemble.decode(probability), per_model)
    labels = {f"bậc {c} — {M.GRADE_NAMES[c]}": float(p)
              for c, p in enumerate(band["probabilities"])}

    outcome = ""
    if truth and truth != "(không biết)":
        result = compare_to_truth(band["grade"], int(truth[0]))
        colour = {"ĐÚNG": "#2e7d32", "SAI, KHÔNG ĐỔI XỬ TRÍ": "#558b2f",
                  "ĐÁNG LƯU Ý": "#ef6c00", "CẢNH BÁO": "#ef6c00",
                  "NGHIÊM TRỌNG": "#c62828"}.get(result["severity"], "#555")
        outcome = (
            f"<div style='padding:10px;border-left:6px solid {colour}'>"
            f"<b>{result['severity']}</b><br>{result['message']}</div>"
        )
    return (original, crop, gate_block, labels, grade_html(band),
            boundary_html(band), per_model_html(band), outcome)


def scan_folder(backend: Backend, folder: str, truth_csv: str, progress=None):
    """Grade a whole folder and hand back a table plus a CSV to download.

    Reuses ``grade_many`` rather than looping over ``analyse``: the batched path
    keeps the GPU busy, which is the difference between roughly 16 and 32 images
    a second at 448, and a folder of a few thousand is where that stops being a
    rounding error.
    """

    import pandas as pd

    if not folder or not Path(folder).is_dir():
        return None, "<i>Chưa chọn thư mục hợp lệ.</i>", None
    paths = list_images(Path(folder))
    if not paths:
        return None, f"<i>Không có ảnh nào trong <code>{folder}</code>.</i>", None

    truths = load_truths(Path(truth_csv)) if truth_csv and Path(truth_csv).is_file() else {}

    def tick(done: int, total: int) -> None:
        if progress is not None:
            progress(done / total, desc=f"{done:,}/{total:,} ảnh")

    rows = grade_many(backend.ensemble, backend.gate, paths, truths=truths,
                      batch_size=8, on_progress=tick)

    out = Path(folder) / "ket_qua.csv"
    write_csv(rows, out)

    frame = pd.DataFrame(rows)
    preferred = [c for c in ("file", "is_fundus", "grade", "band", "confidence",
                             "p_ge2", "p_ge3", "needs_review", "models_disagree",
                             "truth", "severity", "note") if c in frame.columns]
    frame = frame[preferred + [c for c in frame.columns if c not in preferred]]

    summary = batch_summary(rows).replace(chr(10), "<br>")
    # Surface the rows a human should look at first, not the alphabetical order.
    flagged = sum(1 for r in rows if r.get("needs_review"))
    banner = ""
    if flagged:
        banner = (f"<div style='color:#ef6c00'><b>{flagged:,} ảnh cần người đọc lại</b> "
                  "— lọc cột <code>needs_review = 1</code> trong CSV.</div>")
    return frame, f"<div style='padding:10px'>{banner}{summary}</div>", str(out)


def build(backend: Backend):
    import gradio as gr

    with gr.Blocks(title="Chấm mức độ bệnh võng mạc đái tháo đường") as ui:
        gr.Markdown(
            "# Chấm mức độ DR — 5 bậc\n"
            f"`{backend.describe()}`\n\n"
            "Cổng lọc chạy **trước**. Nếu ảnh không phải đáy mắt thì không có bậc nào "
            "được đưa ra — độ tin cậy không phải lưới an toàn: một model từng trả về "
            "0,989 cho ảnh con mèo."
        )
        with gr.Tab("Một ảnh"), gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Thả ảnh vào đây", type="numpy", height=320)
                truth = gr.Dropdown(
                    ["(không biết)"] + [f"{c} — {M.GRADE_NAMES[c]}" for c in range(5)],
                    value="(không biết)", label="Nhãn thật (nếu có)",
                )
                run = gr.Button("Chấm điểm", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    view_original = gr.Image(label="Ảnh gốc + khung võng mạc dò được",
                                             height=300)
                    view_crop = gr.Image(label="448×448 — thứ model thật sự thấy",
                                         height=300)
        gate_out = gr.HTML(label="Cổng lọc")
        grade_out = gr.HTML()
        probs_out = gr.Label(label="Phân bố xác suất", num_top_classes=5)
        bounds_out = gr.HTML()
        models_out = gr.HTML()
        truth_out = gr.HTML()

        def run_one(rgb, t):
            (original, crop, gate_block, labels, grade,
             bounds, models, outcome) = analyse(backend, rgb, t)
            return original, crop, gate_block, grade, labels, bounds, models, outcome

        run.click(
            run_one, [image, truth],
            [view_original, view_crop, gate_out, grade_out, probs_out,
             bounds_out, models_out, truth_out],
        )
        image.change(
            run_one, [image, truth],
            [view_original, view_crop, gate_out, grade_out, probs_out,
             bounds_out, models_out, truth_out],
        )

        with gr.Tab("Cả thư mục"):
            gr.Markdown(
                "Chấm mọi ảnh trong thư mục rồi ghi `ket_qua.csv` vào chính thư mục đó. "
                "Khoảng **0,2 s/ảnh** — 1.000 ảnh mất chừng 3 phút."
            )
            with gr.Row():
                folder = gr.Textbox(label="Đường dẫn thư mục",
                                    placeholder=r"D:\Fundus_Project\demo_images")
                truth_csv = gr.Textbox(
                    label="CSV nhãn thật (tuỳ chọn)",
                    placeholder="cột file/image_id + class_label/grade/level",
                )
            scan = gr.Button("Chấm cả thư mục", variant="primary")
            summary_out = gr.HTML()
            table_out = gr.Dataframe(label="Kết quả", wrap=True, max_height=420)
            csv_out = gr.File(label="Tải CSV")

            def run_folder(f, t, progress=gr.Progress()):
                frame, summary, path = scan_folder(backend, f, t, progress)
                return summary, frame, path

            scan.click(run_folder, [folder, truth_csv],
                       [summary_out, table_out, csv_out])
    return ui


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="App chấm DR chạy local.")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    backend = Backend(args.checkpoints, args.gate)
    print(f"pipeline: {backend.describe()}")
    build(backend).launch(server_port=args.port, share=args.share, inbrowser=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "preprocessing"))
sys.path.insert(0, str(_HERE.parent / "training"))

import metrics as M
from demo import (
    Ensemble, batch_summary, compare_to_truth,
    grade_many, list_images, load_truths, read_image, write_csv,
)
from ood import FundusGate, image_statistics, worst_statistics
from retina import retina_box, standardise

PROJECT_ROOT = _HERE.parents[1]
DEFAULT_DECISION = PROJECT_ROOT / "checkpoints" / "demo_v345" / "decision.json"

BAND_COLOUR = {
    "theo dõi": "#2e7d32",
    "chuyển tuyến": "#ef6c00",
    "chuyển tuyến gấp": "#c62828",
}

class Backend:
    def __init__(self, decision: Path, gate_path: Path) -> None:
        if not decision.is_file():
            raise SystemExit(f"không có bundle mô hình V3+V4+V5 tại {decision}")
        self.ensemble = Ensemble.from_decision(decision)
        self.graders = self.ensemble.graders
        if not gate_path.is_file():
            raise SystemExit(
                f"không có cổng lọc tại {gate_path}. Chạy trước:\n"
                f"  python Source/inference/demo.py --fit-gate --decision {decision} --gate {gate_path}"
            )
        self.gate = FundusGate.load(gate_path)
        missing = [g.name for g in self.graders if g.name not in self.gate.features]
        if missing:
            raise SystemExit(
                f"cổng lọc chưa hiệu chuẩn cho {missing}. Chạy lại --fit-gate "
                "với đủ các checkpoint."
            )

    def describe(self) -> str:
        return self.ensemble.describe()

def with_retina_box(rgb: np.ndarray, max_side: int = 520) -> np.ndarray:
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
        called = b["called"]
        low, high, cut = b["review_low"], b["review_high"], b["threshold"]
        colour = "#f9a825" if uncertain else ("#c62828" if called else "#2e7d32")
        verdict = "CÓ" if called else "không"
        note = " — <b>cần người đọc lại</b>" if uncertain else ""
        blocks.append(
            f"<div style='margin:8px 0'>"
            f"<div>P(bậc ≥ {b['cut']}) = <b>{p:.3f}</b> → {b['label']}: "
            f"<b>{verdict}</b>{note} "
            f"<span style='opacity:.7'>(ngưỡng {cut:.3f}, vùng đọc lại {low:.3f}–{high:.3f})</span></div>"
            f"<div style='position:relative;height:16px;background:#eee;border-radius:3px'>"
            f"<div style='position:absolute;left:{100*low:.1f}%;width:{100*(high-low):.1f}%;height:100%;"
            f"background:#fff3cd'></div>"
            f"<div style='position:absolute;left:0;width:{100*p:.1f}%;height:100%;"
            f"background:{colour};opacity:.75;border-radius:3px'></div>"
            f"<div style='position:absolute;left:{100*cut:.1f}%;width:2px;height:100%;"
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
    if band.get("rule"):
        extra = ""
    if band.get("needs_review"):
        review = ("<div class='review-banner' style='margin-top:8px;padding:6px 10px;background:#fff3cd;color:#000000;border-radius:4px'>"
                  "<b>⚠ CẦN BÁC SĨ ĐỌC LẠI</b> — xác suất nằm trong vùng lưng chừng của một ranh "
                  "giới xử trí. Bậc trên là gợi ý, không phải kết luận.</div>")
    else:
        review = ("<div class='review-banner' style='margin-top:8px;padding:6px 10px;background:#e8f5e9;color:#000000;border-radius:4px'>"
                  "<b>✓ TỰ KẾT LUẬN</b> — không ranh giới xử trí nào ở vùng lưng chừng.</div>")
    return (
        f"<div style='padding:12px;border-left:6px solid {colour}'>"
        f"<div style='font-size:1.5em'><b>Bậc {band['grade']}</b> — {band['name']}</div>"
        f"<div style='color:{colour};font-weight:bold;text-transform:uppercase'>"
        f"{band['band']} — {band['action']}</div>"
        f"<div>tin cậy {band['confidence']:.3f} · bậc kỳ vọng "
        f"{band['expected_grade']:.2f}</div>{extra}{review}</div>"
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
        "<div style='color:#ef6c00'><b>Các model bất đồng</b> — độ bất định này "
        "là thông tin, không phải nhiễu.</div>" if disagree else ""
    )
    return (
        f"<div style='padding:10px'>{banner}"
        "<table style='width:100%;border-collapse:collapse'>"
        "<tr><th align=left>model</th><th align=left>chọn</th>"
        + "".join(f"<th align=right>{c}</th>" for c in range(5))
        + f"</tr>{rows}</table></div>"
    )

def as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(image[..., :3].astype(np.uint8))


def analyse(backend: Backend, rgb: Optional[np.ndarray], truth: str, extra_views=None):
    if rgb is None:
        return None, None, "<i>Chưa có ảnh.</i>", {}, "", "", "", ""
    images = [as_rgb(rgb)]
    names = ["ảnh chính"]
    for item in extra_views or []:
        path = Path(getattr(item, "name", item))
        images.append(read_image(path))
        names.append(path.name)

    original = with_retina_box(images[0])
    crop = standardise(images[0], backend.graders[0].size)

    probability, per_model, features = backend.ensemble.run_views(images)
    blocks, refused = [], False
    for index, (name, image) in enumerate(zip(names, images)):
        statistics = image_statistics(image)
        verdict = backend.gate.judge(statistics, {n: f[index] for n, f in features.items()}, probability)
        block = gate_html(verdict, statistics, backend.gate)
        if len(images) > 1:
            block = f"<div><b>{name}</b></div>{block}"
        blocks.append(block)
        refused |= not verdict.is_fundus

    gate_block = "".join(blocks)
    if refused:
        return original, crop, gate_block, {}, "", "", "", ""

    band = backend.ensemble.band(probability, per_model)
    if len(images) > 1:
        band["name"] += f" · gộp {len(images)} ảnh của cùng mắt"
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
                extra = gr.File(label="Ảnh khác của CÙNG mắt (tuỳ chọn, gộp thành một mắt)",
                                file_count="multiple", file_types=["image"])
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

        def run_one(rgb, t, views):
            (original, crop, gate_block, labels, grade,
             bounds, models, outcome) = analyse(backend, rgb, t, views)
            return original, crop, gate_block, grade, labels, bounds, models, outcome

        run.click(
            run_one, [image, truth, extra],
            [view_original, view_crop, gate_out, grade_out, probs_out,
             bounds_out, models_out, truth_out],
        )
        image.change(
            run_one, [image, truth, extra],
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
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION,
                        help=f"bundle ensemble V3+V4+V5; mặc định {DEFAULT_DECISION}")
    parser.add_argument("--gate", type=Path, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    decision = args.decision
    gate = args.gate or decision.parent / "fundus_gate.json"
    backend = Backend(decision, gate)
    print(f"pipeline: {backend.describe()}")
    css = ".review-banner, .review-banner * { color: #000000 !important; }"
    build(backend).launch(server_port=args.port, share=args.share, inbrowser=True, css=css)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

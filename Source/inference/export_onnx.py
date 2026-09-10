"""Export a trained checkpoint to ONNX, verified against the PyTorch original.

Two things this has to get right that a one-line ``torch.onnx.export`` does not.

**Two outputs, not one.**  The OOD gate scores the *penultimate features*, not
the logits, so a graph that emits only logits cannot answer "is this a fundus
image" -- and a grader that cannot refuse is the thing ``ood.py`` exists to
prevent.  The wrapper emits both from one forward pass.

**Verification is the point of the script.**  An export that runs is not an
export that agrees: layer-norm epsilon placement, GELU approximations and
window-partition reshapes all have more than one legal lowering.  So every
export is replayed against PyTorch on real images and the maximum absolute
disagreement is printed.  If that number is not small, the file is wrong no
matter how cleanly it converted.

Preprocessing stays in Python.  The retina crop is OpenCV connected components,
which has no ONNX equivalent worth writing; the graph starts at the normalised
448x448 tensor, exactly where ``FundusDataset`` hands off.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "preprocessing"))
sys.path.insert(0, str(_HERE.parent / "training"))

from demo import Grader, read_image, to_tensor  # noqa: E402

PROJECT_ROOT = _HERE.parents[1]


class LogitsAndFeatures(nn.Module):
    """One forward pass, both things the pipeline needs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.head

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        return self.head(features), features


def sample_batch(grader: Grader, manifest: Path, count: int) -> torch.Tensor:
    import csv
    import random

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    random.seed(0)
    picks = random.sample(rows, count)
    return torch.stack(
        [to_tensor(read_image(Path(r["processed_path"])), grader.size) for r in picks]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xuất checkpoint sang ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "Data" / "processed" / "img448" / "manifest.csv",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--check-batch", type=int, default=8)
    parser.add_argument("--fp16", action="store_true", help="cũng ghi bản fp16")
    parser.add_argument("--int8", action="store_true", help="cũng ghi bản int8 tĩnh")
    parser.add_argument("--calibration", type=int, default=96,
                        help="số ảnh thật dùng để hiệu chuẩn dải kích hoạt int8")
    return parser.parse_args()


def report_size(path: Path, label: str, reference: float | None = None) -> float:
    size = path.stat().st_size / 2**20
    line = f"  {label:22} {size:8.1f} MB  -> {path.name}"
    if reference:
        line += f"   ({size / reference:.2f}x)"
    print(line)
    return size


def main() -> int:
    args = parse_args()
    grader = Grader(args.checkpoint, device=torch.device("cpu"))
    name = f"{grader.backbone_name}_{grader.head}"
    output = args.output or args.checkpoint.parent / f"{name}.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"{name} @ {grader.size}  <- {args.checkpoint.name}")

    wrapper = LogitsAndFeatures(grader.model).eval()
    images = sample_batch(grader, args.manifest, args.check_batch)
    with torch.no_grad():
        ref_logits, ref_features = wrapper(images)

    torch.onnx.export(
        wrapper,
        images[:1],
        str(output),
        input_names=["images"],
        output_names=["logits", "features"],
        dynamic_axes={
            "images": {0: "batch"},
            "logits": {0: "batch"},
            "features": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(output)))
    print(f"\nđồ thị hợp lệ (opset {args.opset})")

    sizes = {"fp32": report_size(output, "fp32", None)}

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    got = session.run(None, {"images": images.numpy()})
    for label, reference, actual in (
        ("logits", ref_logits, got[0]),
        ("features", ref_features, got[1]),
    ):
        delta = np.abs(reference.numpy() - actual)
        scale = np.abs(reference.numpy()).max()
        print(f"  {label:10} lệch tối đa {delta.max():.3e}  "
              f"(biên độ {scale:.2f}, tương đối {delta.max()/max(scale,1e-9):.2e})")
    same = (ref_logits.numpy().argmax(1) == got[0].argmax(1)).all()
    print(f"  argmax khớp trên {len(images)} ảnh: {same}")
    if not same:
        raise SystemExit("ONNX cho dự đoán KHÁC PyTorch -- không dùng được")

    # A dynamic batch axis is worth nothing if it does not actually work.
    for batch in (1, 4):
        out = session.run(None, {"images": images[:batch].numpy()})
        assert out[0].shape[0] == batch, "trục batch động không hoạt động"
    print(f"  trục batch động: OK (thử batch 1 và 4)")

    started = time.time()
    for _ in range(3):
        session.run(None, {"images": images[:1].numpy()})
    onnx_ms = (time.time() - started) / 3 * 1000
    started = time.time()
    with torch.no_grad():
        for _ in range(3):
            wrapper(images[:1])
    torch_ms = (time.time() - started) / 3 * 1000
    print(f"\n  1 ảnh trên CPU: ONNX {onnx_ms:.0f} ms  |  PyTorch {torch_ms:.0f} ms"
          f"   ({torch_ms/onnx_ms:.2f}x)")

    if args.fp16:
        from onnxconverter_common import float16

        half = output.with_name(f"{name}_fp16.onnx")
        onnx.save(float16.convert_float_to_float16(onnx.load(str(output))), str(half))
        report_size(half, "fp16", sizes["fp32"])

    if args.int8:
        # Static, not dynamic.  Dynamic quantisation lowers convolutions to
        # ConvInteger, which the CPU execution provider has no kernel for -- the
        # file converts, weighs a quarter as much, and then refuses to load:
        #   NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10)
        # Static quantisation records activation ranges from real images first,
        # so it can emit ordinary QDQ nodes that every provider implements.
        from onnxruntime.quantization import (
            CalibrationDataReader, QuantFormat, QuantType, quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process

        calibration = sample_batch(grader, args.manifest, args.calibration)

        class Reader(CalibrationDataReader):
            """Real fundus images, one at a time -- the ranges must come from
            the distribution the model will actually see."""

            def __init__(self, batch: torch.Tensor) -> None:
                self.items = iter([{"images": x.unsqueeze(0).numpy()} for x in batch])

            def get_next(self):
                return next(self.items, None)

        prepared = output.with_name(f"{name}_prepared.onnx")
        quant_pre_process(str(output), str(prepared), skip_symbolic_shape=False)
        quant = output.with_name(f"{name}_int8.onnx")
        print(f"\n  hiệu chuẩn int8 trên {args.calibration} ảnh thật...")
        quantize_static(
            str(prepared), str(quant), Reader(calibration),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        prepared.unlink(missing_ok=True)
        report_size(quant, "int8 (tĩnh, QDQ)", sizes["fp32"])
        try:
            qs = ort.InferenceSession(str(quant), providers=["CPUExecutionProvider"])
            qout = qs.run(None, {"images": images.numpy()})
            agree = (ref_logits.numpy().argmax(1) == qout[0].argmax(1)).mean()
            delta = np.abs(ref_logits.numpy() - qout[0]).max()
            started = time.time()
            for _ in range(3):
                qs.run(None, {"images": images[:1].numpy()})
            print(f"  int8 so với fp32: argmax khớp {100*agree:.0f}%  "
                  f"lệch logit tối đa {delta:.3f}  "
                  f"| 1 ảnh {(time.time()-started)/3*1000:.0f} ms")
        except Exception as exc:  # noqa: BLE001 - reported, not hidden
            print(f"  int8 KHÔNG chạy được: {type(exc).__name__}: {str(exc)[:120]}")

    print(f"\nxong -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

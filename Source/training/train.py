"""Train one backbone on the 448 build, then calibrate it on validation.

Choices here that are answers to specific failures, not defaults:

**bf16, not fp16.**  fp16 produced transient NaN losses on ~0.3 % of steps with
EfficientNetV2-S -- non-reproducible, so not a data bug -- and bf16 was clean.
bf16 has fp32's exponent range, so the overflow that fp16's GradScaler exists to
paper over cannot happen.  On a GPU without bf16 the code falls back to fp16
*with* a scaler rather than silently running fp32.

**Checkpoint selection on the composite, not QWK.**  Measured on this project:
epoch 9 scored QWK 0.8516 with grade-3 recall 0.64; epoch 10 scored QWK 0.8584
with grade-3 recall 0.47.  Selecting on QWK takes epoch 10 and discards exactly
what the balanced sampler was added to buy.

**Workers capped and the sharing strategy left alone.**  At 448, 14 workers
raised ``MemoryError``; 6 is the measured ceiling.  And
``set_sharing_strategy('file_system')`` -- a common "fix" for too-many-open-files
-- leaks into /dev/shm, which is not reclaimable page cache: it is a real leak
that ends in an OOM kill several models into a session.

**Calibration is fitted on val and frozen.**  Temperature and the ordinal
thresholds never see test.  Fitting them on test would recreate, in the metrics,
the same leak the split stage exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:  # progress bars are a convenience, never a dependency
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - environment dependent
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402
from data import (  # noqa: E402
    EVAL_AUGMENT,
    TRAIN_AUGMENT,
    AugmentConfig,
    FundusDataset,
    balanced_sampler,
    class_weights,
    describe,
    load_split,
)
from models import (  # noqa: E402
    ModelConfig,
    build_model,
    coral_loss,
    coral_to_probabilities,
    parameter_groups,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def progress(iterable, desc: str, total: Optional[int] = None, enable: bool = True):
    """A bar when tqdm is installed, the bare iterable when it is not.

    ``leave=False`` so a 30-epoch run does not end with 30 dead bars, and
    ``mininterval=2`` so a committed (headless) Kaggle run writes a bounded
    number of lines into its log instead of one per step.
    """

    if tqdm is None or not enable:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=False,
        mininterval=2.0,
        dynamic_ncols=True,
        unit="batch",
    )


@dataclass
class TrainConfig:
    backbone: str = "convnext_tiny"
    head: str = "softmax"
    image_size: int = 448
    epochs: int = 20
    batch_size: int = 8
    accumulate: int = 4  # effective batch = batch_size * accumulate
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.05
    sampler_strength: float = 0.5
    loss_weight_strength: float = 0.0  # sampler already corrects the prior
    colour_mode: str = "none"
    drop_path_rate: float = 0.1
    workers: int = 6  # 14 raised MemoryError at 448; 6 is the measured ceiling
    seed: int = 42
    amp: str = "auto"  # auto | bf16 | fp16 | off
    progress: bool = True


def build_loaders(
    manifest: Path, splits: Path, config: TrainConfig, image_root: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_samples = load_split(manifest, splits, "train", image_root)
    val_samples = load_split(manifest, splits, "val", image_root)
    test_samples = load_split(manifest, splits, "test", image_root)
    print(f"train {describe(train_samples)}")
    print(f"val   {describe(val_samples)}")
    print(f"test  {describe(test_samples)}")

    train_augment = AugmentConfig(**{**asdict(TRAIN_AUGMENT), "colour_mode": config.colour_mode})
    eval_augment = AugmentConfig(**{**asdict(EVAL_AUGMENT), "colour_mode": config.colour_mode})

    generator = torch.Generator().manual_seed(config.seed)
    sampler = (
        balanced_sampler(train_samples, config.sampler_strength, generator)
        if config.sampler_strength > 0
        else None
    )

    common = dict(
        num_workers=config.workers,
        generator=generator,  # fixes the per-worker augmentation seeds
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
        prefetch_factor=2 if config.workers > 0 else None,
    )
    train_loader = DataLoader(
        FundusDataset(train_samples, train_augment, seed=config.seed),
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        FundusDataset(val_samples, eval_augment),
        batch_size=config.batch_size * 2,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        FundusDataset(test_samples, eval_augment),
        batch_size=config.batch_size * 2,
        shuffle=False,
        **common,
    )
    weights = class_weights(train_samples, config.loss_weight_strength)
    return train_loader, val_loader, test_loader, weights


def has_native_bf16() -> bool:
    """True only where bf16 tensor cores and cuDNN bf16 kernels actually exist.

    ``torch.cuda.is_bf16_supported()`` is the obvious call and it is a trap: it
    defaults to ``including_emulation=True``, and the emulation probe merely
    allocates a bf16 tensor, which succeeds on Turing.  A T4 therefore reports
    bf16 as supported, autocast happily runs in bf16, and the first convolution
    dies with

        RuntimeError: GET was unable to find an engine to execute this computation

    because cuDNN ships no bf16 convolution engines below sm_80.  Compute
    capability is the honest test, and every visible device must pass it -- a
    mixed pair would otherwise fail only on the second GPU.
    """

    try:
        count = torch.cuda.device_count()
        return count > 0 and all(
            torch.cuda.get_device_capability(i)[0] >= 8 for i in range(count)
        )
    except Exception:  # noqa: BLE001 - a probe that throws is a "no"
        return False


def resolve_amp(mode: str, device: torch.device) -> Tuple[Optional[torch.dtype], bool]:
    """Return (autocast dtype, needs GradScaler)."""

    if mode == "off" or device.type != "cuda":
        return None, False
    if mode == "fp16":
        return torch.float16, True

    native = has_native_bf16()
    if mode == "bf16":
        if not native:
            names = ", ".join(
                f"{torch.cuda.get_device_name(i)} (sm_{torch.cuda.get_device_capability(i)[0]}"
                f"{torch.cuda.get_device_capability(i)[1]})"
                for i in range(torch.cuda.device_count())
            )
            raise SystemExit(
                f"--amp bf16 asked for, but this GPU has no bf16 convolution "
                f"kernels: {names}.  Real bf16 needs sm_80 (Ampere) or newer; "
                f"it would fail on the first conv.  Use --amp fp16 or --amp auto."
            )
        return torch.bfloat16, False

    if not native:
        print(
            "[INFO] no native bf16 on this GPU (needs sm_80+); using fp16 + "
            "GradScaler.  Non-finite batches are dropped and counted per epoch."
        )
        return torch.float16, True
    return torch.bfloat16, False


@torch.no_grad()
def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dtype: Optional[torch.dtype],
    tta: bool = False,
    desc: str = "eval",
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Logits and labels for a whole split.

    ``no_grad`` rather than ``inference_mode``: inference tensors cannot be
    saved for backward, and these logits are handed to a calibration step that
    may want autograd.  The memory difference is negligible at eval batch sizes.
    """

    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    # TTA costs 8 forward passes per batch, so say so in the label -- otherwise
    # the bar looks stalled next to the plain pass it follows.
    label = f"{desc} (TTA x8)" if tta else desc
    for images, labels in progress(loader, label, len(loader), show):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            if tta:
                # The 8 square symmetries are lossless on these crops, so
                # averaging over them adds views without inventing content.
                accumulated = None
                for k in range(8):
                    view = torch.rot90(images, k % 4, dims=(2, 3))
                    if k >= 4:
                        view = torch.flip(view, dims=(3,))
                    logits = model(view).float()
                    accumulated = logits if accumulated is None else accumulated + logits
                logits = accumulated / 8.0
            else:
                logits = model(images).float()
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def train(
    config: TrainConfig,
    manifest: Path,
    splits: Path,
    output_dir: Path,
    image_root: Optional[Path] = None,
) -> Dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype, needs_scaler = resolve_amp(config.amp, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, weights = build_loaders(
        manifest, splits, config, image_root
    )
    model = build_model(
        ModelConfig(
            backbone=config.backbone,
            image_size=config.image_size,
            head=config.head,
            drop_path_rate=config.drop_path_rate,
        )
    ).to(device)
    weights = weights.to(device)
    print(f"{config.backbone}/{config.head} on {device} | amp={dtype} | "
          f"effective batch {config.batch_size * config.accumulate}")

    if config.head == "coral":
        def decode(logits: np.ndarray) -> np.ndarray:
            return coral_to_probabilities(
                torch.as_tensor(logits, dtype=torch.float32)
            ).numpy()

        def nll(logits: np.ndarray, labels: np.ndarray) -> float:
            return float(
                coral_loss(
                    torch.as_tensor(logits, dtype=torch.float32),
                    torch.as_tensor(labels, dtype=torch.long),
                )
            )
    else:
        decode, nll = M.softmax_probabilities, M.softmax_nll

    optimizer = torch.optim.AdamW(
        parameter_groups(model, config.weight_decay), lr=config.learning_rate
    )
    steps_per_epoch = max(1, len(train_loader) // config.accumulate)
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(steps_per_epoch * config.warmup_epochs)

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    scaler = torch.amp.GradScaler(device.type, enabled=needs_scaler)

    history: List[Dict] = []
    best_score, best_epoch = -np.inf, -1
    best_path = output_dir / f"{config.backbone}_{config.head}_best.pt"

    for epoch in range(config.epochs):
        model.train()
        started = time.time()
        running, seen, skipped = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        bar = progress(
            train_loader,
            f"{config.backbone} train {epoch + 1}/{config.epochs}",
            len(train_loader),
            config.progress,
        )
        for step, (images, labels) in enumerate(bar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(images)
                if config.head == "coral":
                    loss = coral_loss(logits.float(), labels, weights)
                else:
                    loss = F.cross_entropy(
                        logits.float(),
                        labels,
                        weight=weights,
                        label_smoothing=config.label_smoothing,
                    )
            # fp16 produced a transient non-finite loss on ~0.3 % of steps here,
            # never reproducibly and never on bf16.  Back-propagating one poisons
            # every parameter at once, so the batch is dropped instead: the
            # gradients already accumulated stay valid and the run continues.
            # A rising count means a real problem, not a flake -- so it is
            # printed rather than swallowed.
            if not torch.isfinite(loss):
                skipped += 1
                continue
            scaler.scale(loss / config.accumulate).backward()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

            if (step + 1) % config.accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # After the optimiser step, never before: stepping the scheduler
                # first shifts the whole schedule one step early.
                scheduler.step()

                if seen and hasattr(bar, "set_postfix"):
                    bar.set_postfix(
                        loss=f"{running / seen:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        **({"skip": skipped} if skipped else {}),
                        refresh=False,
                    )

        logits, labels = collect_logits(
            model, val_loader, device, dtype, desc="val", show=config.progress
        )
        # Both of these must go through the head.  CORAL emits 4 cumulative
        # logits, not 5 class scores, so a bare softmax + cross_entropy here
        # raises "Target 4 is out of bounds" -- and would quietly mis-rank the
        # epochs if the label set happened to be smaller.
        predictions = decode(logits).argmax(axis=1)
        values = M.summarise(labels, predictions)
        val_loss = nll(logits, labels)

        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            "skipped_batches": skipped,
            **values,
        }
        history.append(record)
        print(
            f"epoch {epoch + 1}/{config.epochs} | loss {record['train_loss']:.4f} "
            f"| val_loss {val_loss:.4f} | {M.format_metrics(values)} "
            f"| {record['seconds']:.0f}s"
            + (f" | {skipped} non-finite batches skipped" if skipped else "")
        )

        if values["composite"] > best_score:
            best_score, best_epoch = values["composite"], epoch + 1
            # Weights only.  `last.pt` with AdamW state would be 3x this file,
            # and nothing downstream needs the optimiser.
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch + 1,
                    "val": values,
                },
                best_path,
            )

    print(f"best epoch {best_epoch} (composite {best_score:.4f}); reloading it")
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])

    results: Dict[str, Dict] = {}
    for name, loader in (("val", val_loader), ("test", test_loader)):
        for tta in (False, True):
            logits, labels = collect_logits(
                model, loader, device, dtype, tta=tta, desc=name, show=config.progress
            )
            key = f"{name}_tta" if tta else name
            np.savez_compressed(
                output_dir / f"logits_{config.backbone}_{config.head}_{key}.npz",
                logits=logits,
                labels=labels,
            )
            results[key] = {"argmax": M.summarise(labels, decode(logits).argmax(axis=1))}

    # Calibrate on val, apply unchanged to test.
    for suffix in ("", "_tta"):
        val_file = np.load(
            output_dir / f"logits_{config.backbone}_{config.head}_val{suffix}.npz"
        )
        test_file = np.load(
            output_dir / f"logits_{config.backbone}_{config.head}_test{suffix}.npz"
        )
        calibration = M.Calibration.fit(
            val_file["logits"], val_file["labels"], ordinal=True, decoder=decode, nll=nll
        )
        report = calibration.report(test_file["logits"], test_file["labels"])
        results[f"test{suffix}"]["calibrated"] = M.summarise(
            test_file["labels"], calibration.predict(test_file["logits"])
        )
        results[f"test{suffix}"]["calibration"] = report

    summary = {
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_val_composite": best_score,
        "history": history,
        "results": results,
    }
    (output_dir / f"summary_{config.backbone}_{config.head}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    for key, value in results.items():
        line = M.format_metrics(value["argmax"])
        print(f"{key:9} argmax     {line}")
        if "calibrated" in value:
            print(f"{key:9} calibrated {M.format_metrics(value['calibrated'])}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one backbone on the 448 build.")
    defaults = TrainConfig()
    parser.add_argument("--backbone", default=defaults.backbone)
    parser.add_argument("--head", default=defaults.head, choices=["softmax", "coral"])
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--accumulate", type=int, default=defaults.accumulate)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--sampler-strength", type=float, default=defaults.sampler_strength)
    parser.add_argument(
        "--loss-weight-strength", type=float, default=defaults.loss_weight_strength
    )
    parser.add_argument("--colour-mode", default=defaults.colour_mode,
                        choices=["none", "ben_graham", "clahe"])
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--amp", default=defaults.amp, choices=["auto", "bf16", "fp16", "off"])
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument(
        "--data", type=Path, default=PROJECT_ROOT / "Data" / "processed" / "img448"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="re-root the manifest's image paths here (for Kaggle or any move)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        backbone=args.backbone,
        head=args.head,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate=args.accumulate,
        learning_rate=args.learning_rate,
        sampler_strength=args.sampler_strength,
        loss_weight_strength=args.loss_weight_strength,
        colour_mode=args.colour_mode,
        workers=args.workers,
        seed=args.seed,
        amp=args.amp,
        progress=args.progress,
    )
    train(
        config,
        args.data / "manifest.csv",
        args.data / "splits.csv",
        args.output_dir,
        args.image_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

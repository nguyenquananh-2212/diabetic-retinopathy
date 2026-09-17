from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M
from data import (
    AUGMENT_PRESETS,
    EVAL_AUGMENT,
    AugmentConfig,
    EyeBatchSampler,
    FundusDataset,
    balanced_sampler,
    class_weights,
    eye_bag_weights,
    eye_bags,
    describe,
    load_split,
)
from models import (
    ModelConfig,
    build_model,
    coral_loss,
    coral_to_probabilities,
    parameter_groups,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def progress(iterable, desc: str, total: Optional[int] = None, enable: bool = True):
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
    accumulate: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smoothing: float = 0.05
    sampler_strength: float = 0.5
    loss_weight_strength: float = 0.0
    colour_mode: str = "none"
    drop_path_rate: float = 0.1
    workers: int = 6
    seed: int = 42
    amp: str = "auto"
    progress: bool = True
    augment: str = "standard"
    source_balance: float = 0.0
    layer_decay: float = 0.0
    keep_top_k: int = 1
    bag_training: bool = False
    select_on: str = "image"
    patience: int = 0
    min_epochs: int = 0
    max_hours: float = 0.0


def bag_loss(logits, labels, bags, weights=None, label_smoothing: float = 0.0):
    probabilities = logits.float().softmax(dim=1)
    cumulative = probabilities.flip(-1).cumsum(-1).flip(-1)[:, 1:]
    _, inverse = torch.unique(bags, return_inverse=True)
    count = int(inverse.max().item()) + 1
    pooled = torch.zeros(count, cumulative.size(1), device=logits.device, dtype=cumulative.dtype)
    pooled = pooled.scatter_reduce(
        0, inverse[:, None].expand_as(cumulative), cumulative, reduce="amax", include_self=False
    )
    ones = torch.ones(count, 1, device=logits.device, dtype=pooled.dtype)
    edges = torch.cat([ones, pooled, torch.zeros_like(ones)], dim=1)
    bag_probabilities = (edges[:, :-1] - edges[:, 1:]).clamp_min(1e-6)
    bag_labels = torch.zeros(count, dtype=labels.dtype, device=labels.device)
    bag_labels = bag_labels.scatter(0, inverse, labels)

    log_probabilities = bag_probabilities.log()
    picked = log_probabilities.gather(1, bag_labels[:, None]).squeeze(1)
    per_bag = -(1.0 - label_smoothing) * picked
    if label_smoothing:
        per_bag = per_bag - (label_smoothing / log_probabilities.size(1)) * log_probabilities.sum(1)
    if weights is not None:
        scale = weights[bag_labels]
        return (per_bag * scale).sum() / scale.sum().clamp_min(1e-8)
    return per_bag.mean()


def eye_scores(samples: Sequence, probabilities: np.ndarray, true: np.ndarray) -> Dict:
    groups = eye_bags(samples)
    labels = np.array([true[group[0]] for group in groups])
    mean = M.pool_views(probabilities, groups, "mean")
    worst = M.pool_views(probabilities, groups, "max")
    return {"eyes": len(groups),
            "eye_composite_mean": M.composite_score(labels, mean.argmax(axis=1)),
            "eye_composite_worst": M.composite_score(labels, worst.argmax(axis=1))}


def eye_rule(config: TrainConfig) -> str:
    return "worst" if config.bag_training else "mean"


def selection_score(config: TrainConfig, values: Dict, eyes: Dict) -> float:
    if config.select_on == "image":
        return values["composite"]
    return eyes[f"eye_composite_{eye_rule(config)}"]


def optimizer_groups(model: nn.Module, config: TrainConfig) -> List[Dict]:
    if not config.layer_decay:
        return parameter_groups(model, config.weight_decay)

    try:
        from timm.optim import param_groups_layer_decay
    except ImportError as error:
        raise SystemExit(f"layer_decay needs timm's param_groups_layer_decay: {error}")

    groups = list(param_groups_layer_decay(
        model.backbone, weight_decay=config.weight_decay, layer_decay=config.layer_decay
    ))
    for group in parameter_groups(model.head, config.weight_decay):
        if group["params"]:
            groups.append({**group, "lr_scale": 1.0})
    covered = sum(len(g["params"]) for g in groups)
    total = sum(1 for parameter in model.parameters() if parameter.requires_grad)
    if covered != total:
        raise ValueError(f"layer decay covers {covered} tensors, the model has {total}")
    for group in groups:
        group["lr"] = config.learning_rate * group.get("lr_scale", 1.0)
    return groups


def average_states(states: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    averaged = {}
    for key, value in states[0].items():
        if value.is_floating_point():
            stacked = torch.stack([state[key].float() for state in states])
            averaged[key] = stacked.mean(dim=0).to(value.dtype)
        else:
            averaged[key] = value.clone()
    return averaged


def source_metrics(samples: Sequence, true: np.ndarray, pred: np.ndarray) -> Dict:
    index = np.array([sample.dataset_source for sample in samples])
    out = {}
    for source in sorted(set(index.tolist())):
        mask = index == source
        out[f"composite_{source}"] = M.composite_score(true[mask], pred[mask])
    worst = min(sorted(set(index.tolist())), key=lambda s: out[f"composite_{s}"])
    out["worst_source"] = worst
    out["worst_source_composite"] = out[f"composite_{worst}"]
    endpoint = M.binary_endpoint(true, pred, M.URGENT_FROM)
    out["urgent_sensitivity"] = endpoint["sensitivity"]
    out["urgent_specificity"] = endpoint["specificity"]
    return out


@dataclass
class StopRule:
    patience: int = 0
    min_epochs: int = 0
    max_hours: float = 0.0
    best: float = -np.inf
    since: int = 0

    def before_epoch(self, elapsed: float, durations: Sequence[float]) -> Optional[str]:
        if not self.max_hours or not durations:
            return None
        needed = elapsed + max(durations)
        if needed > self.max_hours * 3600:
            return (f"max_hours: {elapsed / 3600:.2f} h used, the next epoch needs up to "
                    f"{max(durations) / 60:.0f} min, budget {self.max_hours:.2f} h")
        return None

    def after_epoch(self, epoch: int, score: float) -> Optional[str]:
        if score > self.best:
            self.best, self.since = score, 0
        else:
            self.since += 1
        if self.patience and epoch >= self.min_epochs and self.since >= self.patience:
            return f"patience: {self.since} epochs without a better composite"
        return None


def build_loaders(
    manifest: Path, splits: Path, config: TrainConfig, image_root: Optional[Path] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    train_samples = load_split(manifest, splits, "train", image_root)
    val_samples = load_split(manifest, splits, "val", image_root)
    test_samples = load_split(manifest, splits, "test", image_root)
    print(f"train {describe(train_samples)}")
    print(f"val   {describe(val_samples)}")
    print(f"test  {describe(test_samples)}")

    if config.augment not in AUGMENT_PRESETS:
        raise ValueError(f"augment must be one of {sorted(AUGMENT_PRESETS)}, got {config.augment!r}")
    preset = AUGMENT_PRESETS[config.augment]
    train_augment = AugmentConfig(**{**asdict(preset), "colour_mode": config.colour_mode})
    eval_augment = AugmentConfig(**{**asdict(EVAL_AUGMENT), "colour_mode": config.colour_mode})

    generator = torch.Generator().manual_seed(config.seed)
    sampler = (
        balanced_sampler(train_samples, config.sampler_strength, generator,
                         config.source_balance)
        if config.sampler_strength > 0 or config.source_balance > 0
        else None
    )

    common = dict(
        num_workers=config.workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
        prefetch_factor=2 if config.workers > 0 else None,
    )
    train_set = FundusDataset(train_samples, train_augment, seed=config.seed,
                              return_bag=config.bag_training)
    if config.bag_training:
        bags = eye_bags(train_samples)
        batches = EyeBatchSampler(
            bags,
            eye_bag_weights(train_samples, bags, config.sampler_strength, config.source_balance),
            config.batch_size, len(train_samples), generator,
        )
        print(f"bag training: {len(bags):,} eyes, {batches.average_views:.2f} views each, "
              f"{batches.expected_views:.2f} per drawn eye, {len(batches):,} batches/epoch")
        train_loader = DataLoader(train_set, batch_sampler=batches, **common)
    else:
        train_loader = DataLoader(
            train_set,
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
    return train_loader, val_loader, test_loader, weights, val_samples


def has_native_bf16() -> bool:
    try:
        count = torch.cuda.device_count()
        return count > 0 and all(
            torch.cuda.get_device_capability(i)[0] >= 8 for i in range(count)
        )
    except Exception:
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
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    label = f"{desc} (TTA x8)" if tta else desc
    for images, labels in progress(loader, label, len(loader), show):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            if tta:
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
    if config.bag_training and config.head != "softmax":
        raise ValueError(
            "bag_training pools softmax probabilities over an eye's views; "
            "the coral head emits cumulative logits and needs its own pooling rule"
        )
    if config.select_on not in ("image", "eye"):
        raise ValueError(f"select_on must be 'image' or 'eye', got {config.select_on!r}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype, needs_scaler = resolve_amp(config.amp, device)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, weights, val_samples = build_loaders(
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
        optimizer_groups(model, config), lr=config.learning_rate
    )
    if config.layer_decay:
        rates = sorted({group["lr"] for group in optimizer.param_groups})
        print(f"layer decay {config.layer_decay}: {len(optimizer.param_groups)} groups, "
              f"lr {min(rates):.2e} .. {max(rates):.2e}")
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
    run_name = f"{config.backbone}_{config.head}"
    best_path = output_dir / f"{run_name}_best.pt"
    precision_plot = output_dir / f"precision_history_{run_name}.png"
    rule = StopRule(config.patience, config.min_epochs, config.max_hours)
    top_k: List[Tuple[float, int, Dict[str, torch.Tensor]]] = []
    stopped = {"reason": "completed", "epoch": config.epochs}
    loop_started = time.time()

    for epoch in range(config.epochs):
        reason = rule.before_epoch(time.time() - loop_started,
                                   [r["seconds"] for r in history])
        if reason:
            stopped = {"reason": reason, "epoch": epoch}
            print(f"stop before epoch {epoch + 1}: {reason}")
            break
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
        for step, batch in enumerate(bar):
            bags = None
            if config.bag_training:
                images, labels, bags = batch
                bags = bags.to(device, non_blocking=True)
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(images)
                if bags is not None:
                    loss = bag_loss(logits, labels, bags, weights, config.label_smoothing)
                elif config.head == "coral":
                    loss = coral_loss(logits.float(), labels, weights)
                else:
                    loss = F.cross_entropy(
                        logits.float(),
                        labels,
                        weight=weights,
                        label_smoothing=config.label_smoothing,
                    )
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
        predictions = decode(logits).argmax(axis=1)
        values = M.summarise(labels, predictions)
        val_loss = nll(logits, labels)

        by_source = source_metrics(val_samples, labels, predictions)
        eyes = eye_scores(val_samples, decode(logits), labels)
        selection = selection_score(config, values, eyes)
        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(1, seen),
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
            "skipped_batches": skipped,
            **values,
            **by_source,
            **eyes,
            "selection_score": selection,
        }
        history.append(record)
        print(
            f"epoch {epoch + 1}/{config.epochs} | loss {record['train_loss']:.4f} "
            f"| val_loss {val_loss:.4f} | {M.format_metrics(values)} "
            f"| {record['seconds']:.0f}s"
            + f" | worst {by_source['worst_source']} {by_source['worst_source_composite']:.4f}"
            + f" | sens>=3 {by_source['urgent_sensitivity']:.3f}"
            + f" | mắt {eyes[f'eye_composite_{eye_rule(config)}']:.4f} ({eye_rule(config)})"
            + (f" | {skipped} non-finite batches skipped" if skipped else "")
        )
        M.save_precision_history(
            history, precision_plot, f"{run_name}: validation precision by epoch"
        )

        if config.keep_top_k > 1:
            top_k.append((selection, epoch + 1,
                          {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}))
            top_k.sort(key=lambda item: item[0], reverse=True)
            del top_k[config.keep_top_k:]

        if selection > best_score:
            best_score, best_epoch = selection, epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch + 1,
                    "val": values,
                },
                best_path,
            )

        reason = rule.after_epoch(epoch + 1, selection)
        if reason:
            stopped = {"reason": reason, "epoch": epoch + 1}
            print(f"stop after epoch {epoch + 1}: {reason}")
            break

    print(f"best epoch {best_epoch} ({config.select_on}-level composite {best_score:.4f}); reloading it")
    model.load_state_dict(torch.load(best_path, map_location=device)["model"])

    judged_by = {"select_on": config.select_on}
    if config.select_on == "eye":
        judged_by["eye_rule"] = eye_rule(config)
    selected = {"source": "best_single", "epoch": best_epoch, "val_composite": best_score, **judged_by}
    if len(top_k) > 1:
        epochs = [epoch for _, epoch, _ in top_k]
        averaged = average_states([state for _, _, state in top_k])
        model.load_state_dict(averaged)
        logits, labels = collect_logits(
            model, val_loader, device, dtype, desc="val (averaged)", show=config.progress
        )
        avg_probabilities = decode(logits)
        avg_values = M.summarise(labels, avg_probabilities.argmax(axis=1))
        avg_eyes = eye_scores(val_samples, avg_probabilities, labels)
        avg_score = selection_score(config, avg_values, avg_eyes)
        print(f"average of epochs {epochs}: {config.select_on}-level composite {avg_score:.4f} "
              f"vs best single {best_score:.4f}")
        if avg_score > best_score:
            selected = {"source": "average", "epochs": epochs, "val_composite": avg_score, **judged_by}
            average_path = output_dir / f"{config.backbone}_{config.head}_avg.pt"
            torch.save({"model": averaged, "config": asdict(config),
                        "epoch": epochs, "val": avg_values}, average_path)
            print(f"using the averaged weights -> {average_path}")
        else:
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
            print("keeping the single best checkpoint")

    results: Dict[str, Dict] = {}
    confusion_artifacts: Dict[str, str] = {}
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
            predictions = decode(logits).argmax(axis=1)
            results[key] = {"argmax": M.summarise(labels, predictions)}
            matrix_path = output_dir / f"confusion_matrix_{run_name}_{key}.png"
            M.save_confusion_matrix(
                labels, predictions, matrix_path,
                f"{run_name} — {key} argmax",
            )
            confusion_artifacts[key] = matrix_path.name

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
        "epochs_run": len(history),
        "stopped": stopped,
        "selected": selected,
        "train_hours": sum(r["seconds"] for r in history) / 3600,
        "history": history,
        "results": results,
        "artifacts": {
            "precision_history": precision_plot.name,
            "confusion_matrices": confusion_artifacts,
        },
    }
    (output_dir / f"summary_{config.backbone}_{config.head}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    for key, value in results.items():
        line = M.format_metrics(value["argmax"])
        print(f"{key:9} argmax     {line}")
        if "calibrated" in value:
            print(f"{key:9} calibrated {M.format_metrics(value['calibrated'])}")
    print(f"precision plot -> {precision_plot}")
    for key, name in confusion_artifacts.items():
        print(f"{key:9} matrix     -> {output_dir / name}")
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
    parser.add_argument("--drop-path-rate", type=float, default=defaults.drop_path_rate)
    parser.add_argument("--augment", default=defaults.augment, choices=sorted(AUGMENT_PRESETS))
    parser.add_argument("--source-balance", type=float, default=defaults.source_balance)
    parser.add_argument("--layer-decay", type=float, default=defaults.layer_decay,
                        help="learning rate multiplier per depth; 0 = off")
    parser.add_argument("--keep-top-k", type=int, default=defaults.keep_top_k,
                        help="average this many best checkpoints; 1 = off")
    parser.add_argument("--bag-training", action="store_true",
                        help="one loss per eye, pooled over its views")
    parser.add_argument("--select-on", default=defaults.select_on, choices=("image", "eye"),
                        help="which composite picks the checkpoint; at the eye, bag runs "
                             "pool views by max and image runs by mean")
    parser.add_argument("--patience", type=int, default=defaults.patience,
                        help="epochs without a better val composite before stopping; 0 = off")
    parser.add_argument("--min-epochs", type=int, default=defaults.min_epochs,
                        help="never stop on patience before this epoch")
    parser.add_argument("--max-hours", type=float, default=defaults.max_hours,
                        help="training budget in hours, leaving time for evaluation; 0 = off")
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
        drop_path_rate=args.drop_path_rate,
        augment=args.augment,
        source_balance=args.source_balance,
        layer_decay=args.layer_decay,
        keep_top_k=args.keep_top_k,
        bag_training=args.bag_training,
        select_on=args.select_on,
        patience=args.patience,
        min_epochs=args.min_epochs,
        max_hours=args.max_hours,
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

"""Shared training utilities for the two fundus models."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_config_path(default_path: Path) -> Path:
    """Read the optional ``--config`` argument."""

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_path,
        help="Path to the model JSON configuration.",
    )
    return parser.parse_args().config.expanduser().resolve()


def load_config(config_path: Path) -> Dict[str, object]:
    """Load a JSON configuration and ensure it contains an object."""

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Config must contain a JSON object")
    return config


def resolve_config_path(value: object) -> Path:
    """Resolve a config path relative to the project root."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_loaders(config: Dict[str, object]) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create ImageFolder train/validation DataLoaders."""

    data_root = resolve_config_path(config["data_root"])
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Training folder does not exist: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Validation folder does not exist: {val_dir}")

    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)
    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Train and validation classes differ: "
            f"train={train_dataset.classes}, val={val_dataset.classes}"
        )

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    return train_loader, val_loader, train_dataset.classes


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    description: str,
) -> float:
    """Train the model for one epoch and return mean loss."""

    model.train()
    running_loss = 0.0
    item_count = 0

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        current_count = labels.size(0)
        running_loss += loss.item() * current_count
        item_count += current_count
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(item_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    description: str,
) -> Dict[str, float]:
    """Calculate validation loss, accuracy, Macro-F1 and QWK."""

    model.eval()
    running_loss = 0.0
    item_count = 0
    targets: List[int] = []
    predictions: List[int] = []

    progress = tqdm(
        loader,
        desc=description,
        leave=False,
        dynamic_ncols=True,
    )
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        current_count = labels.size(0)
        running_loss += loss.item() * current_count
        item_count += current_count
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": running_loss / max(item_count, 1),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "qwk": float(
            cohen_kappa_score(targets, predictions, weights="quadratic")
        ),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    metrics: Dict[str, float],
    classes: List[str],
    model_name: str,
    config: Dict[str, object],
) -> None:
    """Save model state and enough metadata to resume or evaluate later."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": model_name,
            "classes": classes,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def run_training(
    model: nn.Module,
    model_name: str,
    config_path: Path,
) -> None:
    """Run the common training, validation and checkpoint workflow."""

    config = load_config(config_path)
    set_seed(int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(config.get("amp", True))
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available; training will use CPU.")

    train_loader, val_loader, classes = build_loaders(config)
    expected_classes = int(config["num_classes"])
    if len(classes) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} classes, found {len(classes)}: {classes}"
        )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(config["epochs"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = resolve_config_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"

    start_epoch = 0
    best_qwk = float("-inf")
    no_improvement = 0
    history: List[Dict[str, float]] = []

    resume_value = config.get("resume")
    if resume_value:
        resume_path = resolve_config_path(resume_value)
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_qwk = float(checkpoint["metrics"].get("qwk", best_qwk))
        print(f"Resumed from: {resume_path}")

    print(f"Model: {model_name}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Classes: {classes}")
    print(f"AMP: {use_amp}")

    for epoch in range(start_epoch, int(config["epochs"])):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            description=f"Train {epoch + 1}/{int(config['epochs'])}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            description=f"Val   {epoch + 1}/{int(config['epochs'])}",
        )
        scheduler.step()

        metrics = {
            "epoch": float(epoch + 1),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_qwk": float(val_metrics["qwk"]),
        }
        history.append(metrics)

        print(
            f"Epoch {epoch + 1:03d}/{int(config['epochs']):03d} | "
            f"loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['accuracy']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | "
            f"qwk={val_metrics['qwk']:.4f}"
        )

        save_checkpoint(
            last_path,
            model,
            optimizer,
            scheduler,
            epoch + 1,
            val_metrics,
            classes,
            model_name,
            config,
        )

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            no_improvement = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                scheduler,
                epoch + 1,
                val_metrics,
                classes,
                model_name,
                config,
            )
            print(f"  Saved best checkpoint: {best_path}")
        else:
            no_improvement += 1

        if no_improvement >= int(config["patience"]):
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    print(f"Training finished. Best checkpoint: {best_path}")

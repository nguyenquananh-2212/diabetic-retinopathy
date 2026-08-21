"""Evaluate a Swin-T + EfficientNetV2-M weighted ensemble.

Both models must already have been trained and have a ``best.pt`` checkpoint.
The models are run sequentially to reduce GPU memory usage on a laptop GPU.

Run from the project root:

    python Source/ensemble/ensemble_test.py

To use another configuration:

    python Source/ensemble/ensemble_test.py --config path/to/ensemble_config.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "Source" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from train_efficientnet_v2_m import build_model as build_efficientnet  # noqa: E402
from train_swin import build_model as build_swin  # noqa: E402


DEFAULT_CONFIG = Path(__file__).resolve().parent / "ensemble_config.json"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    """Read the optional ensemble configuration path."""

    parser = argparse.ArgumentParser(
        description="Evaluate Swin-T and EfficientNetV2-M with soft voting."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to ensemble_config.json.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    """Load the JSON ensemble configuration."""

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Ensemble config must contain a JSON object")
    return config


def resolve_path(value: object) -> Path:
    """Resolve a path relative to the project root."""

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_test_loader(config: Dict[str, object]) -> Tuple[DataLoader, ImageFolder]:
    """Create a deterministic test DataLoader."""

    test_root = resolve_path(config["test_root"])
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test folder does not exist: {test_root}")

    image_size = int(config["image_size"])
    dataset = ImageFolder(
        test_root,
        transform=transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config["num_workers"]) > 0,
    )
    return loader, dataset


def load_model(
    checkpoint_value: object,
    model_name: str,
    device: torch.device,
    num_classes: int,
) -> torch.nn.Module:
    """Build a model, load its checkpoint and move it to the selected device."""

    checkpoint_path = resolve_path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_num_classes = int(
        checkpoint_config.get("num_classes", num_classes)
    )

    if model_name == "swin_t":
        model = build_swin(checkpoint_num_classes, pretrained=False)
    elif model_name == "efficientnet_v2_m":
        model = build_efficientnet(checkpoint_num_classes, pretrained=False)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded {model_name}: {checkpoint_path}")
    return model


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    """Return class probabilities for every test image in loader order."""

    probabilities: List[torch.Tensor] = []
    model.eval()

    for images, _ in tqdm(
        loader,
        desc=f"Predict {model_name}",
        dynamic_ncols=True,
    ):
        images = images.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(images)
            batch_probabilities = torch.softmax(logits, dim=1)
        probabilities.append(batch_probabilities.float().cpu())

    return torch.cat(probabilities, dim=0)


def make_predictions(
    probabilities: torch.Tensor,
    decision_rule: str,
) -> torch.Tensor:
    """Convert ensemble probabilities into class predictions."""

    if decision_rule == "argmax":
        return probabilities.argmax(dim=1)

    if decision_rule == "expected":
        class_ids = torch.arange(
            probabilities.shape[1],
            dtype=probabilities.dtype,
        )
        expected_class = (probabilities * class_ids).sum(dim=1)
        return expected_class.round().long().clamp(
            min=0,
            max=probabilities.shape[1] - 1,
        )

    raise ValueError("decision_rule must be 'argmax' or 'expected'")


def calculate_metrics(
    targets: List[int],
    predictions: List[int],
) -> Dict[str, float]:
    """Calculate classification metrics for the ensemble."""

    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "qwk": float(
            cohen_kappa_score(targets, predictions, weights="quadratic")
        ),
    }


def save_predictions(
    path: Path,
    dataset: ImageFolder,
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
) -> None:
    """Save one CSV row per test image with labels and probabilities."""

    path.parent.mkdir(parents=True, exist_ok=True)
    class_count = probabilities.shape[1]
    fields = ["path", "true_label", "predicted_label"]
    fields += [f"probability_{index}" for index in range(class_count)]

    with path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for index, (image_path, true_label) in enumerate(dataset.samples):
            row = {
                "path": image_path,
                "true_label": true_label,
                "predicted_label": int(predictions[index]),
            }
            row.update(
                {
                    f"probability_{class_id}": float(
                        probabilities[index, class_id]
                    )
                    for class_id in range(class_count)
                }
            )
            writer.writerow(row)


def main() -> None:
    """Run sequential model inference and weighted probability voting."""

    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and bool(config.get("amp", True))
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader, dataset = build_test_loader(config)
    expected_classes = int(config["num_classes"])
    if len(dataset.classes) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} classes, found {dataset.classes}"
        )

    swin_model = load_model(
        config["swin_checkpoint"],
        "swin_t",
        device,
        expected_classes,
    )
    swin_probabilities = predict_probabilities(
        swin_model,
        "swin_t",
        loader,
        device,
        use_amp,
    )
    del swin_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    efficientnet_model = load_model(
        config["efficientnet_checkpoint"],
        "efficientnet_v2_m",
        device,
        expected_classes,
    )
    efficientnet_probabilities = predict_probabilities(
        efficientnet_model,
        "efficientnet_v2_m",
        loader,
        device,
        use_amp,
    )
    del efficientnet_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    swin_weight = float(config["swin_weight"])
    efficientnet_weight = float(config["efficientnet_weight"])
    weight_sum = swin_weight + efficientnet_weight
    if weight_sum <= 0:
        raise ValueError("Model weights must have a positive sum")

    ensemble_probabilities = (
        swin_weight * swin_probabilities
        + efficientnet_weight * efficientnet_probabilities
    ) / weight_sum
    predictions = make_predictions(
        ensemble_probabilities,
        str(config.get("decision_rule", "argmax")),
    )
    targets = [label for _, label in dataset.samples]
    metrics = calculate_metrics(targets, predictions.tolist())
    metrics.update(
        {
            "swin_weight": swin_weight / weight_sum,
            "efficientnet_weight": efficientnet_weight / weight_sum,
            "decision_rule": str(config.get("decision_rule", "argmax")),
            "test_images": len(dataset),
        }
    )

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_predictions(
        output_dir / "predictions.csv",
        dataset,
        ensemble_probabilities,
        predictions,
    )

    print("\nEnsemble metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to: {output_dir}")


if __name__ == "__main__":
    main()

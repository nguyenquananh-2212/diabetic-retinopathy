"""Train the EfficientNetV2-M model."""

from pathlib import Path

from torch import nn
from torchvision.models import (
    EfficientNet_V2_M_Weights,
    efficientnet_v2_m,
)

from train_utils import load_config, parse_config_path, run_training


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "efficientnet_v2_m.json"
)


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Create EfficientNetV2-M and replace its ImageNet classifier."""

    weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_v2_m(weights=weights)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )
    return model


def main() -> None:
    """Load config, build EfficientNetV2-M and start training."""

    config_path = parse_config_path(DEFAULT_CONFIG)
    config = load_config(config_path)
    model = build_model(
        num_classes=int(config["num_classes"]),
        pretrained=bool(config.get("pretrained", True)),
    )
    run_training(model, "efficientnet_v2_m", config_path)


if __name__ == "__main__":
    main()

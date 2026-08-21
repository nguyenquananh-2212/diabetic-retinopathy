"""Train the Swin Transformer model."""

from pathlib import Path

from torch import nn
from torchvision.models import Swin_T_Weights, swin_t

from train_utils import load_config, parse_config_path, run_training


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "swin_t.json"


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    """Create Swin-T and replace the ImageNet head with a 5-class head."""

    weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = swin_t(weights=weights)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def main() -> None:
    """Load config, build Swin-T and start training."""

    config_path = parse_config_path(DEFAULT_CONFIG)
    config = load_config(config_path)
    model = build_model(
        num_classes=int(config["num_classes"]),
        pretrained=bool(config.get("pretrained", True)),
    )
    run_training(model, "swin_t", config_path)


if __name__ == "__main__":
    main()

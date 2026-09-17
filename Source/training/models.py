from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5


class Backbone(NamedTuple):

    identifier: str
    pretrain_size: int
    alignment: Optional[int]
    windowed: bool
BACKBONES = {
    "convnext_tiny": Backbone("convnext_tiny.fb_in22k_ft_in1k", 224, None, False),
    "swin_t": Backbone(
        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", 224, 224, True
    ),
}


def _require_timm():
    try:
        import timm
    except ImportError as exc:
        raise SystemExit(
            "timm is required for the backbones; install it with `pip install timm`"
        ) from exc
    return timm


class CoralHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features) + self.bias


def coral_targets(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    cuts = torch.arange(num_classes - 1, device=labels.device)
    return (labels[:, None] > cuts[None, :]).float()


def coral_loss(
    logits: torch.Tensor, labels: torch.Tensor, weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    targets = coral_targets(labels, logits.shape[1] + 1)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if weight is not None:
        loss = loss * weight[labels][:, None]
    return loss.mean()


def coral_to_probabilities(logits: torch.Tensor) -> torch.Tensor:
    greater = torch.sigmoid(logits)
    greater, _ = torch.cummin(greater, dim=1)
    ones = torch.ones_like(greater[:, :1])
    zeros = torch.zeros_like(greater[:, :1])
    upper = torch.cat([ones, greater], dim=1)
    lower = torch.cat([greater, zeros], dim=1)
    return (upper - lower).clamp_min(0.0)


@dataclass
class ModelConfig:
    backbone: str = "convnext_tiny"
    image_size: int = 448
    head: str = "softmax"
    drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    pretrained: bool = True


class FundusModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.backbone not in BACKBONES:
            raise ValueError(
                f"unknown backbone {config.backbone!r}; have {sorted(BACKBONES)}"
            )
        spec = BACKBONES[config.backbone]
        if spec.alignment and config.image_size % spec.alignment:
            raise ValueError(
                f"{config.backbone} needs an input that is a multiple of "
                f"{spec.alignment} px so its shifted windows tile every stage "
                f"exactly; {config.image_size} is not.  Nearest valid sizes: "
                f"{spec.alignment * (config.image_size // spec.alignment)} and "
                f"{spec.alignment * (config.image_size // spec.alignment + 1)}."
            )
        timm = _require_timm()

        self.config = config
        self.backbone = timm.create_model(
            spec.identifier,
            pretrained=config.pretrained,
            num_classes=0,
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            img_size=config.image_size if spec.windowed else None,
        )
        self.pretrain_scale = config.image_size / spec.pretrain_size
        features = self.backbone.num_features
        if config.head == "coral":
            self.head: nn.Module = CoralHead(features, NUM_CLASSES)
        elif config.head == "softmax":
            self.head = nn.Linear(features, NUM_CLASSES)
        else:
            raise ValueError(f"unknown head {config.head!r}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(images))

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.head == "coral":
            return coral_to_probabilities(logits)
        return torch.softmax(logits, dim=1)


def build_model(config: ModelConfig) -> FundusModel:
    return FundusModel(config)


def parameter_groups(
    model: nn.Module, weight_decay: float = 0.05
) -> list:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def describe(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    text = f"{total / 1e6:.2f}M parameters ({trainable / 1e6:.2f}M trainable)"
    scale = getattr(model, "pretrain_scale", None)
    if scale and scale != 1.0:
        text += f", running at {scale:.1f}x the pretraining resolution"
    return text

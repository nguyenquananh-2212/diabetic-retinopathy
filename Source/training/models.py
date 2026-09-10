"""The two backbones this project settled on, plus the heads they can wear.

**convnext_tiny** and **swin_t**, both at 448.  The pairing is deliberate: one
convolutional and one attention-based, so an ensemble of the two is not two
copies of the same inductive bias.

Why 448 and not 384: the median retina here is ~1,660 px across and a
microaneurysm is roughly 1/100 of the disc diameter, so at 224 the earliest
lesion of DR lands on ~2.2 px and is physically gone -- moving 224 -> 448 was
worth +0.038 QWK on this data.  And 448 keeps Swin V1's 7 px window aligned
where 384 does not:

    input   feature maps        remainder mod 7
    224     56, 28, 14,  7      0, 0, 0, 0   aligned
    384     96, 48, 24, 12      5, 6, 3, 5   padded at every stage
    448    112, 56, 28, 14      0, 0, 0, 0   aligned
    672    168, 84, 42, 21      0, 0, 0, 0   aligned

timm builds the 384 model without an error or even a warning, which is exactly
why ``FundusModel`` refuses it: a silent 4-stage pad is worse than a stop.
(Swin V2's 8 px window wants 512 instead.)

On a GPU without bf16 (Kaggle's T4 is sm_75) the trainer runs fp16 instead --
see ``resolve_amp`` in train.py, which does not trust
``torch.cuda.is_bf16_supported()``.

Two heads are available.  The plain 5-way softmax is the default.  The ordinal
head predicts 4 cumulative "grade > k" logits (CORAL), which cannot express an
impossible ordering such as P(>2) > P(>1); it pairs with the expected-grade
decoding in ``metrics.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5


class Backbone(NamedTuple):
    """What we need to know about a checkpoint before instantiating it."""

    identifier: str
    pretrain_size: int  # what the *weights* saw, not what we train at
    alignment: Optional[int]  # input must be a multiple of this, or None
    windowed: bool  # needs img_size passed through to rebuild position tables


# Where ``alignment`` comes from, for swin_t: patch 4, window 7, four stages, so
# the last stage sees img/32 and every stage divides cleanly only when
# 4 x 7 x 8 = 224 divides the input.  448 works (448/32 = 14 = 2x7); 384 does not
# (384/32 = 12), and timm silently pads instead of telling you.  ConvNeXt is
# fully convolutional and has no such constraint.
BACKBONES = {
    "convnext_tiny": Backbone("convnext_tiny.fb_in22k_ft_in1k", 224, None, False),
    "swin_t": Backbone(
        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", 224, 224, True
    ),
}


def _require_timm():
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "timm is required for the backbones; install it with `pip install timm`"
        ) from exc
    return timm


class CoralHead(nn.Module):
    """K-1 cumulative logits sharing one projection, with per-cut biases.

    A shared weight vector with separate biases is what makes the cuts
    *ordered*: every threshold reads the same scalar direction in feature space,
    so P(grade > k) can only decrease as k grows.  Independent heads can and do
    produce P(>2) > P(>1), which is not a grade at all.
    """

    def __init__(self, in_features: int, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features) + self.bias


def coral_targets(labels: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """``[label > 0, label > 1, ...]`` as floats."""

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
    """Turn cumulative logits into a proper 5-class distribution.

    P(y=k) = P(y>k-1) - P(y>k), with the ends clamped.  Monotonicity is enforced
    by a cumulative minimum rather than assumed: the shared projection makes
    inversions impossible in theory, but numerically a tie can still order the
    wrong way and produce a negative probability.
    """

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
    head: str = "softmax"  # or "coral"
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
            num_classes=0,  # feature extractor; the head is ours
            drop_rate=config.drop_rate,
            drop_path_rate=config.drop_path_rate,
            # Only windowed models take img_size; timm drops None kwargs, so a
            # convolutional backbone never sees it.
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
    """No weight decay on norms or biases -- decaying them costs accuracy free.

    A LayerNorm gain pulled toward zero shrinks the activation it normalises,
    which the next layer simply undoes by growing; the regularisation buys
    nothing and the optimisation pays for it.
    """

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
        # Worth saying out loud: the weights were fitted on a quarter of this
        # area, so early epochs are also adapting to the new scale, not only to
        # the new task.
        text += f", running at {scale:.1f}x the pretraining resolution"
    return text

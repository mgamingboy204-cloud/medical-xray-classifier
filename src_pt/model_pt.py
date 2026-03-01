from dataclasses import dataclass

import timm
import torch
import torch.nn as nn


class DinoV2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        embed_dim = getattr(backbone, "embed_dim", None) or getattr(backbone, "num_features", None)
        if embed_dim is None:
            raise RuntimeError("Unable to infer DINOv2 embedding dimension.")
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:
            feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        if feats.ndim == 3:
            feats = feats[:, 0]
        return self.head(feats)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        state = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(state[k], alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v)


@dataclass
class ModelBundle:
    model: nn.Module
    ema: EMA | None


def _build_dino(num_classes: int, freeze_backbone: bool, dropout: float) -> nn.Module:
    errors = []
    backbone = None
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    except Exception as e:
        errors.append(f"torch.hub dinov2_vits14 failed: {e}")

    if backbone is None:
        try:
            backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
        except Exception as e:
            errors.append(f"timm vit_small_patch14_dinov2.lvd142m failed: {e}")

    if backbone is None:
        raise RuntimeError("Could not build DINOv2 backbone. " + " | ".join(errors))

    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    return DinoV2Classifier(backbone=backbone, num_classes=num_classes, dropout=dropout)


def create_model(config: dict, num_classes: int) -> ModelBundle:
    name = config["model_name"]
    if name == "convnext_tiny":
        model = timm.create_model("convnext_tiny", pretrained=True, num_classes=num_classes)
    elif name == "tf_efficientnetv2_b3":
        model = timm.create_model("tf_efficientnetv2_b3", pretrained=True, num_classes=num_classes)
    elif name == "convnext_base":
        model = timm.create_model("convnext_base", pretrained=True, num_classes=num_classes)
    elif name == "tf_efficientnetv2_l":
        model = timm.create_model("tf_efficientnetv2_l", pretrained=True, num_classes=num_classes)
    elif name == "swin_base":
        # standard strong Swin baseline in timm
        model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes)
    elif name == "dinov2_vits14":
        model = _build_dino(
            num_classes=num_classes,
            freeze_backbone=bool(config.get("freeze_backbone", True)),
            dropout=float(config.get("head_dropout", 0.2)),
        )
    else:
        raise ValueError(f"Unsupported model_name: {name}")

    ema = EMA(model, decay=float(config.get("ema_decay", 0.999))) if config.get("use_ema", False) else None
    return ModelBundle(model=model, ema=ema)

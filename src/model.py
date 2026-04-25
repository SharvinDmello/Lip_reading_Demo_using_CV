"""
Lip Reading Model
MobileNetV3-Small CNN backbone + BiLSTM temporal encoder + CTC head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional, Dict
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
class SpatialFeatureExtractor(nn.Module):
    """MobileNetV3-Small CNN backbone."""

    def __init__(self, pretrained: bool = True, freeze_layers: int = 5):
        super().__init__()
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            backbone = models.mobilenet_v3_small(weights=weights)
            logger.info("Loaded MobileNetV3-Small with ImageNet pretrained weights")
        else:
            backbone = models.mobilenet_v3_small(weights=None)

        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 576   # fixed for MobileNetV3-Small

        if freeze_layers > 0 and pretrained:
            for i, layer in enumerate(self.features.children()):
                if i < freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 112, 112)
        x = self.features(x)        # (B, 576, H', W')
        x = self.pool(x)            # (B, 576, 1, 1)
        return x.flatten(1)         # (B, 576)


# ════════════════════════════════════════════════════════════════════════════
class TemporalEncoder(nn.Module):
    """
    Linear projection  →  Bidirectional LSTM
    input_size fed to LSTM = hidden_dim (after projection), NOT feature_dim.
    """

    def __init__(
        self,
        input_dim:  int   = 576,
        hidden_dim: int   = 256,
        num_layers: int   = 2,
        dropout:    float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.bidirectional = bidirectional
        directions        = 2 if bidirectional else 1

        # Project CNN features → hidden_dim  (this is what the LSTM sees)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # LSTM input_size == hidden_dim  (256), NOT input_dim (576)
        self.lstm = nn.LSTM(
            input_size  = hidden_dim,    # ← 256
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        self.output_dim    = hidden_dim * directions
        self.output_norm   = nn.LayerNorm(self.output_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T, input_dim)
        x = self.input_proj(x)               # (B, T, hidden_dim)
        out, hidden = self.lstm(x, hidden)   # (B, T, hidden_dim*dirs)
        out = self.output_norm(out)
        out = self.output_dropout(out)
        return out, hidden


# ════════════════════════════════════════════════════════════════════════════
class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)


# ════════════════════════════════════════════════════════════════════════════
class LipReadingModel(nn.Module):
    """
    Complete lip-reading model:
        SpatialEncoder  →  TemporalEncoder  →  (Attention)  →  CTC classifier
    """

    def __init__(
        self,
        vocab_size:           int   = 68,
        feature_dim:          int   = 576,
        hidden_dim:           int   = 256,
        num_lstm_layers:      int   = 2,
        dropout:              float = 0.3,
        pretrained_backbone:  bool  = True,
        use_attention:        bool  = True,
        freeze_backbone_layers: int = 5,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim

        self.spatial_encoder = SpatialFeatureExtractor(
            pretrained    = pretrained_backbone,
            freeze_layers = freeze_backbone_layers,
        )

        self.feature_norm = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout * 0.5),
        )

        self.temporal_encoder = TemporalEncoder(
            input_dim     = feature_dim,
            hidden_dim    = hidden_dim,
            num_layers    = num_lstm_layers,
            dropout       = dropout,
            bidirectional = True,
        )

        temporal_out_dim = hidden_dim * 2

        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(temporal_out_dim, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

        self._init_weights()
        logger.info(
            f"LipReadingModel initialized: vocab={vocab_size}, "
            f"features={feature_dim}, hidden={hidden_dim}, attention={use_attention}"
        )

    # ── weight init ─────────────────────────────────────────────────────────
    def _init_weights(self):
        for module in [self.temporal_encoder, self.classifier]:
            for name, p in module.named_parameters():
                if "weight" in name and p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
        # Forget-gate bias = 1 for better gradient flow
        for name, p in self.temporal_encoder.lstm.named_parameters():
            if "bias" in name:
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)

    # ── forward ─────────────────────────────────────────────────────────────
    def encode_frames(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        feats = self.spatial_encoder(x.view(B * T, C, H, W))   # (B*T, 576)
        feats = feats.view(B, T, -1)                            # (B, T, 576)
        return self.feature_norm(feats)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        feats = self.encode_frames(x)                           # (B, T, 576)
        temporal_out, _ = self.temporal_encoder(feats)          # (B, T, 512)
        if self.use_attention:
            temporal_out = self.attention(temporal_out)
        logits = self.classifier(temporal_out)                  # (B, T, V)
        logits_ctc = logits.permute(1, 0, 2)                    # (T, B, V)
        log_probs  = F.log_softmax(logits_ctc, dim=-1)
        result = {"logits": logits_ctc, "log_probs": log_probs,
                  "probs": F.softmax(logits_ctc, dim=-1)}
        if return_features:
            result["spatial_features"]  = feats
            result["temporal_features"] = temporal_out
        return result

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ════════════════════════════════════════════════════════════════════════════
class ModelLoader:
    """Loads a LipReadingModel from a checkpoint produced by setup_weights.py."""

    def __init__(self, weights_dir: Path = Path("weights")):
        self.weights_dir = weights_dir

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _try_load(module: nn.Module, state: dict, label: str, strict: bool = True):
        try:
            module.load_state_dict(state, strict=strict)
            logger.info(f"Loaded {label} weights (strict={strict})")
        except RuntimeError as exc:
            logger.warning(f"{label} strict load failed, retrying partial: {exc}")
            try:
                module.load_state_dict(state, strict=False)
                logger.info(f"Loaded {label} weights (partial / non-strict)")
            except Exception as exc2:
                logger.warning(f"{label} load failed completely: {exc2}")

    # ── main entry ──────────────────────────────────────────────────────────
    def load(
        self,
        device: torch.device,
        weights_path: Optional[Path] = None,
    ) -> Tuple["LipReadingModel", Dict]:
        if weights_path is None:
            weights_path = self.weights_dir / "lipnet_model.pth"

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weights_path}\n"
                "Run:  python setup_weights.py"
            )

        logger.info(f"Loading checkpoint from {weights_path}")
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)

        cfg        = ckpt.get("model_config", {})
        vocabulary = ckpt.get("vocabulary", {})
        vocab_size = len(vocabulary.get("chars", [])) + 1

        model = LipReadingModel(
            vocab_size          = vocab_size,
            feature_dim         = cfg.get("feature_dim", 576),
            hidden_dim          = cfg.get("hidden_dim",  256),
            num_lstm_layers     = cfg.get("num_layers",  2),
            dropout             = cfg.get("dropout",     0.3),
            pretrained_backbone = True,
            use_attention       = True,
            freeze_backbone_layers = 0,     # no freezing at inference time
        )

        # 1. CNN backbone
        if "backbone_state_dict" in ckpt:
            self._try_load(
                model.spatial_encoder.features,
                ckpt["backbone_state_dict"],
                "CNN backbone",
            )

        # 2. input_proj  (new key in v1.1 checkpoints)
        if "input_proj_state_dict" in ckpt:
            self._try_load(
                model.temporal_encoder.input_proj,
                ckpt["input_proj_state_dict"],
                "input_proj",
            )

        # 3. LSTM
        if "lstm_state_dict" in ckpt:
            self._try_load(
                model.temporal_encoder.lstm,
                ckpt["lstm_state_dict"],
                "LSTM",
            )

        # 4. output_norm
        if "output_norm_state_dict" in ckpt:
            self._try_load(
                model.temporal_encoder.output_norm,
                ckpt["output_norm_state_dict"],
                "output_norm",
            )

        # 5. classifier
        if "classifier_state_dict" in ckpt:
            self._try_load(
                model.classifier,
                ckpt["classifier_state_dict"],
                "classifier",
            )

        model.to(device).eval()
        params = model.count_parameters()
        logger.info(
            f"Model loaded: {params['total']:,} total params, "
            f"{params['trainable']:,} trainable"
        )
        return model, vocabulary

    def load_vocabulary(self) -> Dict:
        path = self.weights_dir / "vocabulary.json"
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary missing: {path}")
        with open(path) as f:
            return json.load(f)
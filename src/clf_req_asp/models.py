import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
import pytorch_lightning as pl
from math import pi
from tqdm import tqdm
import numpy as np
import torchvision
import copy
import os
from typing import Optional
try:
    from .utils import *
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils import *
from functools import partial
import time
import math
from torchmetrics.classification import MulticlassF1Score


class ScalarCondition(nn.Module):
    """
    Embed a 1-D scalar condition (e.g. viewing angle) into a feature vector.
    Shared across multiple classifier backbones to keep conditioning consistent.
    """
    def __init__(self, base_dim: int, out_dim: Optional[int] = None):
        super().__init__()
        self.base_dim = base_dim
        self.out_dim = out_dim or base_dim
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.base_dim, self.out_dim),
        )

    def forward(self, conds: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if conds is None:
            return None
        conds = conds.squeeze(1)  # [B]
        emb = timestep_embedding(conds, self.base_dim, 10)  # [B, base_dim]
        return self.proj(emb)  # [B, out_dim]


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=1000) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError(f"timestep_embedding expects 1-D [B], got {timesteps.shape}")
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]               # [B, half]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, 2*half]
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class ResRP(nn.Module):
    def __init__(self, inplanes, planes, dropout=0., cdim=64, cond_op="film"):
        super(ResRP, self).__init__()
        self.planes=planes
        self.dropout = dropout
        self.cdim=cdim
        self.cond_op = cond_op
        self.cat_dim = cdim if (cond_op == "cat" and cdim is not None) else 0
        self.block_1 = nn.Sequential(
            nn.Conv1d(inplanes + self.cat_dim, planes, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p = self.dropout),
            nn.SiLU(),
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p = self.dropout),
            nn.SiLU(),
        ) 
        if inplanes != planes or self.cat_dim > 0:
            self.residual = nn.Conv1d(inplanes + self.cat_dim, planes, kernel_size=1, stride=1, padding=0)
        else:
            self.residual = nn.Identity()
        
        if cdim is not None and cond_op == "film":
            # FiLM (scale + shift) when using film conditioning
            self.scal_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cdim, planes * 2),
            )
        else:
            self.scal_proj = None

        if cond_op == "cbn" and cdim is not None:
            self.bn1 = nn.BatchNorm1d(planes, affine=False)
            self.bn2 = nn.BatchNorm1d(planes, affine=False)
            self.cbn_proj1 = nn.Linear(cdim, planes * 2)
            self.cbn_proj2 = nn.Linear(cdim, planes * 2)
        else:
            self.bn1 = None
            self.bn2 = None
            self.cbn_proj1 = None
            self.cbn_proj2 = None

    def forward(self, x, scal_emb):
        x_in = x
        if self.cond_op == "cat" and scal_emb is not None:
            cond_map = scal_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x_in = torch.cat([x_in, cond_map], dim=1)
        out = self.block_1(x_in)
        if self.cond_op == "cbn" and scal_emb is not None and self.cbn_proj1 is not None:
            gamma, beta = self.cbn_proj1(scal_emb).chunk(2, dim=-1)
            out = self.bn1(out)
            out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            out = F.silu(out)
        # Inject condition only when provided; necessary for masked cond_levels (film)
        if self.cond_op == "film" and self.scal_proj is not None and scal_emb is not None:
            gamma, beta = self.scal_proj(scal_emb).chunk(2, dim=-1)
            out = out * (1 + gamma).reshape(-1, self.planes, 1) + beta.reshape(-1, self.planes, 1)
        out = self.block_2(out)
        if self.cond_op == "cbn" and scal_emb is not None and self.cbn_proj2 is not None:
            gamma, beta = self.cbn_proj2(scal_emb).chunk(2, dim=-1)
            out = self.bn2(out)
            out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            out = F.silu(out)
        out += self.residual(x_in)
        return out
    
class EmbedLinear(nn.Module):
    def __init__(self, in_ch:int, out_ch:int):
        super().__init__()
        self.layer1 = nn.Linear(in_ch, out_ch)
        self.layernorm = nn.LayerNorm(out_ch)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.layernorm(self.layer1(x))
    
class ResNetRP(nn.Module):
    def __init__(self, ch_mul=[1, 2, 4, 8], mod_ch=16, emb_dim=512, num_res=4, cond=False, cond_op="film", dropout=0.1, cond_levels=None, **kwargs):
        super().__init__()
        self.conv_in = nn.Conv1d(1, ch_mul[0]*mod_ch, kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList([nn.Conv1d(ch_mul[i]*mod_ch, ch_mul[i]*mod_ch, kernel_size=5, stride=2, padding=2) for i in range(len(ch_mul)-1)])
        self.ch_mul, self.num_res = ch_mul, num_res
        self.net_blocks = nn.ModuleList([])
        self.cond = cond
        self.cond_op = cond_op
        self.mod_ch = mod_ch
        # None = inject at all levels; otherwise list/set of level indices (0 = first down block)
        self.cond_levels = None if cond_levels is None else set(cond_levels)
        cdim = emb_dim//16
        self.cond_dim = cdim if cond else 0
        if self.cond:
            self.scal_emb_layer = EmbedLinear(self.mod_ch, cdim)
        for i in range(len(ch_mul)-1):
            in_ch = ch_mul[i]*mod_ch
            out_ch = ch_mul[i+1]*mod_ch
            self.net_blocks.append(ResRP(in_ch, out_ch, dropout=dropout, cdim=cdim if cond else None, cond_op=cond_op))
            for j in range(num_res-1):
                self.net_blocks.append(ResRP(out_ch, out_ch, dropout=dropout, cdim=cdim if cond else None, cond_op=cond_op))
        self.conv_out = nn.Conv1d(ch_mul[-1]*mod_ch, emb_dim, kernel_size=7, stride=1, padding=3)
        self.output_dim = emb_dim + (self.cond_dim if (self.cond and self.cond_op == "cat") else 0)

    def forward(self, x, conds):  # x: [B, 1, R]
        if self.cond:
            conds = conds.squeeze(1)  # [B, 1]
            scal = torch.as_tensor(timestep_embedding(conds, self.mod_ch, 10), device=conds.device)
            scal = self.scal_emb_layer(scal)  # [B, D]
        x = self.conv_in(x)  # [B, D, R]
        for i in range(len(self.ch_mul)-1):
            x = self.down[i](x)  # Downsample RP features
            for j in range(self.num_res):
                if self.cond and self.cond_op == "cat":
                    inject = scal  # always inject for cat
                elif self.cond and self.cond_op in {"film", "cbn"} and (self.cond_levels is None or i in self.cond_levels):
                    inject = scal
                else:
                    inject = None
                x  = self.net_blocks[self.num_res*i+j](x, inject)  # [B, D, R]
        x = self.conv_out(x)  # [B, D, R//8]
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B, D]
        if self.cond_op == "cat":
            x = torch.cat([x, scal], dim=1)  # [B, D + cond_dim]
        return x
    
class MLPHead(nn.Module):
    def __init__(self, in_h, num_class=10):
        super().__init__()
        self.head = nn.Linear(in_h, num_class)

    def forward(self, x):
        return self.head(x)


class Conv1dBackbone(nn.Module):
    """
    Lightweight 1D CNN backbone without residual connections.
    Good baseline for HRRP classification when we want something simpler than ResNetRP.
    """
    def __init__(
        self,
        ch_mul=[1, 2, 3, 4],
        mod_ch=32,
        emb_dim=128,
        kernel_size=5,
        dropout=0.1,
        cond=False,
        cond_op="film",
        cond_levels=None,
        cond_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.mod_ch = mod_ch
        self.cond = cond
        self.cond_op = cond_op
        self.cond_levels = None if cond_levels is None else set(cond_levels)
        self.kernel_size = kernel_size
        self.dropout = dropout

        # Embed scalar conditioning once for all blocks
        self.cond_dim = cond_dim or max(4, emb_dim // 8)
        self.cond_emb_layer = ScalarCondition(base_dim=self.mod_ch, out_dim=self.cond_dim) if cond else None

        blocks = []
        bn_film = []
        bn_cbn = []
        film_layers = []
        in_ch = 1
        for i, mult in enumerate(ch_mul):
            out_ch = mult * mod_ch
            stride = 1 if i == 0 else 2
            extra_in = self.cond_dim if (self.cond and cond_op == "cat") else 0
            conv = nn.Conv1d(in_ch + extra_in, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
            drop = nn.Dropout(dropout)
            blocks.append(nn.ModuleList([conv, drop]))
            # BN affine pour film/cat, BN sans affine pour cbn
            bn_film.append(nn.BatchNorm1d(out_ch, affine=True))
            bn_cbn.append(nn.BatchNorm1d(out_ch, affine=False))
            film_layers.append(nn.Linear(self.cond_dim, out_ch * 2) if (self.cond and cond_op in {"film", "cbn"}) else None)
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.bn_film = nn.ModuleList(bn_film)
        self.bn_cbn = nn.ModuleList(bn_cbn)
        self.film_layers = nn.ModuleList([f if f is not None else nn.Identity() for f in film_layers])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_ch, emb_dim),
        )
        self.output_dim = emb_dim + (self.cond_dim if (self.cond and self.cond_op == "cat") else 0)

    def forward(self, x, conds):
        cond_emb = self.cond_emb_layer(conds) if self.cond else None  # [B, cond_dim]
        for i, block in enumerate(self.blocks):
            conv, drop = block
            x_in = x
            if self.cond and self.cond_op == "cat" and cond_emb is not None:
                # concat condition along channel dimension, broadcast over temporal length
                cond_map = cond_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                x_in = torch.cat([x_in, cond_map], dim=1)
            x = conv(x_in)
            if self.cond_op == "cbn":
                x = self.bn_cbn[i](x)
            elif self.cond_op != "cat":
                x = self.bn_film[i](x)
            if self.cond and cond_emb is not None and self.cond_op in {"film", "cbn"} and (self.cond_levels is None or i in self.cond_levels):
                gamma, beta = self.film_layers[i](cond_emb).chunk(2, dim=-1)
                # FiLM for both film and cbn
                x = x * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
            x = F.silu(x)
            x = drop(x)
        x = self.head(x)  # [B, emb_dim]
        if self.cond and self.cond_op == "cat" and cond_emb is not None:
            x = torch.cat([x, cond_emb], dim=1)
        return x


class LSTMBackbone(nn.Module):
    """
    Sequence model for HRRP signals. Handles conditioning either by FiLM (film/cbn)
    on the pooled representation or by concatenation.
    """
    def __init__(
        self,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        emb_dim=128,
        dropout=0.1,
        mod_ch=32,
        cond=False,
        cond_op="cat",
        cond_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.cond = cond
        self.cond_op = cond_op
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mod_ch = mod_ch
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.feature_dim = hidden_size * (2 if bidirectional else 1)

        self.cond_dim = cond_dim or max(4, emb_dim // 8)
        self.cond_emb_layer = ScalarCondition(base_dim=self.mod_ch, out_dim=self.cond_dim) if cond else None
        self.film_layer = nn.Linear(self.cond_dim, self.feature_dim * 2) if cond and cond_op in {"film", "cbn"} else None
        self.proj = nn.Linear(self.feature_dim, emb_dim)
        self.output_dim = emb_dim + (self.cond_dim if (self.cond and self.cond_op == "cat") else 0)

    def forward(self, x, conds):
        seq = x.squeeze(1).unsqueeze(-1)  # [B, L, 1]
        out, _ = self.lstm(seq)           # [B, L, H]
        feat = out.mean(dim=1)            # pooled representation
        cond_emb = self.cond_emb_layer(conds) if self.cond else None
        if self.cond and self.cond_op in {"film", "cbn"} and cond_emb is not None:
            gamma, beta = self.film_layer(cond_emb).chunk(2, dim=-1)
            feat = feat * (1 + gamma) + beta
        feat = self.proj(feat)
        if self.cond and self.cond_op == "cat" and cond_emb is not None:
            feat = torch.cat([feat, cond_emb], dim=1)
        return feat


class MLPBackbone(nn.Module):
    """
    Simple MLP baseline operating on the flattened HRRP.
    Useful when we want a non-sequential, lightweight classifier.
    """
    def __init__(
        self,
        emb_dim=128,
        hidden_dims=None,
        dropout=0.1,
        mod_ch=32,
        cond=False,
        cond_op="cat",
        cond_dim=None,
        input_length=200,
        **kwargs,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        self.cond = cond
        self.cond_op = cond_op
        self.mod_ch = mod_ch
        self.cond_dim = cond_dim or max(4, emb_dim // 8)
        self.input_length = input_length
        self.cond_emb_layer = ScalarCondition(base_dim=self.mod_ch, out_dim=self.cond_dim) if cond else None
        self.film = nn.Linear(self.cond_dim, emb_dim * 2) if cond and cond_op in {"film", "cbn"} else None
        # Optional normalization between layers to stabilize unconditioned MLP training
        self.use_ln = bool(kwargs.get("use_ln", False))
        self.use_bn = bool(kwargs.get("use_bn", False))
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.emb_dim = emb_dim
        # Will be (re)built lazily in forward based on actual input length
        self.hidden_linears: Optional[nn.ModuleList] = None
        self.hidden_norms: Optional[nn.ModuleList] = None
        self.hidden_films: Optional[nn.ModuleList] = None
        self.hidden_dropouts: Optional[nn.ModuleList] = None
        self.final_linear: Optional[nn.Linear] = None
        self._mlp_in_features = None

        # Build once at init so model size is reflected before any forward().
        # Forward still rebuilds if it receives a different input length.
        self._build_mlp(self.input_length, device=torch.device("cpu"))

        self.output_dim = emb_dim + (self.cond_dim if (self.cond and self.cond_op == "cat") else 0)

    def forward(self, x, conds):
        # x: [B, 1, L]
        b, _, l = x.shape
        flat = x.view(b, l)  # [B, L]
        cond_emb = self.cond_emb_layer(conds) if self.cond else None
        feat = flat
        # (Re)build the MLP if input feature size differs from expected (e.g., HRRP length 201)
        in_features = feat.shape[1]
        if self._mlp_in_features != in_features or self.hidden_linears is None:
            self._build_mlp(in_features, device=feat.device)

        # Forward with per-layer FiLM injection (regular intervals) for film/cbn
        for i, linear in enumerate(self.hidden_linears):
            z_in = feat
            if self.cond and self.cond_op == "cat" and cond_emb is not None:
                z_in = torch.cat([z_in, cond_emb], dim=1)
            z = linear(z_in)
            norm = self.hidden_norms[i]
            if not isinstance(norm, nn.Identity):
                z = norm(z)
            if self.cond and self.cond_op in {"film", "cbn"} and cond_emb is not None:
                gamma, beta = self.hidden_films[i](cond_emb).chunk(2, dim=-1)
                z = z * (1 + gamma) + beta
            z = F.silu(z)
            z = self.hidden_dropouts[i](z)
            feat = z

        feat_in = feat
        if self.cond and self.cond_op == "cat" and cond_emb is not None:
            feat_in = torch.cat([feat_in, cond_emb], dim=1)
        feat = self.final_linear(feat_in)
        if self.cond and self.cond_op == "cat" and cond_emb is not None:
            feat = torch.cat([feat, cond_emb], dim=1)
        return feat

    def _build_mlp(self, in_features: int, device):
        self.hidden_linears = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        self.hidden_films = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()

        curr = in_features
        for h in self.hidden_dims:
            in_dim = curr + (self.cond_dim if (self.cond and self.cond_op == "cat") else 0)
            self.hidden_linears.append(nn.Linear(in_dim, h))
            if self.cond and self.cond_op == "cbn":
                norm_layer = nn.BatchNorm1d(h, affine=False)
            elif self.cond and self.cond_op == "film":
                norm_layer = nn.BatchNorm1d(h, affine=True)
            elif self.use_ln:
                norm_layer = nn.LayerNorm(h)
            elif self.use_bn:
                norm_layer = nn.BatchNorm1d(h)
            else:
                norm_layer = nn.Identity()
            self.hidden_norms.append(norm_layer)

            if self.cond and self.cond_op in {"film", "cbn"}:
                self.hidden_films.append(nn.Linear(self.cond_dim, h * 2))
            else:
                self.hidden_films.append(nn.Identity())

            self.hidden_dropouts.append(nn.Dropout(self.dropout))
            curr = h

        final_in = curr + (self.cond_dim if (self.cond and self.cond_op == "cat") else 0)
        self.final_linear = nn.Linear(final_in, self.emb_dim)

        # move to correct device
        self.hidden_linears.to(device)
        self.hidden_norms.to(device)
        self.hidden_films.to(device)
        self.hidden_dropouts.to(device)
        self.final_linear.to(device)
        self._mlp_in_features = in_features


def create_feature_extractor(config):
    """
    Factory to instantiate the requested classifier backbone.
    Defaults to ResNetRP to preserve historical behaviour.
    """
    clf_cfg = copy.deepcopy(config.get("clf", {}))
    cond_cfg = config.get("conditionned", {})
    arch = str(clf_cfg.get("arch", "resnet_rp")).lower()

    cond_flag = bool(cond_cfg.get("bool", clf_cfg.get("cond", False)))
    cond_op = cond_cfg.get("cond_op", clf_cfg.get("cond_op", "film"))
    clf_cfg["cond"] = cond_flag
    clf_cfg["cond_op"] = cond_op

    if arch in {"resnet", "resnet_rp"}:
        return ResNetRP(**clf_cfg)
    if arch in {"cnn", "conv", "conv1d"}:
        return Conv1dBackbone(**clf_cfg)
    if arch in {"lstm", "rnn", "gru"}:
        return LSTMBackbone(**clf_cfg)
    if arch in {"mlp", "ffn", "feedforward"}:
        return MLPBackbone(**clf_cfg)
    raise ValueError(f"Unknown classifier architecture '{arch}'")

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_class = config.get("num_classes", 10 if config["dataset"] == "MSTAR" else 30)
        self.global_feature = config.get("global_feature", None)

        if self.global_feature == "total_energy":
            # Simple classifier for a single global feature
            self.feature_extractor = None
            self.classifier = nn.Sequential(
                nn.Linear(1, 64),
                nn.SiLU(),
                nn.Linear(64, num_class)
            )
        else:
            self.feature_extractor = create_feature_extractor(config)
            in_h = getattr(self.feature_extractor, "output_dim", config["clf"]["emb_dim"])
            self.classifier = MLPHead(in_h=in_h, num_class=num_class)

    def forward(self, x, conds):
        if self.global_feature == "total_energy":
            features = x.view(x.size(0), -1)  # [B, 1]
            return self.classifier(features)
        else:
            features = self.feature_extractor(x, conds)
            logits = self.classifier(features)
            return logits

class ClassifierPL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(config)
        self.num_classes = config.get("num_classes", 10 if config["dataset"] == "MSTAR" else 30)
        # Optional class weighting for imbalanced datasets
        class_weights = config.get("class_weights", None)
        if class_weights is not None and len(class_weights) != self.num_classes:
            raise ValueError(f"class_weights length {len(class_weights)} != num_classes {self.num_classes}")
        self.register_buffer(
            "class_weights_tensor",
            torch.tensor(class_weights, dtype=torch.float) if class_weights is not None else None
        )
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        self.lr = config["lr"]
        # Macro-F1 over all classes (logged on epoch for reliable aggregation)
        self.val_macro_f1_metric = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.test_macro_f1_metric = MulticlassF1Score(num_classes=self.num_classes, average="macro")

    def forward(self, x, conds):
        return self.model(x, conds)

    def training_step(self, batch, batch_idx):
        x, conds, y = batch
        x, conds = x.view(x.size(0), 1, -1), conds.view(conds.size(0), 1)
        y = y.view(-1).long()
        logits = self(x, conds)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        self.log('train_acc', (torch.argmax(logits, dim=1) == y).float().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        x, conds, y = batch
        x, conds = x.view(x.size(0), 1, -1), conds.view(conds.size(0), 1)
        y = y.view(-1).long()
        if y.numel() == 0:
            return
        logits = self(x, conds)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.val_macro_f1_metric.update(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    def on_validation_epoch_start(self):
        self.val_macro_f1_metric.reset()

    def on_validation_epoch_end(self):
        self.log('val_macro_f1', self.val_macro_f1_metric.compute(), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, conds, y = batch
        x, conds = x.view(x.size(0), 1, -1), conds.view(conds.size(0), 1)
        y = y.view(-1).long()
        if y.numel() == 0:
            return
        logits = self(x, conds)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.test_macro_f1_metric.update(preds, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        self.test_macro_f1_metric.reset()

    def on_test_epoch_end(self):
        macro_f1 = self.test_macro_f1_metric.compute()
        self.log('test_macro_f1', macro_f1, prog_bar=True, sync_dist=True)
        self.log('test_f1', macro_f1, prog_bar=False, sync_dist=True)

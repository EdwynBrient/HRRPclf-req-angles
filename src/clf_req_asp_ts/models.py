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
try:
    from .utils import *
except ImportError:
    from utils import *
from functools import partial
import time
import math
from torchmetrics.classification import MulticlassF1Score


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:
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
        
        if cdim is not None and cond_op == "inres":
            self.scal_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cdim, planes),
            )
        elif cdim is not None and cond_op == "film":
            # FiLM (scale + shift) as in the Classification project
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
        # Inject condition only when provided; necessary for masked cond_levels (inres)
        if self.cond_op == "inres" and self.scal_proj is not None and scal_emb is not None:
            out += self.scal_proj(scal_emb).reshape(-1, self.planes, 1)
        elif self.cond_op == "film" and self.scal_proj is not None and scal_emb is not None:
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
        if self.cond:
            self.scal_emb_layer = EmbedLinear(self.mod_ch, cdim)
        for i in range(len(ch_mul)-1):
            in_ch = ch_mul[i]*mod_ch
            out_ch = ch_mul[i+1]*mod_ch
            self.net_blocks.append(ResRP(in_ch, out_ch, dropout=dropout, cdim=cdim if cond else None, cond_op=cond_op))
            for j in range(num_res-1):
                self.net_blocks.append(ResRP(out_ch, out_ch, dropout=dropout, cdim=cdim if cond else None, cond_op=cond_op))
        self.conv_out = nn.Conv1d(ch_mul[-1]*mod_ch, emb_dim, kernel_size=7, stride=1, padding=3)

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
                    inject = scal  # always inject for concatenation
                elif self.cond and self.cond_op in {"inres", "cbn", "film"} and (self.cond_levels is None or i in self.cond_levels):
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

def sinusoidal_positional_encoding(length: int, dim: int, device: torch.device):
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [L, D]


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=4, num_layers=2, dropout=0.1, pooling="cls", dim_feedforward=1024):
        super().__init__()
        self.pooling = pooling
        self.proj = nn.Linear(input_dim, d_model) if input_dim != d_model else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=dim_feedforward,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls_token = None
        self.output_dim = d_model

    def forward(self, x):
        # x: [B, L, D]
        if self.proj is not None:
            x = self.proj(x)
        b, l, d = x.shape
        pos = sinusoidal_positional_encoding(l + (1 if self.pooling == "cls" else 0), d, x.device)
        pos = pos.unsqueeze(0)  # [1, L(+1), D]
        if self.pooling == "cls":
            cls_token = self.cls_token.expand(b, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        x = x + pos[:, : x.size(1)]
        out = self.encoder(x)
        if self.pooling == "cls":
            return out[:, 0, :]
        return out.mean(dim=1)

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_class = config.get("num_classes", 10 if config["dataset"] == "MSTAR" else 30)
        self.global_feature = config.get("global_feature", None)
        self.sequence_length = int(config.get("sequence_length", 1))
        abl_cfg = config.get("ablations", {}) or {}
        self.input_mode = abl_cfg.get("input_mode", "sequence")
        self.cond_fusion = abl_cfg.get("cond_fusion", "none")
        self.logits_pool = self.input_mode == "logit_pool"
        # Late fusion appends a small cond embedding right before the classifier
        self.cond_late_dim = 0 if self.logits_pool else (2 if self.cond_fusion == "late_cat" else 0)
        self._nan_logged = False

        if self.global_feature == "total_energy":
            # Classifieur simple pour feature global unique
            n_global_feat = 1 + int(config.get("global_feature_include_meta", False))
            self.feature_extractor = None
            self.classifier = nn.Sequential(
                nn.Linear(n_global_feat, 64),
                nn.SiLU(),
                nn.Linear(64, num_class)
            )
        else:
            # When conditions are concatenated, the feature size grows by emb_dim//16 (not a float factor)
            emb_dim = config["clf"]["emb_dim"]
            cond_cat = config["conditionned"]["bool"] and config["conditionned"]["cond_op"] == "cat"
            in_h = emb_dim + emb_dim // 16 if cond_cat else emb_dim
            self.feature_extractor = ResNetRP(**config["clf"])
            # Agrégation temporelle configurable (par défaut: moyenne sur la séquence)
            temporal_cfg = config.get("temporal", {})
            self.temporal_type = temporal_cfg.get("type", "mean")
            self.temporal_cond_op = str(temporal_cfg.get("cond_op", "none")).lower()
            self.temporal_cond_dim = int(temporal_cfg.get("cond_dim", 32))
            self.temporal_cond_hidden = int(temporal_cfg.get("cond_hidden", max(self.temporal_cond_dim * 2, emb_dim // 4 if emb_dim > 0 else 32)))
            self.temporal_cond_dropout = float(temporal_cfg.get("cond_dropout", 0.0))
            self.temporal_cond_reduction = str(temporal_cfg.get("cond_reduction", "none")).lower()
            cond_base_dim = 2  # sin/cos
            self._temporal_use_cond = self.temporal_cond_op in {"concat", "film"}
            if self._temporal_use_cond:
                cond_proj_dim = int(temporal_cfg.get("cond_proj_dim", self.temporal_cond_dim))
                if cond_proj_dim > cond_base_dim:
                    self.temporal_cond_proj = nn.Sequential(
                        nn.Linear(cond_base_dim, cond_proj_dim),
                        nn.SiLU(),
                    )
                else:
                    self.temporal_cond_proj = None
                self._temporal_cond_feat_dim = cond_proj_dim if cond_proj_dim > cond_base_dim else cond_base_dim
            else:
                self.temporal_cond_proj = None
                self._temporal_cond_feat_dim = 0
            if self.temporal_cond_op == "film":
                film_out_dim = in_h
                self.temporal_film = nn.Sequential(
                    nn.Linear(self._temporal_cond_feat_dim, self.temporal_cond_hidden),
                    nn.SiLU(),
                    nn.Dropout(self.temporal_cond_dropout),
                    nn.Linear(self.temporal_cond_hidden, film_out_dim * 2),
                )
            else:
                self.temporal_film = None
            temporal_input_dim = in_h + (self._temporal_cond_feat_dim if self.temporal_cond_op == "concat" else 0)
            hidden_dim = int(temporal_cfg.get("hidden_dim", in_h))
            num_layers = int(temporal_cfg.get("num_layers", 1))
            bidir = bool(temporal_cfg.get("bidirectional", False))
            dropout = float(temporal_cfg.get("dropout", 0.0))
            if self.temporal_type == "gru":
                self.temporal = nn.GRU(
                    input_size=temporal_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidir,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                classifier_in = hidden_dim * (2 if bidir else 1)
            elif self.temporal_type == "lstm":
                self.temporal = nn.LSTM(
                    input_size=temporal_input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidir,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                classifier_in = hidden_dim * (2 if bidir else 1)
            elif self.temporal_type == "transformer":
                d_model = int(temporal_cfg.get("d_model", temporal_input_dim))
                nhead = int(temporal_cfg.get("nhead", 4))
                pooling = temporal_cfg.get("pooling", "cls")
                dim_ff = int(temporal_cfg.get("dim_feedforward", 1024))
                self.temporal = TemporalTransformer(
                    input_dim=temporal_input_dim,
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=num_layers,
                    dropout=dropout,
                    pooling=pooling,
                    dim_feedforward=dim_ff,
                )
                classifier_in = self.temporal.output_dim
            else:
                self.temporal = None
                classifier_in = in_h
            classifier_in += self.cond_late_dim
            self.classifier = MLPHead(in_h=classifier_in, num_class=num_class)

    def forward(self, x, conds):
        if (not self._nan_logged) and ((torch.isnan(x).any() or torch.isinf(x).any()) or (conds is not None and (torch.isnan(conds).any() or torch.isinf(conds).any()))):
            print("[DEBUG] NaN/Inf detected in model input "
                  f"x nan={torch.isnan(x).sum().item()} inf={torch.isinf(x).sum().item()} "
                  f"conds nan={(torch.isnan(conds).sum().item() if conds is not None else 'n/a')} "
                  f"inf={(torch.isinf(conds).sum().item() if conds is not None else 'n/a')} "
                  f"x min={x.min().item():.4f} max={x.max().item():.4f}")
            self._nan_logged = True
        if self.global_feature == "total_energy":
            # x: [B, L, n_feat] ou [B, n_feat]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            # Agrégation simple: moyenne temporelle
            features = x.mean(dim=1)
            return self.classifier(features)

        # x: [B, L, R] ou [B, R]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if conds is None:
            conds = torch.zeros(x.size(0), x.size(1), device=x.device)
        if conds.dim() == 1:
            conds = conds.unsqueeze(1)

        b, l, r = x.shape
        if self.input_mode == "concat" and l > 1:
            # concatène les HRRP sur la dimension range
            x = x.reshape(b, 1, l * r)
            conds = conds.mean(dim=1, keepdim=True)
            l = 1
            r = x.size(-1)

        x_flat = x.view(b * l, 1, r)
        conds_flat = conds.view(b * l, 1)
        step_features = self.feature_extractor(x_flat, conds_flat)  # [B*L, D]
        step_features = step_features.view(b, l, -1)  # [B, L, D]

        if self.logits_pool:
            logits = self.classifier(step_features)  # [B, L, C]
            logits = logits.view(b, l, -1)
            return logits.mean(dim=1)

        if self._temporal_use_cond:
            temporal_cond = self._build_temporal_cond(conds)  # [B, L, Cc]
            if self.temporal_cond_reduction == "mean":
                temporal_cond = temporal_cond.mean(dim=1, keepdim=True)  # [B, 1, Cc]
                # pour concat, on répète au long de la séquence
                if self.temporal_cond_op == "concat":
                    temporal_cond = temporal_cond.expand(-1, l, -1)
        else:
            temporal_cond = None
        temporal_in = step_features
        if self.temporal_cond_op == "concat" and temporal_cond is not None:
            temporal_in = torch.cat([temporal_in, temporal_cond], dim=-1)
        elif self.temporal_cond_op == "film" and temporal_cond is not None:
            gamma, beta = self.temporal_film(temporal_cond).chunk(2, dim=-1)
            temporal_in = temporal_in * (1 + gamma) + beta

        if self.temporal_type in {"gru", "lstm"}:
            out, h_state = self.temporal(temporal_in)
            h = h_state if self.temporal_type == "gru" else h_state[0]
            if self.temporal.bidirectional:
                # h shape: [num_layers*2, B, H]
                h_fwd = h[-2]
                h_bwd = h[-1]
                features = torch.cat([h_fwd, h_bwd], dim=-1)
            else:
                features = h[-1]
        elif self.temporal_type == "transformer":
            features = self.temporal(temporal_in)
        else:
            # Pooling moyen par défaut
            features = temporal_in.mean(dim=1)

        if self.cond_late_dim:
            cond_vec = self._build_late_cond(conds)
            features = torch.cat([features, cond_vec], dim=1)

        logits = self.classifier(features)
        if (not self._nan_logged) and (torch.isnan(logits).any() or torch.isinf(logits).any()):
            print("[DEBUG] NaN/Inf detected in model logits")
            self._nan_logged = True
        return logits

    def _build_late_cond(self, conds: torch.Tensor) -> torch.Tensor:
        """
        Construit un embedding angulaire compact (sin/cos) pour late fusion.
        conds: [B, L]
        """
        if conds.dim() == 1:
            conds = conds.unsqueeze(1)
        cond_mean = conds.mean(dim=1)
        return torch.stack([torch.sin(cond_mean), torch.cos(cond_mean)], dim=1)

    def _build_temporal_cond(self, conds: torch.Tensor) -> torch.Tensor:
        """
        Encode les angles de visée pour le conditionnement temporel.
        Retourne [B, L, C] (sin/cos puis éventuelle projection).
        """
        if conds.dim() == 1:
            conds = conds.unsqueeze(1)
        cond_vec = torch.stack([torch.sin(conds), torch.cos(conds)], dim=-1)
        if self.temporal_cond_proj is not None:
            cond_vec = self.temporal_cond_proj(cond_vec)
        return cond_vec

class ClassifierPL(pl.LightningModule):
    def __init__(self, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = Classifier(config)
        self.num_classes = int(config.get("num_classes", 10 if config["dataset"] == "MSTAR" else 30))
        # class_weights (Tensor) optionnel pour pondérer la CE
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            raise ValueError("class_weights must be provided for CrossEntropyLoss")
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        self.lr = float(config["lr"])
        self.val_macro_f1_metric = MulticlassF1Score(num_classes=self.num_classes, average="macro")
        self.test_macro_f1_metric = MulticlassF1Score(num_classes=self.num_classes, average="macro")

    def _unpack_batch(self, batch):
        if len(batch) == 3:
            x, conds, y = batch
            lengths = None
        elif len(batch) == 4:
            x, conds, y, lengths = batch
        else:
            raise ValueError(f"Unexpected batch format: {len(batch)} items")
        return x, conds, y, lengths

    def forward(self, x, conds):
        return self.model(x, conds)

    def training_step(self, batch, batch_idx):
        x, conds, y, lengths = self._unpack_batch(batch)
        y = y.view(-1).long()
        logits = self(x, conds)
        loss = self.criterion(logits, y)
        if not torch.isfinite(loss).all():
            print(f"[DEBUG] Non-finite loss at batch {batch_idx} | "
                  f"logits nan={torch.isnan(logits).sum().item()} inf={torch.isinf(logits).sum().item()} "
                  f"logits min={logits.min().item():.4f} max={logits.max().item():.4f} "
                  f"x min={x.min().item():.4f} max={x.max().item():.4f} "
                  f"conds min={(conds.min().item() if conds is not None else 0):.4f} "
                  f"conds max={(conds.max().item() if conds is not None else 0):.4f}")
        self.log('train_loss', loss)
        self.log('train_acc', (torch.argmax(logits, dim=1) == y).float().mean())
        return loss

    def validation_step(self, batch, batch_idx):
        x, conds, y, lengths = self._unpack_batch(batch)
        y = y.view(-1).long()
        if y.numel() == 0:
            return
        logits = self(x, conds)
        raw_loss = self.criterion(logits, y)
        nan_frac = torch.isnan(raw_loss).float().mean()
        loss = torch.nan_to_num(raw_loss, nan=1e4, posinf=1e4, neginf=1e4)
        preds = torch.argmax(logits, dim=1)
        acc = torch.nan_to_num((preds == y).float().mean(), nan=0.0)
        self.val_macro_f1_metric.update(preds, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_loss_nan_frac', nan_frac, prog_bar=False, sync_dist=True)
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)

    def on_validation_epoch_start(self):
        self.val_macro_f1_metric.reset()

    def on_validation_epoch_end(self):
        self.log('val_macro_f1', self.val_macro_f1_metric.compute(), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        sched_cfg = self.hparams["config"].get("lr_scheduler", None)
        if sched_cfg is None:
            return optimizer
        # Accept string or dict
        if isinstance(sched_cfg, str):
            sched_type = sched_cfg.lower()
            sched_params = {}
        else:
            sched_type = str(sched_cfg.get("type", "cosine")).lower()
            sched_params = sched_cfg
        if sched_type == "cosine":
            t_max = int(sched_params.get("t_max", self.hparams["config"].get("epochs", 1)))
            eta_min = float(sched_params.get("eta_min", self.lr / 50))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, conds, y, lengths = self._unpack_batch(batch)
        y = y.view(-1).long()
        if y.numel() == 0:
            return
        logits = self(x, conds)
        raw_loss = self.criterion(logits, y)
        nan_frac = torch.isnan(raw_loss).float().mean()
        loss = torch.nan_to_num(raw_loss, nan=1e4, posinf=1e4, neginf=1e4)
        preds = torch.argmax(logits, dim=1)
        acc = torch.nan_to_num((preds == y).float().mean(), nan=0.0)
        self.test_macro_f1_metric.update(preds, y)
        self.log('test_loss', loss, prog_bar=True, sync_dist=True)
        self.log('test_loss_nan_frac', nan_frac, prog_bar=False, sync_dist=True)
        self.log('test_acc', acc, prog_bar=True, sync_dist=True)
        # Accuracies par longueur de séquence pour mesurer l'apport d'information
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), device=x.device)
        max_len = int(self.hparams["config"].get("sequence_length", int(lengths.max().item())))
        lengths = torch.clamp(lengths, max=max_len)
        for l in range(2, max_len + 1):
            mask = lengths == l
            if mask.any():
                acc_l = torch.nan_to_num((preds[mask] == y[mask]).float().mean(), nan=0.0)
                self.log(f'test_acc_len{l}', acc_l, prog_bar=False, sync_dist=True)
        self.log('test_len_mean', lengths.float().mean(), prog_bar=False, sync_dist=True)

    def on_test_epoch_start(self):
        self.test_macro_f1_metric.reset()

    def on_test_epoch_end(self):
        self.log('test_macro_f1', self.test_macro_f1_metric.compute(), prog_bar=True, sync_dist=True)

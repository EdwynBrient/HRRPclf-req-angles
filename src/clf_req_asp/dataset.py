import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
try:
    from .utils import *
except ImportError:  # pragma: no cover - fallback for direct script execution
    from utils import *
import torch.nn.functional as F
import torch
from torchvision import transforms
from torchvision.models import vgg16
import matplotlib.pyplot as plt
import skimage
import timm
from tqdm import tqdm
import pandas as pd
from pathlib import Path

selectRP = [str(i) for i in range(200)]

TAU = 2*np.pi
EPS = 1e-9


def approx_viewing_angle_from_xy_kalman_track(
    x_meas: np.ndarray,
    y_meas: np.ndarray,
    dts=None,
    base_va=None,
    *,
    turn: bool = True,
    temp_gap: bool = True,
    max_dt_gap: float = 20 * 60,
    smooth_window: int = 2,
    use_filtered_pos_for_az: bool = True,
    sigma_pos: float = 60.0,
    sigma_acc: float = 0.1,
    sigma_acc_turn: float = 2.0,
    turn_ang_thr: float = 0.15,
    min_speed: float = 0.2,
):
    """
    Estime une suite de viewing_angle (rad dans [0, 2π)) le long d'une trajectoire (x,y)
    via un Kalman constant-velocity (x, y, vx, vy). Retourne (va, los).
    """
    n = len(x_meas)
    va_out = np.full(n, np.nan, dtype=float)
    los_out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return va_out, los_out

    if dts is None:
        dts = np.ones(n - 1, dtype=float)
    dts = np.asarray(dts, dtype=float)
    dts = np.where(dts > EPS, dts, EPS)

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]], dtype=float)
    R = (sigma_pos ** 2) * np.eye(2, dtype=float)

    def init_state(i0, i1):
        dt0 = float(dts[max(0, i1 - 1)])
        vx0 = (x_meas[i1] - x_meas[i0]) / dt0
        vy0 = (y_meas[i1] - y_meas[i0]) / dt0
        x0 = np.array([x_meas[i0], y_meas[i0], vx0, vy0], dtype=float)
        P0 = np.diag([sigma_pos ** 2, sigma_pos ** 2, (10 * sigma_pos) ** 2, (10 * sigma_pos) ** 2]).astype(float)
        head0 = np.arctan2((y_meas[i1] - y_meas[i0]), (x_meas[i1] - x_meas[i0]))
        return x0, P0, head0

    x_state, P, prev_meas_heading = init_state(0, 1)
    va_out[0] = np.nan
    los_out[0] = np.nan

    xs_f, ys_f, vxs_f, vys_f = x_meas.astype(float).copy(), y_meas.astype(float).copy(), np.zeros(n), np.zeros(n)
    xs_f[0], ys_f[0], vxs_f[0], vys_f[0] = x_state

    for i in range(1, n):
        dt = float(dts[i - 1]) if i - 1 < len(dts) else 1.0

        # reset sur gap temporel important
        if temp_gap and dt > max_dt_gap and i < n:
            x_state, P, prev_meas_heading = init_state(i - 1, i)
            xs_f[i], ys_f[i], vxs_f[i], vys_f[i] = x_state
            if base_va is not None:
                va_out[i] = base_va[i]
            los_out[i] = np.mod(np.arctan2(y_meas[i], x_meas[i]), TAU)
            continue

        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)

        if turn and i >= 2:
            dhx = x_meas[i] - x_meas[i - 1]
            dhy = y_meas[i] - y_meas[i - 1]
            cur_meas_heading = np.arctan2(dhy, dhx)
            dhead = (cur_meas_heading - prev_meas_heading + np.pi) % (2 * np.pi) - np.pi
            prev_meas_heading = cur_meas_heading
            sigma_a = sigma_acc_turn if abs(dhead) > turn_ang_thr else sigma_acc
        else:
            sigma_a = sigma_acc

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = sigma_a ** 2
        Q = q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                          [0, dt4 / 4, 0, dt3 / 2],
                          [dt3 / 2, 0, dt2, 0],
                          [0, dt3 / 2, 0, dt2]], dtype=float)

        x_pred = F @ x_state
        P_pred = F @ P @ F.T + Q

        z = np.array([x_meas[i], y_meas[i]], dtype=float)
        y_res = z - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_state = x_pred + K @ y_res
        P = (np.eye(4) - K @ H) @ P_pred

        xs_f[i], ys_f[i], vxs_f[i], vys_f[i] = x_state

        x_last = xs_f[i] if use_filtered_pos_for_az else x_meas[i]
        y_last = ys_f[i] if use_filtered_pos_for_az else y_meas[i]
        vx_last, vy_last = vxs_f[i], vys_f[i]

        speed = float(np.hypot(vx_last, vy_last))
        if (not np.isfinite(speed)) or speed < min_speed:
            vx_last = (x_meas[i] - x_meas[i - 1]) / dt
            vy_last = (y_meas[i] - y_meas[i - 1]) / dt

        heading_ais = (np.pi / 2 - np.arctan2(vy_last, vx_last)) % TAU
        Az = np.arctan2(y_last, x_last) % TAU
        heading_rot = (-((heading_ais - np.pi / 2) % TAU)) % TAU
        viewing_angle = (heading_rot - Az) % TAU

        va_out[i] = viewing_angle
        los_out[i] = Az

    if base_va is not None:
        base_va_arr = np.asarray(base_va, dtype=float)
        nan_mask = np.isnan(va_out)
        va_out[nan_mask] = base_va_arr[nan_mask]
    if np.isnan(va_out[0]) and n > 1:
        va_out[0] = va_out[1]
        los_out[0] = los_out[1]

    return va_out, los_out


def approx_viewing_angle_from_xy_causal(
    df: pd.DataFrame,
    x_col="X",
    y_col="Y",
    time_col="unix_seconds",
    az_col=None,
    smooth_window=5,
    eval_idx=-1,
    max_dt_gap=20 * 60,      # 20 minutes en secondes
    max_history=9,
    # détection virage
    micro_thr=0.05,
    monotone_frac_thr=0.85,
    total_turn_thr=0.4,
    # lissage léger en virage
    turn_sigma=1.0,
    return_heading=False,
):
    """
    Estime l'angle de vue AU POINT eval_idx en utilisant uniquement le passé,
    en coupant l'historique au dernier gap temporel > max_dt_gap et en se
    limitant à max_history points antérieurs.
    """

    if df.shape[0] < 2:
        raise ValueError("Besoin d'au moins 2 points.")

    if time_col:
        df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    if eval_idx < 0:
        eval_idx = n + eval_idx
    if not (1 <= eval_idx < n):
        raise ValueError(f"eval_idx doit être dans [1, {n-1}]")

    # ============================
    # 1) COUPURE SUR GAP TEMPOREL + HISTOIRE LIMITÉE
    # ============================
    t = df[time_col].to_numpy()

    # borne de fenêtre d'historique (<= max_history pas en arrière)
    history_start = max(0, eval_idx - max_history)
    t_window = t[history_start:eval_idx + 1]
    dt = np.diff(t_window)

    # indices où dt > seuil (dans la fenêtre réduite)
    gap_idx = np.where(dt > max_dt_gap)[0]

    if len(gap_idx) > 0:
        # on coupe juste après le DERNIER gap dans la fenêtre
        cut = history_start + gap_idx[-1] + 1
    else:
        cut = history_start

    sub = df.iloc[cut:eval_idx + 1]

    if len(sub) < 2:
        # pas assez de points après coupure → fallback minimal
        sub = df.iloc[max(eval_idx-1, 0):eval_idx+1]

    # ============================
    # 2) GRADIENTS CAUSAUX
    # ============================
    x = sub[x_col].to_numpy()
    y = sub[y_col].to_numpy()
    t = sub[time_col].to_numpy()

    dx = np.gradient(x, t)
    dy = np.gradient(y, t)

    W = int(max(2, min(smooth_window, len(dx))))
    dx_w = dx[-W:]
    dy_w = dy[-W:]

    # ============================
    # 3) DÉTECTION VIRAGE
    # ============================
    ang = np.arctan2(dy_w, dx_w)
    ang_u = np.unwrap(ang)
    d_ang = np.diff(ang_u)

    d_sig = d_ang[np.abs(d_ang) > micro_thr]
    is_monotone_turn = False
    if len(d_sig) >= 3:
        frac_pos = np.mean(d_sig > 0)
        frac_neg = np.mean(d_sig < 0)
        monotone_strength = max(frac_pos, frac_neg)
        total_turn = float(np.sum(np.abs(d_sig)))
        is_monotone_turn = (
            monotone_strength > monotone_frac_thr
            and total_turn > total_turn_thr
        )

    # ============================
    # 4) LISSAGE CAUSAL
    # ============================
    if is_monotone_turn:
        idx = np.arange(W, dtype=float)
        w = np.exp(-0.5 * ((idx - (W - 1)) / float(turn_sigma)) ** 2)
        w /= np.sum(w)
        dx_s = float(np.sum(dx_w * w))
        dy_s = float(np.sum(dy_w * w))
    else:
        dx_s = float(np.mean(dx_w))
        dy_s = float(np.mean(dy_w))

    # fallback sécurité
    if not np.isfinite(dx_s) or not np.isfinite(dy_s) or (dx_s == 0 and dy_s == 0):
        dx_s = float(x[-1] - x[-2])
        dy_s = float(y[-1] - y[-2])

    # ============================
    # 5) ANGLES
    # ============================
    heading_ais = (np.pi / 2 - np.arctan2(dy_s, dx_s)) % (2 * np.pi)

    if az_col and az_col in df.columns:
        az_val = df.loc[eval_idx, az_col]
        if pd.isna(az_val) or not np.isfinite(az_val):
            Az = np.arctan2(y[-1], x[-1]) % (2 * np.pi)
        else:
            Az = np.mod(float(az_val), 2 * np.pi)
    else:
        Az = np.arctan2(y[-1], x[-1]) % (2 * np.pi)

    heading_rot = (-((heading_ais - np.pi / 2) % (2 * np.pi))) % (2 * np.pi)
    viewing_angle = (heading_rot - Az) % (2 * np.pi)

    if return_heading:
        return viewing_angle, heading_ais
    return viewing_angle


def angle_to_bucket(a, orientation_rad_dict):
    """
    a: np.ndarray d'angles (rad), quelconques -> normalisés dans [0, 2π)
    orientation_rad_dict: dict { bucket:int -> list[[lo,hi], [lo,hi], ...] }
    retourne: np.ndarray int8 (0..3, ou -1 si aucun intervalle)
    """
    a = np.asarray(a, dtype=np.float64)
    a = np.mod(a, TAU)  # normalise dans [0, 2π)
    out = np.full(a.shape, -1, dtype=np.int8)

    for b in range(4):
        ranges = orientation_rad_dict[b]  # p.ex. [[lo1,hi1],[lo2,hi2]]
        sel_b = np.zeros_like(a, dtype=bool)
        for lo, hi in ranges:
            lo = lo % TAU
            hi = hi % TAU
            if lo <= hi:
                sel = (a >= lo) & (a < hi)
            else:
                # wrap-around: [lo, 2π) U [0, hi)
                sel = (a >= lo) | (a < hi)
            sel_b |= sel
        out[sel_b] = b
    return out

class RP_ImageDataset(Dataset):
    def __init__(self, config, path_rp="../data/all_df_10_25"):
        path_rp = config.get("path_rp", path_rp)
        path_rp_obj = Path(path_rp)
        if path_rp_obj.suffix.lower() in {".pt", ".pth"}:
            self.df = _load_ship_hrrp_pt(path_rp_obj, target_len=len(selectRP))
        else:
            self.df = pd.read_csv(path_rp)
        self.df = self.df[~(self.df.length==0)]
        self.config=config
        self.mmsi_selection_cfg = config.get("mmsi_selection", {}) or {}
        self.min_mmsi_samples = int(
            config.get(
                "min_mmsi_samples",
                config.get("min_samples_per_mmsi", self.mmsi_selection_cfg.get("min_samples", 30))
            )
        )
        self.min_mmsi_samples = max(1, self.min_mmsi_samples)
        self.viewing_angle_source = str(config.get("viewing_angle_source", "real")).lower()
        # Par défaut on préfère l'angle réel, sauf pour les modes kalman où l'on veut utiliser l'estimation après sélection.
        default_prefer_true = False if self.viewing_angle_source in {"positions_kalman", "kalman"} else True
        self.prefer_true_viewing_angle = bool(config.get("prefer_true_viewing_angle", default_prefer_true))
        self.viewing_angle_col = config.get("viewing_angle_col", None)
        if self.viewing_angle_col is None:
            for cand in ["viewing_angle_true", "viewing_angle_real", "viewing_angle"]:
                if cand in self.df.columns:
                    self.viewing_angle_col = cand
                    break
        if self.viewing_angle_col is None:
            raise KeyError("Aucune colonne d'angle de vue trouvée (ex: viewing_angle)")
        self.viewing_angle_true_col = self.viewing_angle_col
        # Copie brute des angles réels pour les sélections et fallbacks
        self.df["viewing_angle_true"] = self.df[self.viewing_angle_true_col].astype(float)
        # Version alignée sur la convention historique
        self.df["viewing_angle"] = self._align_viewing_angle(self.df["viewing_angle_true"])
        self.kalman_params = config.get("kalman_params", {})
        self.position_cols = (
            self.kalman_params.get("x_col", "X"),
            self.kalman_params.get("y_col", "Y"),
        )
        self.viewing_angle_history = int(config.get("viewing_angle_history", 9))
        self.viewing_angle_history = max(1, self.viewing_angle_history)
        self.viewing_angle_smooth_window = int(config.get("viewing_angle_smooth_window", 5))
        self.va_fallback_jitter = float(config.get("va_fallback_jitter", 0.06))
        self.va_max_dt_gap = float(config.get("va_max_dt_gap", 20 * 60))
        self._va_rng = np.random.default_rng(config.get("va_rng_seed", None))
        self.df["viewing_angle_real"] = self.df.viewing_angle
        # Utilisation optionnelle d'un feature global (ex: énergie totale)
        self.global_feature = config.get("global_feature", None)
        self.global_feature_include_meta = bool(config.get("global_feature_include_meta", False))
        self.lim_data = int(config.get("lim_data", 0))
        self.base = "../data/"
        self.sel_mmsi = config.get("sel_mmsi", [])
        self.va_jitter_std = float(config.get("va_jitter_std", 0.17))
        self.processor  = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
        ])
        # Limiting data for quick experiments
        self.df = self.df.iloc[:self.lim_data] if self.lim_data != 0 else self.df
        count_mmsi = self.df.mmsi.value_counts()
        self.df = self.df[self.df.mmsi.map(count_mmsi) >= self.min_mmsi_samples]
        if self.df.empty:
            raise ValueError(
                f"Aucune donnée après filtrage MMSI (min_mmsi_samples={self.min_mmsi_samples})."
            )
        # Normalisation du mode (évite les espaces/casse) et n cible par défaut
        self.mmsi_selection_mode = str(self.mmsi_selection_cfg.get("mode", "")).strip().lower()
        n_target = int(self.mmsi_selection_cfg.get("n", 30))

        if len(self.sel_mmsi) > 0:
            self.df = self.df[self.df.mmsi.isin(self.sel_mmsi)]
        elif self.mmsi_selection_mode == "similar_size":
            selected_mmsi = self._select_mmsi_similar_size(
                n=n_target,
                length_band=self.mmsi_selection_cfg.get("length_band", 0.1),
                width_band=self.mmsi_selection_cfg.get("width_band", 0.1),
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        elif self.mmsi_selection_mode == "largest":
            selected_mmsi = self._select_mmsi_largest(
                n=n_target,
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        elif self.mmsi_selection_mode == "angle_diverse":
            selected_mmsi = self._select_mmsi_angle_diverse(
                n=n_target,
                bins=self.mmsi_selection_cfg.get("angle_bins", 12),
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        elif self.mmsi_selection_mode == "angle_dense":
            selected_mmsi = self._select_mmsi_angle_dense(
                n=n_target,
                bins=self.mmsi_selection_cfg.get("angle_bins", 12),
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        elif self.mmsi_selection_mode == "angle_dense_length_cap":
            selected_mmsi = self._select_mmsi_angle_dense_length_cap(
                n=n_target,
                bins=self.mmsi_selection_cfg.get("angle_bins", 12),
                bin_width=float(self.mmsi_selection_cfg.get("bin_width", 20.0)),
                max_per_bin=int(self.mmsi_selection_cfg.get("max_per_bin", 2)),
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        elif self.mmsi_selection_mode == "angle_dense_length_prune":
            selected_mmsi = self._select_mmsi_angle_dense_length_prune(
                n=n_target,
                bins=self.mmsi_selection_cfg.get("angle_bins", 12),
                bin_width=float(self.mmsi_selection_cfg.get("bin_width", 20.0)),
                max_per_bin=int(self.mmsi_selection_cfg.get("max_per_bin", 2)),
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        elif self.mmsi_selection_mode == "angle_diverse_length_prune":
            selected_mmsi = self._select_mmsi_angle_diverse_length_prune(
                n=n_target,
                bins=self.mmsi_selection_cfg.get("angle_bins", 12),
                bin_width=float(self.mmsi_selection_cfg.get("bin_width", 20.0)),
                max_per_bin=int(self.mmsi_selection_cfg.get("max_per_bin", 2)),
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        elif self.mmsi_selection_mode == "length_bin_cap":
            selected_mmsi = self._select_mmsi_length_bin_cap(
                bin_width=float(self.mmsi_selection_cfg.get("bin_width", 20.0)),
                max_per_bin=int(self.mmsi_selection_cfg.get("max_per_bin", 2)),
                n=self.mmsi_selection_cfg.get("n", None),
                seed=self.mmsi_selection_cfg.get("seed", None),
            )
            self.df = self.df[self.df.mmsi.isin(selected_mmsi)]
        else:
            # Fallback aligné avec TimeSeriesClf: prend les 30 premiers MMSI uniques
            self.df = self.df[self.df.mmsi.isin(self.df.mmsi.unique()[:30])]
        if self.df.empty:
            raise ValueError("Aucune donnée après sélection MMSI (sel_mmsi/mmsi_selection).")
        requested_n = int(self.mmsi_selection_cfg.get("n", 0) or 0)
        selected_n = int(self.df.mmsi.nunique())
        self.requested_mmsi_n = requested_n
        self.selected_mmsi_n = selected_n
        self.mmsi_selection_mode = self.mmsi_selection_mode or "default"
        if requested_n > 0 and selected_n < requested_n:
            print(
                f"[WARN] mmsi_selection.n={requested_n} mais seulement {selected_n} MMSI sélectionnés "
                f"(min_samples={self.min_mmsi_samples}, mode={self.mmsi_selection_mode})."
            )
        # Détail supplémentaire par bins de longueur si la sélection est incomplète
        unique_mmsi = self.df.mmsi.unique()
        if requested_n > 0 and len(unique_mmsi) < requested_n:
            len_by_bin = (self.df.groupby('mmsi')['length'].mean() // float(self.mmsi_selection_cfg.get("bin_width", 20.0))).astype(int)
            print(f"[mmsi_selection] Seulement {len(unique_mmsi)} MMSI retenus sur {requested_n} demandés. Comptes par bin de longueur (20m par défaut):")
            print(len_by_bin.value_counts().sort_index().to_dict())

        # Recalcule éventuel des angles (post-sélection uniquement)
        should_estimate = (
            self.viewing_angle_source in {"positions_kalman", "kalman", "positions", "positions_causal", "causal"}
            and not self.prefer_true_viewing_angle
        )
        if should_estimate:
            if self.viewing_angle_source in {"positions_kalman", "kalman"}:
                va_from_xy, los_angles = self._estimate_viewing_angle_from_positions_kalman()
                if va_from_xy is not None:
                    self.df["viewing_angle_estimated"] = va_from_xy
                    self.df.viewing_angle = self._align_viewing_angle(va_from_xy)
                    if los_angles is not None:
                        self.df["los_angle"] = los_angles
                else:
                    print("viewing_angle_source=positions_kalman mais positions/temps indisponibles, fallback sur l'angle réel.")
            elif self.viewing_angle_source != "real":
                va_from_xy, heading_est = self._estimate_viewing_angle_from_positions_causal()
                if va_from_xy is not None:
                    self.df["viewing_angle_estimated"] = va_from_xy
                    self.df.viewing_angle = self._align_viewing_angle(va_from_xy)
                    if heading_est is not None:
                        self.df["heading_estimated"] = heading_est
                else:
                    print("viewing_angle_source != 'real' mais positions/temps indisponibles, fallback sur l'angle réel.")
        else:
            # On conserve les angles réels alignés par défaut
            self.df.viewing_angle = self._align_viewing_angle(self.df.viewing_angle_true)

        self.min_rp, self.max_rp = self.normalize_df()
        self.preprocess_vars()
            
    def normalize_df(self):
        # Conserve les valeurs originales pour pouvoir revenir aux mètres si besoin
        self.length_scale = float(self.df.length.max())
        self.width_scale = float(self.df.width.max())
        self.df["length_raw"] = self.df.length
        self.df["width_raw"] = self.df.width

        self.df.length = self.df.length / self.length_scale
        self.df.width = self.df.width / self.width_scale
        min_rp, max_rp = 0., self.df[selectRP].max().max()
        self.df[selectRP] = (self.df[selectRP]- min_rp) / (max_rp - min_rp)
        self.df[selectRP] = self.df[selectRP] * 2 - 1
        return min_rp, max_rp

    def get_key_for_angle(self, angle):
        for key, ranges in self.orientation_rad_dict.items():
            if key != 1:
                for r in ranges:
                    if r[0] <= angle <= r[1]:
                        return key
            else:
                r1, r2 = ranges
                if ((r1[0] <= angle) or (angle <= r1[1]) or (r2[0] <= angle <= r2[1])):
                    return key

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _align_viewing_angle(angle_values):
        """
        Applique la convention historique: (pi - angle) mod 2π puis inversion.
        """
        ang = np.asarray(angle_values, dtype=np.float64)
        ang = np.mod(ang - np.pi, TAU)
        ang = np.mod(-ang, TAU)
        return ang

    def _estimate_viewing_angle_from_positions_kalman(self):
        """
        Estime l'angle de vue via un Kalman (x, y, vx, vy) par MMSI.
        """
        x_col = self.kalman_params.get("x_col", "X")
        y_col = self.kalman_params.get("y_col", "Y")
        required_cols = {x_col, y_col, "mmsi"}
        if not required_cols.issubset(self.df.columns):
            return None, None

        time_col = self.kalman_params.get("time_col", None)
        if time_col is None:
            for cand in ["unix_seconds", "robin_timestamp", "timestamp", "time_of_day"]:
                if cand in self.df.columns:
                    time_col = cand
                    break

        kp = self.kalman_params
        smooth_w = max(1, int(kp.get("smooth_window", 2)))

        def process_group(g):
            g = g.sort_values(time_col) if time_col else g
            xs_raw = g[x_col].to_numpy(dtype=float)
            ys_raw = g[y_col].to_numpy(dtype=float)
            if smooth_w > 1:
                xs = pd.Series(xs_raw).rolling(smooth_w, min_periods=1, center=True).mean().to_numpy(dtype=float)
                ys = pd.Series(ys_raw).rolling(smooth_w, min_periods=1, center=True).mean().to_numpy(dtype=float)
            else:
                xs, ys = xs_raw, ys_raw
            base_col = getattr(self, "viewing_angle_true_col", None)
            if base_col and base_col in g.columns:
                base_va = g[base_col].to_numpy(dtype=float)
            else:
                base_va = g["viewing_angle"].to_numpy(dtype=float) if "viewing_angle" in g else None
            dts = None
            if time_col:
                t = g[time_col].to_numpy(dtype=float)
                dts = np.diff(t)
            va, los = approx_viewing_angle_from_xy_kalman_track(
                xs, ys, dts, base_va=base_va,
                turn=bool(kp.get("turn", True)),
                temp_gap=bool(kp.get("temp_gap", True)),
                max_dt_gap=float(kp.get("max_dt_gap", 20 * 60)),
                smooth_window=smooth_w,
                use_filtered_pos_for_az=bool(kp.get("use_filtered_pos_for_az", True)),
                sigma_pos=float(kp.get("sigma_pos", 60.0)),
                sigma_acc=float(kp.get("sigma_acc", 0.1)),
                sigma_acc_turn=float(kp.get("sigma_acc_turn", 2.0)),
                turn_ang_thr=float(kp.get("turn_ang_thr", 0.15)),
                min_speed=float(kp.get("min_speed", 0.2)),
            )
            return pd.Series(va, index=g.index), pd.Series(los, index=g.index)

        va_list, los_list = [], []
        for _, grp in self.df.groupby("mmsi", sort=False):
            va_grp, los_grp = process_group(grp)
            va_list.append(va_grp)
            los_list.append(los_grp)

        va_all = pd.concat(va_list).sort_index()
        los_all = pd.concat(los_list).sort_index()

        if va_all.isna().any():
            base_col = getattr(self, "viewing_angle_true_col", None)
            fallback_va = self.df[base_col] if base_col and base_col in self.df.columns else self.df.get("viewing_angle")
            if fallback_va is None:
                fallback_va = 0.0
            va_all = va_all.fillna(fallback_va)
        if los_all.isna().any():
            if "Az(rad)" in self.df.columns:
                los_fallback = self.df["Az(rad)"]
            else:
                los_fallback = np.arctan2(self.df[y_col].astype(np.float64), self.df[x_col].astype(np.float64))
            los_all = los_all.fillna(los_fallback)

        va_all = np.nan_to_num(va_all.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        los_all = np.nan_to_num(los_all.to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        va_all = np.mod(va_all, TAU)
        los_all = np.mod(los_all, TAU)
        va_all = pd.Series(va_all, index=self.df.index)
        los_all = pd.Series(los_all, index=self.df.index)
        return va_all.reindex(self.df.index), los_all.reindex(self.df.index)

    def _estimate_viewing_angle_from_positions_causal(self):
        """
        Recalcule un angle de vue causal en utilisant au plus
        `viewing_angle_history` mesures précédentes par MMSI.
        """
        required_cols = {"X", "Y", "mmsi"}
        if not required_cols.issubset(self.df.columns):
            return None, None

        time_col = None
        for cand in ["unix_seconds", "robin_timestamp", "timestamp", "time_of_day"]:
            if cand in self.df.columns:
                time_col = cand
                break
        if time_col is None:
            return None, None

        az_col = "Az(rad)" if "Az(rad)" in self.df.columns else None
        df_sorted = self.df.sort_values(["mmsi", time_col]).copy()
        va_series = pd.Series(index=self.df.index, dtype=np.float64)
        heading_series = pd.Series(index=self.df.index, dtype=np.float64)

        for _, group in df_sorted.groupby("mmsi", sort=False):
            group = group.reset_index().rename(columns={"index": "orig_idx"})
            for i in range(len(group)):
                orig_idx = group.loc[i, "orig_idx"]
                if i == 0:
                    base_col = getattr(self, "viewing_angle_true_col", None)
                    col_name = base_col if base_col and base_col in group.columns else "viewing_angle"
                    base_angle = float(group.loc[i, col_name])
                    jitter = float(self._va_rng.normal(0.0, self.va_fallback_jitter)) if self.va_fallback_jitter > 0 else 0.0
                    va_val = (base_angle + jitter) % TAU
                    heading_val = np.nan
                else:
                    start = max(0, i - self.viewing_angle_history)
                    window = group.loc[start:i].copy()
                    try:
                        va_val, heading_val = approx_viewing_angle_from_xy_causal(
                            window,
                            x_col="X",
                            y_col="Y",
                            time_col=time_col,
                            az_col=az_col,
                            smooth_window=min(self.viewing_angle_smooth_window, len(window)),
                            eval_idx=len(window) - 1,
                            max_dt_gap=self.va_max_dt_gap,
                            max_history=self.viewing_angle_history,
                            return_heading=True,
                        )
                    except Exception:
                        base_col = getattr(self, "viewing_angle_true_col", None)
                        col_name = base_col if base_col and base_col in group.columns else "viewing_angle"
                        base_angle = float(group.loc[i, col_name])
                        jitter = float(self._va_rng.normal(0.0, self.va_fallback_jitter)) if self.va_fallback_jitter > 0 else 0.0
                        va_val = (base_angle + jitter) % TAU
                        heading_val = np.nan
                va_series.at[orig_idx] = va_val
                heading_series.at[orig_idx] = heading_val

        return va_series.sort_index(), heading_series.sort_index()
    
    def preprocess_vars(self):
        self.df = self.df.sort_values(["length", "width"], axis=0)
        self.viewing_angles = torch.tensor(self.df.viewing_angle.values, dtype=torch.float32)
        self.hrrps = torch.tensor(np.array(self.df[selectRP]), dtype=torch.float32)
        # Energie totale sur les HRRP normalisés (L2 classique)
        self.total_energy = torch.sum(self.hrrps ** 2, dim=1, keepdim=True)
        # Labels factorisés sur les MMSI sélectionnés
        self.labels = torch.tensor(pd.factorize(self.df.mmsi)[0], dtype=torch.long)
        self.mmsi_values = self.df.mmsi.unique()


    def __getitem__(self, idx):
        # Mode "global feature": on ne renvoie que l'énergie totale pour tester les biais
        va = self.viewing_angles[idx:idx+1]
        if self.va_jitter_std > 0:
            va = (va + torch.randn_like(va) * self.va_jitter_std) % TAU
        label = self.labels[idx]
        if self.global_feature == "total_energy":
            feats = self.total_energy[idx]  # [1]
            if self.global_feature_include_meta:
                # Ajoute uniquement l'angle de visée, jamais length/width
                feats = torch.cat([feats, va], dim=0)
            vars = feats.unsqueeze(0)       # [1, n_feat]
            return vars, va, label

        hrrp = self.hrrps[idx]
        # Ne pas feed length/width : uniquement HRRP (+ angle pour éventuelle cond)
        vars = torch.cat([hrrp.unsqueeze(0), va.unsqueeze(0)], dim=1)
        return vars, va, label
    
    def _select_mmsi_similar_size(self, n=30, length_band=0.1, width_band=0.1, seed=None):
        """
        Sélectionne n MMSI dont la taille moyenne (length/width) est proche de la médiane globale,
        pour créer un set de classes difficiles à distinguer.
        length_band/width_band sont des tolérances relatives autour des médianes (par défaut ±10%).
        Si pas assez de MMSI, la bande est élargie progressivement.
        """
        rng = np.random.default_rng(seed)
        agg = self.df.groupby("mmsi")[["length", "width"]].mean()
        med_len = agg["length"].median()
        med_wid = agg["width"].median()
        selected = []
        lb, wb = float(length_band), float(width_band)

        # Evite division par zéro si médiane = 0
        med_len = med_len if med_len != 0 else 1e-6
        med_wid = med_wid if med_wid != 0 else 1e-6

        while len(selected) < n and len(agg) > 0:
            mask = (agg["length"].sub(med_len).abs() <= med_len * lb) & \
                   (agg["width"].sub(med_wid).abs() <= med_wid * wb)
            pool = agg[mask].index.to_numpy()
            if len(pool) == 0:
                lb *= 1.5
                wb *= 1.5
                continue
            rng.shuffle(pool)
            for m in pool:
                if m not in selected:
                    selected.append(m)
                    if len(selected) >= n:
                        break
            lb *= 1.2
            wb *= 1.2
        return selected[:n]

    def _select_mmsi_angle_dense_length_prune(self, n=30, bins=12, bin_width=20.0, max_per_bin=2, seed=None):
        """
        Variante « angle_dense » avec prune après coup :
        1) Sélectionne les MMSI par gain de couverture angulaire (greedy set cover).
        2) Si un bin de longueur contient plus de `max_per_bin` navires, on supprime
           en priorité ceux qui couvrent le moins de bins d'angle (puis les moins présents).
        3) On complète ensuite pour revenir à n navires sans dépasser le cap par bin.
        """
        rng = np.random.default_rng(seed)
        angles = self.df[["mmsi", "viewing_angle", "length"]].copy()
        bin_size = 2 * np.pi / float(bins)
        angles["angle_bin"] = ((angles["viewing_angle"] % (2 * np.pi)) // bin_size).astype(int) % bins

        mmsi_bins = angles.groupby("mmsi")["angle_bin"].apply(lambda x: set(x.to_list()))
        mmsi_counts = angles.mmsi.value_counts()
        agg_len = self.df.groupby("mmsi")["length"].mean()

        def len_bin(m):
            return int(agg_len.get(m, 0) // float(bin_width))

        covered = set()
        selected = []
        candidates = list(mmsi_counts.index)
        rng.shuffle(candidates)

        def coverage_gain(m):
            return len(mmsi_bins[m] - covered)

        # Étape 1 : greedy couverture angulaire
        while candidates and len(covered) < bins:
            best = None
            best_gain = -1
            best_count = -1
            for m in candidates:
                gain = coverage_gain(m)
                count = mmsi_counts[m]
                if gain > best_gain or (gain == best_gain and count > best_count):
                    best_gain = gain
                    best_count = count
                    best = m
            if best is None:
                break
            selected.append(best)
            covered |= mmsi_bins[best]
            candidates.remove(best)
            if len(selected) >= n and len(covered) == bins:
                break

        # Étape 1b : compléter jusqu'à n si besoin (priorité au volume restant)
        if len(selected) < n and candidates:
            candidates_sorted = sorted(candidates, key=lambda m: mmsi_counts[m], reverse=True)
            selected.extend(candidates_sorted[: n - len(selected)])

        # Étape 2 : prune par bin de longueur en gardant les mieux couvrants
        pruned = []
        bins_to_mmsi = {}
        for m in selected:
            b = len_bin(m)
            bins_to_mmsi.setdefault(b, []).append(m)

        for b, ms in bins_to_mmsi.items():
            if len(ms) <= max_per_bin:
                pruned.extend(ms)
                continue
            ms_sorted = sorted(
                ms,
                key=lambda m: (len(mmsi_bins[m]), mmsi_counts[m]),
                reverse=True,
            )
            pruned.extend(ms_sorted[:max_per_bin])

        # Étape 3 : compléter après prune en respectant le cap
        if len(pruned) < n:
            remaining = [m for m in mmsi_counts.index if m not in pruned]
            remaining_sorted = sorted(
                remaining,
                key=lambda m: (len(mmsi_bins[m]), mmsi_counts[m]),
                reverse=True,
            )
            bins_used = {}
            for m in pruned:
                b = len_bin(m)
                bins_used[b] = bins_used.get(b, 0) + 1
            for m in remaining_sorted:
                b = len_bin(m)
                bins_used.setdefault(b, 0)
                if bins_used[b] >= max_per_bin:
                    continue
                pruned.append(m)
                bins_used[b] += 1
                if len(pruned) >= n:
                    break

        return pruned[:n]

    def _select_mmsi_largest(self, n=30, seed=None):
        """
        Sélectionne les n MMSI avec les plus grandes surfaces moyennes (length*width).
        """
        _ = np.random.default_rng(seed)  # pour homogénéité d'API
        agg = self.df.groupby("mmsi")[["length", "width"]].mean()
        agg = agg.assign(area=agg["length"] * agg["width"])
        agg = agg.sort_values("area", ascending=False)
        return agg.index.to_list()[:n]

    def _select_mmsi_angle_diverse(self, n=30, bins=12, seed=None):
        """
        Sélectionne des MMSI couvrant un maximum de bins d'angles de visée (greedy set cover).
        """
        rng = np.random.default_rng(seed)
        angles = self.df[["mmsi", "viewing_angle"]].copy()
        bin_size = 2 * np.pi / float(bins)
        angles["bin"] = ((angles["viewing_angle"] % (2 * np.pi)) // bin_size).astype(int) % bins

        mmsi_bins = angles.groupby("mmsi")["bin"].apply(lambda x: set(x.to_list()))
        covered = set()
        selected = []
        candidates = list(mmsi_bins.index)
        rng.shuffle(candidates)

        while candidates and len(selected) < n:
            best = None
            best_gain = -1
            for m in candidates:
                gain = len(mmsi_bins[m] - covered)
                if gain > best_gain:
                    best_gain = gain
                    best = m
            if best is None:
                break
            selected.append(best)
            covered |= mmsi_bins[best]
            candidates.remove(best)

        if len(selected) < n and candidates:
            rng.shuffle(candidates)
            selected.extend(candidates[: n - len(selected)])
        return selected[:n]
    
    def _select_mmsi_angle_dense(self, n=30, bins=12, seed=None):
        """
        Sélectionne des MMSI qui couvrent les bins d'angle tout en priorisant ceux
        qui ont le plus d'échantillons (greedy par gain de couverture puis volume).
        """
        rng = np.random.default_rng(seed)
        angles = self.df[["mmsi", "viewing_angle"]].copy()
        bin_size = 2 * np.pi / float(bins)
        angles["bin"] = ((angles["viewing_angle"] % (2 * np.pi)) // bin_size).astype(int) % bins

        # Ensemble de bins couverts par MMSI + comptage total d'échantillons
        mmsi_bins = angles.groupby("mmsi")["bin"].apply(lambda x: set(x.to_list()))
        mmsi_counts = angles.mmsi.value_counts()

        covered = set()
        selected = []
        # Ordre initial : par nombre d'échantillons décroissant
        candidates = list(mmsi_counts.index)

        def coverage_gain(m):
            return len(mmsi_bins[m] - covered)

        while candidates and len(selected) < n and len(covered) < bins:
            # choisir le meilleur gain, tie-break sur le volume
            best = None
            best_gain = -1
            best_count = -1
            for m in candidates:
                gain = coverage_gain(m)
                count = mmsi_counts[m]
                if gain > best_gain or (gain == best_gain and count > best_count):
                    best_gain = gain
                    best_count = count
                    best = m
            if best is None:
                break
            selected.append(best)
            covered |= mmsi_bins[best]
            candidates.remove(best)

        # Si on n'a pas assez de MMSI, on complète par les plus gros restants
        if len(selected) < n and candidates:
            # Tri décroissant sur le volume restant
            candidates_sorted = sorted(candidates, key=lambda m: mmsi_counts[m], reverse=True)
            selected.extend(candidates_sorted[: n - len(selected)])
        return selected[:n]

    def _select_mmsi_angle_dense_length_cap(self, n=30, bins=12, bin_width=20.0, max_per_bin=2, seed=None):
        """
        Combine la contrainte de 2 navires par tranche de `bin_width` mètres
        et la couverture angulaire maximale (greedy angle_dense) sur les candidats filtrés.
        """
        # Étape 1 : filtrer les MMSI avec un cap par bin de longueur
        base_candidates = self._select_mmsi_length_bin_cap(bin_width=bin_width, max_per_bin=max_per_bin, n=None, seed=seed)
        if len(base_candidates) == 0:
            return []

        rng = np.random.default_rng(seed)
        angles = self.df[self.df.mmsi.isin(base_candidates)][["mmsi", "viewing_angle"]].copy()
        bin_size = 2 * np.pi / float(bins)
        angles["bin"] = ((angles["viewing_angle"] % (2 * np.pi)) // bin_size).astype(int) % bins

        mmsi_bins = angles.groupby("mmsi")["bin"].apply(lambda x: set(x.to_list()))
        mmsi_counts = angles.mmsi.value_counts()

        covered = set()
        selected = []
        candidates = list(mmsi_counts.index)
        rng.shuffle(candidates)

        def coverage_gain(m):
            return len(mmsi_bins[m] - covered)

        while candidates and len(selected) < n and len(covered) < bins:
            best = None
            best_gain = -1
            best_count = -1
            for m in candidates:
                gain = coverage_gain(m)
                count = mmsi_counts[m]
                if gain > best_gain or (gain == best_gain and count > best_count):
                    best_gain = gain
                    best_count = count
                    best = m
            if best is None:
                break
            selected.append(best)
            covered |= mmsi_bins[best]
            candidates.remove(best)

        if len(selected) < n and candidates:
            candidates_sorted = sorted(candidates, key=lambda m: mmsi_counts[m], reverse=True)
            selected.extend(candidates_sorted[: n - len(selected)])
        return selected[:n]

    def _select_mmsi_angle_diverse_length_prune(self, n=10, bins=12, bin_width=20.0, max_per_bin=2, seed=None):
        """
        1) Priorise les MMSI ayant le plus de bins d'angles distincts (greedy set cover).
        2) Garde au plus `max_per_bin` navires par tranche de longueur `bin_width`,
           en retirant les moins divers (puis les moins présents) quand il y a surplus.
           Si on tombe sous n après prune, on complète avec les candidats restants
           en respectant le cap de longueur.
        """
        rng = np.random.default_rng(seed)
        angles = self.df[["mmsi", "viewing_angle", "length"]].copy()
        bin_size = 2 * np.pi / float(bins)
        angles["angle_bin"] = ((angles["viewing_angle"] % (2 * np.pi)) // bin_size).astype(int) % bins

        mmsi_angle_bins = angles.groupby("mmsi")["angle_bin"].apply(lambda x: set(x.to_list()))
        mmsi_counts = angles.mmsi.value_counts()

        covered = set()
        selected = []
        candidates = list(mmsi_counts.index)
        rng.shuffle(candidates)

        def coverage_gain(m):
            return len(mmsi_angle_bins[m] - covered)

        while candidates and len(selected) < n and len(covered) < bins:
            best = None
            best_gain = -1
            best_span = -1  # nombre total de bins distincts pour tie-break
            best_count = -1
            for m in candidates:
                gain = coverage_gain(m)
                span = len(mmsi_angle_bins[m])
                count = mmsi_counts[m]
                if gain > best_gain or (
                    gain == best_gain and (span > best_span or (span == best_span and count > best_count))
                ):
                    best_gain = gain
                    best_span = span
                    best_count = count
                    best = m
            if best is None:
                break
            selected.append(best)
            covered |= mmsi_angle_bins[best]
            candidates.remove(best)

        if len(selected) < n and candidates:
            candidates_sorted = sorted(
                candidates,
                key=lambda m: (len(mmsi_angle_bins[m]), mmsi_counts[m]),
                reverse=True,
            )
            selected.extend(candidates_sorted[: n - len(selected)])

        # Étape 2 : prune pour respecter le cap de longueur
        agg_len = self.df.groupby("mmsi")["length"].mean()
        def bin_of(m):
            return int(agg_len.get(m, 0) // float(bin_width))

        pruned = []
        bins_to_mmsi = {}
        for m in selected:
            b = bin_of(m)
            bins_to_mmsi.setdefault(b, []).append(m)

        for b, ms in bins_to_mmsi.items():
            if len(ms) <= max_per_bin:
                pruned.extend(ms)
                continue
            # Surplus : on garde les plus divers puis les plus fournis
            ms_sorted = sorted(
                ms,
                key=lambda m: (len(mmsi_angle_bins[m]), mmsi_counts[m]),
                reverse=True,
            )
            pruned.extend(ms_sorted[:max_per_bin])

        # Compléter si on a perdu trop d'items après prune
        if len(pruned) < n:
            # candidats restants = ceux qui n'étaient pas dans selected
            remaining = [m for m in mmsi_counts.index if m not in selected]
            # tri par diversité puis volume
            remaining_sorted = sorted(
                remaining,
                key=lambda m: (len(mmsi_angle_bins[m]), mmsi_counts[m]),
                reverse=True,
            )
            # Comptage correct des bin déjà occupés
            bins_used = {}
            for m in pruned:
                b = bin_of(m)
                bins_used[b] = bins_used.get(b, 0) + 1
            for m in remaining_sorted:
                b = bin_of(m)
                bins_used.setdefault(b, 0)
                if bins_used[b] >= max_per_bin:
                    continue
                pruned.append(m)
                bins_used[b] += 1
                if len(pruned) >= n:
                    break

        return pruned[:n]

    def _select_mmsi_length_bin_cap(self, bin_width=20.0, max_per_bin=2, n=None, seed=None):
        """
        Limite la sélection à `max_per_bin` navires par tranche de longueur `bin_width` (mètres).
        Les MMSI sont triés par nombre d'échantillons décroissant dans chaque bin, avec un
        shuffle optionnel pour varier (seed). Optionnellement, on tronque à n MMSI au total.
        """
        rng = np.random.default_rng(seed)
        agg = self.df.groupby("mmsi").agg(mean_len=("length", "mean"), count=("mmsi", "size"))
        agg["bin"] = (agg["mean_len"] // float(bin_width)).astype(int)

        selected = []
        for _, group in agg.groupby("bin"):
            # Classement décroissant sur le volume d'échantillons
            group_sorted = group.sort_values("count", ascending=False)
            idxs = group_sorted.index.to_numpy()
            # Shuffle léger pour éviter toujours les mêmes si counts proches
            if seed is not None:
                idxs = rng.permutation(idxs)
            selected.extend(list(idxs[:max_per_bin]))

        # Optionnel : réduire au top-n (conservant l'ordre par bin puis count)
        if n is not None:
            try:
                n_int = int(n)
                if n_int > 0:
                    selected = selected[:n_int]
            except Exception:
                pass

        return selected

def _load_dataframe(path):
    """Chargement tolérant (pkl ou csv avec fallback latin1)."""
    p = Path(path)
    if p.suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    try:
        return pd.read_csv(p)
    except UnicodeDecodeError:
        return pd.read_csv(p, encoding="latin1")


def _load_ship_hrrp_pt(path, target_len=200):
    """Load a ship HRRP tensor file (.pt/.pth) into an RP_ImageDataset-compatible DataFrame."""
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "hrrps" not in payload:
        raise ValueError("Expected a dict with key 'hrrps' in ship HRRP tensor file.")

    hrrps = payload["hrrps"]
    if isinstance(hrrps, torch.Tensor):
        hrrps = hrrps.detach().cpu().numpy()
    else:
        hrrps = np.asarray(hrrps)

    if hrrps.ndim == 3 and hrrps.shape[1] == 1:
        hrrps = hrrps[:, 0, :]
    if hrrps.ndim != 2:
        raise ValueError(f"Expected hrrps with shape [N, L], got {hrrps.shape}.")

    n, l = hrrps.shape
    if l > target_len:
        start = (l - target_len) // 2
        hrrps = hrrps[:, start:start + target_len]
    elif l < target_len:
        pad = target_len - l
        left = pad // 2 + (pad % 2)
        right = pad // 2
        hrrps = np.pad(hrrps, ((0, 0), (left, right)), mode="constant")

    aspect = payload.get("aspect_angles", np.zeros(n, dtype=np.float32))
    if isinstance(aspect, torch.Tensor):
        aspect = aspect.detach().cpu().numpy()
    aspect = np.asarray(aspect, dtype=np.float32).reshape(-1)
    if len(aspect) != n:
        aspect = np.resize(aspect, n)

    dims = payload.get("ship_dims", np.ones((n, 2), dtype=np.float32))
    if isinstance(dims, torch.Tensor):
        dims = dims.detach().cpu().numpy()
    dims = np.asarray(dims, dtype=np.float32)
    if dims.ndim != 2 or dims.shape[1] < 2:
        dims = np.ones((n, 2), dtype=np.float32)
    if dims.shape[0] != n:
        dims = np.resize(dims, (n, 2))

    labels_src = payload.get("labels", None)
    if labels_src is None:
        labels_src = payload.get("mmsi", None)
    if labels_src is None:
        labels_src = payload.get("label", None)

    if labels_src is not None:
        if isinstance(labels_src, torch.Tensor):
            labels_src = labels_src.detach().cpu().numpy()
        labels_src = np.asarray(labels_src).reshape(-1)
        if len(labels_src) != n:
            labels_src = np.resize(labels_src, n)
        mmsi_vals = pd.factorize(labels_src)[0]
    else:
        # Fallback: derive pseudo-classes from ship dimensions
        dims_key = np.round(dims[:, :2], 4)
        mmsi_vals = pd.factorize([f"{a:.4f}_{b:.4f}" for a, b in dims_key])[0]

    df = pd.DataFrame(hrrps.astype(np.float32), columns=selectRP)
    df["viewing_angle"] = np.mod(aspect.astype(np.float64), TAU)
    df["length"] = dims[:, 0].astype(np.float32)
    df["width"] = dims[:, 1].astype(np.float32)
    df["mmsi"] = mmsi_vals.astype(np.int64)
    return df


def _pick_column(df, preferred):
    for c in preferred:
        if c in df.columns:
            return c
    return None


def _parse_and_pad(rp_entry, target_len=128):
    if isinstance(rp_entry, (list, np.ndarray)):
        rp = np.asarray(rp_entry, dtype=np.float32).ravel()
    else:
        rp = np.fromstring(str(rp_entry).strip("[]"), sep=" ", dtype=np.float32)
    diff = target_len - len(rp)
    if diff < 0:
        rp = rp[:target_len]
    else:
        left = diff // 2 + (diff % 2)
        right = diff // 2
        rp = np.pad(rp, (left, right), mode="constant")
    return rp


class MSTAR_dataset(Dataset):
    def __init__(self, config, path="../data/MSTAR_data/mstar_data.csv"):
        self.path = path
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        # Petit jitter d'angle (radians) pour éviter la mémo trop stricte
        self.va_jitter_std = float(config.get("va_jitter_std", np.deg2rad(2.0)))
        self.data_df = _load_dataframe(self.path)
        self.va_col = _pick_column(self.data_df, ["azimuth_deg", "azimuth", "az"])
        self.label_col = _pick_column(self.data_df, ["class_name", "label", "target", "class"])
        if self.label_col is None:
            self.label_col = self.data_df.columns[-1]
        self.hrrp_col = _pick_column(self.data_df, ["hrrp", "rp", "profile"])
        if self.hrrp_col is None:
            self.hrrp_col = self.data_df.columns[0]
        self.normalize()
    
    def normalize(self):
        self.hrrps = np.stack([_parse_and_pad(s) for s in self.data_df[self.hrrp_col]])
        self.min_rp, self.max_rp = self.hrrps.min(), self.hrrps.max()
        self.hrrps = (self.hrrps - self.min_rp) / (self.max_rp - self.min_rp) * 2 - 1
        self.labels, _ = pd.factorize(self.data_df[self.label_col])
        if self.va_col is not None:
            # Convertit explicitement les degrés en radians pour rester borné
            self.va = np.deg2rad(self.data_df[self.va_col].to_numpy(dtype=np.float32))
        else:
            self.va = np.zeros(len(self.data_df), dtype=np.float32)
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        hrrp = torch.tensor(self.hrrps[idx], dtype=torch.float32)
        va = torch.tensor(self.va[idx], dtype=torch.float32)
        if self.va_jitter_std > 0:
            va = (va + torch.randn((), dtype=torch.float32) * self.va_jitter_std) % TAU
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return hrrp, va, label

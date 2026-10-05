#!/usr/bin/env python3
"""
texturelab_batch.py — Cálculo em lote (sem interface) de descritores de textura de pavimento
a partir de varreduras 3dT (BASt) em LAZ.

Uso:
    python texturelab_batch.py --input /dados/laz --output /dados/resultados [--workers N]

Dependências: numpy, scipy, pandas, laspy[lazrs]   (Python >= 3.10)

Cadeias implementadas (todas sobre a grade em mm, eixo maior = sentido da via):
  A  MPD/MSD/ETD ............ ISO 13473-1:2019 (medição pontual): faixas 0,5 mm, reamostragem 0,5 mm,
                               spikes Anexo E, passa-baixa Butterworth 2ª ordem projetado em 2,40 mm
                               ida e volta (Tab. D.2), supressão de rampa por segmento, ETD = 1,1·MPD.
  E  Espectro ................ ISO 13473-4:2024 método 1: terços de oitava (IEC 61260-1, Butterworth),
                               ref. 1 µm, l >= 12·λmax, espelhamento Anexo F, spikes Anexo D.
  S  Areal ................... ISO 25178-3/-2 com gaussiano ISO 16610-61:
                               SF   : S = 0,05 mm, F = plano
                               SL5  : S = 0,05 mm, F = plano, L = 5 mm   (par 100:1 da Tab. 1;
                                      N_is compatível com dx = 11 µm pela Tab. 3)
                               MICRO: sem S digital, F = plano, L = 0,5 mm (largura de banda fora do
                                      padrão da Tab. 1 -> PROVISÓRIO, ver resumo)
  Rk  Família Rk/Sk ........... ISO 13565-2 (secante 40 %, reta de mínimos quadrados, triângulos).
  Perfil (segmentos 100 mm) .. Rq, Rsk, Rku, Rk... sobre o perfil 0,5 mm da cadeia A sem passa-baixa
                               — NÃO é ISO 21920 (filtros de perfil não aplicados), uso descritivo.
  H  Hurst e dimensão fractal . descritivo, sem norma: inclinação da PSD 1D (PSD ∝ f^-(1+2H)),
                               D_perfil = 2 - H, D_superficie = 3 - H; faixas micro (0,05–0,5 mm) e
                               macro (0,5–20 mm) ajustadas separadamente.
Não implementado de propósito (pendente de conferência normativa): g-factor e ENDT (ISO 10844).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Um fio por processo: o paralelismo é entre arquivos.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import ndimage, signal

VERSION = "3.0.0"

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy < 2.0

CFG = {
    # cadeia A (ISO 13473-1)
    "A_strip_width_mm": 0.5,        # largura total das linhas paralelas por perfil (0,5–1 mm)
    "A_resample_mm": 0.5,
    "A_spike_alpha": 3.0,
    "A_lp_design_mm": 2.40,         # -3 dB de projeto; ida e volta -> "3 mm"
    "A_segment_mm": 100.0,
    "A_max_dropout_frac": 0.10,
    "A_max_spike_frac": 0.05,
    "A_extrap_max_mm": 5.0,
    "ETD_factor": 1.1,
    # espectro (ISO 13473-4)
    "E_strip_width_mm": 0.5,
    "E_decimate_target_mm": 0.1,    # passo após anti-aliasing (múltiplo inteiro de dx)
    "E_min_wavelength_mm": 0.4,
    "E_spike_alpha": 3.0,
    # areal (ISO 25178-3)
    "S_nis_mm": 0.05,
    "S_L_macro_mm": 5.0,
    "S_L_micro_mm": 0.5,
    "S_acf_threshold": 0.2,
    # Hurst / dimensão fractal (descritivo, não normativo)
    "H_nperseg_mm": 90.0,           # segmento de Welch (potência de 2 em amostras)
    "H_bands_mm": {"micro": (0.05, 0.5), "macro": (0.5, 20.0)},   # faixas de comprimento de onda
    "H_n_bins": 20,                 # bins log-espaçados no ajuste
    "preview_step": 8,
}

ALPHA_G = np.sqrt(np.log(2.0) / np.pi)          # ISO 16610-61
NOMINAL_TO = [500, 400, 315, 250, 200, 160, 125, 100, 80, 63, 50, 40, 31.5, 25, 20, 16, 12.5, 10,
              8, 6.3, 5, 4, 3.15, 2.5, 2, 1.6, 1.25, 1, 0.8, 0.63, 0.5, 0.4]   # ISO 13473-4 Tab. 2


# ======================================================================================
# Leitura
# ======================================================================================
def parse_name(stem: str) -> dict:
    parts = stem.split("_")
    info = {"arquivo": stem, "trecho": parts[0] if parts else stem,
            "revestimento": parts[1] if len(parts) > 1 else "", "mp": "", "nr": "", "data": ""}
    for p in parts:
        if re.fullmatch(r"MP\d+|REF", p):
            info["mp"] = p
        elif re.fullmatch(r"NR\d+", p):
            info["nr"] = p
        elif re.fullmatch(r"\d{8}", p):
            info["data"] = p
    info["trecho"] = re.sub(r"x+$", "", info["trecho"])          # "B6xxx" -> "B6"
    info["revestimento"] = re.sub(r"x+$", "", info["revestimento"])
    return info


def read_laz(path: str):
    """Lê LAZ do 3dT e devolve Z[n_largura, n_via] (float32, mm), dx (mm) e metadados.

    Caminho rápido: grade completa gravada linha a linha (verificado). Caso contrário,
    gradeamento genérico por índices inteiros com média por célula.
    """
    import laspy
    las = laspy.read(path)
    h = las.header
    X = np.asarray(las.X, dtype=np.int64)
    Y = np.asarray(las.Y, dtype=np.int64)
    Zi = np.asarray(las.Z, dtype=np.int64)
    sx, sy, sz = (float(s) for s in h.scales)
    meta = {"n_pontos": int(len(X)), "escala": [sx, sy, sz]}

    ux = np.unique(X)
    uy = np.unique(Y)
    stepx = int(np.median(np.diff(ux))) if len(ux) > 1 else 1
    stepy = int(np.median(np.diff(uy))) if len(uy) > 1 else 1
    dx = stepx * sx
    dy = stepy * sy
    z = Zi * sz + float(h.offsets[2])

    grid = None
    nx, ny = len(ux), len(uy)
    if nx * ny == len(X):
        if Y[1] != Y[0]:
            g = z.reshape(nx, ny)                                       # Y mais rápido
            if np.array_equal(X.reshape(nx, ny)[:, 0], ux) and \
               (X.reshape(nx, ny) == X.reshape(nx, ny)[:, :1]).all() and \
               (Y.reshape(nx, ny) == Y.reshape(nx, ny)[:1, :]).all():
                grid = g                                                # [x, y]
        if grid is None:
            g = z.reshape(ny, nx)                                       # X mais rápido
            if (Y.reshape(ny, nx) == Y.reshape(ny, nx)[:, :1]).all() and \
               (X.reshape(ny, nx) == X.reshape(ny, nx)[:1, :]).all():
                grid = g.T if np.all(np.diff(Y.reshape(ny, nx)[:, 0]) > 0) else g[::-1].T
                if not np.all(np.diff(X.reshape(ny, nx)[0]) > 0):
                    grid = grid[::-1]
        meta["leitura"] = "reshape" if grid is not None else "generica"
    if grid is None:
        ix = ((X - X.min()) // stepx).astype(np.int64)
        iy = ((Y - Y.min()) // stepy).astype(np.int64)
        nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
        lin = ix * ny + iy
        s = np.bincount(lin, weights=z, minlength=nx * ny)
        c = np.bincount(lin, minlength=nx * ny)
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = (s / c).reshape(nx, ny)
        meta["leitura"] = "generica"
    del X, Y, Zi, z

    # unidades: o 3dT grava em mm. Passo < 0,001 indica metros.
    if dx < 1e-3:
        grid = grid * 1000.0
        dx *= 1000.0
        dy *= 1000.0
        meta["aviso_unidade"] = "passo < 0,001: assumido metros e convertido para mm"
    if abs(dx - dy) > 1e-9:
        meta["aviso_passo"] = f"dx ({dx}) != dy ({dy}); usado dx no eixo da via"
    # eixo da via = eixo maior
    if grid.shape[1] < grid.shape[0]:
        grid = grid.T
        dx, dy = dy, dx
        meta["transposto"] = True
    meta.update({"n_largura": int(grid.shape[0]), "n_via": int(grid.shape[1]),
                 "dx_mm": dx, "largura_mm": grid.shape[0] * dy, "comprimento_mm": grid.shape[1] * dx,
                 "frac_invalidos": float(np.mean(~np.isfinite(grid)))})
    return grid.astype(np.float32), dx, meta


# ======================================================================================
# Utilidades de perfil
# ======================================================================================
def make_strips(Z: np.ndarray, dx: float, width_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Média de linhas paralelas adjacentes (largura total >= width_mm). Devolve [n_faixas, n_via]."""
    k = int(np.ceil(width_mm / dx - 1e-9))
    n = Z.shape[0] // k
    blk = Z[: n * k].reshape(n, k, Z.shape[1]).astype(np.float64)
    with np.errstate(invalid="ignore"):
        strips = np.nanmean(blk, axis=1)
    centers = (np.arange(n) * k + (k - 1) / 2.0) * dx
    return strips, centers


def resample_mean(P: np.ndarray, dx: float, step: float) -> np.ndarray:
    """Média de todas as amostras dentro de cada intervalo de 'step' (ISO 13473-1 §7.4). Só intervalos completos."""
    n = P.shape[1]
    edges = np.floor(np.arange(n) * dx / step + 1e-9).astype(np.int64)
    nb = int(edges[-1])                       # descarta o último intervalo (incompleto)
    starts = np.searchsorted(edges, np.arange(nb))
    v = np.where(np.isfinite(P), P, 0.0)
    c = np.isfinite(P).astype(np.float64)
    s = np.add.reduceat(v, starts, axis=1)[:, :nb]
    cnt = np.add.reduceat(c, starts, axis=1)[:, :nb]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / cnt, np.nan)


def interp_invalid(p: np.ndarray, step: float, extrap_max_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolação linear de amostras inválidas; extrapolação constante nas pontas até extrap_max_mm.
    Devolve perfil e máscara 'inutilizável' (pontas além do limite)."""
    p = p.copy()
    bad = ~np.isfinite(p)
    unusable = np.zeros(p.shape, bool)
    if not bad.any():
        return p, unusable
    good = np.flatnonzero(~bad)
    if len(good) == 0:
        return p, ~unusable
    x = np.arange(len(p))
    p[bad] = np.interp(x[bad], good, p[good])          # np.interp já extrapola constante
    lim = int(np.floor(extrap_max_mm / step))
    if good[0] > lim:
        unusable[: good[0] - lim] = True
    if len(p) - 1 - good[-1] > lim:
        unusable[good[-1] + 1 + lim:] = True
    return p, unusable


def spike_mask(p: np.ndarray, step: float, alpha: float) -> np.ndarray:
    """ISO 13473-1 Anexo E / 13473-4 Anexo D: z_i - z_{i-1} >= alpha·Δx, ida e volta."""
    thr = alpha * step
    m = np.zeros(p.shape, bool)
    d = np.diff(p)
    m[1:] |= d >= thr            # ida:   z_i - z_{i-1}
    m[:-1] |= (-d) >= thr        # volta: z_i - z_{i+1}
    return m


def iso13473_lp(x: np.ndarray, step: float, design_mm: float) -> np.ndarray:
    """Butterworth 2ª ordem (bilinear) aplicado ida e volta como na ISO 13473-1 D.4, Fórmula (D.1):
    as duas primeiras amostras ficam inalteradas em cada passagem."""
    b, a = signal.butter(2, 1.0 / design_mm, btype="low", fs=1.0 / step)
    A0, A1, A2 = 1.0 / b[0], -a[2], -a[1]

    def one_pass(u):
        y = u.copy()
        for i in range(2, u.shape[-1]):
            y[..., i] = (u[..., i] + 2 * u[..., i - 1] + u[..., i - 2]) / A0 + A1 * y[..., i - 2] + A2 * y[..., i - 1]
        return y

    y = one_pass(x)
    return one_pass(y[..., ::-1])[..., ::-1]


def material_ratio_curve(h: np.ndarray, n: int = 10001) -> tuple[np.ndarray, np.ndarray]:
    """c(Mr): altura em função do material ratio (0..100 %), decrescente."""
    mr = np.linspace(0.0, 100.0, n)
    c = np.quantile(h, 1.0 - mr / 100.0)
    return mr, c


def rk_family(h: np.ndarray, n: int = 10001) -> dict:
    """ISO 13565-2: zona central de 40 % com menor inclinação da secante; reta de mínimos quadrados
    nessa zona; Rk, Mr1, Mr2; Rpk e Rvk pelos triângulos de área equivalente."""
    h = h[np.isfinite(h)]
    if h.size < 20:
        return {}
    mr, c = material_ratio_curve(h, n)
    w = int(round(40.0 / (mr[1] - mr[0])))
    slope = c[w:] - c[:-w]                      # negativo; menor inclinação = maior valor
    i0 = int(np.argmax(slope))                  # primeiro máximo = primeira zona encontrada
    seg_mr, seg_c = mr[i0:i0 + w + 1], c[i0:i0 + w + 1]
    k, q = np.polyfit(seg_mr, seg_c, 1)
    z_top, z_bot = q, k * 100.0 + q
    rk = z_top - z_bot
    mr1 = float(np.interp(-z_top, -c, mr))      # c decrescente
    mr2 = float(np.interp(-z_bot, -c, mr))
    m1 = mr <= mr1
    a1 = _trapz(np.clip(c[m1] - z_top, 0, None), mr[m1]) if m1.sum() > 1 else 0.0
    m2 = mr >= mr2
    a2 = _trapz(np.clip(z_bot - c[m2], 0, None), mr[m2]) if m2.sum() > 1 else 0.0
    rpk = 2 * a1 / mr1 if mr1 > 0 else 0.0
    rvk = 2 * a2 / (100 - mr2) if mr2 < 100 else 0.0
    return {"k": float(rk), "pk": float(rpk), "vk": float(rvk), "mr1": mr1, "mr2": mr2}


def volume_params(h: np.ndarray, p: float = 10.0, q: float = 80.0, n: int = 10001) -> dict:
    """ISO 25178-2 Vmp, Vmc, Vvc, Vvv em mm³/mm² (= mm; ×1000 -> ml/m²)."""
    mr, c = material_ratio_curve(h[np.isfinite(h)], n)

    def vm(t):
        m = mr <= t
        return _trapz(c[m] - np.interp(t, mr, c), mr[m]) / 100.0

    def vv(t):
        m = mr >= t
        return _trapz(np.interp(t, mr, c) - c[m], mr[m]) / 100.0

    return {"Vmp": float(vm(p)), "Vmc": float(vm(q) - vm(p)),
            "Vvc": float(vv(p) - vv(q)), "Vvv": float(vv(q))}


def height_stats(h: np.ndarray, prefix: str) -> dict:
    h = h[np.isfinite(h)].astype(np.float64)
    h = h - h.mean()
    sq = np.sqrt(np.mean(h ** 2))
    return {f"{prefix}a": float(np.mean(np.abs(h))), f"{prefix}q": float(sq),
            f"{prefix}sk": float(np.mean(h ** 3) / sq ** 3) if sq > 0 else 0.0,
            f"{prefix}ku": float(np.mean(h ** 4) / sq ** 4) if sq > 0 else 0.0,
            f"{prefix}p": float(h.max()), f"{prefix}v": float(-h.min()), f"{prefix}z": float(h.max() - h.min())}


# ======================================================================================
# Cadeia A — ISO 13473-1:2019
# ======================================================================================
def chain_a(Z: np.ndarray, dx: float, cfg: dict) -> tuple[dict, pd.DataFrame]:
    step = cfg["A_resample_mm"]
    strips, centers = make_strips(Z, dx, cfg["A_strip_width_mm"])
    P = resample_mean(strips, dx, step)                     # [faixas, amostras 0,5 mm]
    nseg_pts = int(round(cfg["A_segment_mm"] / step))
    n = P.shape[1]
    nseg = n // nseg_pts
    if nseg == 0:
        return {"A_erro": f"perfil de {n * step:.1f} mm < segmento de {cfg['A_segment_mm']} mm"}, pd.DataFrame()
    start = (n - nseg * nseg_pts) // 2                      # segmentos centralizados

    cleaned = np.empty_like(P)
    drop = ~np.isfinite(P)
    unusable = np.zeros_like(drop)
    spikes = np.zeros_like(drop)
    for i in range(P.shape[0]):
        p, un = interp_invalid(P[i], step, cfg["A_extrap_max_mm"])
        sm = spike_mask(p, step, cfg["A_spike_alpha"])
        if sm.any():
            q = p.copy()
            q[sm] = np.nan
            p, _ = interp_invalid(q, step, cfg["A_extrap_max_mm"])
        cleaned[i], unusable[i], spikes[i] = p, un, sm
    ok_rows = np.isfinite(cleaned).all(axis=1)
    lp = np.full_like(cleaned, np.nan)
    if ok_rows.any():
        lp[ok_rows] = iso13473_lp(cleaned[ok_rows], step, cfg["A_lp_design_mm"])

    rows = []
    t = np.arange(nseg_pts, dtype=np.float64)
    half = nseg_pts // 2
    for i in range(P.shape[0]):
        for s in range(nseg):
            a, b = start + s * nseg_pts, start + (s + 1) * nseg_pts
            seg = lp[i, a:b]
            fd = float(drop[i, a:b].mean())
            fs = float(spikes[i, a:b].mean())
            valid = bool(ok_rows[i] and fd <= cfg["A_max_dropout_frac"] and fs <= cfg["A_max_spike_frac"]
                         and not unusable[i, a:b].any())
            r = {"faixa": i, "x_centro_mm": round(float(centers[i]), 4), "segmento": s,
                 "inicio_mm": a * step, "dropout_frac": fd, "spike_frac": fs, "valido": valid}
            if valid:
                seg = seg - np.polyval(np.polyfit(t, seg, 1), t)                 # supressão de rampa
                r["MSD"] = float((seg[:half].max() + seg[half:].max()) / 2.0 - seg.mean())
                raw = cleaned[i, a:b] - np.polyval(np.polyfit(t, cleaned[i, a:b], 1), t)
                r.update(height_stats(raw, "R"))
                r.update({("R" + k): v for k, v in rk_family(raw, 2001).items()})
            rows.append(r)
    df = pd.DataFrame(rows)
    v = df[df["valido"]] if len(df) else df
    out = {"A_n_faixas": int(P.shape[0]), "A_n_segmentos_total": int(len(df)),
           "A_n_segmentos_validos": int(len(v)), "A_segmentos_por_faixa": nseg,
           "A_largura_faixa_mm": float(int(np.ceil(cfg["A_strip_width_mm"] / dx - 1e-9)) * dx),
           "A_inicio_primeiro_segmento_mm": start * step,
           "A_spike_frac_media": float(spikes.mean()), "A_dropout_frac_media": float(drop.mean())}
    if len(v):
        mpd = float(v["MSD"].mean())
        out.update({"MPD": mpd, "MPD_desvio": float(v["MSD"].std(ddof=1)) if len(v) > 1 else 0.0,
                    "ETD": cfg["ETD_factor"] * mpd})
        for k in ["Rq", "Rsk", "Rku", "Rk", "Rpk", "Rvk", "Rmr1", "Rmr2"]:
            if k in v:
                out[f"perfil_{k}_media"] = float(v[k].mean())
    return out, df


# ======================================================================================
# Espectro — ISO 13473-4:2024, método 1
# ======================================================================================
def chain_spectrum(Z: np.ndarray, dx: float, cfg: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    strips, _ = make_strips(Z, dx, cfg["E_strip_width_mm"])
    dec = max(1, int(round(cfg["E_decimate_target_mm"] / dx)))
    step = dec * dx
    # anti-aliasing (§6.5): plano (<0,4 dB) até a menor banda, >= 60 dB no Nyquist do novo passo
    fs0 = 1.0 / dx
    sos_aa = signal.butter(10, 0.66 * (0.5 / step), btype="low", fs=fs0, output="sos")
    good_rows = np.isfinite(strips).all(axis=1)
    S = strips[good_rows]
    if dec > 1:
        S = signal.sosfiltfilt(sos_aa, S, axis=1)[:, ::dec]
    l_eval = S.shape[1] * step
    # bandas: λ >= 2,5·passo (§6.7) e l >= 12·λ (§6.2)
    bands = []
    for nb in range(3, 40):
        lam_exact = 1000.0 / 10 ** (nb / 10.0)                       # mm
        nominal = min(NOMINAL_TO, key=lambda v: abs(np.log(v / lam_exact)))
        if lam_exact < cfg["E_min_wavelength_mm"] * 0.99 or lam_exact < 2.5 * step:
            continue
        if 12 * nominal > l_eval:
            continue
        bands.append((nominal, lam_exact))
    lead = int(round(12 * max(b[0] for b in bands) / step)) if bands else 0
    lead = min(lead, S.shape[1] - 1)
    levels = np.full((S.shape[0], len(bands)), np.nan)
    spk = []
    for i in range(S.shape[0]):
        p, _ = interp_invalid(S[i], step, 5.0)
        sm = spike_mask(p, step, cfg["E_spike_alpha"])
        spk.append(sm.mean())
        if sm.mean() > 0.05:
            continue
        if sm.any():
            q = p.copy(); q[sm] = np.nan
            p, _ = interp_invalid(q, step, 5.0)
        t = np.arange(len(p))
        p = p - np.polyval(np.polyfit(t, p, 1), t)                      # Anexo F.1
        mir = 2 * p[0] - p[1:lead + 1][::-1]                             # Anexo F.2 (lead-in)
        x = np.concatenate([mir, p])
        for j, (_, lam) in enumerate(bands):
            fm = 1.0 / lam
            f1, f2 = fm * 10 ** (-0.05), fm * 10 ** (0.05)               # IEC 61260-1, base 10, b = 3
            sos = signal.butter(3, [f1, f2], btype="bandpass", fs=1.0 / step, output="sos")
            y = signal.sosfilt(sos, x)[len(mir):]
            rms_mm = np.sqrt(np.mean(y ** 2))
            levels[i, j] = 20 * np.log10(rms_mm * 1e-3 / 1e-6)          # ref 1 µm
    df_bands = pd.DataFrame(levels, columns=[f"{b[0]:g}" for b in bands])
    summary = pd.DataFrame({
        "lambda_centro_mm": [b[0] for b in bands],
        "L_tx_dB_media": np.nanmean(levels, axis=0) if len(levels) else [],
        "L_tx_dB_desvio": np.nanstd(levels, axis=0, ddof=1) if len(levels) > 1 else np.nan,
        "n_perfis": np.sum(np.isfinite(levels), axis=0)})
    out = {"E_passo_mm": step, "E_comprimento_avaliacao_mm": l_eval, "E_n_perfis": int(S.shape[0]),
           "E_lambda_max_mm": max(b[0] for b in bands) if bands else None,
           "E_lambda_min_mm": min(b[0] for b in bands) if bands else None,
           "E_spike_frac_media": float(np.mean(spk)) if spk else None}
    return out, summary, df_bands


# ======================================================================================
# Hurst e dimensão fractal — descritivo (sem norma)
# ======================================================================================
def chain_hurst(Z: np.ndarray, dx: float, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Expoente de Hurst pela inclinação da PSD 1D no sentido da via: PSD ∝ f^-(1+2H).

    Usa linhas individuais na resolução nativa (sem média entre linhas, que atenuaria a microtextura),
    uma a cada largura de faixa da cadeia A. PSD de Welch (Hann, 50 % de sobreposição, tendência linear
    removida por segmento), média entre linhas, ajuste linear em log-log com bins log-espaçados.
    D_perfil = 2 - H; D_superficie = 3 - H (supõe isotropia). Faixas ajustadas separadamente."""
    k = max(1, int(np.ceil(cfg["A_strip_width_mm"] / dx - 1e-9)))
    lines = Z[::k].astype(np.float64)
    lines = lines[np.isfinite(lines).all(axis=1)]
    if lines.shape[0] == 0:
        return {"H_erro": "nenhuma linha sem valores inválidos"}, pd.DataFrame()
    nper = int(2 ** np.ceil(np.log2(cfg["H_nperseg_mm"] / dx)))
    nper = min(nper, lines.shape[1])
    f, P = signal.welch(lines, fs=1.0 / dx, window="hann", nperseg=nper, noverlap=nper // 2,
                        detrend="linear", axis=1)
    Pm = P.mean(axis=0)
    out = {"H_n_linhas": int(lines.shape[0]), "H_segmento_mm": nper * dx}
    nb = cfg["H_n_bins"]
    for name, (lmin, lmax) in cfg["H_bands_mm"].items():
        m = (f >= 1.0 / lmax) & (f <= 1.0 / lmin) & (Pm > 0)
        if m.sum() < 5:
            out[f"H_{name}_aviso"] = "pontos insuficientes na faixa"
            continue
        lf, lp = np.log10(f[m]), np.log10(Pm[m])
        edges = np.linspace(lf.min(), lf.max(), nb + 1)
        idx = np.clip(np.digitize(lf, edges[1:-1]), 0, nb - 1)
        bx = np.array([lf[idx == i].mean() for i in range(nb) if np.any(idx == i)])
        by = np.array([lp[idx == i].mean() for i in range(nb) if np.any(idx == i)])
        slope, icpt = np.polyfit(bx, by, 1)
        resid = by - (slope * bx + icpt)
        r2 = 1.0 - np.sum(resid ** 2) / np.sum((by - by.mean()) ** 2)
        beta = -slope
        H = (beta - 1.0) / 2.0
        out.update({f"H_{name}": float(H), f"D_perfil_{name}": float(2.0 - H),
                    f"D_superficie_{name}": float(3.0 - H), f"H_{name}_beta": float(beta),
                    f"H_{name}_R2": float(r2), f"H_{name}_faixa_mm": f"{lmin}-{lmax}",
                    f"H_{name}_aviso": "" if 0.0 < H < 1.0 else
                    "H fora de (0,1): a faixa não é auto-afim; D sem significado físico"})
    if "micro" in cfg["H_bands_mm"]:
        out["H_micro_status"] = "PROVISORIO: depende da resolução óptica lateral do 3dT (desconhecida)"
    df_psd = pd.DataFrame({"f_por_mm": f[1:], "lambda_mm": 1.0 / f[1:], "PSD_mm3": Pm[1:]})
    return out, df_psd


# ======================================================================================
# Areal — ISO 25178-3 / 25178-2
# ======================================================================================
def gauss(Z: np.ndarray, nesting_mm: float, dx: float) -> np.ndarray:
    sigma = ALPHA_G * nesting_mm / np.sqrt(2 * np.pi) / dx            # = 0,1874·λc em células
    return ndimage.gaussian_filter(Z, sigma=sigma, mode="reflect", truncate=4.0)


def fill_nan(Z: np.ndarray) -> np.ndarray:
    if np.isfinite(Z).all():
        return Z
    m = np.isfinite(Z)
    idx = ndimage.distance_transform_edt(~m, return_distances=False, return_indices=True)
    return Z[tuple(idx)]


def remove_plane(Z: np.ndarray, dx: float, sub: int = 8) -> np.ndarray:
    ny, nx = Z.shape
    yy, xx = np.mgrid[0:ny:sub, 0:nx:sub]
    A = np.c_[xx.ravel() * dx, yy.ravel() * dx, np.ones(xx.size)]
    c, *_ = np.linalg.lstsq(A, Z[::sub, ::sub].ravel().astype(np.float64), rcond=None)
    out = Z.astype(np.float32, copy=True)
    out -= (c[0] * dx * np.arange(nx, dtype=np.float32))[None, :]
    out -= (c[1] * dx * np.arange(ny, dtype=np.float32))[:, None]
    out -= np.float32(c[2])
    return out


def acf_params(Z: np.ndarray, dx: float, thr: float, max_cells: int = 4_000_000) -> dict:
    """Sal e Str (ISO 25178-2) via ACF por FFT com zero-padding. Decima se a superfície for grande."""
    f = 1
    while (Z.shape[0] // f) * (Z.shape[1] // f) > max_cells:
        f += 1
    z = Z[: (Z.shape[0] // f) * f, : (Z.shape[1] // f) * f]
    if f > 1:
        z = z.reshape(z.shape[0] // f, f, z.shape[1] // f, f).mean(axis=(1, 3))
    z = (z - z.mean()).astype(np.float64)
    ny, nx = z.shape
    F = np.fft.rfft2(z, s=(2 * ny, 2 * nx))
    acf = np.fft.irfft2(np.abs(F) ** 2, s=(2 * ny, 2 * nx))
    acf = np.fft.fftshift(acf / acf[0, 0])
    cy, cx = ny, nx
    acf = acf[cy - ny // 2: cy + ny // 2 + 1, cx - nx // 2: cx + nx // 2 + 1]   # lags até metade
    cy, cx = ny // 2, nx // 2
    yy, xx = np.indices(acf.shape)
    r = np.hypot((yy - cy), (xx - cx)) * dx * f
    below = acf <= thr
    if not below.any():
        return {"Sal": None, "Str": None, "acf_aviso": "ACF não decai a 0,2 na janela"}
    sal = float(r[below].min())
    lab, _ = ndimage.label(~below)
    central = lab == lab[cy, cx]
    edge = central & ~ndimage.binary_erosion(central)
    touches = central[0, :].any() or central[-1, :].any() or central[:, 0].any() or central[:, -1].any()
    rmax = float(r[edge].max())
    return {"Sal": sal, "Str": sal / rmax if rmax > 0 else None,
            "acf_aviso": "região central toca a borda da janela: Str subestimado" if touches else ""}


def areal_params(S: np.ndarray, dx: float, prefix: str, cfg: dict, acf: bool = True) -> dict:
    out = height_stats(S, "S")
    gy, gx = np.gradient(S, dx)
    g2 = gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2
    out["Sdq"] = float(np.sqrt(np.mean(g2)))
    out["Sdr_pct"] = float(np.mean(np.sqrt(1 + g2) - 1) * 100)
    del gx, gy, g2
    flat = S.ravel()
    if flat.size > 20_000_000:
        flat = flat[:: int(np.ceil(flat.size / 20_000_000))]
    out.update({("S" + k): v for k, v in rk_family(flat).items()})
    out.update(volume_params(flat))
    if acf:
        out.update(acf_params(S, dx, cfg["S_acf_threshold"]))
    return {f"{prefix}_{k}": v for k, v in out.items()}


def chain_areal(Z: np.ndarray, dx: float, cfg: dict) -> dict:
    Z = fill_nan(Z)
    out = {}
    Ss = gauss(Z, cfg["S_nis_mm"], dx)
    SF = remove_plane(Ss, dx)
    del Ss
    m = int(np.ceil(cfg["S_L_macro_mm"] / dx))                       # margem = L (efeito de borda)
    out.update(areal_params(SF[m:-m, m:-m], dx, "SF", cfg))
    SL = SF - gauss(SF, cfg["S_L_macro_mm"], dx)
    out.update(areal_params(SL[m:-m, m:-m], dx, "SL5", cfg))
    del SL, SF
    P = remove_plane(Z, dx)
    mu = int(np.ceil(cfg["S_L_micro_mm"] / dx))
    MI = P - gauss(P, cfg["S_L_micro_mm"], dx)
    del P
    c = MI[mu:-mu, mu:-mu]
    out.update(areal_params(c, dx, "MICRO", cfg, acf=False))
    cy, cx = c.shape[0] // 2, c.shape[1] // 2
    win = c[max(0, cy - 1024): cy + 1024, max(0, cx - 1024): cx + 1024]
    out.update({f"MICRO_{k}": v for k, v in acf_params(win, dx, cfg["S_acf_threshold"]).items()})
    out["MICRO_status"] = ("PROVISORIO: banda ~0,05–0,5 mm, sem S-filter digital (limite óptico do 3dT "
                           "desconhecido), razão L/S fora da Tab. 1 da ISO 25178-3")
    return out


# ======================================================================================
# Processamento de um arquivo
# ======================================================================================
def process_file(path: str, out_root: str, cfg: dict) -> dict:
    t0 = time.time()
    stem = Path(path).stem
    info = parse_name(stem)
    od = Path(out_root) / (info["trecho"] or "sem_trecho") / stem
    od.mkdir(parents=True, exist_ok=True)
    res = {**info, "caminho": str(path), "versao_script": VERSION}
    tim = {}
    try:
        t = time.time(); Z, dx, meta = read_laz(path); tim["leitura_s"] = time.time() - t
        res.update(meta)
        t = time.time(); a, df_a = chain_a(Z, dx, cfg); tim["cadeia_A_s"] = time.time() - t
        res.update(a); df_a.to_csv(od / "cadeiaA_segmentos.csv", index=False)
        t = time.time(); e, df_e, df_eb = chain_spectrum(Z, dx, cfg); tim["espectro_s"] = time.time() - t
        res.update(e); df_e.to_csv(od / "espectro_terco_oitava.csv", index=False)
        df_eb.to_csv(od / "espectro_por_perfil.csv", index=False)
        t = time.time(); hu, df_psd = chain_hurst(Z, dx, cfg); tim["hurst_s"] = time.time() - t
        res.update(hu); df_psd.to_csv(od / "psd_media.csv", index=False)
        t = time.time(); res.update(chain_areal(Z, dx, cfg)); tim["areal_s"] = time.time() - t
        s = cfg["preview_step"]
        np.savez_compressed(od / "previa.npz", z=Z[::s, ::s].astype(np.float16), passo_mm=dx * s)
        res["status"] = "ok"
    except Exception as ex:                                          # noqa: BLE001
        res["status"] = "erro"
        res["erro"] = f"{type(ex).__name__}: {ex}"
        (od / "erro.txt").write_text(traceback.format_exc(), encoding="utf-8")
    tim["total_s"] = time.time() - t0
    res.update(tim)
    with open(od / "resumo.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
    return res


def main():
    ap = argparse.ArgumentParser(description="Descritores de textura (ISO 13473-1/-4, 25178, 13565-2) em lote.")
    ap.add_argument("--input", required=True, help="pasta com arquivos .laz (busca recursiva)")
    ap.add_argument("--output", required=True, help="pasta de resultados")
    ap.add_argument("--workers", type=int, default=0, help="processos paralelos (0 = automático)")
    ap.add_argument("--pattern", default="*.laz", help="filtro de nomes (ex.: '*B6*MP1*.laz')")
    ap.add_argument("--skip-done", action="store_true", help="pula arquivos com resumo.json status ok")
    args = ap.parse_args()

    files = sorted(str(p) for p in Path(args.input).rglob(args.pattern))
    if not files:
        sys.exit(f"Nenhum arquivo '{args.pattern}' em {args.input}")
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.skip_done:
        def done(f):
            info = parse_name(Path(f).stem)
            r = out / (info["trecho"] or "sem_trecho") / Path(f).stem / "resumo.json"
            return r.exists() and json.loads(r.read_text(encoding="utf-8")).get("status") == "ok"
        files = [f for f in files if not done(f)]
    ncpu = os.cpu_count() or 1
    try:
        mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, AttributeError, OSError):
        mem_gb = 16.0
    workers = args.workers or max(1, min(len(files), ncpu, int(mem_gb // 4)))   # ~4 GB por arquivo
    print(f"{len(files)} arquivo(s) | {workers} processo(s) | CPU {ncpu} | RAM {mem_gb:.0f} GB", flush=True)

    run = {"versao_script": VERSION, "inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "config": CFG,
           "python": sys.version, "numpy": np.__version__, "plataforma": platform.platform(),
           "entrada": str(Path(args.input).resolve()), "n_arquivos": len(files), "workers": workers}
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_file, f, str(out), CFG): f for f in files}
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            results.append(r)
            msg = (f"MPD={r.get('MPD', float('nan')):.3f} mm" if r.get("status") == "ok" else r.get("erro"))
            print(f"[{k}/{len(files)}] {r['arquivo']}: {r['status']} {msg} ({r.get('total_s', 0):.0f} s)", flush=True)
    run["fim"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["duracao_s"] = time.time() - t0
    with open(out / "execucao.json", "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False, default=str)
    allres = []
    for r in out.rglob("resumo.json"):
        try:
            allres.append(json.loads(r.read_text(encoding="utf-8")))
        except Exception:                                            # noqa: BLE001
            pass
    pd.DataFrame(allres).sort_values(["trecho", "mp", "arquivo"]).to_csv(out / "resumo_geral.csv", index=False)
    n_err = sum(r.get("status") != "ok" for r in results)
    print(f"Concluído em {run['duracao_s'] / 60:.1f} min. Erros: {n_err}. Resumo: {out / 'resumo_geral.csv'}")


if __name__ == "__main__":
    main()

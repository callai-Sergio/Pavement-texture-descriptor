#!/usr/bin/env python3
"""
texturelab_batch.py — Cálculo em lote (sem interface) de descritores de textura de pavimento
a partir de varreduras 3dT (BASt) em LAZ.

Núcleo único do TextureLab: o mesmo cálculo alimenta o lote, a tese e o app (que só lê os resultados,
ver "Projeto" abaixo). Cálculos da v3.0.0 (conferidos contra as normas) + Hurst, com as otimizações
de memória e CPU da v2 do servidor:
  - leitura do LAZ em blocos (int32, sem cópias int64 nem do registro completo de pontos);
  - faixas (strips) calculadas uma vez e compartilhadas entre cadeia A e espectro;
  - cadeia areal sem cópias simultâneas da grade (filtros subtraídos no lugar, estatísticas,
    gradiente e decimação da ACF por blocos de linhas);
  - curva de material ratio calculada uma vez por superfície (Rk e volume);
  - filtros de terço de oitava projetados uma vez e aplicados a todos os perfis juntos;
  - cada processo trata um único arquivo e é reiniciado (max_tasks_per_child=1);
  - número de processos pela RAM DISPONÍVEL e pelo tamanho dos arquivos (não pela RAM total).
Diferenças numéricas esperadas em relação à v3.0.0 sem otimizações: apenas arredondamento de somas
(~1e-12 relativo) nas estatísticas areais calculadas por blocos.

Uso:
    python texturelab_batch.py --input /dados/laz --output /dados/projeto [--workers N] [--mem-gb G]
                               [--config ajustes.json] [--skip-done] [--zip]

Projeto (pasta de saída, lida pelo app sem recalcular; formato em docs/FORMATO_PROJETO.md):
    projeto.json                         índice: versão do núcleo, configuração, lista de arquivos
    resumo_geral.csv, execucao.json
    <trecho>/<arquivo>/resumo.json       parâmetros + "receita" (versão do núcleo + config + LAZ)
                     /cadeiaA_segmentos.csv, espectro_terco_oitava.csv, espectro_por_perfil.csv,
                     /psd_media.csv, previa.npz, visualizacao.npz
    Com --skip-done só são recalculados os arquivos cuja receita mudou (código, configuração ou LAZ).
    Com --zip a pasta também é empacotada em <saída>.tlproj (zip, sem pickle).

Dependências: numpy, scipy, pandas, laspy[lazrs], PyWavelets   (Python >= 3.11)

Cadeias implementadas (todas sobre a grade em mm, eixo maior = sentido da via):
  A  MPD/MSD ................ ISO 13473-1:2019 (medição pontual): faixas 0,5 mm, reamostragem 0,5 mm,
                               spikes Anexo E, passa-baixa Butterworth 2ª ordem projetado em 2,40 mm
                               ida e volta (Tab. D.2), supressão de rampa por segmento. (ETD = 1,1·MPD
                               não é mais gravado: é só uma escala do MPD.)
  G  g-factor e assimetria ... DIN ISO 10844:2024-11, 5.3.2 e Anexo B: por segmento de 100 mm do MPD,
                               perfil da ISO 13473-1 sem passa-baixa, média zero; z_mid = (z_max+z_min)/2;
                               g = distribuição cumulativa (0 % no ponto mais alto) onde z_mid cruza a
                               curva de Abbott; média dos segmentos por ponto de medição (arquivo).
                               Assimetria = Rsk por segmento (ISO 13473-2), média = perfil_Rsk_media.
  W  Ondaletas ............... descritivo, sem norma: DWT ortonormal (Daubechies db4, periodização).
                               Perfil: linhas nativas (uma por faixa de 0,5 mm), energia por oitava
                               λ ∈ [2^j·dx, 2^(j+1)·dx], em µm rms e dB re 1 µm; parcelas micro/macro.
                               2D: superfície SL5, energia por oitava ao longo da via, transversal e
                               diagonal; anisotropia = (E_via − E_transv)/(E_via + E_transv).
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
import copy
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Um fio por processo: o paralelismo é entre arquivos.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import ndimage, signal

VERSION = "3.2.0"
FORMAT_VERSION = 1              # formato do projeto (projeto.json / visualizacao.npz)

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
    # ondaletas (descritivo, não normativo)
    "W_wavelet": "db4",             # Daubechies 4 (ortonormal: energia dos coeficientes = variância)
    "W_lambda_max_mm": 50.0,        # maior banda de perfil (limite superior da oitava)
    "W_micro_limit_mm": 0.5,        # bandas com λ central < 0,5 mm contam como micro
    # saídas para o app (não alteram os parâmetros)
    "preview_step": 8,
    "V_abbott_points": 201,         # pontos da curva de Abbott-Firestone gravados por cadeia areal
    "V_hist_bins": 120,             # classes do histograma de alturas (PDF)
    "V_n_profiles": 5,              # perfis da cadeia A gravados para o app
    "V_native_profile_mm": 100.0,   # trecho de linha na resolução nativa gravado para o app
}

ALPHA_G = np.sqrt(np.log(2.0) / np.pi)          # ISO 16610-61
NOMINAL_TO = [500, 400, 315, 250, 200, 160, 125, 100, 80, 63, 50, 40, 31.5, 25, 20, 16, 12.5, 10,
              8, 6.3, 5, 4, 3.15, 2.5, 2, 1.6, 1.25, 1, 0.8, 0.63, 0.5, 0.4]   # ISO 13473-4 Tab. 2

READ_CHUNK = 5_000_000          # pontos por bloco na leitura do LAZ
BLOCK_CELLS = 4_000_000         # células por bloco nas estatísticas areais
BYTES_PER_POINT = 32            # pico estimado de memória por ponto (v2); ver "pico_memoria_processo_gb"
BASE_GB = 1.0                   # memória fixa por processo (Python, bibliotecas, buffers)


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
    Leitura em blocos: só X, Y, Z inteiros (int32) ficam na memória, não o registro completo.
    """
    import laspy
    with laspy.open(path) as f:
        h = f.header
        n = int(h.point_count)
        X = np.empty(n, dtype=np.int32)
        Y = np.empty(n, dtype=np.int32)
        Zi = np.empty(n, dtype=np.int32)
        i = 0
        for pts in f.chunk_iterator(READ_CHUNK):
            m = len(pts)
            X[i:i + m] = pts.X
            Y[i:i + m] = pts.Y
            Zi[i:i + m] = pts.Z
            i += m
        sx, sy, sz = (float(s) for s in h.scales)
        z_off = float(h.offsets[2])
    if i != n:
        X, Y, Zi = X[:i], Y[:i], Zi[:i]
    meta = {"n_pontos": int(len(X)), "escala": [sx, sy, sz]}

    ux = np.unique(X)
    uy = np.unique(Y)
    stepx = int(np.median(np.diff(ux))) if len(ux) > 1 else 1
    stepy = int(np.median(np.diff(uy))) if len(uy) > 1 else 1
    dx = stepx * sx
    dy = stepy * sy
    # unidades: o 3dT grava em mm. Passo < 0,001 indica metros.
    metros = dx < 1e-3

    grid = None
    nx, ny = len(ux), len(uy)
    if nx * ny == len(X):
        # z em float32, calculado em float64 por bloco (mesmo arredondamento da v1)
        z = np.empty(len(Zi), dtype=np.float32)
        for a in range(0, len(Zi), READ_CHUNK):
            zz = Zi[a:a + READ_CHUNK] * sz + z_off
            if metros:
                zz = zz * 1000.0
            z[a:a + READ_CHUNK] = zz
            del zz
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
        del g, z
    if grid is None:
        z = Zi * sz + z_off
        ix = ((X.astype(np.int64) - X.min()) // stepx).astype(np.int64)
        iy = ((Y.astype(np.int64) - Y.min()) // stepy).astype(np.int64)
        nx, ny = int(ix.max()) + 1, int(iy.max()) + 1
        lin = ix * ny + iy
        del ix, iy
        s = np.bincount(lin, weights=z, minlength=nx * ny)
        c = np.bincount(lin, minlength=nx * ny)
        del lin, z
        with np.errstate(invalid="ignore", divide="ignore"):
            grid = (s / c).reshape(nx, ny)
        del s, c
        if metros:
            grid = grid * 1000.0
        meta["leitura"] = "generica"
    del X, Y, Zi

    if metros:
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
    """Média de linhas paralelas adjacentes (largura total >= width_mm). Devolve [n_faixas, n_via].
    Uma faixa por vez: não converte a grade inteira para float64."""
    k = int(np.ceil(width_mm / dx - 1e-9))
    n = Z.shape[0] // k
    strips = np.empty((n, Z.shape[1]), dtype=np.float64)
    with np.errstate(invalid="ignore"):
        for i in range(n):
            strips[i] = np.nanmean(Z[i * k:(i + 1) * k].astype(np.float64), axis=0)
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


def rk_family(h: np.ndarray, n: int = 10001, curve=None) -> dict:
    """ISO 13565-2: zona central de 40 % com menor inclinação da secante; reta de mínimos quadrados
    nessa zona; Rk, Mr1, Mr2; Rpk e Rvk pelos triângulos de área equivalente.
    'curve' = (mr, c) já calculada sobre h finito, para não repetir o quantil."""
    if curve is None:
        h = h[np.isfinite(h)]
        if h.size < 20:
            return {}
        mr, c = material_ratio_curve(h, n)
    else:
        mr, c = curve
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


def volume_params(h: np.ndarray, p: float = 10.0, q: float = 80.0, n: int = 10001, curve=None) -> dict:
    """ISO 25178-2 Vmp, Vmc, Vvc, Vvv em mm³/mm² (= mm; ×1000 -> ml/m²)."""
    mr, c = curve if curve is not None else material_ratio_curve(h[np.isfinite(h)], n)

    def vm(t):
        m = mr <= t
        return _trapz(c[m] - np.interp(t, mr, c), mr[m]) / 100.0

    def vv(t):
        m = mr >= t
        return _trapz(np.interp(t, mr, c) - c[m], mr[m]) / 100.0

    return {"Vmp": float(vm(p)), "Vmc": float(vm(q) - vm(p)),
            "Vvc": float(vv(p) - vv(q)), "Vvv": float(vv(q))}


def g_factor(seg: np.ndarray) -> tuple[float, np.ndarray, float]:
    """Shape factor (g-factor), DIN ISO 10844:2024-11 Anexo B, de um segmento de 100 mm já processado
    (ISO 13473-1 sem passa-baixa). B.1: média zero. B.2: ordena do mais alto ao mais baixo, z_mid =
    (z_max + z_min)/2. B.3: D_cum,i = (i-1)·100 %/(n-1). B.4: g = D_cum onde z_mid cruza a curva
    (interpolação linear entre os dois pontos vizinhos). Devolve (g [%], z ordenado, z_mid)."""
    z = np.sort(seg[np.isfinite(seg)] - np.nanmean(seg))[::-1]
    n = z.size
    if n < 2 or z[0] == z[-1]:
        return float("nan"), z, float("nan")
    d = np.arange(n) * 100.0 / (n - 1)
    z_mid = (z[0] + z[-1]) / 2.0
    g = float(np.interp(-z_mid, -z, d))          # z decrescente -> -z crescente
    return g, z, float(z_mid)


def height_stats(h: np.ndarray, prefix: str) -> dict:
    h = h[np.isfinite(h)].astype(np.float64)
    h = h - h.mean()
    sq = np.sqrt(np.mean(h ** 2))
    return {f"{prefix}a": float(np.mean(np.abs(h))), f"{prefix}q": float(sq),
            f"{prefix}sk": float(np.mean(h ** 3) / sq ** 3) if sq > 0 else 0.0,
            f"{prefix}ku": float(np.mean(h ** 4) / sq ** 4) if sq > 0 else 0.0,
            f"{prefix}p": float(h.max()), f"{prefix}v": float(-h.min()), f"{prefix}z": float(h.max() - h.min())}


def _row_blocks(S: np.ndarray, min_rows: int = 1):
    rows = max(min_rows, BLOCK_CELLS // max(1, S.shape[1]))
    return range(0, S.shape[0], rows), rows


def height_stats_2d(S: np.ndarray, prefix: str) -> dict:
    """Igual a height_stats, mas por blocos de linhas (sem cópias float64 da superfície inteira)."""
    starts, rows = _row_blocks(S)
    tot, cnt = 0.0, 0
    for a in starts:
        b = S[a:a + rows]
        b = b[np.isfinite(b)].astype(np.float64)
        tot += float(b.sum())
        cnt += b.size
    mean = tot / cnt
    s1 = s2 = s3 = s4 = 0.0
    hmax, hmin = -np.inf, np.inf
    for a in starts:
        b = S[a:a + rows]
        b = b[np.isfinite(b)].astype(np.float64) - mean
        if b.size == 0:
            continue
        s1 += float(np.abs(b).sum())
        s2 += float((b ** 2).sum())
        s3 += float((b ** 3).sum())
        s4 += float((b ** 4).sum())
        hmax = max(hmax, float(b.max()))
        hmin = min(hmin, float(b.min()))
    sq = np.sqrt(s2 / cnt)
    return {f"{prefix}a": s1 / cnt, f"{prefix}q": float(sq),
            f"{prefix}sk": float(s3 / cnt / sq ** 3) if sq > 0 else 0.0,
            f"{prefix}ku": float(s4 / cnt / sq ** 4) if sq > 0 else 0.0,
            f"{prefix}p": hmax, f"{prefix}v": -hmin, f"{prefix}z": hmax - hmin}


def gradient_stats(S: np.ndarray, dx: float) -> dict:
    """Sdq e Sdr por blocos de linhas, com 1 linha de sobreposição: valores do gradiente idênticos
    aos de np.gradient na superfície inteira."""
    ny = S.shape[0]
    starts, rows = _row_blocks(S, min_rows=2)
    s_g2 = s_dr = 0.0
    for a in starts:
        b = min(ny, a + rows)
        lo, hi = max(0, a - 1), min(ny, b + 1)
        gy, gx = np.gradient(S[lo:hi], dx)
        keep = slice(a - lo, a - lo + (b - a))
        g2 = gx[keep].astype(np.float64) ** 2 + gy[keep].astype(np.float64) ** 2
        del gx, gy
        s_g2 += float(g2.sum())
        s_dr += float((np.sqrt(1 + g2) - 1).sum())
    n = S.shape[0] * S.shape[1]
    return {"Sdq": float(np.sqrt(s_g2 / n)), "Sdr_pct": float(s_dr / n * 100)}


def ravel_subsample(S: np.ndarray, max_n: int) -> np.ndarray:
    """Equivale a S.ravel()[::k] (k = ceil(tamanho/max_n)) sem copiar a superfície inteira."""
    if S.size <= max_n:
        return S.ravel()
    k = int(np.ceil(S.size / max_n))
    idx = np.arange(0, S.size, k, dtype=np.int64)
    out = np.empty(idx.size, dtype=S.dtype)
    for a in range(0, idx.size, BLOCK_CELLS):
        r, c = np.divmod(idx[a:a + BLOCK_CELLS], S.shape[1])
        out[a:a + BLOCK_CELLS] = S[r, c]
    return out


# ======================================================================================
# Cadeia A — ISO 13473-1:2019
# ======================================================================================
def chain_a(Z: np.ndarray, dx: float, cfg: dict, strips=None, extras: dict | None = None
            ) -> tuple[dict, pd.DataFrame]:
    """'extras' (opcional) recebe perfis de exemplo para o app: limpos e após o passa-baixa."""
    step = cfg["A_resample_mm"]
    strips, centers = strips if strips is not None else make_strips(Z, dx, cfg["A_strip_width_mm"])
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
    if extras is not None and ok_rows.any():
        sel = np.flatnonzero(ok_rows)
        sel = sel[np.unique(np.linspace(0, len(sel) - 1, min(cfg["V_n_profiles"], len(sel))).round().astype(int))]
        extras.update({"A_perfil_faixa": sel.astype(np.int32),
                       "A_perfil_y_mm": centers[sel].astype(np.float32),
                       "A_perfil_passo_mm": np.float64(step),
                       "A_perfil_limpo": cleaned[sel].astype(np.float32),
                       "A_perfil_passa_baixa": lp[sel].astype(np.float32),
                       "A_segmento_inicio_mm": np.float64(start * step),
                       "A_segmento_mm": np.float64(cfg["A_segment_mm"]), "A_n_segmentos": np.int32(nseg)})

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
                g, gz, gmid = g_factor(raw)                                       # ISO 10844 Anexo B
                r["g_pct"] = g
                if extras is not None and "g_seg_z" not in extras and i >= P.shape[0] // 2:
                    extras.update({"g_seg_z": gz.astype(np.float32), "g_seg_zmid": np.float64(gmid),
                                   "g_seg_g": np.float64(g), "g_seg_faixa": np.int32(i),
                                   "g_seg_y_mm": np.float64(centers[i]), "g_seg_inicio_mm": np.float64(a * step)})
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
        out.update({"MPD": mpd, "MPD_desvio": float(v["MSD"].std(ddof=1)) if len(v) > 1 else 0.0})
        gv = v["g_pct"].dropna() if "g_pct" in v else pd.Series(dtype=float)
        if len(gv):
            out.update({"g_factor_media": float(gv.mean()),
                        "g_factor_desvio": float(gv.std(ddof=1)) if len(gv) > 1 else 0.0,
                        "g_factor_n_segmentos": int(len(gv))})
        for k in ["Rq", "Rsk", "Rku", "Rk", "Rpk", "Rvk", "Rmr1", "Rmr2"]:
            if k in v:
                out[f"perfil_{k}_media"] = float(v[k].mean())
    return out, df


# ======================================================================================
# Espectro — ISO 13473-4:2024, método 1
# ======================================================================================
def chain_spectrum(Z: np.ndarray, dx: float, cfg: dict, strips=None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    if strips is None:
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
    rows_ok, xs, n_mir = [], [], 0
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
        n_mir = len(mir)
        rows_ok.append(i)
        xs.append(np.concatenate([mir, p]))
    if rows_ok and bands:
        X = np.vstack(xs)
        del xs
        for j, (_, lam) in enumerate(bands):
            fm = 1.0 / lam
            f1, f2 = fm * 10 ** (-0.05), fm * 10 ** (0.05)               # IEC 61260-1, base 10, b = 3
            sos = signal.butter(3, [f1, f2], btype="bandpass", fs=1.0 / step, output="sos")
            y = signal.sosfilt(sos, X, axis=-1)[:, n_mir:]               # todos os perfis de uma vez
            rms_mm = np.sqrt(np.mean(y ** 2, axis=1))
            levels[rows_ok, j] = 20 * np.log10(rms_mm * 1e-3 / 1e-6)    # ref 1 µm
        del X
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
# Ondaletas — descritivo (sem norma)
# ======================================================================================
def _wavelet_bands(dx: float, levels: int) -> pd.DataFrame:
    """Oitava do detalhe de nível j (1 = mais fino): λ entre 2^j·dx e 2^(j+1)·dx, centro geométrico."""
    j = np.arange(1, levels + 1)
    lo, hi = 2.0 ** j * dx, 2.0 ** (j + 1) * dx
    return pd.DataFrame({"nivel": j, "lambda_min_mm": lo, "lambda_max_mm": hi, "lambda_centro_mm": np.sqrt(lo * hi)})


def chain_wavelet_profile(Z: np.ndarray, dx: float, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Espectro de ondaletas do perfil no sentido da via. Linhas na resolução nativa (sem média entre
    linhas), uma a cada largura de faixa da cadeia A, tendência linear removida. DWT ortonormal com
    periodização: soma(d_j²)/N = parcela da variância do perfil na oitava j."""
    import pywt
    k = max(1, int(np.ceil(cfg["A_strip_width_mm"] / dx - 1e-9)))
    lines = Z[::k].astype(np.float64)
    lines = lines[np.isfinite(lines).all(axis=1)]
    if lines.shape[0] == 0:
        return {"W_erro": "nenhuma linha sem valores inválidos"}, pd.DataFrame()
    n = lines.shape[1]
    t = np.arange(n, dtype=np.float64)
    coef = np.polyfit(t, lines.T, 1)
    lines -= (np.outer(coef[0], t) + coef[1][:, None])
    w = pywt.Wavelet(cfg["W_wavelet"])
    max_lev = pywt.dwt_max_level(n, w.dec_len)
    levels = int(min(max_lev, np.floor(np.log2(cfg["W_lambda_max_mm"] / dx)) - 1))
    c = pywt.wavedec(lines, w, mode="periodization", level=levels, axis=1)
    bands = _wavelet_bands(dx, levels)
    var = np.stack([(c[-j] ** 2).sum(axis=1) / n for j in range(1, levels + 1)], axis=1)   # [linhas, níveis]
    with np.errstate(divide="ignore"):
        L = 10 * np.log10(var / 1e-6)                                                     # (1 µm)² = 1e-6 mm²
    bands["rms_um"] = np.sqrt(var.mean(axis=0)) * 1e3
    bands["L_w_dB_media"] = 10 * np.log10(var.mean(axis=0) / 1e-6)
    bands["L_w_dB_desvio"] = L.std(axis=0, ddof=1) if L.shape[0] > 1 else 0.0
    bands["fracao_energia"] = var.mean(axis=0) / var.mean(axis=0).sum()
    micro = bands["lambda_centro_mm"] < cfg["W_micro_limit_mm"]
    out = {"W_n_linhas": int(lines.shape[0]), "W_ondaleta": cfg["W_wavelet"], "W_n_niveis": levels,
           "W_rms_micro_um": float(np.sqrt(var.mean(axis=0)[micro.to_numpy()].sum()) * 1e3),
           "W_rms_macro_um": float(np.sqrt(var.mean(axis=0)[~micro.to_numpy()].sum()) * 1e3),
           "W_fracao_micro": float(bands.loc[micro, "fracao_energia"].sum())}
    return out, bands


def wavelet_2d(S: np.ndarray, dx: float, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Energia de ondaletas 2D por oitava e direção (eixo 0 = largura, eixo 1 = via). pywt: cV = passa-alta
    ao longo do eixo 1 -> variação no sentido da via; cH -> variação transversal; cD -> diagonal. Em float32
    para limitar a memória. Níveis até a oitava cujo limite inferior fica abaixo do L do SL5."""
    import pywt
    w = pywt.Wavelet(cfg["W_wavelet"])
    max_lev = pywt.dwt_max_level(min(S.shape), w.dec_len)
    levels = int(min(max_lev, np.floor(np.log2(cfg["S_L_macro_mm"] / dx))))
    c = pywt.wavedec2(np.ascontiguousarray(S, dtype=np.float32), w, mode="periodization", level=levels)
    ncell = float(S.size)
    bands = _wavelet_bands(dx, levels)
    e = np.array([[float((c[-j][q].astype(np.float64) ** 2).sum()) / ncell for q in (1, 0, 2)]
                  for j in range(1, levels + 1)])                                         # via, transversal, diagonal
    del c
    bands["rms_via_um"] = np.sqrt(e[:, 0]) * 1e3
    bands["rms_transversal_um"] = np.sqrt(e[:, 1]) * 1e3
    bands["rms_diagonal_um"] = np.sqrt(e[:, 2]) * 1e3
    with np.errstate(invalid="ignore", divide="ignore"):
        bands["anisotropia"] = (e[:, 0] - e[:, 1]) / (e[:, 0] + e[:, 1])
    micro = (bands["lambda_centro_mm"] < cfg["W_micro_limit_mm"]).to_numpy()
    out = {}
    for name, m in (("micro", micro), ("macro", ~micro)):
        ev, et = e[m, 0].sum(), e[m, 1].sum()
        out[f"W2_anisotropia_{name}"] = float((ev - et) / (ev + et)) if ev + et > 0 else None
    out["W2_n_niveis"] = levels
    return out, bands


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
    """Sal e Str (ISO 25178-2) via ACF por FFT com zero-padding. Decima se a superfície for grande
    (decimação por blocos de linhas, sem copiar a superfície inteira)."""
    f = 1
    while (Z.shape[0] // f) * (Z.shape[1] // f) > max_cells:
        f += 1
    if f > 1:
        nyo, nxo = Z.shape[0] // f, Z.shape[1] // f
        R = max(1, BLOCK_CELLS // (f * Z.shape[1]))
        parts = []
        for j in range(0, nyo, R):
            r = min(R, nyo - j)
            blk = Z[j * f:(j + r) * f, : nxo * f]
            parts.append(blk.reshape(r, f, nxo, f).mean(axis=(1, 3)))
        z = np.concatenate(parts, axis=0)
        del parts
    else:
        z = Z
    z = (z - z.mean()).astype(np.float64)
    ny, nx = z.shape
    F = np.fft.rfft2(z, s=(2 * ny, 2 * nx))
    acf = np.fft.irfft2(np.abs(F) ** 2, s=(2 * ny, 2 * nx))
    del F
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


def _view_curves(h: np.ndarray, curve, prefix: str, cfg: dict, extras: dict) -> None:
    """Curva de Abbott-Firestone reduzida e histograma de alturas (centradas na média) para o app."""
    mr, c = curve
    idx = np.unique(np.linspace(0, len(mr) - 1, cfg["V_abbott_points"]).round().astype(int))
    mean = float(np.mean(h, dtype=np.float64))
    cnt, edges = np.histogram(h.astype(np.float64) - mean, bins=cfg["V_hist_bins"])
    extras.update({f"{prefix}_abbott_mr_pct": mr[idx].astype(np.float32),
                   f"{prefix}_abbott_altura_mm": (c[idx] - mean).astype(np.float32),
                   f"{prefix}_hist_bordas_mm": edges.astype(np.float32),
                   f"{prefix}_hist_contagem": cnt.astype(np.int64)})


def areal_params(S: np.ndarray, dx: float, prefix: str, cfg: dict, acf: bool = True,
                 extras: dict | None = None) -> dict:
    out = height_stats_2d(S, "S")
    out.update(gradient_stats(S, dx))
    flat = ravel_subsample(S, 20_000_000)
    h = flat[np.isfinite(flat)]
    del flat
    curve = material_ratio_curve(h, 10001) if h.size else None    # uma vez para Rk e volume
    out.update({("S" + k): v for k, v in (rk_family(h, curve=curve) if h.size >= 20 else {}).items()})
    out.update(volume_params(h, curve=curve))
    if extras is not None and curve is not None:
        _view_curves(h, curve, prefix, cfg, extras)
    del h, curve
    if acf:
        out.update(acf_params(S, dx, cfg["S_acf_threshold"]))
    return {f"{prefix}_{k}": v for k, v in out.items()}


def chain_areal(zbox: list, dx: float, cfg: dict, extras: dict | None = None,
                tables: dict | None = None) -> dict:
    """Recebe a grade dentro de uma lista (zbox) para poder liberá-la no meio do cálculo.
    Ordem: MICRO primeiro (usa Z), depois SF e SL5; nunca mais de ~3 grades na memória.
    'extras' (opcional) recebe curvas de Abbott, histogramas e prévias filtradas para o app."""
    Z = fill_nan(zbox.pop())

    # MICRO: F = plano, L = 0,5 mm
    mu = int(np.ceil(cfg["S_L_micro_mm"] / dx))
    P = remove_plane(Z, dx)
    MI = gauss(P, cfg["S_L_micro_mm"], dx)
    np.subtract(P, MI, out=MI)                                         # MI = P - gauss(P)
    del P
    c = MI[mu:-mu, mu:-mu]
    micro = areal_params(c, dx, "MICRO", cfg, acf=False, extras=extras)
    cy, cx = c.shape[0] // 2, c.shape[1] // 2
    win = c[max(0, cy - 1024): cy + 1024, max(0, cx - 1024): cx + 1024]
    micro.update({f"MICRO_{k}": v for k, v in acf_params(win, dx, cfg["S_acf_threshold"]).items()})
    micro["MICRO_status"] = ("PROVISORIO: banda ~0,05–0,5 mm, sem S-filter digital (limite óptico do 3dT "
                             "desconhecido), razão L/S fora da Tab. 1 da ISO 25178-3")
    del c, win, MI

    # SF: S = 0,05 mm, F = plano
    Ss = gauss(Z, cfg["S_nis_mm"], dx)
    del Z
    SF = remove_plane(Ss, dx)
    del Ss
    m = int(np.ceil(cfg["S_L_macro_mm"] / dx))                       # margem = L (efeito de borda)
    out = areal_params(SF[m:-m, m:-m], dx, "SF", cfg, extras=extras)

    # SL5: + L = 5 mm
    SL = gauss(SF, cfg["S_L_macro_mm"], dx)
    np.subtract(SF, SL, out=SL)                                        # SL = SF - gauss(SF)
    del SF
    out.update(areal_params(SL[m:-m, m:-m], dx, "SL5", cfg, extras=extras))
    w2, df_w2 = wavelet_2d(SL[m:-m, m:-m], dx, cfg)                     # ondaletas 2D na SL5
    out.update(w2)
    if tables is not None:
        tables["ondaletas_2d.csv"] = df_w2
    if extras is not None:
        st = cfg["preview_step"]
        extras["SL5_previa"] = SL[m:-m:st, m:-m:st].astype(np.float16)     # superfície filtrada (3D)
    del SL

    out.update(micro)                                                  # mesma ordem de colunas da v1
    return out


# ======================================================================================
# Processamento de um arquivo
# ======================================================================================
def _peak_mem_gb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # kB no Linux
    except Exception:                                                # noqa: BLE001
        return None


def laz_identity(path: str, chunk: int = 4 << 20) -> dict:
    """Identifica o LAZ sem ler o arquivo inteiro: nome, tamanho e SHA-256 do primeiro e do último
    bloco de 4 MB (detecta arquivo trocado ou regravado; não depende da data de modificação)."""
    size = os.path.getsize(path)
    hsh = hashlib.sha256()
    with open(path, "rb") as fh:
        hsh.update(fh.read(chunk))
        if size > chunk:
            fh.seek(max(chunk, size - chunk))
            hsh.update(fh.read(chunk))
    return {"nome": Path(path).name, "tamanho_bytes": size, "sha256_pontas": hsh.hexdigest()}


def recipe(path: str, cfg: dict) -> dict:
    """Receita do resultado: versão do núcleo + configuração + identificação do LAZ.
    Mesma receita -> mesmo resultado; o hash decide o que precisa ser recalculado."""
    r = {"versao_nucleo": VERSION, "config": cfg, "laz": laz_identity(path)}
    r["hash"] = hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return r


def out_dir(out_root: str | Path, path: str) -> Path:
    info = parse_name(Path(path).stem)
    return Path(out_root) / (info["trecho"] or "sem_trecho") / Path(path).stem


def process_file(path: str, out_root: str, cfg: dict, rec: dict | None = None) -> dict:
    t0 = time.time()
    stem = Path(path).stem
    info = parse_name(stem)
    od = out_dir(out_root, path)
    od.mkdir(parents=True, exist_ok=True)
    (od / "erro.txt").unlink(missing_ok=True)
    rec = rec or recipe(path, cfg)
    res = {**info, "caminho": str(path), "versao_script": VERSION, "receita_hash": rec["hash"]}
    tim = {}
    view = {"versao_formato": np.int32(FORMAT_VERSION)}
    try:
        t = time.time(); Z, dx, meta = read_laz(path); tim["leitura_s"] = time.time() - t
        res.update(meta)
        s = cfg["preview_step"]
        previa = Z[::s, ::s].astype(np.float16)
        cy, n_nat = Z.shape[0] // 2, int(round(cfg["V_native_profile_mm"] / dx))
        a0 = max(0, (Z.shape[1] - n_nat) // 2)
        view.update({"nativo_perfil": Z[cy, a0:a0 + n_nat].astype(np.float32), "nativo_passo_mm": np.float64(dx),
                     "nativo_y_mm": np.float64(cy * dx), "nativo_inicio_mm": np.float64(a0 * dx)})
        t = time.time()
        strips_a = make_strips(Z, dx, cfg["A_strip_width_mm"])
        a, df_a = chain_a(Z, dx, cfg, strips_a, extras=view); tim["cadeia_A_s"] = time.time() - t
        res.update(a); df_a.to_csv(od / "cadeiaA_segmentos.csv", index=False)
        strips_e = strips_a[0] if cfg["E_strip_width_mm"] == cfg["A_strip_width_mm"] else None
        t = time.time(); e, df_e, df_eb = chain_spectrum(Z, dx, cfg, strips_e); tim["espectro_s"] = time.time() - t
        res.update(e); df_e.to_csv(od / "espectro_terco_oitava.csv", index=False)
        df_eb.to_csv(od / "espectro_por_perfil.csv", index=False)
        del strips_a, strips_e
        t = time.time(); hu, df_psd = chain_hurst(Z, dx, cfg); tim["hurst_s"] = time.time() - t
        res.update(hu); df_psd.to_csv(od / "psd_media.csv", index=False)
        t = time.time(); wv, df_wv = chain_wavelet_profile(Z, dx, cfg); tim["ondaletas_s"] = time.time() - t
        res.update(wv); df_wv.to_csv(od / "ondaletas_perfil.csv", index=False)
        zbox = [Z]
        del Z                                                        # chain_areal libera a grade
        tables = {}
        t = time.time(); res.update(chain_areal(zbox, dx, cfg, extras=view, tables=tables)); tim["areal_s"] = time.time() - t
        for name, df_t in tables.items():
            df_t.to_csv(od / name, index=False)
        np.savez_compressed(od / "previa.npz", z=previa, passo_mm=dx * s)
        np.savez_compressed(od / "visualizacao.npz", **view)
        res["status"] = "ok"
    except Exception as ex:                                          # noqa: BLE001
        res["status"] = "erro"
        res["erro"] = f"{type(ex).__name__}: {ex}"
        (od / "erro.txt").write_text(traceback.format_exc(), encoding="utf-8")
    tim["total_s"] = time.time() - t0
    res.update(tim)
    res["pico_memoria_processo_gb"] = _peak_mem_gb()                 # 1 arquivo por processo: pico do arquivo
    res["receita"] = rec
    with open(od / "resumo.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=float)
    return res


# ======================================================================================
# Projeto (índice lido pelo app)
# ======================================================================================
def write_project_index(out: Path) -> dict:
    """Varre os resumo.json da pasta e grava projeto.json e resumo_geral.csv. Pode ser rodado
    sozinho (--index-only), por exemplo sobre resultados antigos."""
    out = Path(out)
    allres, items = [], []
    for r in sorted(out.rglob("resumo.json")):
        try:
            d = json.loads(r.read_text(encoding="utf-8"))
        except Exception:                                            # noqa: BLE001
            continue
        allres.append({k: v for k, v in d.items() if k != "receita"})
        items.append({"arquivo": d.get("arquivo", r.parent.name), "trecho": d.get("trecho", ""),
                      "revestimento": d.get("revestimento", ""), "mp": d.get("mp", ""), "nr": d.get("nr", ""),
                      "data": d.get("data", ""), "pasta": r.parent.relative_to(out).as_posix(),
                      "status": d.get("status", ""), "versao_nucleo": d.get("versao_script", ""),
                      "receita_hash": d.get("receita_hash", ""),
                      "visualizacao": (r.parent / "visualizacao.npz").exists()})
    if allres:
        pd.DataFrame(allres).sort_values(["trecho", "mp", "arquivo"]).to_csv(out / "resumo_geral.csv", index=False)
    versions = sorted({i["versao_nucleo"] for i in items if i["versao_nucleo"]})
    proj = {"formato": "texturelab-projeto", "versao_formato": FORMAT_VERSION, "versao_nucleo": VERSION,
            "versoes_nos_resultados": versions, "atualizado": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": CFG, "n_arquivos": len(items), "n_ok": sum(i["status"] == "ok" for i in items),
            "arquivos": items}
    if len(versions) > 1:
        proj["aviso"] = "resultados de versões diferentes do núcleo misturados: recalcule com --skip-done"
    with open(out / "projeto.json", "w", encoding="utf-8") as fh:
        json.dump(proj, fh, indent=2, ensure_ascii=False, default=str)
    return proj


PROJECT_FILES = ("projeto.json", "resumo_geral.csv", "execucao.json", "resumo.json", "cadeiaA_segmentos.csv",
                 "espectro_terco_oitava.csv", "espectro_por_perfil.csv", "psd_media.csv", "previa.npz",
                 "visualizacao.npz", "ondaletas_perfil.csv", "ondaletas_2d.csv", "erro.txt")


def pack_project(out: Path) -> Path:
    """Empacota só os arquivos do projeto (sem outras pastas de análise) em <saída>.tlproj (zip)."""
    out = Path(out)
    dest = out.with_suffix(".tlproj")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as z:   # NPZ já é comprimido
        z.write(out / "projeto.json", "projeto.json")
        proj = json.loads((out / "projeto.json").read_text(encoding="utf-8"))
        for name in ("resumo_geral.csv", "execucao.json"):
            if (out / name).exists():
                z.write(out / name, name)
        for it in proj["arquivos"]:
            for f in sorted((out / it["pasta"]).iterdir()):
                if f.name in PROJECT_FILES:
                    z.write(f, f"{it['pasta']}/{f.name}")
    return dest


# ======================================================================================
# Memória e número de processos
# ======================================================================================
def mem_available_gb() -> float:
    """RAM disponível agora (MemAvailable), não a total: o servidor é compartilhado."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES") / 1e9
    except (ValueError, AttributeError, OSError):
        return 16.0


def estimate_file_gb(path: str) -> float:
    """Pico de memória estimado para um arquivo, pelo número de pontos do cabeçalho."""
    try:
        import laspy
        with laspy.open(path) as f:
            n = int(f.header.point_count)
        return n * BYTES_PER_POINT / 1e9 + BASE_GB
    except Exception:                                                # noqa: BLE001
        return 8.0


def load_config(path: str | None) -> dict:
    """CFG padrão com os ajustes de um JSON (só as chaves informadas; chaves desconhecidas são erro)."""
    cfg = copy.deepcopy(CFG)
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(user) - set(cfg))
        if unknown:
            sys.exit(f"Chaves desconhecidas em {path}: {', '.join(unknown)}")
        cfg.update(user)
    return cfg


def is_current(out: Path, path: str, rec: dict) -> bool:
    """Resultado existente com status ok e a mesma receita (código, configuração e LAZ)."""
    r = out_dir(out, path) / "resumo.json"
    if not r.exists():
        return False
    try:
        d = json.loads(r.read_text(encoding="utf-8"))
    except Exception:                                                # noqa: BLE001
        return False
    return d.get("status") == "ok" and d.get("receita_hash") == rec["hash"]


def main():
    ap = argparse.ArgumentParser(description="Descritores de textura (ISO 13473-1/-4, 25178, 13565-2) em lote.")
    ap.add_argument("--input", help="pasta com arquivos .laz (busca recursiva)")
    ap.add_argument("--output", required=True, help="pasta do projeto (resultados)")
    ap.add_argument("--workers", type=int, default=0, help="processos paralelos (0 = automático pela RAM livre)")
    ap.add_argument("--mem-gb", type=float, default=0.0,
                    help="RAM máxima a usar no total, em GB (0 = 50 %% da RAM disponível agora)")
    ap.add_argument("--pattern", default="*.laz", help="filtro de nomes (ex.: '*B6*MP1*.laz')")
    ap.add_argument("--config", help="JSON com ajustes da configuração (ex.: {\"A_lp_design_mm\": 2.4})")
    ap.add_argument("--skip-done", action="store_true",
                    help="pula arquivos já calculados com a mesma receita (versão do núcleo + config + LAZ)")
    ap.add_argument("--index-only", action="store_true", help="só regrava projeto.json e resumo_geral.csv")
    ap.add_argument("--zip", action="store_true", help="empacota o projeto em <saída>.tlproj ao final")
    args = ap.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if args.index_only:
        proj = write_project_index(out)
        print(f"projeto.json: {proj['n_arquivos']} arquivo(s), {proj['n_ok']} ok")
        if args.zip:
            print(f"Pacote: {pack_project(out)}")
        return
    if not args.input:
        ap.error("--input é obrigatório (exceto com --index-only)")
    cfg = load_config(args.config)
    files = sorted(str(p) for p in Path(args.input).rglob(args.pattern))
    if not files:
        sys.exit(f"Nenhum arquivo '{args.pattern}' em {args.input}")
    recs = {f: recipe(f, cfg) for f in files}
    if args.skip_done:
        n0 = len(files)
        files = [f for f in files if not is_current(out, f, recs[f])]
        print(f"--skip-done: {n0 - len(files)} arquivo(s) já calculados com a mesma receita", flush=True)
        if not files:
            write_project_index(out)
            if args.zip:
                print(f"Pacote: {pack_project(out)}")
            print("Nada a recalcular: todos os arquivos estão atualizados.")
            return
    ncpu = os.cpu_count() or 1
    avail_gb = mem_available_gb()
    budget_gb = args.mem_gb or 0.5 * avail_gb
    per_file_gb = max(estimate_file_gb(f) for f in files)
    workers = args.workers or max(1, min(len(files), ncpu // 4, int(budget_gb // per_file_gb)))
    print(f"{len(files)} arquivo(s) | {workers} processo(s) | CPU {ncpu} | RAM livre {avail_gb:.0f} GB | "
          f"limite {budget_gb:.0f} GB | estimativa {per_file_gb:.1f} GB/arquivo", flush=True)

    run = {"versao_script": VERSION, "inicio": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg,
           "python": sys.version, "numpy": np.__version__, "plataforma": platform.platform(),
           "entrada": str(Path(args.input).resolve()), "n_arquivos": len(files), "workers": workers,
           "ram_livre_gb": avail_gb, "limite_ram_gb": budget_gb, "estimativa_gb_por_arquivo": per_file_gb}
    results = []
    t0 = time.time()
    # max_tasks_per_child=1: cada processo trata 1 arquivo e é encerrado, devolvendo toda a memória
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        futs = {ex.submit(process_file, f, str(out), cfg, recs[f]): f for f in files}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                r = fu.result()
            except Exception as err:                                 # noqa: BLE001  (ex.: processo morto por falta de RAM)
                r = {"arquivo": Path(futs[fu]).stem, "status": "erro", "erro": f"{type(err).__name__}: {err}"}
            results.append(r)
            msg = (f"MPD={r.get('MPD', float('nan')):.3f} mm" if r.get("status") == "ok" else r.get("erro"))
            pico = r.get("pico_memoria_processo_gb")
            pico = f", pico {pico:.1f} GB" if pico else ""
            print(f"[{k}/{len(files)}] {r['arquivo']}: {r['status']} {msg} ({r.get('total_s', 0):.0f} s{pico})",
                  flush=True)
    run["fim"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run["duracao_s"] = time.time() - t0
    with open(out / "execucao.json", "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False, default=str)
    write_project_index(out)
    n_err = sum(r.get("status") != "ok" for r in results)
    print(f"Concluído em {run['duracao_s'] / 60:.1f} min. Erros: {n_err}. Projeto: {out / 'projeto.json'}")
    if args.zip:
        print(f"Pacote: {pack_project(out)}")


if __name__ == "__main__":
    main()

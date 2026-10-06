"""
TextureLab – Visualizador de projetos calculados no servidor.

O app não calcula descritores: abre um projeto gravado por pipeline/texturelab_batch.py
(pasta de resultados ou .tlproj) e só renderiza tabelas, gráficos, 3D, comparações e PCA.
O cálculo antigo (v2.4.0, ver docs/DIAGNOSTICO.md) ficou em app_legacy.py.
Interface em inglês, alemão e português (components/i18n.py); opções internas usam códigos
fixos, e só os rótulos mudam com o idioma.

Copyright (c) 2025 Sergio Callai
Licensed under CC BY-NC 4.0
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from components.i18n import LANGUAGES, DEFAULT_LANGUAGE, set_language, t  # noqa: E402
from components.project_reader import Project, ProjectError  # noqa: E402

APP_VERSION = "3.2.0"
APP_AUTHOR = "Sergio Callai"
APP_YEAR = "2026"
SHOW_PCA = False        # PCA desativada por enquanto na vista Estatística; True para reativar

st.set_page_config(page_title="TextureLab", page_icon="🔬", layout="wide")

# Idioma: escolhido na barra lateral, guardado na sessão e na URL (?lang=de) para sobreviver a recarregar.
if "lang" not in st.session_state:
    st.session_state["lang"] = st.query_params.get("lang", DEFAULT_LANGUAGE)
    if st.session_state["lang"] not in LANGUAGES:
        st.session_state["lang"] = DEFAULT_LANGUAGE
set_language(st.session_state["lang"])

st.markdown("""
<style>
    .hero-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; line-height: 1.2;
                  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .version-badge { display: inline-block; background: rgba(102, 126, 234, 0.15); color: #667eea;
                     border-radius: 20px; padding: 2px 10px; font-size: 0.7rem; font-weight: 600; }
    div[data-testid="stMetric"] { border: 1px solid rgba(102, 126, 234, 0.25); border-radius: 8px;
                                  padding: 0.35rem 0.6rem; }
    .license-footer { font-size: 0.7rem; color: #9ca3af; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ===================================================================
# Parâmetros e rótulos
# ===================================================================
UNITS = {
    "W_rms_micro_um": "µm", "W_rms_macro_um": "µm",
    "MPD": "mm", "MPD_desvio": "mm", "g_factor_media": "%", "g_factor_desvio": "%",
    "perfil_Rq_media": "mm", "perfil_Rk_media": "mm", "perfil_Rpk_media": "mm", "perfil_Rvk_media": "mm",
    "perfil_Rmr1_media": "%", "perfil_Rmr2_media": "%",
}
for _c in ("SF", "SL5", "MICRO"):
    for _p in ("Sa", "Sq", "Sp", "Sv", "Sz", "Sk", "Spk", "Svk", "Sal", "Vmp", "Vmc", "Vvc", "Vvv"):
        UNITS[f"{_c}_{_p}"] = "mm"
    UNITS[f"{_c}_Sdr_pct"] = "%"
    UNITS[f"{_c}_Smr1"] = "%"
    UNITS[f"{_c}_Smr2"] = "%"

PARAM_GROUPS = {
    "chainA": ["MPD", "MPD_desvio", "A_n_segmentos_validos"],
    "iso10844": ["g_factor_media", "g_factor_desvio", "perfil_Rsk_media"],
    "profile": ["perfil_Rq_media", "perfil_Rsk_media", "perfil_Rku_media", "perfil_Rk_media",
                            "perfil_Rpk_media", "perfil_Rvk_media", "perfil_Rmr1_media", "perfil_Rmr2_media"],
    "wavelets": ["W_rms_micro_um", "W_rms_macro_um", "W_fracao_micro", "W2_anisotropia_micro",
                 "W2_anisotropia_macro"],
    "hurst": ["H_macro", "D_superficie_macro", "H_macro_R2",
                                     "H_micro", "D_superficie_micro", "H_micro_R2"],
}
_AREAL = ["Sa", "Sq", "Ssk", "Sku", "Sp", "Sv", "Sz", "Sdq", "Sdr_pct", "Sk", "Spk", "Svk", "Smr1", "Smr2",
          "Vmp", "Vmc", "Vvc", "Vvv", "Sal", "Str"]
for _c in ("SF", "SL5", "MICRO"):
    PARAM_GROUPS[_c] = [f"{_c}_{p}" for p in _AREAL]

DEFAULT_PARAMS = ["MPD", "g_factor_media", "SF_Sq", "SL5_Sq", "SL5_Sdr_pct", "SL5_Ssk", "H_macro"]
ID_COLS = ["arquivo", "trecho", "revestimento", "mp", "nr", "data"]
# "section_surface" (trecho + revestimento) identifica cada pavimento: agrupa os 5 MPs da mesma superfície.
# Trecho sozinho mistura revestimentos (A1, B244); revestimento sozinho junta trechos (SMA11, ISO).
# As pistas ISO 10844 (IKA, ISO-A, ISO-C, ISO-P) são 4 superfícies distintas com o mesmo nome de camada
# "ISO" no arquivo: no agrupamento por revestimento cada uma vira "ISO (<trecho>)" (coluna _revest_grupo).
GROUP_OPTIONS = {"file": ["arquivo"], "section_surface": ["trecho", "revestimento"], "section": ["trecho"],
                 "surface": ["_revest_grupo"], "mp": ["mp"], "section_mp": ["trecho", "mp"]}
GROUPINGS = [g for g in GROUP_OPTIONS if g != "file"]   # agrupamentos com mais de um arquivo por grupo
AREAL_CHAINS = ["SF", "SL5", "MICRO"]


def pg_label(code: str) -> str:
    return t(f"pg_{code}")


def grp_label(code: str) -> str:
    return t(f"grp_{code}")


def ch_label(code: str) -> str:
    return t(f"ch_{code}")


def id_col_config() -> dict:
    """Rótulos traduzidos das colunas de identificação (os nomes no arquivo não mudam)."""
    names = {"arquivo": t("grp_file"), "trecho": t("grp_section"), "revestimento": t("grp_surface"),
             "mp": "MP", "nr": t("col_nr"), "data": t("col_date"), "_revest_grupo": t("grp_surface")}
    return {k: st.column_config.Column(v) for k, v in names.items()}


def label(p: str) -> str:
    u = UNITS.get(p)
    return f"{p} [{u}]" if u else p


def numeric_params(df: pd.DataFrame) -> list[str]:
    known = [p for g in PARAM_GROUPS.values() for p in g if p in df and pd.api.types.is_numeric_dtype(df[p])]
    return known


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return f"{v:.{digits}g}" if abs(v) < 1e-3 or abs(v) >= 1e4 else f"{v:.{digits}f}"
    return str(v)


def group_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg(" · ".join, axis=1)


RK_KEYS = ["Sk", "Spk", "Svk", "Smr1", "Smr2"]
COLORS = px.colors.qualitative.Plotly


def rk_table(df: pd.DataFrame, ch: str) -> pd.DataFrame:
    """Família Sk (ISO 13565-2) já calculada no servidor + inclinação da reta equivalente.
    A reta vai de (0 %, topo do núcleo) a (100 %, base do núcleo): inclinação = -Sk / 100 % [mm/%]."""
    cols = {f"{ch}_{k}": k for k in RK_KEYS + ["Vmp", "Vmc", "Vvc", "Vvv"] if f"{ch}_{k}" in df}
    tb = df[list(cols)].rename(columns=cols)
    if "Sk" in tb:
        tb.insert(min(5, tb.shape[1]), t("col_slope"), -tb["Sk"] / 100.0)
    return tb


def rk_overlay(fig: go.Figure, mr: np.ndarray, h: np.ndarray, p, color: str, group: str,
               labels: bool = True, annotate: bool = False, **pos) -> None:
    """Desenha sobre a curva de Abbott a reta equivalente, os pontos Smr1/Smr2 e linhas verticais
    em Smr1 e Smr2 (só desenho: Sk, Smr1 e Smr2 vêm do resumo.json). O topo do núcleo é a altura da
    curva em Smr1. 'annotate' escreve os valores (Smr1, Smr2, Sk e inclinação) no gráfico."""
    sk, mr1, mr2 = (float(p.get(k, np.nan)) for k in ("Sk", "Smr1", "Smr2"))
    if not np.all(np.isfinite([sk, mr1, mr2])):
        return
    z_top = float(np.interp(mr1, mr, h))
    z_bot = z_top - sk
    fig.add_scatter(x=[0, 100], y=[z_top, z_bot], mode="lines", line=dict(color=color, dash="dash", width=1),
                    legendgroup=group, showlegend=False, hoverinfo="skip", **pos)
    for x, name, side in ((mr1, "Smr1", "right"), (mr2, "Smr2", "left")):
        fig.add_vline(x=x, line=dict(color=color, dash="dot", width=1), opacity=0.8,
                      annotation_text=f"{name} = {x:.1f} %" if annotate else None,
                      annotation_position=f"top {side}", annotation_font=dict(size=9), **pos)
    if annotate:
        fig.add_annotation(x=50, y=z_top - sk / 2, text=f"Sk = {sk:.3f} mm<br>{t('slope')} = {-sk / 100:.4f} mm/%",
                           showarrow=False, yshift=18, font=dict(size=9),
                           bgcolor="rgba(128,128,128,0.12)", bordercolor=color, borderwidth=1, **pos)
    fig.add_scatter(x=[mr1, mr2], y=[z_top, z_bot], mode="markers+text" if labels else "markers",
                    legendgroup=group, showlegend=False,
                    marker=dict(color=color, size=9, symbol="diamond", line=dict(width=1, color="white")),
                    text=["Smr1", "Smr2"], textposition=["top right", "bottom left"],
                    textfont=dict(size=10, color=color),
                    hovertemplate=[f"{group}<br>Smr1 = {mr1:.1f} %<br>{t('core_top')} = {z_top:.3f} mm<extra></extra>",
                                   f"{group}<br>Smr2 = {mr2:.1f} %<br>Sk = {sk:.3f} mm<br>{t('slope')} = "
                                   f"{-sk / 100:.4f} mm/%<extra></extra>"], **pos)


def spectrum_axis(fig: go.Figure, lam) -> None:
    """Eixo λ em escala log, crescente (menores λ perto da origem), rótulos nas bandas nominais."""
    lam = sorted(set(float(x) for x in lam))
    lo, hi = min(lam), max(lam)
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    major = [m * 10.0 ** d for d in decades for m in (1, 2, 5) if lo * 0.95 <= m * 10.0 ** d <= hi * 1.05]
    minor = [m * 10.0 ** d for d in decades for m in range(1, 10) if lo * 0.95 <= m * 10.0 ** d <= hi * 1.05]
    fig.update_xaxes(type="log", tickvals=major, ticktext=[f"{x:g}" for x in major], showgrid=True,
                     gridcolor="rgba(128,128,128,0.45)",
                     minor=dict(tickvals=minor, showgrid=True, gridcolor="rgba(128,128,128,0.15)", ticks="outside"),
                     range=[np.log10(lo) - 0.05, np.log10(hi) + 0.05],
                     title=t("ax_lambda_band"))


def abbott_grid(curves: list, ncols: int, yrange=None, height_row: int = 230, share_y: bool = True) -> go.Figure:
    """Um gráfico pequeno por curva (mesmos eixos), com reta equivalente e Smr1/Smr2.
    curves = [(titulo, mr, altura, params, cor)]."""
    n = len(curves)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=False, shared_yaxes=share_y,   # eixo x numerado em todos
                        subplot_titles=[c[0] for c in curves], horizontal_spacing=0.04,
                        vertical_spacing=min(0.12, 0.6 / max(1, nrows)))
    for i, (title, mr, h, p, col) in enumerate(curves):
        r, c = i // ncols + 1, i % ncols + 1
        fig.add_scatter(x=mr, y=h, name=title, line=dict(color=col), showlegend=False, row=r, col=c)
        if p is not None:
            rk_overlay(fig, mr, h, p, col, title, labels=False, annotate=True, row=r, col=c)
    titles = {c[0] for c in curves}
    fig.for_each_annotation(lambda a: a.update(font_size=10) if a.text in titles else None)   # títulos
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    fig.update_xaxes(range=[0, 100], dtick=20)
    fig.update_xaxes(showticklabels=True, ticksuffix=" %")
    fig.update_layout(height=max(300, (height_row + 40) * nrows + 60), margin=dict(t=40, l=40, r=10, b=40))
    return fig


def core_range(heights) -> list:
    """Faixa do eixo de alturas cobrindo o núcleo (material ratio entre 1 % e 99 %)."""
    lo, hi = float(np.min(heights)), float(np.max(heights))
    pad = 0.05 * (hi - lo)
    return [lo - pad, hi + pad]


# ===================================================================
# Estado e carregamento
# ===================================================================
def _proj() -> Project | None:
    return st.session_state.get("projeto")


def _open(source, name=None):
    try:
        st.session_state["projeto"] = Project(source, name=name)
        st.session_state["proj_token"] = st.session_state.get("proj_token", 0) + 1
        st.session_state.pop("open_error", None)
    except (ProjectError, OSError, ValueError) as e:
        st.session_state["open_error"] = (e.key, e.kw) if isinstance(e, ProjectError) else ("err_open", {"msg": str(e)})


@st.cache_data(show_spinner=False, max_entries=4)
def all_spectra(_proj: Project, token: int) -> pd.DataFrame:
    """Espectros de terço de oitava de todos os arquivos (formato longo). Só leitura de CSV."""
    parts = []
    for arq in _proj.index["arquivo"]:
        t = _proj.table(arq, "espectro_terco_oitava.csv")
        if t is not None and len(t):
            parts.append(t.assign(arquivo=arq))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def wavelet_axis(fig: go.Figure, lam) -> None:
    """Eixo λ (centro de cada oitava de ondaleta) em escala log, crescente."""
    lam = sorted(set(float(x) for x in lam))
    fig.update_xaxes(type="log", tickvals=lam, ticktext=[f"{x:.3g}" for x in lam], title=t("ax_lambda_octave"))


@st.cache_data(show_spinner=False, max_entries=8)
def all_tables(_proj: Project, token: int, filename: str) -> pd.DataFrame:
    """Uma tabela CSV (ex.: ondaletas_perfil.csv) de todos os arquivos, em formato longo."""
    parts = []
    for arq in _proj.index["arquivo"]:
        tb = _proj.table(arq, filename)
        if tb is not None and len(tb):
            parts.append(tb.assign(arquivo=arq))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=4)
def all_abbott(_proj: Project, token: int, chain: str) -> pd.DataFrame:
    parts = []
    for arq in _proj.index["arquivo"][_proj.index["visualizacao"].astype(bool)]:
        v = _proj.view(arq)
        if f"{chain}_abbott_mr_pct" in v:
            parts.append(pd.DataFrame({"mr_pct": v[f"{chain}_abbott_mr_pct"],
                                       "altura_mm": v[f"{chain}_abbott_altura_mm"], "arquivo": arq}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ===================================================================
# Barra lateral
# ===================================================================
with st.sidebar:
    st.markdown('<p class="hero-title">🔬 TextureLab</p>', unsafe_allow_html=True)
    st.markdown(f'<span class="version-badge">v{APP_VERSION} · {t("viewer")}</span>', unsafe_allow_html=True)
    lang = st.segmented_control(t("language"), list(LANGUAGES), format_func=str.upper,
                                default=st.session_state["lang"], key="lang_sel",
                                help=" · ".join(f"{k.upper()} = {v}" for k, v in LANGUAGES.items()))
    if lang and lang != st.session_state["lang"]:
        st.session_state["lang"] = lang
        st.query_params["lang"] = lang
        st.rerun()
    st.markdown(f"### {t('open_project')}")
    path = st.text_input(t("path_label"), key="proj_path", placeholder=t("path_placeholder"), help=t("path_help"))
    if st.button(t("open_btn"), type="primary", width="stretch") and path.strip():
        _open(path.strip().strip('"'))
    up = st.file_uploader(t("upload_label"), type=["tlproj", "zip"], key="proj_upload")
    if up is not None and st.button(t("open_upload_btn"), width="stretch"):
        _open(up.getvalue(), name=up.name)
    if st.session_state.get("open_error"):
        key, kw = st.session_state["open_error"]
        st.error(t(key, **kw))

    proj = _proj()
    if proj is not None:
        st.markdown("---")
        st.markdown(f"**{proj.name}**")
        m = proj.meta
        st.caption(t("project_caption", ok=m.get("n_ok", 0), n=m.get("n_arquivos", 0),
                     core=", ".join(m.get("versoes_nos_resultados", [])) or "?", fmt=m.get("versao_formato")))
        for key, kw in proj.warnings:
            st.warning(t(key, **kw), icon="⚠️")
        if st.button(t("close_project"), width="stretch"):
            for k in ("projeto", "open_error"):
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown(f'<div class="license-footer">© {APP_YEAR} {APP_AUTHOR} · '
                '<a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a></div>',
                unsafe_allow_html=True)


# ===================================================================
# Sem projeto: instruções
# ===================================================================
proj = _proj()
if proj is None:
    st.markdown(f"## {t('viewer_title')}")
    st.info(t("viewer_info"))
    st.markdown(t("server_help"))
    st.stop()

# ===================================================================
# Filtros
# ===================================================================
summary = proj.summary()
ok = summary[summary.get("status", "ok") == "ok"].copy() if "status" in summary else summary.copy()
params_all = numeric_params(ok)
if "revestimento" in ok:
    _rv = ok["revestimento"].astype(str)
    ok["_revest_grupo"] = _rv.where(~_rv.str.upper().str.startswith("ISO"), "ISO (" + ok["trecho"].astype(str) + ")")

with st.expander(t("filters"), expanded=False):
    fc = st.columns(3)
    sel = {}
    for (col, code), c in zip((("trecho", "section"), ("revestimento", "surface"), ("mp", "mp")), fc):
        opts = sorted(x for x in ok[col].dropna().astype(str).unique()) if col in ok else []
        sel[col] = c.multiselect(grp_label(code), opts, default=[], key=f"f_{col}", placeholder=t("all"))
    for col, vals in sel.items():
        if vals:
            ok = ok[ok[col].astype(str).isin(vals)]
st.caption(t("n_after_filters", n=len(ok)))
if ok.empty:
    st.warning(t("no_files_filters"))
    st.stop()

VIEWS = ["summary", "file", "compare", "stats", "server"]
view = st.segmented_control(t("view"), VIEWS, format_func=lambda v: t(f"view_{v}"), default=VIEWS[0], key="view",
                            label_visibility="collapsed") or VIEWS[0]


# ===================================================================
# Resumo
# ===================================================================
def view_summary():
    st.markdown(f"### {t('params_by_file')}")
    groups = st.multiselect(t("param_groups"), list(PARAM_GROUPS), format_func=pg_label,
                            default=["chainA", "iso10844", "SL5"], key="sum_groups")
    cols = [p for g in groups for p in PARAM_GROUPS[g] if p in ok]
    tbl = ok[[c for c in ID_COLS if c in ok] + cols]
    st.dataframe(tbl, hide_index=True, column_config={**id_col_config(),
                 **{p: st.column_config.NumberColumn(label(p), format="%.4g") for p in cols}})

    st.markdown(f"### {t('mean_by_group')}")
    gcode = st.selectbox(t("group_by"), GROUPINGS, format_func=grp_label, index=0, key="sum_gby")
    gname = grp_label(gcode)
    gcols = GROUP_OPTIONS[gcode]
    if cols:
        agg = ok.groupby(gcols)[cols].agg(["mean", "std", "count"])
        agg.columns = [f"{p} ({t('agg_' + s)})" for p, s in agg.columns]
        st.dataframe(agg.reset_index().style.format(precision=4, na_rep="–"), hide_index=True,
                     column_config=id_col_config())
        st.caption(t("std_caption"))

    c1, c2 = st.columns(2)
    c1.download_button(t("dl_full_csv"), ok.to_csv(index=False).encode("utf-8"),
                       file_name=f"{proj.name}_resumo.csv", mime="text/csv", width="stretch")
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf) as xw:
            ok.to_excel(xw, sheet_name="resumo", index=False)
            if cols:
                agg.reset_index().to_excel(xw, sheet_name=f"{t('agg_mean')}_{gcode}"[:31], index=False)
        c2.download_button(t("dl_excel"), buf.getvalue(), file_name=f"{proj.name}_resumo.xlsx", width="stretch")
    except ImportError:
        c2.caption(t("need_openpyxl"))


# ===================================================================
# Arquivo
# ===================================================================
def _plane_removed(z: np.ndarray) -> np.ndarray:
    """Só para exibir: remove o plano médio da prévia (o cálculo usa a grade completa no servidor)."""
    ny, nx = z.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    m = np.isfinite(z)
    A = np.c_[xx[m], yy[m], np.ones(m.sum())]
    c, *_ = np.linalg.lstsq(A, z[m].astype(np.float64), rcond=None)
    return z - (c[0] * xx + c[1] * yy + c[2])


def _surface_fig(z: np.ndarray, step: float, max_pts: int, kind: str, title: str, exag: float = 1.0) -> go.Figure:
    """Mapa ou 3D da prévia. No 3D os eixos x, y e z ficam na mesma escala física, multiplicada
    por 'exag' só no z (1 = escala real). Cores cortadas em P1–P99 para poucos vales profundos
    não dominarem a escala (afeta só a cor, não a geometria)."""
    k = max(1, int(np.ceil(np.sqrt(z.size / max_pts))))
    zz = z[::k, ::k]
    x = np.arange(zz.shape[1]) * step * k
    y = np.arange(zz.shape[0]) * step * k
    cmin, cmax = (float(v) for v in np.nanpercentile(zz, [1, 99]))
    if kind == "3D":
        fig = go.Figure(go.Surface(z=zz, x=x, y=y, colorscale="Viridis", cmin=cmin, cmax=cmax,
                                   colorbar=dict(title="z [mm]")))
        span = [x[-1], y[-1], max(1e-9, float(np.nanmax(zz) - np.nanmin(zz)))]
        r = max(span[:2])
        fig.update_layout(scene=dict(xaxis_title=t("ax_road"), yaxis_title=t("ax_width"), zaxis_title="z [mm]",
                                     aspectmode="manual",
                                     aspectratio=dict(x=span[0] / r * 2, y=span[1] / r * 2, z=span[2] / r * 2 * exag),
                                     camera=dict(eye=dict(x=0.45, y=-1.45, z=0.75))),
                          height=600, margin=dict(l=0, r=0, t=30, b=0), title=title)
    else:
        fig = go.Figure(go.Heatmap(z=zz, x=x, y=y, colorscale="Viridis", zmin=cmin, zmax=cmax,
                                   colorbar=dict(title="z [mm]")))
        fig.update_layout(xaxis_title=t("ax_road"), yaxis_title=t("ax_width"), height=380,
                          margin=dict(l=0, r=0, t=30, b=0), title=title)
        fig.update_yaxes(scaleanchor="x")
    return fig


def view_file():
    files = ok["arquivo"].tolist()
    arq = st.selectbox(t("file"), files, key="file_sel")
    r = proj.resumo(arq)
    v = proj.view(arq)
    st.markdown(f"#### {arq}")
    if r.get("status") != "ok":
        st.error(r.get("erro", t("calc_error")))
        return
    m = st.columns(6)
    for c, k in zip(m, ["MPD", "g_factor_media", "SF_Sq", "SL5_Sq", "SL5_Sdr_pct", "H_macro"]):
        c.metric(label(k), fmt(r.get(k)))
    st.caption(t("file_caption", L=r.get("comprimento_mm", 0), W=r.get("largura_mm", 0), dx=r.get("dx_mm"),
                 n=r.get("n_pontos", 0) / 1e6, core=r.get("versao_script"), rec=r.get("receita_hash", "–")))
    if not v:
        st.info(t("no_view_data"))

    tabs = st.tabs([t(k) for k in ("tab_surface", "tab_profiles", "tab_spectrum", "tab_abbott", "tab_psd",
                                   "tab_segments", "tab_wavelets", "tab_all_params")])
    with tabs[0]:
        c = st.columns(5)
        src = c[0].radio(t("surface"), ["raw", "sl5"] if "SL5_previa" in v else ["raw"],
                         format_func=lambda x: t(f"surf_{x}"), key="surf_src")
        kind = c[1].radio(t("type"), ["map", "3d"], format_func=lambda x: t(f"kind_{x}"), key="surf_kind")
        max_pts = c[2].select_slider(t("points_plot"), [50_000, 150_000, 300_000, 600_000, 1_200_000],
                                     value=150_000, key="surf_pts")
        exag = c[3].select_slider(t("vert_exag"), [1, 2, 3, 5, 10], value=2, format_func=lambda e: f"{e}×",
                                  key="surf_exag", disabled=kind != "3d", help=t("vert_exag_help"))
        detr = c[4].checkbox(t("remove_plane"), value=True, key="surf_detr", disabled=src == "sl5",
                             help=t("remove_plane_help"))
        if src == "sl5":
            pstep = r.get("receita", {}).get("config", {}).get("preview_step", 8)
            z, step = v["SL5_previa"].astype(np.float32), float(r["dx_mm"]) * pstep
        else:
            pv = proj.preview(arq)
            if pv is None:
                st.info(t("no_preview"))
                z = None
            else:
                z, step = pv
                if detr:
                    z = _plane_removed(z)
        if z is not None:
            title = t(f"surf_{src}") + (f" · {t('vert_exag')} {exag}×" if kind == "3d" else "")
            st.plotly_chart(_surface_fig(z, step, max_pts, "3D" if kind == "3d" else "map", title, exag),
                            key="surf_fig")
            st.caption(t("preview_caption", step=step, k=int(round(step / r["dx_mm"]))))
    with tabs[1]:
        if "A_perfil_limpo" in v:
            step = float(v["A_perfil_passo_mm"])
            fig = go.Figure()
            for i, y in enumerate(v["A_perfil_y_mm"]):
                xs = np.arange(v["A_perfil_limpo"].shape[1]) * step
                vis = True if i == len(v["A_perfil_y_mm"]) // 2 else "legendonly"
                fig.add_scatter(x=xs, y=v["A_perfil_limpo"][i], name=t("prof_clean", y=y), visible=vis,
                                line=dict(width=1))
                fig.add_scatter(x=xs, y=v["A_perfil_passa_baixa"][i], name=t("prof_lp", y=y), visible=vis)
            a0, L = float(v["A_segmento_inicio_mm"]), float(v["A_segmento_mm"])
            for s in range(int(v["A_n_segmentos"]) + 1):
                fig.add_vline(x=a0 + s * L, line_dash="dot", line_color="gray")
            fig.update_layout(title=t("prof_title"), xaxis_title=t("ax_road"), yaxis_title="z [mm]", height=420)
            st.plotly_chart(fig, key="prof_a")
        if "nativo_perfil" in v:
            xs = float(v["nativo_inicio_mm"]) + np.arange(len(v["nativo_perfil"])) * float(v["nativo_passo_mm"])
            fig = px.line(x=xs, y=v["nativo_perfil"], labels={"x": t("ax_road"), "y": "z [mm]"},
                          title=t("native_title", dx=float(v["nativo_passo_mm"]), y=float(v["nativo_y_mm"])))
            fig.update_traces(line=dict(width=1))
            st.plotly_chart(fig, key="prof_nat")
        if not v:
            st.info(t("no_profiles"))
    with tabs[2]:
        sp = proj.table(arq, "espectro_terco_oitava.csv")
        if sp is not None and len(sp):
            fig = go.Figure(go.Scatter(x=sp["lambda_centro_mm"], y=sp["L_tx_dB_media"], mode="lines+markers",
                                       error_y=dict(array=sp["L_tx_dB_desvio"], visible=True), name=t("mean_sd")))
            spectrum_axis(fig, sp["lambda_centro_mm"])
            fig.update_layout(yaxis_title="L_tx [dB re 1 µm]", height=420, title=t("spec_title"))
            st.plotly_chart(fig, key="spec")
            st.dataframe(sp, hide_index=True)
    with tabs[3]:
        chains = [c for c in AREAL_CHAINS if f"{c}_abbott_mr_pct" in v]
        if chains:
            modo = st.radio(t("display"), ["overlay", "side"], format_func=lambda x: t(f"mode_{x}"),
                            horizontal=True, key="arq_abbott_mode")
            if modo == "side":
                curves = [(ch_label(ch), v[f"{ch}_abbott_mr_pct"], v[f"{ch}_abbott_altura_mm"],
                           {k: r.get(f"{ch}_{k}", np.nan) for k in RK_KEYS}, COLORS[i % len(COLORS)])
                          for i, ch in enumerate(chains)]
                fga = abbott_grid(curves, ncols=len(curves), height_row=330, share_y=False)
                fga.update_yaxes(title_text=t("ax_height"), col=1)
                fga.update_xaxes(title_text=t("ax_mr"))
                st.plotly_chart(fga, key="abbott_grid")
            c1, c2 = st.columns(2)
            fa, fh = go.Figure(), go.Figure()
            for i, ch in enumerate(chains):
                col = COLORS[i % len(COLORS)]
                mr, h = v[f"{ch}_abbott_mr_pct"], v[f"{ch}_abbott_altura_mm"]
                fa.add_scatter(x=mr, y=h, name=ch_label(ch), line=dict(color=col), legendgroup=ch)
                rk_overlay(fa, mr, h, {k: r.get(f"{ch}_{k}", np.nan) for k in RK_KEYS}, col, ch)
                e = v[f"{ch}_hist_bordas_mm"]
                cnt = v[f"{ch}_hist_contagem"].astype(float)
                dens = cnt / (cnt.sum() * np.diff(e))
                fh.add_scatter(x=(e[:-1] + e[1:]) / 2, y=dens, name=ch_label(ch), mode="lines")
            fa.update_layout(title=t("abbott_title"), xaxis_title=t("ax_mr"), yaxis_title=t("ax_height"), height=420)
            fh.update_layout(title=t("pdf_title"), xaxis_title=t("ax_height"), yaxis_title=t("ax_density"), height=420)
            if modo != "side":
                c1.plotly_chart(fa, key="abbott")
                c2.plotly_chart(fh, key="hist")
            else:
                c1.plotly_chart(fh, key="hist")
            one = pd.DataFrame([r])
            rk = pd.concat({ch_label(ch): rk_table(one, ch).iloc[0] for ch in chains}, axis=1)
            st.caption(t("abbott_caption_file"))
            st.dataframe(rk)
        else:
            st.info(t("no_abbott"))
    with tabs[4]:
        psd = proj.table(arq, "psd_media.csv")
        if psd is not None and len(psd):
            fig = go.Figure(go.Scatter(x=psd["lambda_mm"], y=psd["PSD_mm3"], mode="lines", name=t("psd_mean"),
                                       line=dict(width=1)))
            for band, (lo, hi) in {"micro": (0.05, 0.5), "macro": (0.5, 20.0)}.items():
                beta = r.get(f"H_{band}_beta")
                fig.add_shape(type="rect", xref="x", yref="paper", x0=lo, x1=hi, y0=0, y1=1,
                              fillcolor="gray", opacity=0.08, line_width=0)
                fig.add_annotation(x=np.log10(np.sqrt(lo * hi)), y=1, xref="x", yref="paper",   # eixo log: log10
                                   text=band, showarrow=False, yanchor="bottom")
                if beta is not None:
                    m = (psd["lambda_mm"] >= lo) & (psd["lambda_mm"] <= hi) & (psd["PSD_mm3"] > 0)
                    if m.any():
                        lam = psd.loc[m, "lambda_mm"].to_numpy()
                        y0 = 10 ** np.mean(np.log10(psd.loc[m, "PSD_mm3"]))
                        l0 = 10 ** np.mean(np.log10(lam))
                        xx = np.array([lam.min(), lam.max()])
                        fig.add_scatter(x=xx, y=y0 * (xx / l0) ** beta, mode="lines", line=dict(dash="dash"),
                                        name=f"{band}: H={r.get(f'H_{band}'):.3f} (R²={r.get(f'H_{band}_R2'):.3f})")
            fig.update_xaxes(type="log", title="λ [mm]")
            fig.update_yaxes(type="log", title="PSD [mm³]")
            fig.update_layout(height=450, title=t("psd_title"))
            st.plotly_chart(fig, key="psd")
            st.caption(t("psd_caption"))
        else:
            st.info(t("no_psd"))
    with tabs[5]:
        seg = proj.table(arq, "cadeiaA_segmentos.csv")
        if seg is not None and len(seg):
            val = seg[seg["valido"]]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(val, x="MSD", nbins=30, title=t("msd_hist_title"),
                                         labels={"MSD": "MSD [mm]"}), key="msd_hist")
            c2.plotly_chart(px.scatter(val, x="x_centro_mm", y="MSD", color=val["segmento"].astype(str),
                                       title=t("msd_pos_title"),
                                       labels={"x_centro_mm": t("ax_pos_width"), "color": t("segment")}), key="msd_pos")
            if "g_pct" in val:
                st.markdown(f"#### {t('g_section')}")
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.histogram(val, x="g_pct", nbins=30, title=t("g_hist_title"),
                                             labels={"g_pct": "g [%]"}), key="g_hist")
                if "g_seg_z" in v:
                    gz = v["g_seg_z"].astype(float)
                    d = np.arange(gz.size) * 100.0 / (gz.size - 1)
                    gval, zmid = float(v["g_seg_g"]), float(v["g_seg_zmid"])
                    fg = go.Figure(go.Scatter(x=d, y=gz, mode="lines", name=t("abbott_curves")))
                    fg.add_hline(y=zmid, line=dict(dash="dashdot", color="gray"),
                                 annotation_text=f"z_mid = {zmid:.3f} mm", annotation_position="top left")
                    fg.add_vline(x=gval, line=dict(dash="dot", color="red"),
                                 annotation_text=f"g = {gval:.1f} %", annotation_position="bottom left")
                    fg.add_scatter(x=[gval], y=[zmid], mode="markers", marker=dict(size=12, symbol="circle-open",
                                   color="red", line=dict(width=2)), showlegend=False)
                    fg.update_layout(title=t("g_seg_title", y=float(v["g_seg_y_mm"]), x0=float(v["g_seg_inicio_mm"])),
                                     xaxis_title=t("g_ax_dcum"), yaxis_title=t("ax_height"), height=420,
                                     showlegend=False)
                    fg.update_xaxes(range=[0, 100], dtick=10)
                    c2.plotly_chart(fg, key="g_seg")
                st.caption(t("g_caption"))
            st.dataframe(seg, hide_index=True)
    with tabs[6]:
        wp, w2 = proj.table(arq, "ondaletas_perfil.csv"), proj.table(arq, "ondaletas_2d.csv")
        if wp is None or not len(wp):
            st.info(t("no_wavelets"))
        else:
            c1, c2 = st.columns(2)
            fw = go.Figure(go.Scatter(x=wp["lambda_centro_mm"], y=wp["L_w_dB_media"], mode="lines+markers",
                                      error_y=dict(array=wp["L_w_dB_desvio"], visible=True), name=t("mean_sd")))
            wavelet_axis(fw, wp["lambda_centro_mm"])
            fw.add_vline(x=0.5, line=dict(dash="dot", color="gray"), annotation_text="micro | macro")
            fw.update_layout(title=t("w_profile_title"), yaxis_title=t("ax_w_level"), height=420)
            c1.plotly_chart(fw, key="w_prof")
            if w2 is not None and len(w2):
                f2 = make_subplots(specs=[[{"secondary_y": True}]])
                for col, name in (("rms_via_um", t("w_dir_road")), ("rms_transversal_um", t("w_dir_cross")),
                                  ("rms_diagonal_um", t("w_dir_diag"))):
                    f2.add_scatter(x=w2["lambda_centro_mm"], y=w2[col], mode="lines+markers", name=name)
                f2.add_bar(x=w2["lambda_centro_mm"], y=w2["anisotropia"], name=t("w_anis"), opacity=0.3,
                           secondary_y=True)
                wavelet_axis(f2, w2["lambda_centro_mm"])
                f2.update_yaxes(title_text=t("ax_rms_um"), type="log", secondary_y=False)
                f2.update_yaxes(title_text=t("w_anis"), range=[-1, 1], secondary_y=True, showgrid=False)
                f2.update_layout(title=t("w_2d_title"), height=420, legend=dict(orientation="h", y=-0.25))
                c2.plotly_chart(f2, key="w_2d")
            st.caption(t("w_caption"))
            st.dataframe(wp, hide_index=True)
            if w2 is not None and len(w2):
                st.dataframe(w2, hide_index=True)
    with tabs[7]:
        flat = {k: v_ for k, v_ in r.items() if not isinstance(v_, (dict, list))}
        st.dataframe(pd.DataFrame({t("param"): list(flat), t("value"): [str(x) for x in flat.values()]}),
                     hide_index=True, height=600)
        if "receita" in r:
            with st.expander(t("recipe_expander")):
                st.json(r["receita"])


# ===================================================================
# Comparar
# ===================================================================
def view_compare():
    c = st.columns([2, 3])
    gcode = c[0].selectbox(t("compare_by"), list(GROUP_OPTIONS), format_func=grp_label, index=1, key="cmp_gby")
    gname = grp_label(gcode)
    gcols = GROUP_OPTIONS[gcode]
    keys = group_key(ok, gcols)
    groups = sorted(keys.unique())
    chosen = c[1].multiselect(t("groups"), groups, default=groups[:20], key=f"cmp_groups_{gcode}")
    if not chosen:
        st.info(t("pick_group"))
        return
    sub = ok[keys.isin(chosen)].assign(_grupo=keys[keys.isin(chosen)])

    params = st.multiselect(t("parameters"), params_all, default=[p for p in DEFAULT_PARAMS if p in params_all],
                            format_func=label, key="cmp_params")
    kind = st.radio(t("chart"), ["bar", "box"], format_func=lambda x: t(f"chart_{x}"), horizontal=True,
                    key="cmp_kind")
    for p in params:
        if kind == "bar":
            g = sub.groupby("_grupo")[p].agg(["mean", "std", "count"]).reindex(chosen).reset_index()
            fig = px.bar(g, x="_grupo", y="mean", error_y="std", hover_data=["count"],
                         labels={"_grupo": gname, "mean": label(p)}, title=label(p))
        else:
            fig = px.box(sub, x="_grupo", y=p, points="all", hover_data=["arquivo"],
                         labels={"_grupo": gname, p: label(p)}, title=label(p),
                         category_orders={"_grupo": chosen})
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, key=f"cmp_{p}")

    st.markdown(f"### {t('spectra_title')}")
    spec = all_spectra(proj, st.session_state["proj_token"])
    if len(spec):
        s = spec.merge(sub[["arquivo", "_grupo"]], on="arquivo")
        g = s.groupby(["_grupo", "lambda_centro_mm"])["L_tx_dB_media"].mean().reset_index()
        fig = px.line(g, x="lambda_centro_mm", y="L_tx_dB_media", color="_grupo", markers=True,
                      category_orders={"_grupo": chosen}, color_discrete_sequence=COLORS,
                      labels={"lambda_centro_mm": "λ [mm]", "L_tx_dB_media": "L_tx [dB re 1 µm]", "_grupo": gname})
        spectrum_axis(fig, g["lambda_centro_mm"])
        fig.update_layout(height=450)
        st.plotly_chart(fig, key="cmp_spec")
        wide = s.pivot_table(index=["_grupo", "arquivo"], columns="lambda_centro_mm", values="L_tx_dB_media")
        wide = wide[sorted(wide.columns)]
        wide.columns = [f"{c:g} mm" for c in wide.columns]
        means = wide.groupby(level=0).mean().reindex([c for c in chosen if c in wide.index.get_level_values(0)])
        st.markdown(t("ltx_by_band"))
        st.dataframe(means.round(2).rename_axis(gname))
        with st.expander(t("all_samples", n=len(wide)), expanded=True):
            st.dataframe(wide.round(2).reset_index().rename(columns={"_grupo": gname}), hide_index=True,
                         column_config=id_col_config())
        st.download_button(t("dl_spectra"), wide.reset_index().rename(columns={"_grupo": gname})
                           .to_csv(index=False).encode("utf-8"), file_name=f"{proj.name}_espectros.csv",
                           mime="text/csv", key="dl_spec")

    st.markdown(f"### {t('abbott_curves')}")
    ch = st.radio(t("areal_chain"), AREAL_CHAINS, format_func=ch_label, horizontal=True, index=1,
                  key="cmp_abbott_chain")
    ab = all_abbott(proj, st.session_state["proj_token"], ch)
    if len(ab):
        c0, c1, c2, c3 = st.columns([2, 3, 3, 1])
        modo = c0.radio(t("display"), ["overlay", "grid"], format_func=lambda x: t(f"mode_{x}"), horizontal=True,
                        key="cmp_abbott_mode")
        show_rk = c1.checkbox(t("show_rk"), value=True, key="cmp_rk")
        zoom = c2.checkbox(t("zoom_core"), value=True, key="cmp_zoom")
        ncols = c3.number_input(t("columns"), 1, 8, 4, key="cmp_cols", disabled=modo == "overlay")
        a = ab.merge(sub[["arquivo", "_grupo"]], on="arquivo")
        g = a.groupby(["_grupo", "mr_pct"])["altura_mm"].mean().reset_index()
        tbl = rk_table(sub, ch)
        tbl.insert(0, "arquivo", sub["arquivo"])
        if gcode != "file":
            tbl.insert(0, gname, sub["_grupo"])
            gmean = tbl.drop(columns="arquivo").groupby(gname).mean()
        else:
            gmean = tbl.set_index("arquivo").rename_axis(gname)
        yr = core_range(g[(g["mr_pct"] >= 1) & (g["mr_pct"] <= 99)]["altura_mm"]) if zoom else None
        if modo == "overlay":
            fig = go.Figure()
            for i, grp in enumerate(chosen):
                d = g[g["_grupo"] == grp]
                if d.empty:
                    continue
                col = COLORS[i % len(COLORS)]
                fig.add_scatter(x=d["mr_pct"], y=d["altura_mm"], name=grp, line=dict(color=col), legendgroup=grp)
                if show_rk and grp in gmean.index:
                    rk_overlay(fig, d["mr_pct"].to_numpy(), d["altura_mm"].to_numpy(), gmean.loc[grp], col, grp,
                               labels=False)
            if yr:
                fig.update_yaxes(range=yr)
            fig.update_layout(height=500, xaxis_title=t("ax_mr"), yaxis_title=t("ax_height"),
                              legend_title_text=gname)
            st.plotly_chart(fig, key="cmp_abbott")
            st.caption(t("legend_hint"))
        else:
            curves = []
            for i, grp in enumerate(chosen):
                d = g[g["_grupo"] == grp]
                if not d.empty:
                    curves.append((grp, d["mr_pct"].to_numpy(), d["altura_mm"].to_numpy(),
                                   gmean.loc[grp] if show_rk and grp in gmean.index else None,
                                   COLORS[i % len(COLORS)]))
            st.plotly_chart(abbott_grid(curves, int(ncols), yr), key="cmp_abbott_grid")
        st.caption(t("abbott_caption_cmp"))
        fmt_cols = {c: st.column_config.NumberColumn(c, format="%.4g") for c in tbl.columns
                    if pd.api.types.is_numeric_dtype(tbl[c])}
        fmt_cols.update(id_col_config())
        st.markdown(t("sk_family_mean"))
        st.dataframe(gmean.reindex([c for c in chosen if c in gmean.index]), column_config=fmt_cols)
        with st.expander(t("all_samples", n=len(tbl)), expanded=True):
            st.dataframe(tbl, hide_index=True, column_config=fmt_cols)
        st.download_button(t("dl_sk"), tbl.to_csv(index=False).encode("utf-8"),
                           file_name=f"{proj.name}_abbott_{ch}.csv", mime="text/csv", key="dl_abbott")
    else:
        st.info(t("no_abbott_project"))

    st.markdown(f"### {t('w_section')}")
    compare_wavelets(sub, chosen, gname)


def compare_wavelets(sub: pd.DataFrame, chosen: list, gname: str) -> None:
    """Espectro de ondaletas do perfil e anisotropia 2D por grupo (média das energias dos arquivos)."""
    wp = all_tables(proj, st.session_state["proj_token"], "ondaletas_perfil.csv")
    if not len(wp):
        st.info(t("no_wavelets"))
        return
    s = wp.merge(sub[["arquivo", "_grupo"]], on="arquivo")
    s["var"] = (s["rms_um"] * 1e-3) ** 2
    g = s.groupby(["_grupo", "lambda_centro_mm"])["var"].mean().reset_index()
    g["L_w_dB"] = 10 * np.log10(g["var"] / 1e-6)
    c1, c2 = st.columns(2)
    fig = px.line(g, x="lambda_centro_mm", y="L_w_dB", color="_grupo", markers=True,
                  category_orders={"_grupo": chosen}, color_discrete_sequence=COLORS,
                  labels={"lambda_centro_mm": "λ [mm]", "L_w_dB": t("ax_w_level"), "_grupo": gname})
    wavelet_axis(fig, g["lambda_centro_mm"])
    fig.update_layout(height=450, title=t("w_profile_title"))
    c1.plotly_chart(fig, key="cmp_w_prof")
    w2 = all_tables(proj, st.session_state["proj_token"], "ondaletas_2d.csv")
    if len(w2):
        s2 = w2.merge(sub[["arquivo", "_grupo"]], on="arquivo")
        s2["ev"], s2["et"] = s2["rms_via_um"] ** 2, s2["rms_transversal_um"] ** 2
        g2 = s2.groupby(["_grupo", "lambda_centro_mm"])[["ev", "et"]].mean().reset_index()
        g2["anis"] = (g2["ev"] - g2["et"]) / (g2["ev"] + g2["et"])
        fig2 = px.line(g2, x="lambda_centro_mm", y="anis", color="_grupo", markers=True,
                       category_orders={"_grupo": chosen}, color_discrete_sequence=COLORS,
                       labels={"lambda_centro_mm": "λ [mm]", "anis": t("w_anis"), "_grupo": gname})
        wavelet_axis(fig2, g2["lambda_centro_mm"])
        fig2.add_hline(y=0, line=dict(color="gray", dash="dot"))
        fig2.update_yaxes(range=[-1, 1])
        fig2.update_layout(height=450, title=t("w_2d_anis_title"))
        c2.plotly_chart(fig2, key="cmp_w_2d")
    wide = s.pivot_table(index=["_grupo", "arquivo"], columns="lambda_centro_mm", values="L_w_dB_media")
    wide.columns = [f"{c:.3g} mm" for c in sorted(wide.columns)]
    st.caption(t("w_caption"))
    with st.expander(t("all_samples", n=len(wide)), expanded=False):
        st.dataframe(wide.round(2).reset_index().rename(columns={"_grupo": gname}), hide_index=True,
                     column_config=id_col_config())


# ===================================================================
# Estatística
# ===================================================================
def view_stats():
    params = st.multiselect(t("parameters"), params_all,
                            default=[p for p in ["MPD", "SF_Sq", "SL5_Sq", "SL5_Ssk", "SL5_Sku", "SL5_Sdr_pct",
                                                 "SL5_Sal", "MICRO_Sq", "H_macro"] if p in params_all],
                            format_func=label, key="st_params")
    gcode = st.selectbox(t("color_by"), GROUPINGS, format_func=grp_label, index=0, key="st_color")
    gname = grp_label(gcode)
    color = group_key(ok, GROUP_OPTIONS[gcode])
    if len(params) < 2:
        st.info(t("pick_two"))
        return
    X = ok[params].apply(pd.to_numeric, errors="coerce")

    st.markdown(f"### {t('corr_title')}")
    corr = X.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=120 + 40 * len(params))
    st.plotly_chart(fig, key="st_corr")

    st.markdown(f"### {t('scatter_title')}")
    c = st.columns(2)
    xp = c[0].selectbox("X", params, index=0, format_func=label, key="st_x")
    yp = c[1].selectbox("Y", params, index=1, format_func=label, key="st_y")
    fig = px.scatter(ok.assign(_g=color), x=xp, y=yp, color="_g", hover_data=["arquivo"], trendline=None,
                     labels={xp: label(xp), yp: label(yp), "_g": gname})
    fig.update_layout(height=450)
    st.plotly_chart(fig, key="st_scatter")

    if not SHOW_PCA:
        return
    st.markdown(f"### {t('pca_title')}")
    Xc = X.dropna()
    if len(Xc) < 3:
        st.info(t("pca_few"))
        return
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(Xc)
    pca = PCA(n_components=min(len(params), len(Xc), 5)).fit(Z)
    sc = pca.transform(Z)
    ev = pca.explained_variance_ratio_ * 100
    c1, c2 = st.columns([3, 2])
    df = pd.DataFrame({"PC1": sc[:, 0], "PC2": sc[:, 1], "arquivo": ok.loc[Xc.index, "arquivo"],
                       "_g": color.loc[Xc.index]})
    fig = px.scatter(df, x="PC1", y="PC2", color="_g", hover_data=["arquivo"],
                     labels={"PC1": f"PC1 ({ev[0]:.0f} %)", "PC2": f"PC2 ({ev[1]:.0f} %)", "_g": gname})
    scale = np.abs(sc[:, :2]).max() / max(1e-12, np.abs(pca.components_[:2]).max())
    for j, p in enumerate(params):
        fig.add_annotation(x=pca.components_[0, j] * scale, y=pca.components_[1, j] * scale, ax=0, ay=0,
                           axref="x", ayref="y", text=p, showarrow=True, arrowhead=2, opacity=0.6)
    fig.update_layout(height=500)
    c1.plotly_chart(fig, key="st_pca")
    c2.plotly_chart(px.bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, labels={"x": "", "y": t("variance")},
                           title=t("explained_var")), key="st_pca_var")
    load = pd.DataFrame(pca.components_.T, index=params, columns=[f"PC{i + 1}" for i in range(len(ev))])
    c2.dataframe(load.style.format("{:.2f}"))
    if len(Xc) < len(ok):
        st.caption(t("pca_dropped", n=len(ok) - len(Xc)))


# ===================================================================
# Calcular no servidor
# ===================================================================
def view_server():
    st.markdown(f"### {t('server_title')}")
    st.markdown(t("server_intro"))
    st.markdown(t("server_help"))
    cfg = proj.meta.get("config") or {}
    if cfg:
        with st.expander(t("config_used")):
            st.json(cfg)
    idx = proj.index.copy()
    st.markdown(f"#### {t('file_status')}")
    st.dataframe(idx[["arquivo", "status", "versao_nucleo", "receita_hash", "visualizacao"]], hide_index=True,
                 column_config=id_col_config())


{"summary": view_summary, "file": view_file, "compare": view_compare,
 "stats": view_stats, "server": view_server}[view]()

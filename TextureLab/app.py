"""
TextureLab – Visualizador de projetos calculados no servidor.

O app não calcula descritores: abre um projeto gravado por pipeline/texturelab_batch.py
(pasta de resultados ou .tlproj) e só renderiza tabelas, gráficos, 3D, comparações e PCA.
O cálculo antigo (v2.4.0, ver docs/DIAGNOSTICO.md) ficou em app_legacy.py.

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

from components.project_reader import Project, ProjectError  # noqa: E402

APP_VERSION = "3.1.0"
APP_AUTHOR = "Sergio Callai"
APP_YEAR = "2026"

st.set_page_config(page_title="TextureLab", page_icon="🔬", layout="wide")

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
    "MPD": "mm", "MPD_desvio": "mm", "ETD": "mm",
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
    "Cadeia A – ISO 13473-1": ["MPD", "MPD_desvio", "ETD", "A_n_segmentos_validos"],
    "Perfil (descritivo)": ["perfil_Rq_media", "perfil_Rsk_media", "perfil_Rku_media", "perfil_Rk_media",
                            "perfil_Rpk_media", "perfil_Rvk_media", "perfil_Rmr1_media", "perfil_Rmr2_media"],
    "Hurst / fractal (descritivo)": ["H_macro", "D_superficie_macro", "H_macro_R2",
                                     "H_micro", "D_superficie_micro", "H_micro_R2"],
}
_AREAL = ["Sa", "Sq", "Ssk", "Sku", "Sp", "Sv", "Sz", "Sdq", "Sdr_pct", "Sk", "Spk", "Svk", "Smr1", "Smr2",
          "Vmp", "Vmc", "Vvc", "Vvv", "Sal", "Str"]
PARAM_GROUPS["Areal SF – ISO 25178 (S 0,05 mm, F plano)"] = [f"SF_{p}" for p in _AREAL]
PARAM_GROUPS["Areal SL5 – ISO 25178 (+ L 5 mm)"] = [f"SL5_{p}" for p in _AREAL]
PARAM_GROUPS["Areal MICRO (L 0,5 mm, provisório)"] = [f"MICRO_{p}" for p in _AREAL]

DEFAULT_PARAMS = ["MPD", "ETD", "SF_Sq", "SL5_Sq", "SL5_Sdr_pct", "SL5_Ssk", "H_macro"]
ID_COLS = ["arquivo", "trecho", "revestimento", "mp", "nr", "data"]
GROUP_OPTIONS = {"Arquivo": ["arquivo"], "Trecho": ["trecho"], "Revestimento": ["revestimento"],
                 "Ponto de medição (MP)": ["mp"], "Trecho + MP": ["trecho", "mp"]}
AREAL_CHAINS = {"SF": "SF (S 0,05 mm, F plano)", "SL5": "SL5 (+ L 5 mm)", "MICRO": "MICRO (L 0,5 mm, provisório)"}


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
    t = df[list(cols)].rename(columns=cols)
    if "Sk" in t:
        t.insert(min(5, t.shape[1]), "inclinacao_mm_por_pct", -t["Sk"] / 100.0)
    return t


def rk_overlay(fig: go.Figure, mr: np.ndarray, h: np.ndarray, p, color: str, group: str,
               labels: bool = True, **pos) -> None:
    """Desenha sobre a curva de Abbott a reta equivalente e os pontos Smr1/Smr2 (só desenho:
    Sk, Smr1 e Smr2 vêm do resumo.json). O topo do núcleo é a altura da curva em Smr1."""
    sk, mr1, mr2 = (float(p.get(k, np.nan)) for k in ("Sk", "Smr1", "Smr2"))
    if not np.all(np.isfinite([sk, mr1, mr2])):
        return
    z_top = float(np.interp(mr1, mr, h))
    z_bot = z_top - sk
    fig.add_scatter(x=[0, 100], y=[z_top, z_bot], mode="lines", line=dict(color=color, dash="dash", width=1),
                    legendgroup=group, showlegend=False, hoverinfo="skip", **pos)
    fig.add_scatter(x=[mr1, mr2], y=[z_top, z_bot], mode="markers+text" if labels else "markers",
                    legendgroup=group, showlegend=False,
                    marker=dict(color=color, size=9, symbol="diamond", line=dict(width=1, color="white")),
                    text=["Smr1", "Smr2"], textposition=["top right", "bottom left"],
                    textfont=dict(size=10, color=color),
                    hovertemplate=[f"{group}<br>Smr1 = {mr1:.1f} %<br>topo do núcleo = {z_top:.3f} mm<extra></extra>",
                                   f"{group}<br>Smr2 = {mr2:.1f} %<br>Sk = {sk:.3f} mm<br>inclinação = "
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
                     title="λ centro do terço de oitava [mm] — escala log (grade: 1–9 por década)")


def abbott_grid(curves: list, ncols: int, yrange=None, height_row: int = 230, share_y: bool = True) -> go.Figure:
    """Um gráfico pequeno por curva (mesmos eixos), com reta equivalente e Smr1/Smr2.
    curves = [(titulo, mr, altura, params, cor)]."""
    n = len(curves)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=True, shared_yaxes=share_y,
                        subplot_titles=[c[0] for c in curves], horizontal_spacing=0.03,
                        vertical_spacing=min(0.08, 0.35 / max(1, nrows)))
    for i, (title, mr, h, p, col) in enumerate(curves):
        r, c = i // ncols + 1, i % ncols + 1
        fig.add_scatter(x=mr, y=h, name=title, line=dict(color=col), showlegend=False, row=r, col=c)
        if p is not None:
            rk_overlay(fig, mr, h, p, col, title, labels=False, row=r, col=c)
    fig.update_annotations(font_size=10)
    if yrange is not None:
        fig.update_yaxes(range=yrange)
    fig.update_xaxes(range=[0, 100], dtick=20)
    fig.update_layout(height=max(300, height_row * nrows + 60), margin=dict(t=40, l=40, r=10, b=40))
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
        st.session_state["open_error"] = str(e)


@st.cache_data(show_spinner=False, max_entries=4)
def all_spectra(_proj: Project, token: int) -> pd.DataFrame:
    """Espectros de terço de oitava de todos os arquivos (formato longo). Só leitura de CSV."""
    parts = []
    for arq in _proj.index["arquivo"]:
        t = _proj.table(arq, "espectro_terco_oitava.csv")
        if t is not None and len(t):
            parts.append(t.assign(arquivo=arq))
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
    st.markdown(f'<span class="version-badge">v{APP_VERSION} · visualizador</span>', unsafe_allow_html=True)
    st.markdown("### 📂 Abrir projeto")
    path = st.text_input("Pasta do projeto ou arquivo .tlproj", key="proj_path",
                         placeholder=r"C:\TextureLab\Resultados_v3  ou  /data/.../Resultados_v3",
                         help="Pasta de saída do pipeline (com projeto.json) ou o pacote .tlproj. "
                              "Também abre pastas antigas (Resultados_v2), com menos gráficos.")
    if st.button("Abrir pasta / arquivo", type="primary", width="stretch") and path.strip():
        _open(path.strip().strip('"'))
    up = st.file_uploader("…ou envie um .tlproj", type=["tlproj", "zip"], key="proj_upload")
    if up is not None and st.button("Abrir pacote enviado", width="stretch"):
        _open(up.getvalue(), name=up.name)
    if st.session_state.get("open_error"):
        st.error(st.session_state["open_error"])

    proj = _proj()
    if proj is not None:
        st.markdown("---")
        st.markdown(f"**{proj.name}**")
        m = proj.meta
        st.caption(f"{m.get('n_ok', 0)}/{m.get('n_arquivos', 0)} arquivos ok · núcleo "
                   f"{', '.join(m.get('versoes_nos_resultados', [])) or '?'} · formato {m.get('versao_formato')}")
        for w in proj.warnings:
            st.warning(w, icon="⚠️")
        if st.button("Fechar projeto", width="stretch"):
            for k in ("projeto", "open_error"):
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown(f'<div class="license-footer">© {APP_YEAR} {APP_AUTHOR} · '
                '<a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a></div>',
                unsafe_allow_html=True)


# ===================================================================
# Sem projeto: instruções
# ===================================================================
SERVER_HELP = """
**Fluxo:** o servidor calcula uma vez, o app só abre e desenha.

1. Copie os LAZ para o servidor (pasta `LAZ/`).
2. No servidor, rode o lote com uso controlado (CPU, RAM e disco limitados; ver `pipeline/README.md`):
```bash
cd /data/callai/workspace/tyron/texturelab_repo/pipeline
./servidor.sh iniciar /data/callai/workspace/tyron/LAZ /data/callai/workspace/tyron/Resultados_v3
./servidor.sh status        # também: pausar, continuar, parar, log
```
3. Copie `Resultados_v3.tlproj` (ou a pasta `Resultados_v3`) para o PC, por `scp` ou Nextcloud:
```bash
scp sergio@137.226.169.235:/data/callai/workspace/tyron/Resultados_v3.tlproj .
```
4. Abra aqui o `.tlproj` ou a pasta.

`--skip-done` só recalcula o que mudou (versão do núcleo, configuração ou o próprio LAZ).
Para outra configuração: `EXTRA="--config ajustes.json" ./servidor.sh iniciar LAZ Resultados_outra` (ex.: `{"A_lp_design_mm": 2.4}`).
"""

proj = _proj()
if proj is None:
    st.markdown("## Visualizador de projetos")
    st.info("Abra um projeto calculado no servidor pela barra lateral. Nada é recalculado neste app.")
    st.markdown(SERVER_HELP)
    st.stop()

# ===================================================================
# Filtros
# ===================================================================
summary = proj.summary()
ok = summary[summary.get("status", "ok") == "ok"].copy() if "status" in summary else summary.copy()
params_all = numeric_params(ok)

with st.expander("🔎 Filtros", expanded=False):
    fc = st.columns(3)
    sel = {}
    for col, c in zip(("trecho", "revestimento", "mp"), fc):
        opts = sorted(x for x in ok[col].dropna().astype(str).unique()) if col in ok else []
        sel[col] = c.multiselect(col.capitalize(), opts, default=[], key=f"f_{col}",
                                 placeholder="todos")
    for col, vals in sel.items():
        if vals:
            ok = ok[ok[col].astype(str).isin(vals)]
st.caption(f"{len(ok)} arquivo(s) após filtros")
if ok.empty:
    st.warning("Nenhum arquivo com os filtros atuais.")
    st.stop()

VIEWS = ["📋 Resumo", "🔎 Arquivo", "📊 Comparar", "🧮 Estatística", "🖥️ Calcular no servidor"]
view = st.segmented_control("Vista", VIEWS, default=VIEWS[0], key="view", label_visibility="collapsed") or VIEWS[0]


# ===================================================================
# Resumo
# ===================================================================
def view_summary():
    st.markdown("### Parâmetros por arquivo")
    groups = st.multiselect("Grupos de parâmetros", list(PARAM_GROUPS),
                            default=list(PARAM_GROUPS)[:1] + [list(PARAM_GROUPS)[4]], key="sum_groups")
    cols = [p for g in groups for p in PARAM_GROUPS[g] if p in ok]
    tbl = ok[[c for c in ID_COLS if c in ok] + cols]
    st.dataframe(tbl, hide_index=True, column_config={p: st.column_config.NumberColumn(label(p), format="%.4g")
                                                      for p in cols})

    st.markdown("### Média por grupo")
    gname = st.selectbox("Agrupar por", list(GROUP_OPTIONS)[1:], index=3, key="sum_gby")
    gcols = GROUP_OPTIONS[gname]
    if cols:
        agg = ok.groupby(gcols)[cols].agg(["mean", "std", "count"])
        agg.columns = [f"{p} ({s})" for p, s in agg.columns]
        st.dataframe(agg.reset_index(), hide_index=True)

    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Tabela completa (CSV)", ok.to_csv(index=False).encode("utf-8"),
                       file_name=f"{proj.name}_resumo.csv", mime="text/csv", width="stretch")
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf) as xw:
            ok.to_excel(xw, sheet_name="resumo", index=False)
            if cols:
                agg.reset_index().to_excel(xw, sheet_name=f"media_{gname[:20]}", index=False)
        c2.download_button("⬇️ Excel", buf.getvalue(), file_name=f"{proj.name}_resumo.xlsx", width="stretch")
    except ImportError:
        c2.caption("Instale openpyxl para exportar Excel.")


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


def _surface_fig(z: np.ndarray, step: float, max_pts: int, kind: str, title: str) -> go.Figure:
    k = max(1, int(np.ceil(np.sqrt(z.size / max_pts))))
    zz = z[::k, ::k]
    x = np.arange(zz.shape[1]) * step * k
    y = np.arange(zz.shape[0]) * step * k
    if kind == "3D":
        fig = go.Figure(go.Surface(z=zz, x=x, y=y, colorscale="Viridis", colorbar=dict(title="z [mm]")))
        span = [x[-1], y[-1], max(1e-9, float(np.nanmax(zz) - np.nanmin(zz)))]
        r = max(span[:2])
        fig.update_layout(scene=dict(xaxis_title="via [mm]", yaxis_title="largura [mm]", zaxis_title="z [mm]",
                                     aspectmode="manual",
                                     aspectratio=dict(x=span[0] / r * 2, y=span[1] / r * 2, z=0.4)),
                          height=600, margin=dict(l=0, r=0, t=30, b=0), title=title)
    else:
        fig = go.Figure(go.Heatmap(z=zz, x=x, y=y, colorscale="Viridis", colorbar=dict(title="z [mm]")))
        fig.update_layout(xaxis_title="via [mm]", yaxis_title="largura [mm]", height=380,
                          margin=dict(l=0, r=0, t=30, b=0), title=title)
        fig.update_yaxes(scaleanchor="x")
    return fig


def view_file():
    files = ok["arquivo"].tolist()
    arq = st.selectbox("Arquivo", files, key="file_sel")
    r = proj.resumo(arq)
    v = proj.view(arq)
    st.markdown(f"#### {arq}")
    if r.get("status") != "ok":
        st.error(r.get("erro", "erro no cálculo"))
        return
    m = st.columns(6)
    for c, k in zip(m, ["MPD", "ETD", "SF_Sq", "SL5_Sq", "SL5_Sdr_pct", "H_macro"]):
        c.metric(label(k), fmt(r.get(k)))
    st.caption(f"{r.get('comprimento_mm', 0):.0f} × {r.get('largura_mm', 0):.0f} mm · dx {r.get('dx_mm')} mm · "
               f"{r.get('n_pontos', 0) / 1e6:.0f} M pontos · núcleo {r.get('versao_script')} · "
               f"receita {r.get('receita_hash', '–')}")
    if not v:
        st.info("Este resultado não tem visualizacao.npz (núcleo antigo): perfis, Abbott e histograma não "
                "estão disponíveis. Recalcule no servidor com a versão atual para ver tudo.")

    tabs = st.tabs(["Superfície", "Perfis", "Espectro", "Abbott / alturas", "PSD / Hurst", "Segmentos", "Todos os parâmetros"])
    with tabs[0]:
        c = st.columns(4)
        src = c[0].radio("Superfície", ["Bruta (prévia)", "SL5 filtrada"] if "SL5_previa" in v else ["Bruta (prévia)"],
                         key="surf_src")
        kind = c[1].radio("Tipo", ["Mapa", "3D"], key="surf_kind")
        max_pts = c[2].select_slider("Pontos no gráfico", [50_000, 150_000, 300_000, 600_000, 1_200_000],
                                     value=150_000, key="surf_pts")
        detr = c[3].checkbox("Remover plano (exibição)", value=True, key="surf_detr")
        if src.startswith("SL5"):
            pstep = r.get("receita", {}).get("config", {}).get("preview_step", 8)
            z, step = v["SL5_previa"].astype(np.float32), float(r["dx_mm"]) * pstep
        else:
            pv = proj.preview(arq)
            if pv is None:
                st.info("Sem prévia.")
                z = None
            else:
                z, step = pv
                if detr:
                    z = _plane_removed(z)
        if z is not None:
            st.plotly_chart(_surface_fig(z, step, max_pts, "3D" if kind == "3D" else "map", src),
                            key="surf_fig")
            st.caption(f"Prévia com passo {step:.3f} mm (a cada {int(round(step / r['dx_mm']))} pontos); "
                       "os parâmetros foram calculados na grade completa.")
    with tabs[1]:
        if "A_perfil_limpo" in v:
            step = float(v["A_perfil_passo_mm"])
            fig = go.Figure()
            for i, y in enumerate(v["A_perfil_y_mm"]):
                xs = np.arange(v["A_perfil_limpo"].shape[1]) * step
                vis = True if i == len(v["A_perfil_y_mm"]) // 2 else "legendonly"
                fig.add_scatter(x=xs, y=v["A_perfil_limpo"][i], name=f"y={y:.1f} mm limpo", visible=vis,
                                line=dict(width=1))
                fig.add_scatter(x=xs, y=v["A_perfil_passa_baixa"][i], name=f"y={y:.1f} mm passa-baixa", visible=vis)
            a0, L = float(v["A_segmento_inicio_mm"]), float(v["A_segmento_mm"])
            for s in range(int(v["A_n_segmentos"]) + 1):
                fig.add_vline(x=a0 + s * L, line_dash="dot", line_color="gray")
            fig.update_layout(title="Cadeia A: perfis de 0,5 mm (limpos e após passa-baixa); linhas = segmentos de 100 mm",
                              xaxis_title="via [mm]", yaxis_title="z [mm]", height=420)
            st.plotly_chart(fig, key="prof_a")
        if "nativo_perfil" in v:
            xs = float(v["nativo_inicio_mm"]) + np.arange(len(v["nativo_perfil"])) * float(v["nativo_passo_mm"])
            fig = px.line(x=xs, y=v["nativo_perfil"], labels={"x": "via [mm]", "y": "z [mm]"},
                          title=f"Linha na resolução nativa (dx {float(v['nativo_passo_mm'])} mm, y = "
                                f"{float(v['nativo_y_mm']):.1f} mm)")
            fig.update_traces(line=dict(width=1))
            st.plotly_chart(fig, key="prof_nat")
        if not v:
            st.info("Sem perfis neste resultado.")
    with tabs[2]:
        sp = proj.table(arq, "espectro_terco_oitava.csv")
        if sp is not None and len(sp):
            fig = go.Figure(go.Scatter(x=sp["lambda_centro_mm"], y=sp["L_tx_dB_media"], mode="lines+markers",
                                       error_y=dict(array=sp["L_tx_dB_desvio"], visible=True), name="média ± dp"))
            spectrum_axis(fig, sp["lambda_centro_mm"])
            fig.update_layout(yaxis_title="L_tx [dB ref. 1 µm]", height=420,
                              title="Espectro de textura – ISO 13473-4 (método 1)")
            st.plotly_chart(fig, key="spec")
            st.dataframe(sp, hide_index=True)
    with tabs[3]:
        chains = [c for c in AREAL_CHAINS if f"{c}_abbott_mr_pct" in v]
        if chains:
            modo = st.radio("Exibição", ["Sobrepostas", "Lado a lado (uma por cadeia)"], horizontal=True,
                            key="arq_abbott_mode")
            if modo.startswith("Lado"):
                curves = [(AREAL_CHAINS[ch], v[f"{ch}_abbott_mr_pct"], v[f"{ch}_abbott_altura_mm"],
                           {k: r.get(f"{ch}_{k}", np.nan) for k in RK_KEYS}, COLORS[i % len(COLORS)])
                          for i, ch in enumerate(chains)]
                fga = abbott_grid(curves, ncols=len(curves), height_row=330, share_y=False)
                fga.update_yaxes(title_text="altura [mm]", col=1)
                fga.update_xaxes(title_text="Material ratio [%]")
                st.plotly_chart(fga, key="abbott_grid")
            c1, c2 = st.columns(2)
            fa, fh = go.Figure(), go.Figure()
            for i, ch in enumerate(chains):
                col = COLORS[i % len(COLORS)]
                mr, h = v[f"{ch}_abbott_mr_pct"], v[f"{ch}_abbott_altura_mm"]
                fa.add_scatter(x=mr, y=h, name=AREAL_CHAINS[ch], line=dict(color=col), legendgroup=ch)
                rk_overlay(fa, mr, h, {k: r.get(f"{ch}_{k}", np.nan) for k in RK_KEYS}, col, ch)
                e = v[f"{ch}_hist_bordas_mm"]
                cnt = v[f"{ch}_hist_contagem"].astype(float)
                dens = cnt / (cnt.sum() * np.diff(e))
                fh.add_scatter(x=(e[:-1] + e[1:]) / 2, y=dens, name=AREAL_CHAINS[ch], mode="lines")
            fa.update_layout(title="Curva de Abbott-Firestone (alturas em relação à média)",
                             xaxis_title="Material ratio [%]", yaxis_title="altura [mm]", height=420)
            fh.update_layout(title="Distribuição de alturas (PDF)", xaxis_title="altura [mm]",
                             yaxis_title="densidade [1/mm]", height=420)
            if not modo.startswith("Lado"):
                c1.plotly_chart(fa, key="abbott")
                c2.plotly_chart(fh, key="hist")
            else:
                c1.plotly_chart(fh, key="hist")
            one = pd.DataFrame([r])
            rk = pd.concat({AREAL_CHAINS[ch]: rk_table(one, ch).iloc[0] for ch in chains}, axis=1)
            st.caption("Tracejado: reta equivalente da ISO 13565-2 (de 0 % a 100 %); losangos: Smr1 e Smr2. "
                       "Inclinação = −Sk/100 % [mm/%].")
            st.dataframe(rk)
        else:
            st.info("Sem curvas de Abbott neste resultado.")
    with tabs[4]:
        psd = proj.table(arq, "psd_media.csv")
        if psd is not None and len(psd):
            fig = go.Figure(go.Scatter(x=psd["lambda_mm"], y=psd["PSD_mm3"], mode="lines", name="PSD média",
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
            fig.update_layout(height=450, title="PSD 1D no sentido da via (Welch) e ajuste de Hurst – descritivo")
            st.plotly_chart(fig, key="psd")
            st.caption("Retas tracejadas: inclinação ajustada no servidor (PSD ∝ λ^β), desenhadas pela média "
                       "geométrica da faixa.")
        else:
            st.info("Sem PSD neste resultado (núcleo anterior à v3).")
    with tabs[5]:
        seg = proj.table(arq, "cadeiaA_segmentos.csv")
        if seg is not None and len(seg):
            val = seg[seg["valido"]]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(val, x="MSD", nbins=30, title="MSD por segmento de 100 mm (válidos)",
                                         labels={"MSD": "MSD [mm]"}), key="msd_hist")
            c2.plotly_chart(px.scatter(val, x="x_centro_mm", y="MSD", color=val["segmento"].astype(str),
                                       title="MSD ao longo da largura", labels={"x_centro_mm": "posição na largura [mm]",
                                                                               "color": "segmento"}), key="msd_pos")
            st.dataframe(seg, hide_index=True)
    with tabs[6]:
        flat = {k: v_ for k, v_ in r.items() if not isinstance(v_, (dict, list))}
        st.dataframe(pd.DataFrame({"parâmetro": list(flat), "valor": [str(x) for x in flat.values()]}),
                     hide_index=True, height=600)
        if "receita" in r:
            with st.expander("Receita (versão do núcleo, configuração, LAZ)"):
                st.json(r["receita"])


# ===================================================================
# Comparar
# ===================================================================
def view_compare():
    c = st.columns([2, 3])
    gname = c[0].selectbox("Comparar por", list(GROUP_OPTIONS), index=1, key="cmp_gby")
    gcols = GROUP_OPTIONS[gname]
    keys = group_key(ok, gcols)
    groups = sorted(keys.unique())
    chosen = c[1].multiselect("Grupos", groups, default=groups[:12], key=f"cmp_groups_{gname}")
    if not chosen:
        st.info("Escolha pelo menos um grupo.")
        return
    sub = ok[keys.isin(chosen)].assign(_grupo=keys[keys.isin(chosen)])

    params = st.multiselect("Parâmetros", params_all, default=[p for p in DEFAULT_PARAMS if p in params_all],
                            format_func=label, key="cmp_params")
    kind = st.radio("Gráfico", ["Barras (média ± dp)", "Caixa (todos os arquivos)"], horizontal=True, key="cmp_kind")
    for p in params:
        if kind.startswith("Barras"):
            g = sub.groupby("_grupo")[p].agg(["mean", "std", "count"]).reindex(chosen).reset_index()
            fig = px.bar(g, x="_grupo", y="mean", error_y="std", hover_data=["count"],
                         labels={"_grupo": gname, "mean": label(p)}, title=label(p))
        else:
            fig = px.box(sub, x="_grupo", y=p, points="all", hover_data=["arquivo"],
                         labels={"_grupo": gname, p: label(p)}, title=label(p),
                         category_orders={"_grupo": chosen})
        fig.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig, key=f"cmp_{p}")

    st.markdown("### Espectros de terço de oitava")
    spec = all_spectra(proj, st.session_state["proj_token"])
    if len(spec):
        s = spec.merge(sub[["arquivo", "_grupo"]], on="arquivo")
        g = s.groupby(["_grupo", "lambda_centro_mm"])["L_tx_dB_media"].mean().reset_index()
        fig = px.line(g, x="lambda_centro_mm", y="L_tx_dB_media", color="_grupo", markers=True,
                      category_orders={"_grupo": chosen}, color_discrete_sequence=COLORS,
                      labels={"lambda_centro_mm": "λ [mm]", "L_tx_dB_media": "L_tx [dB ref. 1 µm]", "_grupo": gname})
        spectrum_axis(fig, g["lambda_centro_mm"])
        fig.update_layout(height=450)
        st.plotly_chart(fig, key="cmp_spec")
        wide = s.pivot_table(index=["_grupo", "arquivo"], columns="lambda_centro_mm", values="L_tx_dB_media")
        wide = wide[sorted(wide.columns)]
        wide.columns = [f"{c:g} mm" for c in wide.columns]
        means = wide.groupby(level=0).mean().reindex([c for c in chosen if c in wide.index.get_level_values(0)])
        st.markdown("**L_tx [dB ref. 1 µm] por banda – média por grupo**")
        st.dataframe(means.round(2).rename_axis(gname))
        with st.expander(f"Todas as amostras ({len(wide)})", expanded=True):
            st.dataframe(wide.round(2).reset_index().rename(columns={"_grupo": gname}), hide_index=True)
        st.download_button("⬇️ Espectros (CSV)", wide.reset_index().rename(columns={"_grupo": gname})
                           .to_csv(index=False).encode("utf-8"), file_name=f"{proj.name}_espectros.csv",
                           mime="text/csv", key="dl_spec")

    st.markdown("### Curvas de Abbott-Firestone")
    ch = st.radio("Cadeia areal", list(AREAL_CHAINS), format_func=AREAL_CHAINS.get, horizontal=True, index=1,
                  key="cmp_abbott_chain")
    ab = all_abbott(proj, st.session_state["proj_token"], ch)
    if len(ab):
        c0, c1, c2, c3 = st.columns([2, 2, 2, 1])
        modo = c0.radio("Exibição", ["Sobrepostas", "Uma por gráfico"], horizontal=True, key="cmp_abbott_mode")
        show_rk = c1.checkbox("Reta equivalente e Smr1/Smr2 (ISO 13565-2)", value=True, key="cmp_rk")
        zoom = c2.checkbox("Zoom no núcleo (alturas entre 1 % e 99 %)", value=True, key="cmp_zoom")
        ncols = c3.number_input("Colunas", 1, 8, 4, key="cmp_cols", disabled=modo == "Sobrepostas")
        a = ab.merge(sub[["arquivo", "_grupo"]], on="arquivo")
        g = a.groupby(["_grupo", "mr_pct"])["altura_mm"].mean().reset_index()
        tbl = rk_table(sub, ch)
        tbl.insert(0, "arquivo", sub["arquivo"])
        tbl.insert(0, gname, sub["_grupo"])
        gmean = tbl.drop(columns="arquivo").groupby(gname).mean()
        yr = core_range(g[(g["mr_pct"] >= 1) & (g["mr_pct"] <= 99)]["altura_mm"]) if zoom else None
        if modo == "Sobrepostas":
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
            fig.update_layout(height=500, xaxis_title="Material ratio [%]", yaxis_title="altura [mm]",
                              legend_title_text=gname)
            st.plotly_chart(fig, key="cmp_abbott")
            st.caption("Clique em um item da legenda para esconder a curva; duplo clique para ver só ela.")
        else:
            curves = []
            for i, grp in enumerate(chosen):
                d = g[g["_grupo"] == grp]
                if not d.empty:
                    curves.append((grp, d["mr_pct"].to_numpy(), d["altura_mm"].to_numpy(),
                                   gmean.loc[grp] if show_rk and grp in gmean.index else None,
                                   COLORS[i % len(COLORS)]))
            st.plotly_chart(abbott_grid(curves, int(ncols), yr), key="cmp_abbott_grid")
        st.caption("Curva = média das curvas do grupo; tracejado = reta equivalente; losangos = Smr1 e Smr2 "
                   "(passe o mouse para ver os valores). Smr1, Smr2 e Sk são médias do grupo, calculados por arquivo "
                   "no servidor. Inclinação da reta equivalente = −Sk/100 % [mm/%].")
        fmt_cols = {c: st.column_config.NumberColumn(c, format="%.4g") for c in tbl.columns[2:]}
        st.markdown("**Família Sk – média por grupo**")
        st.dataframe(gmean.reindex([c for c in chosen if c in gmean.index]), column_config=fmt_cols)
        with st.expander(f"Todas as amostras ({len(tbl)})", expanded=True):
            st.dataframe(tbl, hide_index=True, column_config=fmt_cols)
        st.download_button("⬇️ Família Sk (CSV)", tbl.to_csv(index=False).encode("utf-8"),
                           file_name=f"{proj.name}_abbott_{ch}.csv", mime="text/csv", key="dl_abbott")
    else:
        st.info("Sem curvas de Abbott no projeto (resultados do núcleo antigo).")


# ===================================================================
# Estatística
# ===================================================================
def view_stats():
    params = st.multiselect("Parâmetros", params_all,
                            default=[p for p in ["MPD", "SF_Sq", "SL5_Sq", "SL5_Ssk", "SL5_Sku", "SL5_Sdr_pct",
                                                 "SL5_Sal", "MICRO_Sq", "H_macro"] if p in params_all],
                            format_func=label, key="st_params")
    gname = st.selectbox("Cor por", list(GROUP_OPTIONS)[1:], index=0, key="st_color")
    color = group_key(ok, GROUP_OPTIONS[gname])
    if len(params) < 2:
        st.info("Escolha pelo menos dois parâmetros.")
        return
    X = ok[params].apply(pd.to_numeric, errors="coerce")

    st.markdown("### Correlação (Pearson)")
    corr = X.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=120 + 40 * len(params))
    st.plotly_chart(fig, key="st_corr")

    st.markdown("### Dispersão")
    c = st.columns(2)
    xp = c[0].selectbox("X", params, index=0, format_func=label, key="st_x")
    yp = c[1].selectbox("Y", params, index=1, format_func=label, key="st_y")
    fig = px.scatter(ok.assign(_g=color), x=xp, y=yp, color="_g", hover_data=["arquivo"], trendline=None,
                     labels={xp: label(xp), yp: label(yp), "_g": gname})
    fig.update_layout(height=450)
    st.plotly_chart(fig, key="st_scatter")

    st.markdown("### PCA (parâmetros padronizados)")
    Xc = X.dropna()
    if len(Xc) < 3:
        st.info("Poucos arquivos completos para PCA.")
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
    c2.plotly_chart(px.bar(x=[f"PC{i + 1}" for i in range(len(ev))], y=ev, labels={"x": "", "y": "variância [%]"},
                           title="Variância explicada"), key="st_pca_var")
    load = pd.DataFrame(pca.components_.T, index=params, columns=[f"PC{i + 1}" for i in range(len(ev))])
    c2.dataframe(load.style.format("{:.2f}"))
    if len(Xc) < len(ok):
        st.caption(f"{len(ok) - len(Xc)} arquivo(s) sem algum dos parâmetros ficaram fora da PCA.")


# ===================================================================
# Calcular no servidor
# ===================================================================
def view_server():
    st.markdown("### Calcular ou recalcular no servidor")
    st.markdown("Este app não calcula. O que faltar (arquivos novos, outra configuração, núcleo mais novo) "
                "é calculado no servidor pelo pipeline, e o resultado é aberto aqui.")
    st.markdown(SERVER_HELP)
    cfg = proj.meta.get("config") or {}
    if cfg:
        with st.expander("Configuração usada neste projeto"):
            st.json(cfg)
    idx = proj.index.copy()
    st.markdown("#### Situação dos arquivos")
    st.dataframe(idx[["arquivo", "status", "versao_nucleo", "receita_hash", "visualizacao"]], hide_index=True)


{VIEWS[0]: view_summary, VIEWS[1]: view_file, VIEWS[2]: view_compare,
 VIEWS[3]: view_stats, VIEWS[4]: view_server}[view]()

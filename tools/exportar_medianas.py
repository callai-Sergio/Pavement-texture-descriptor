#!/usr/bin/env python3
"""
exportar_medianas.py – Excel com a mediana por superfície (trecho + revestimento) dos parâmetros 3D
(SL5 e MICRO), dos parâmetros normativos de perfil (marcados "perfil") e dos vetores (L_tx por terço
de oitava, PSD 1D e radial, PDF das alturas).

Só lê o projeto (pasta ou .tlproj); não calcula descritores.

    python tools/exportar_medianas.py Resultados_v3.3 [-o Resultados_v3.3_medianas.xlsx]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "TextureLab"))
from components.project_reader import Project  # noqa: E402

CHAINS = ["SL5", "MICRO"]
AREAL = [("Ssk", "-", "Skewness (ISO 25178-2)"), ("Sku", "-", "Kurtosis (ISO 25178-2)"),
         ("Sk", "mm", "Altura do núcleo (ISO 25178-2 / 13565-2)"), ("Spk", "mm", "Altura reduzida dos picos"),
         ("Svk", "mm", "Profundidade reduzida dos vales"), ("Smr1", "%", "Material ratio superior do núcleo"),
         ("Smr2", "%", "Material ratio inferior do núcleo"), ("Vvv", "mm", "Volume de vazios dos vales (mm³/mm²)"),
         ("Vmp", "mm", "Volume de material dos picos (mm³/mm²)"),
         ("Str", "-", "Texture aspect ratio (ISO 25178-2)"), ("Sal", "mm", "Autocorrelation length (ISO 25178-2)"),
         ("P50", "mm", "Percentil 50 das alturas (rel. à média)"), ("P70", "mm", "Percentil 70 das alturas"),
         ("P90", "mm", "Percentil 90 das alturas"), ("P95", "mm", "Percentil 95 das alturas"),
         ("P90_P70", "-", "Razão P90/P70 (cauda superior da PDF)"),
         ("g3D_pct", "%", "g-factor 3D: análogo do Anexo B da DIN ISO 10844 na curva de Abbott da superfície (sem norma)")]
PROFILE = [("MPD", "mm", "Mean Profile Depth (ISO 13473-1)"),
           ("g_factor_media", "%", "g-factor (DIN ISO 10844:2024 Anexo B), média dos segmentos de 100 mm"),
           ("perfil_Rsk_media", "-", "Skewness de perfil (ISO 13473-2 / DIN ISO 10844 5.3.2)"),
           ("perfil_Rku_media", "-", "Kurtosis de perfil"),
           ("perfil_Rk_media", "mm", "Rk (ISO 13565-2)"), ("perfil_Rpk_media", "mm", "Rpk (ISO 13565-2)"),
           ("perfil_Rvk_media", "mm", "Rvk (ISO 13565-2)"), ("perfil_Rmr1_media", "%", "Mr1 (ISO 13565-2)"),
           ("perfil_Rmr2_media", "%", "Mr2 (ISO 13565-2)")]
HURST = [("H_micro", "-", "Hurst 1D (perfil), 0,05–0,5 mm", "perfil"), ("H_macro", "-", "Hurst 1D (perfil), 0,5–20 mm", "perfil"),
         ("H2D_micro", "-", "Hurst 3D pela PSD radial (SF), 0,05–0,5 mm", "3D"),
         ("H2D_macro", "-", "Hurst 3D pela PSD radial (SF), 0,5–20 mm", "3D")]


def surface_key(df: pd.DataFrame) -> pd.Series:
    return df["trecho"].astype(str) + " · " + df["revestimento"].astype(str)


def vector_median(proj: Project, files: pd.DataFrame, filename: str, xcol: str, ycol: str) -> pd.DataFrame:
    """Vetor (ex.: L_tx por banda) de cada arquivo -> mediana por superfície, uma coluna por x."""
    parts = []
    for _, r in files.iterrows():
        tb = proj.table(r["arquivo"], filename)
        if tb is not None and len(tb):
            parts.append(pd.DataFrame({"superficie": r["superficie"], "x": tb[xcol].round(6), "y": tb[ycol]}))
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts)
    return long.groupby(["superficie", "x"])["y"].median().unstack("x").sort_index(axis=1)


def pdf_median(proj: Project, files: pd.DataFrame, chain: str, n: int = 200) -> pd.DataFrame:
    """PDF das alturas (rel. à média) de cada arquivo, reamostrada numa grade comum -> mediana."""
    dens = {}
    for _, r in files.iterrows():
        v = proj.view(r["arquivo"])
        if f"{chain}_hist_bordas_mm" in v:
            e = v[f"{chain}_hist_bordas_mm"].astype(float)
            c = v[f"{chain}_hist_contagem"].astype(float)
            dens[r["arquivo"]] = ((e[:-1] + e[1:]) / 2, c / (c.sum() * np.diff(e)), r["superficie"])
    if not dens:
        return pd.DataFrame()
    lo = np.percentile([d[0][0] for d in dens.values()], 50)
    hi = np.percentile([d[0][-1] for d in dens.values()], 50)
    grid = np.linspace(lo, hi, n)
    rows = [pd.Series(np.interp(grid, x, y, left=0, right=0), index=np.round(grid, 4), name=s) for x, y, s in dens.values()]
    return pd.DataFrame(rows).groupby(level=0).median()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto", help="pasta do projeto ou .tlproj")
    ap.add_argument("-o", "--saida", help="arquivo .xlsx (padrão: <projeto>_medianas.xlsx)")
    args = ap.parse_args()
    src = Path(args.projeto)
    out = Path(args.saida) if args.saida else src.parent / (src.name.removesuffix(".tlproj") + "_medianas.xlsx")
    proj = Project(src)
    s = proj.summary()
    s = s[s["status"] == "ok"].copy()
    s["superficie"] = surface_key(s)

    cols, legend = [], []
    for k, u, d in PROFILE:
        cols.append((f"perfil | {k}", k)); legend.append(("perfil (norma)", k, u, d))
    for ch in CHAINS:
        for k, u, d in AREAL:
            cols.append((f"{ch} | {k}", f"{ch}_{k}")); legend.append((f"3D {ch}", f"{ch}_{k}", u, d))
    for k, u, d, kind in HURST:
        cols.append((f"{kind} | {k}", k)); legend.append((kind, k, u, d))
    present = [(name, c) for name, c in cols if c in s]
    data = s[["superficie", "arquivo"] + [c for _, c in present]].rename(columns={c: name for name, c in present})
    num = [name for name, _ in present]
    med = data.groupby("superficie")[num].median()
    med.insert(0, "n_MPs", data.groupby("superficie").size())

    files = s[["arquivo", "superficie"]]
    ltx = vector_median(proj, files, "espectro_terco_oitava.csv", "lambda_centro_mm", "L_tx_dB_media")
    psd1 = vector_median(proj, files, "psd_media.csv", "lambda_mm", "PSD_mm3")
    psd2 = vector_median(proj, files, "psd_radial.csv", "lambda_mm", "PSD2D_mm4")

    leia = pd.DataFrame(legend, columns=["tipo", "parâmetro", "unidade", "descrição"])
    notas = pd.DataFrame({"nota": [
        f"Projeto: {proj.name} · núcleo {', '.join(proj.meta.get('versoes_nos_resultados', []))} · {len(s)} arquivos.",
        "Uma linha por superfície = trecho + revestimento (as 4 pistas ISO são superfícies distintas); valor = mediana dos MPs.",
        "3D SL5: superfície S 0,05 mm + F plano + L 5 mm (textura dos agregados). 3D MICRO: F plano + L 0,5 mm (provisória).",
        "Parâmetros 'perfil' são definidos por norma só para perfis (ISO 13473-1/-2, ISO 13565-2, DIN ISO 10844); mantidos como referência normativa.",
        "Percentis P50–P95: alturas em relação à média da superfície; P90/P70 descreve a cauda superior da PDF.",
        "L_tx por terço de oitava (ISO 13473-4) é vetorial e de perfil: aba 'Ltx_terco_oitava' (dB re 1 µm, colunas = λ central em mm).",
        "PSD 1D: perfil no sentido da via (mm³). PSD radial: superfície SF em 2D, média radial (mm⁴). Colunas = λ em mm.",
        "PDF: densidade (1/mm) das alturas, reamostrada numa grade comum; colunas = altura em mm.",
    ]})
    with pd.ExcelWriter(out) as xw:
        notas.to_excel(xw, sheet_name="Leia-me", index=False)
        leia.to_excel(xw, sheet_name="Leia-me", index=False, startrow=len(notas) + 2)
        med.to_excel(xw, sheet_name="Medianas")
        for name, df in (("Ltx_terco_oitava", ltx), ("PSD_1D_perfil", psd1), ("PSD_radial_3D", psd2)):
            if len(df):
                df.to_excel(xw, sheet_name=name)
        for ch in CHAINS:
            pdf = pdf_median(proj, files, ch)
            if len(pdf):
                pdf.to_excel(xw, sheet_name=f"PDF_{ch}")
        data.sort_values(["superficie", "arquivo"]).to_excel(xw, sheet_name="Dados_por_MP", index=False)
    print(f"{out}  ({len(med)} superfícies, {len(num)} parâmetros escalares)")


if __name__ == "__main__":
    main()

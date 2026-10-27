#!/usr/bin/env python3
"""
exportar_medianas.py – Excel (em inglês) com a mediana por superfície (trecho + revestimento) dos parâmetros
3D (SL5 e MICRO), dos parâmetros normativos de perfil (marcados "profile") e dos vetores (L_tx por terço
de oitava, PSD 1D e radial, PDF das alturas). A aba "All_data" junta tudo numa tabela só (uma linha por
superfície); as outras abas trazem os mesmos dados separados.

Só lê o projeto (pasta ou .tlproj); não calcula descritores.

    python tools/exportar_medianas.py Resultados_v3.4 [-o Resultados_v3.4_medians.xlsx]
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
         ("Sk", "mm", "Core height (ISO 25178-2 / 13565-2)"), ("Spk", "mm", "Reduced peak height"),
         ("Svk", "mm", "Reduced valley depth"), ("Smr1", "%", "Upper material ratio of the core"),
         ("Smr2", "%", "Lower material ratio of the core"), ("Vvv", "mm", "Dale void volume (mm³/mm²)"),
         ("Vmp", "mm", "Peak material volume (mm³/mm²)"),
         ("Str", "-", "Texture aspect ratio (ISO 25178-2)"), ("Sal", "mm", "Autocorrelation length (ISO 25178-2)"),
         ("P50", "mm", "50th percentile of heights (relative to the mean)"), ("P70", "mm", "70th percentile of heights"),
         ("P90", "mm", "90th percentile of heights"), ("P95", "mm", "95th percentile of heights"),
         ("P90_P70", "-", "Ratio P90/P70 (upper tail of the PDF)"),
         ("g3D_pct", "%", "3D g-factor (no standard): DIN ISO 10844 Annex B on 100 mm × full-width patches, "
                          "0.5 mm cells; mean of the patches. SL5 only (resampling would erase the MICRO texture)")]
# (coluna no resumo, nome curto, unidade, descrição)
PROFILE = [("MPD", "MPD", "mm", "Mean Profile Depth (ISO 13473-1)"),
           ("g_factor_media", "g", "%", "g-factor (DIN ISO 10844:2024 Annex B), mean of the 100 mm segments"),
           ("perfil_Rsk_media", "Rsk", "-", "Profile skewness (ISO 13473-2 / DIN ISO 10844 5.3.2)"),
           ("perfil_Rku_media", "Rku", "-", "Profile kurtosis"),
           ("perfil_Rk_media", "Rk", "mm", "Rk (ISO 13565-2)"), ("perfil_Rpk_media", "Rpk", "mm", "Rpk (ISO 13565-2)"),
           ("perfil_Rvk_media", "Rvk", "mm", "Rvk (ISO 13565-2)"), ("perfil_Rmr1_media", "Mr1", "%", "Mr1 (ISO 13565-2)"),
           ("perfil_Rmr2_media", "Mr2", "%", "Mr2 (ISO 13565-2)")]
HURST = [("H_micro", "H micro", "-", "1D Hurst (profile PSD), 0.05–0.5 mm", "profile"),
         ("H_macro", "H macro", "-", "1D Hurst (profile PSD), 0.5–20 mm", "profile"),
         ("H2D_micro", "H micro", "-", "3D Hurst (radial PSD of the SF surface), 0.05–0.5 mm", "3D"),
         ("H2D_macro", "H macro", "-", "3D Hurst (radial PSD of the SF surface), 0.5–20 mm", "3D")]
LAMBDA_GRID = np.logspace(np.log10(0.025), np.log10(80.0), 61)          # λ comum para as PSD (mm)


def surface_key(df: pd.DataFrame) -> pd.Series:
    return df["trecho"].astype(str) + " · " + df["revestimento"].astype(str)


def vector_median(proj: Project, files: pd.DataFrame, filename: str, xcol: str, ycol: str,
                  grid=None) -> pd.DataFrame:
    """Vetor (ex.: L_tx por banda) de cada arquivo -> mediana por superfície, uma coluna por x. Com 'grid',
    cada vetor é antes interpolado em log-log nesse eixo (PSD: grade comum de λ, fora da faixa = vazio)."""
    parts = []
    for _, r in files.iterrows():
        tb = proj.table(r["arquivo"], filename)
        if tb is None or not len(tb):
            continue
        if grid is not None:
            tb = tb[(tb[xcol] > 0) & (tb[ycol] > 0)].sort_values(xcol)
            lx, ly = np.log10(tb[xcol].to_numpy()), np.log10(tb[ycol].to_numpy())
            y = 10 ** np.interp(np.log10(grid), lx, ly, left=np.nan, right=np.nan)
            parts.append(pd.DataFrame({"surface": r["surface"], "x": np.round(grid, 4), "y": y}))
        else:
            parts.append(pd.DataFrame({"surface": r["surface"], "x": tb[xcol].round(6), "y": tb[ycol]}))
    if not parts:
        return pd.DataFrame()
    long = pd.concat(parts)
    return long.groupby(["surface", "x"])["y"].median().unstack("x").sort_index(axis=1)


def pdf_median(proj: Project, files: pd.DataFrame, chain: str, n: int = 200) -> pd.DataFrame:
    """PDF das alturas (rel. à média) de cada arquivo, reamostrada numa grade comum -> mediana."""
    dens = {}
    for _, r in files.iterrows():
        v = proj.view(r["arquivo"])
        if f"{chain}_hist_bordas_mm" in v:
            e = v[f"{chain}_hist_bordas_mm"].astype(float)
            c = v[f"{chain}_hist_contagem"].astype(float)
            dens[r["arquivo"]] = ((e[:-1] + e[1:]) / 2, c / (c.sum() * np.diff(e)), r["surface"])
    if not dens:
        return pd.DataFrame()
    lo = np.percentile([d[0][0] for d in dens.values()], 50)
    hi = np.percentile([d[0][-1] for d in dens.values()], 50)
    grid = np.linspace(lo, hi, n)
    rows = [pd.Series(np.interp(grid, x, y, left=0, right=0), index=np.round(grid, 4), name=s) for x, y, s in dens.values()]
    out = pd.DataFrame(rows).groupby(level=0).median()
    out.index.name = "surface"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto", help="pasta do projeto ou .tlproj")
    ap.add_argument("-o", "--saida", help="arquivo .xlsx (padrão: <projeto>_medians.xlsx)")
    args = ap.parse_args()
    src = Path(args.projeto)
    out = Path(args.saida) if args.saida else src.parent / (src.name.removesuffix(".tlproj") + "_medians.xlsx")
    proj = Project(src)
    s = proj.summary()
    s = s[s["status"] == "ok"].copy()
    s["surface"] = surface_key(s)

    cols, legend = [], []
    for k, short, u, d in PROFILE:
        name = f"profile | {short} [{u}]"
        cols.append((name, k)); legend.append((name, "profile (standard)", k, d))
    for ch in CHAINS:
        for k, u, d in AREAL:
            if ch == "MICRO" and k == "g3D_pct":
                continue
            name = f"{ch} | {k.replace('_pct', '').replace('P90_P70', 'P90/P70')} [{u}]"
            cols.append((name, f"{ch}_{k}")); legend.append((name, f"3D {ch}", f"{ch}_{k}", d))
    for k, short, u, d, kind in HURST:
        name = f"{kind} | {short} [{u}]"
        cols.append((name, k)); legend.append((name, kind, k, d))
    present = [(name, c) for name, c in cols if c in s]
    data = s[["surface", "arquivo"] + [c for _, c in present]].rename(
        columns={"arquivo": "file", **{c: name for name, c in present}})
    num = [name for name, _ in present]
    med = data.groupby("surface")[num].median()
    med.insert(0, "n_MPs", data.groupby("surface").size())

    files = s[["arquivo", "surface"]]
    ltx = vector_median(proj, files, "espectro_terco_oitava.csv", "lambda_centro_mm", "L_tx_dB_media")
    psd1 = vector_median(proj, files, "psd_media.csv", "lambda_mm", "PSD_mm3", grid=LAMBDA_GRID)
    psd2 = vector_median(proj, files, "psd_radial.csv", "lambda_mm", "PSD2D_mm4", grid=LAMBDA_GRID)
    psd1, psd2 = (d.dropna(axis=1, how="all") for d in (psd1, psd2))
    pdfs = {ch: pdf_median(proj, files, ch) for ch in CHAINS}

    # tabela única: escalares + vetores lado a lado, cada coluna com o prefixo do bloco e o x com unidade
    blocks = [med,
              ltx.add_prefix("Ltx [dB re 1 µm] | λ = ").add_suffix(" mm"),
              psd1.add_prefix("PSD 1D profile [mm³] | λ = ").add_suffix(" mm"),
              psd2.add_prefix("PSD radial 3D [mm⁴] | λ = ").add_suffix(" mm")]
    blocks += [p.add_prefix(f"PDF {ch} [1/mm] | z = ").add_suffix(" mm") for ch, p in pdfs.items() if len(p)]
    all_data = pd.concat([b for b in blocks if len(b)], axis=1)
    all_data.index.name = "surface"

    leg = pd.DataFrame(legend, columns=["column", "type", "field in resumo.json", "description"])
    notes = pd.DataFrame({"note": [
        f"Project: {proj.name} · core {', '.join(proj.meta.get('versoes_nos_resultados', []))} · {len(s)} files.",
        "One row per surface = section + pavement (the 4 ISO tracks are separate surfaces); value = median of the "
        "measuring points (MPs).",
        "Sheet 'All_data': every median in one table (scalars, L_tx, PSD 1D, radial PSD, PDFs side by side). "
        "The other sheets hold the same data split by type.",
        "3D SL5: surface with S 0.05 mm + F plane + L 5 mm (aggregate texture, 0.05–5 mm). "
        "3D MICRO: F plane + L 0.5 mm, no digital S (provisional: bandwidth outside ISO 25178-3 Table 1).",
        "'profile' parameters are defined by standards for profiles only (ISO 13473-1/-2, ISO 13565-2, "
        "DIN ISO 10844); kept as the normative reference.",
        "Percentiles P50–P95: heights relative to the surface mean; P90/P70 describes the upper tail of the PDF.",
        "L_tx per third octave (ISO 13473-4) is a profile vector: sheet 'Ltx_third_octave' (dB re 1 µm, "
        "columns = centre λ in mm).",
        "PSD 1D: profile along the road (mm³). Radial PSD: SF surface in 2D, radial average (mm⁴). Columns = λ in mm "
        "(60 log-spaced values; each file interpolated in log-log before the median).",
        "The 3D g-factor is higher than the profile g: each 100 mm × width patch has ~85× more points than a profile, "
        "so z_min is deeper and z_mid drops; use the profile g as the normative value.",
        "PDF: height density (1/mm), resampled on a common grid; columns = height in mm.",
    ]})
    with pd.ExcelWriter(out) as xw:
        notes.to_excel(xw, sheet_name="Read_me", index=False)
        leg.to_excel(xw, sheet_name="Read_me", index=False, startrow=len(notes) + 2)
        all_data.to_excel(xw, sheet_name="All_data")
        med.to_excel(xw, sheet_name="Medians")
        for name, df in (("Ltx_third_octave", ltx), ("PSD_1D_profile", psd1), ("PSD_radial_3D", psd2)):
            if len(df):
                df.to_excel(xw, sheet_name=name)
        for ch, pdf in pdfs.items():
            if len(pdf):
                pdf.to_excel(xw, sheet_name=f"PDF_{ch}")
        data.sort_values(["surface", "file"]).to_excel(xw, sheet_name="Data_per_MP", index=False)
    print(f"{out}  ({len(med)} surfaces, {len(num)} scalar parameters, {all_data.shape[1]} columns in All_data)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
exportar_superficies.py – Exporta de um projeto TextureLab (pasta de resultados) um pacote pequeno e
autocontido de superfícies, para uso fora do TextureLab (por exemplo, o simulador visual do pavimento).

É o ÚNICO ponto de contato entre o TextureLab e o simulador: o simulador lê só este JSON, nunca a pasta
de resultados. Só lê o projeto; não calcula descritores.

De cada superfície (trecho + revestimento) entra UMA medição: o ponto de medição cujo MPD é o mais
próximo da mediana da superfície (empate: o menor MP). Cada superfície leva:

  - a superfície SL5 (visualizacao.npz, SL5_previa) recortada no centro, reduzida pela média de blocos
    e gravada em inteiros de 16 bits (1 µm), em base64 (z_b64, nx, ny, passo_mm, escala_um);
  - os perfis (chave "perfis"): o perfil nativo de 100 mm, os perfis da cadeia A (bruto limpo e passa-baixa,
    5 faixas) e as curvas de Abbott e histogramas das cadeias MICRO, SF e SL5;
  - os parâmetros oficiais da medição inteira (MPD, g, Rsk, Sq, Ssk, ...).

Formato "texturelab-superficies", versão 1 (descrito em docs/FORMATO_SUPERFICIES.md).

    python tools/exportar_superficies.py dados/Resultados_v3.5 [-o superficies.json]
"""
import argparse
import base64
import json
import sys
import warnings
from pathlib import Path

import numpy as np

FORMATO = "texturelab-superficies"
VERSAO_FORMATO = 1
ESCALA_UM = 1.0                                   # 1 unidade int16 = 1 µm (faixa ±32,7 mm)
SEM_DADO = -32768                                 # int16 reservado: ponto sem valor (só nos perfis)
CADEIAS_ABBOTT = ("MICRO", "SF", "SL5")

# Parâmetros oficiais (da medição inteira) gravados junto de cada superfície.
PARAMETROS = ["MPD", "g_factor_media", "perfil_Rsk_media", "SL5_Sq", "SL5_Ssk", "SL5_Sku",
              "SL5_Sk", "SL5_Spk", "SL5_Svk", "SF_Sq", "H_macro"]


def escolher_medicoes(projeto: dict, pasta: Path) -> list[dict]:
    """Uma medição por superfície: a de MPD mais próximo da mediana (empate: o menor MP)."""
    grupos: dict[tuple[str, str], list[dict]] = {}
    for a in projeto["arquivos"]:
        if a.get("status") != "ok" or not a.get("visualizacao"):
            continue
        resumo = json.loads((pasta / a["pasta"] / "resumo.json").read_text(encoding="utf-8"))
        grupos.setdefault((a["trecho"], a["revestimento"]), []).append({**a, "resumo": resumo})
    escolhidas = []
    for chave in sorted(grupos):
        itens = sorted(grupos[chave], key=lambda i: i["mp"])
        mpd = np.array([i["resumo"].get("MPD", np.nan) for i in itens], dtype=float)
        if np.all(np.isnan(mpd)):
            escolhidas.append(itens[0])
            continue
        escolhidas.append(itens[int(np.nanargmin(np.abs(mpd - np.nanmedian(mpd))))])
    return escolhidas


def reduzir(z: np.ndarray, fator: int) -> np.ndarray:
    """Média de blocos fator × fator (ignora NaN); descarta a sobra das bordas."""
    if fator <= 1:
        return z
    ny, nx = (z.shape[0] // fator) * fator, (z.shape[1] // fator) * fator
    blocos = z[:ny, :nx].reshape(ny // fator, fator, nx // fator, fator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # bloco só de NaN: fica NaN, tratado depois
        return np.nanmean(blocos, axis=(1, 3))


def preparar_superficie(z: np.ndarray, passo_mm: float, comprimento_mm: float, fator: int) -> dict:
    """Recorte central no sentido da via + redução + int16. z tem forma (largura, via), em mm."""
    z = np.asarray(z, dtype=np.float64)
    n_via = min(z.shape[1], int(round(comprimento_mm / passo_mm)))
    ini = (z.shape[1] - n_via) // 2
    z = reduzir(z[:, ini:ini + n_via], fator)
    invalidos = ~np.isfinite(z)
    frac_invalidos = float(invalidos.mean())
    if invalidos.all():
        raise ValueError("superfície sem pontos válidos")
    z = z - np.nanmean(z)
    z[invalidos] = np.nanmin(z)                    # buraco sem dado vira fundo, não pico
    q = np.clip(np.round(z * 1000.0 / ESCALA_UM), -32767, 32767).astype("<i2")
    return {"nx": int(q.shape[1]), "ny": int(q.shape[0]), "passo_mm": float(passo_mm * fator),
            "escala_um": ESCALA_UM, "frac_invalidos": frac_invalidos,
            "z_b64": base64.b64encode(q.tobytes()).decode("ascii")}


def decodificar(sup: dict) -> np.ndarray:
    """Inverso de preparar_superficie: matriz (ny, nx) em mm."""
    q = np.frombuffer(base64.b64decode(sup["z_b64"]), dtype="<i2").reshape(sup["ny"], sup["nx"])
    return q.astype(np.float64) * sup["escala_um"] / 1000.0


def codificar_perfil(p: np.ndarray) -> dict:
    """Perfil (1-D ou 2-D) em mm → int16 em µm, base64. NaN vira SEM_DADO."""
    p = np.asarray(p, dtype=np.float64)
    q = np.where(np.isfinite(p), np.clip(np.round(p * 1000.0 / ESCALA_UM), -32767, 32767), SEM_DADO)
    return {"forma": list(p.shape), "escala_um": ESCALA_UM, "sem_dado": SEM_DADO,
            "b64": base64.b64encode(q.astype("<i2").tobytes()).decode("ascii")}


def decodificar_perfil(d: dict) -> np.ndarray:
    """Inverso de codificar_perfil; ponto sem dado volta como NaN. Valores em mm."""
    q = np.frombuffer(base64.b64decode(d["b64"]), dtype="<i2").reshape(d["forma"])
    p = q.astype(np.float64) * d["escala_um"] / 1000.0
    p[q == d["sem_dado"]] = np.nan
    return p


def _lista(a: np.ndarray, casas: int) -> list:
    return np.round(np.asarray(a, dtype=np.float64), casas).tolist()


def montar_perfis(v) -> dict:
    """Perfis e curvas do visualizacao.npz (v: NpzFile). Chaves ausentes (formatos antigos) são omitidas."""
    perfis: dict = {}
    if "nativo_perfil" in v.files:
        perfis["nativo"] = {"passo_mm": float(v["nativo_passo_mm"]), "y_mm": float(v["nativo_y_mm"]),
                            "inicio_mm": float(v["nativo_inicio_mm"]), **codificar_perfil(v["nativo_perfil"])}
    if "A_perfil_limpo" in v.files:
        perfis["cadeia_A"] = {"passo_mm": float(v["A_perfil_passo_mm"]),
                              "faixas": [int(i) for i in v["A_perfil_faixa"]],
                              "y_mm": _lista(v["A_perfil_y_mm"], 4),
                              "limpo": codificar_perfil(v["A_perfil_limpo"]),
                              "passa_baixa": codificar_perfil(v["A_perfil_passa_baixa"])}
    abbott = {}
    for c in CADEIAS_ABBOTT:
        if f"{c}_abbott_mr_pct" in v.files:
            abbott[c] = {"mr_pct": _lista(v[f"{c}_abbott_mr_pct"], 3),
                         "altura_mm": _lista(v[f"{c}_abbott_altura_mm"], 5),
                         "hist_bordas_mm": _lista(v[f"{c}_hist_bordas_mm"], 5),
                         "hist_contagem": [int(i) for i in v[f"{c}_hist_contagem"]]}
    if abbott:
        perfis["abbott"] = abbott
    return perfis


def montar_dados(pasta: Path, comprimento_mm: float, fator: int) -> dict:
    projeto = json.loads((pasta / "projeto.json").read_text(encoding="utf-8"))
    superficies = []
    for a in escolher_medicoes(projeto, pasta):
        with np.load(pasta / a["pasta"] / "visualizacao.npz", allow_pickle=False) as v:
            if "SL5_previa" not in v.files:
                print(f"  sem SL5_previa, pulado: {a['arquivo']}", file=sys.stderr)
                continue
            z = v["SL5_previa"]
            perfis = montar_perfis(v)
        r = a["resumo"]
        passo = float(r["dx_mm"]) * int(projeto["config"]["preview_step"])
        sup = preparar_superficie(z, passo, comprimento_mm, fator)
        sup.update({"arquivo": a["arquivo"], "trecho": a["trecho"], "revestimento": a["revestimento"],
                    "mp": a["mp"], "data": a["data"],
                    "parametros": {k: (float(r[k]) if isinstance(r.get(k), (int, float)) else None)
                                   for k in PARAMETROS},
                    "perfis": perfis})
        superficies.append(sup)
        print(f"  {a['trecho']:6s} {a['revestimento']:7s} {a['mp']}  {sup['ny']} × {sup['nx']} pontos")
    if not superficies:
        raise SystemExit("Nenhuma medição com SL5_previa no projeto.")
    return {"formato": FORMATO, "versao_formato": VERSAO_FORMATO, "projeto": pasta.name,
            "versao_nucleo": projeto.get("versao_nucleo"), "cadeia": "SL5", "superficies": superficies}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto", type=Path, help="pasta de resultados (com projeto.json)")
    ap.add_argument("-o", "--saida", type=Path, default=None,
                    help="arquivo JSON (padrão: <projeto>_superficies.json ao lado da pasta)")
    ap.add_argument("--comprimento-mm", type=float, default=100.0, help="recorte no sentido da via (padrão 100)")
    ap.add_argument("--reducao", type=int, default=2, help="média de blocos N × N da prévia SL5 (padrão 2)")
    args = ap.parse_args(argv)

    pasta = args.projeto.resolve()
    if not (pasta / "projeto.json").exists():
        raise SystemExit(f"{pasta} não tem projeto.json (use a pasta de resultados, não o .tlproj).")
    saida = args.saida or pasta.with_name(pasta.name + "_superficies.json")
    dados = montar_dados(pasta, args.comprimento_mm, args.reducao)
    saida.write_text(json.dumps(dados, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{saida}  ({saida.stat().st_size / 1e6:.1f} MB, {len(dados['superficies'])} superfícies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
test_exportar_superficies.py – Exportador de superfícies para o simulador (tools/exportar_superficies.py).

Usa um projeto sintético mínimo (só projeto.json, resumo.json e visualizacao.npz).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))

import exportar_superficies as ex  # noqa: E402

DX, PASSO_PREVIA = 0.011, 8


def _projeto(pasta: Path, mpds=(0.8, 1.0, 1.3), forma=(60, 400), completo=True) -> Path:
    """Projeto com uma superfície (T1 / SMA) e um MP por valor de MPD."""
    rng = np.random.default_rng(0)
    arquivos = []
    for i, mpd in enumerate(mpds, start=1):
        nome = f"T1_SMA_MP{i}"
        od = pasta / "T1" / nome
        od.mkdir(parents=True)
        dados = {"SL5_previa": rng.normal(0, 0.3, forma).astype(np.float16) + np.float16(i)}
        if completo:
            nativo = rng.normal(0, 0.5, 900).astype(np.float32)
            nativo[10] = np.nan
            dados.update(
                nativo_perfil=nativo, nativo_passo_mm=DX, nativo_y_mm=5.0, nativo_inicio_mm=0.0,
                A_perfil_faixa=np.arange(5, dtype=np.int32), A_perfil_y_mm=np.linspace(0.2, 10, 5, dtype=np.float32),
                A_perfil_passo_mm=0.5, A_perfil_limpo=rng.normal(0, 1, (5, 50)).astype(np.float32),
                A_perfil_passa_baixa=rng.normal(0, 1, (5, 50)).astype(np.float32),
                SL5_abbott_mr_pct=np.linspace(0, 100, 201, dtype=np.float32),
                SL5_abbott_altura_mm=np.linspace(1, -1, 201, dtype=np.float32),
                SL5_hist_bordas_mm=np.linspace(-1, 1, 121, dtype=np.float32),
                SL5_hist_contagem=np.arange(120, dtype=np.int64))
        np.savez_compressed(od / "visualizacao.npz", **dados)
        (od / "resumo.json").write_text(json.dumps({"dx_mm": DX, "MPD": mpd, "SL5_Sq": 0.3, "status": "ok"}))
        arquivos.append({"arquivo": nome, "trecho": "T1", "revestimento": "SMA", "mp": f"MP{i}", "nr": "NR01",
                         "data": "20240101", "pasta": f"T1/{nome}", "status": "ok", "visualizacao": True})
    (pasta / "projeto.json").write_text(json.dumps({
        "formato": "texturelab-projeto", "versao_formato": 1, "versao_nucleo": "9.9.9",
        "config": {"preview_step": PASSO_PREVIA}, "arquivos": arquivos}))
    return pasta


def test_superficie_ida_e_volta():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 0.4, (40, 300))
    z[3, 100] = np.nan                                              # coluna 7 do recorte
    sup = ex.preparar_superficie(z, 0.088, comprimento_mm=10.0, fator=1)
    assert (sup["ny"], sup["nx"]) == (40, 114)                      # 10 mm / 0,088 mm = 114 pontos
    volta = ex.decodificar(sup)
    recorte = z[:, 93:93 + 114]
    esperado = recorte - np.nanmean(recorte)
    ok = np.isfinite(esperado)
    assert np.abs(volta[ok] - esperado[ok]).max() <= 0.0005 + 1e-9  # meio passo de quantização (1 µm)
    assert volta[3, 7] == pytest.approx(volta.min())                 # ponto sem dado vira fundo
    assert sup["frac_invalidos"] == pytest.approx(1 / (40 * 114))


def test_reducao_por_blocos():
    z = np.arange(24, dtype=float).reshape(4, 6)
    assert np.allclose(ex.reduzir(z, 2), [[3.5, 5.5, 7.5], [15.5, 17.5, 19.5]])
    sup = ex.preparar_superficie(z, 0.1, comprimento_mm=100.0, fator=2)
    assert (sup["ny"], sup["nx"], sup["passo_mm"]) == (2, 3, pytest.approx(0.2))


def test_perfil_ida_e_volta_com_nan():
    p = np.array([[0.0, 1.2345, np.nan], [-2.0, 0.0004, 3.0]])
    volta = ex.decodificar_perfil(ex.codificar_perfil(p))
    assert volta.shape == p.shape
    assert np.isnan(volta[0, 2]) and not np.isnan(volta[1]).any()
    ok = np.isfinite(p)
    assert np.abs(volta[ok] - p[ok]).max() <= 0.0005 + 1e-9


def test_escolhe_mp_da_mediana(tmp_path):
    pasta = _projeto(tmp_path / "Proj")
    projeto = json.loads((pasta / "projeto.json").read_text())
    escolhidas = ex.escolher_medicoes(projeto, pasta)
    assert [e["mp"] for e in escolhidas] == ["MP2"]                  # MPD 1,0 é a mediana de 0,8 / 1,0 / 1,3


def test_exporta_pacote_completo(tmp_path):
    pasta = _projeto(tmp_path / "Proj")
    saida = tmp_path / "pacote.json"
    assert ex.main([str(pasta), "-o", str(saida), "--comprimento-mm", "20"]) == 0
    d = json.loads(saida.read_text(encoding="utf-8"))
    assert (d["formato"], d["versao_formato"], d["versao_nucleo"]) == (ex.FORMATO, 1, "9.9.9")
    assert len(d["superficies"]) == 1
    s = d["superficies"][0]
    assert s["mp"] == "MP2" and s["parametros"]["MPD"] == 1.0 and s["parametros"]["SL5_Ssk"] is None
    assert s["passo_mm"] == pytest.approx(DX * PASSO_PREVIA * 2)
    assert ex.decodificar(s).shape == (30, 113)                      # 60/2 × (20 mm / 0,088 mm)/2
    p = s["perfis"]
    nativo = ex.decodificar_perfil(p["nativo"])
    assert nativo.shape == (900,) and np.isnan(nativo[10]) and p["nativo"]["passo_mm"] == DX
    assert ex.decodificar_perfil(p["cadeia_A"]["limpo"]).shape == (5, 50)
    assert p["cadeia_A"]["passo_mm"] == 0.5 and len(p["cadeia_A"]["y_mm"]) == 5
    ab = p["abbott"]["SL5"]
    assert len(ab["mr_pct"]) == len(ab["altura_mm"]) == 201 and len(ab["hist_contagem"]) == 120
    assert set(p["abbott"]) == {"SL5"}                               # só as cadeias presentes


def test_formato_antigo_sem_perfis(tmp_path):
    """Resultado sem perfis no npz exporta a superfície mesmo assim, com "perfis" vazio."""
    pasta = _projeto(tmp_path / "Proj", completo=False)
    dados = ex.montar_dados(pasta, 20.0, 2)
    assert dados["superficies"][0]["perfis"] == {}


def test_recusa_tlproj_e_pasta_sem_projeto(tmp_path):
    with pytest.raises(SystemExit):
        ex.main([str(tmp_path)])

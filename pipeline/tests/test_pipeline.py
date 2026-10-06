"""
test_pipeline.py – Pipeline em lote, receita e leitura do projeto pelo app.

Usa um LAZ sintético pequeno (220 × 20 mm, dx = 0,05 mm). Um teste opcional confere um arquivo real
contra resultados já calculados: defina TEXTURELAB_REF_LAZ (arquivo .laz) e TEXTURELAB_REF_DIR
(pasta com o resumo.json desse arquivo, ex.: Resultados_v2/A1/<arquivo>).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "TextureLab"))

import texturelab_batch as tb  # noqa: E402
from components.project_reader import Project  # noqa: E402

laspy = pytest.importorskip("laspy")


def _write_laz(path: Path, seed: int = 0, nx: int = 4400, ny: int = 400, dx: float = 0.05):
    """Superfície auto-afim sintética gravada como grade completa (Y mais rápido), em mm."""
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(ny, dx)[:, None]
    fx = np.fft.rfftfreq(nx, dx)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1.0
    spec = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) * f ** -1.6
    spec[0, 0] = 0
    z = np.fft.irfft2(spec, s=(ny, nx))
    z = z / z.std() * 0.4                                    # ~0,4 mm rms
    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")    # [x, y], Y mais rápido
    hdr = laspy.LasHeader(point_format=0, version="1.2")
    hdr.scales = np.array([1e-3, 1e-3, 1e-6])
    hdr.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(hdr)
    las.X = (xi.ravel() * 50).astype(np.int32)              # 50 × 1e-3 = 0,05 mm
    las.Y = (yi.ravel() * 50).astype(np.int32)
    las.Z = np.round(z.T.ravel() / 1e-6).astype(np.int32)
    las.write(str(path))


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    d = tmp_path_factory.mktemp("proj")
    laz = d / "laz"
    laz.mkdir()
    for i, name in enumerate(["T1xx_AC_MP1_3DT_NR01_20240101", "T1xx_AC_MP2_3DT_NR01_20240101"]):
        _write_laz(laz / f"{name}.laz", seed=i)
    out = d / "out"
    cfg = tb.load_config(None)
    for f in sorted(laz.glob("*.laz")):
        r = tb.process_file(str(f), str(out), cfg)
        assert r["status"] == "ok", r.get("erro")
    tb.write_project_index(out)
    return {"laz": laz, "out": out, "cfg": cfg}


def test_results_and_viewer_data(project):
    od = project["out"] / "T1" / "T1xx_AC_MP1_3DT_NR01_20240101"
    r = json.loads((od / "resumo.json").read_text(encoding="utf-8"))
    assert r["versao_script"] == tb.VERSION
    assert r["receita"]["hash"] == r["receita_hash"]
    assert 0 < r["MPD"] < 5 and r["ETD"] == pytest.approx(1.1 * r["MPD"])
    assert r["A_n_segmentos_validos"] > 0
    assert "H_macro" in r and "SL5_Sq" in r and "MICRO_Sq" in r
    for name in ("cadeiaA_segmentos.csv", "espectro_terco_oitava.csv", "psd_media.csv", "previa.npz",
                 "visualizacao.npz"):
        assert (od / name).exists(), name
    with np.load(od / "visualizacao.npz", allow_pickle=False) as v:
        assert int(v["versao_formato"]) == tb.FORMAT_VERSION
        for ch in ("SF", "SL5", "MICRO"):
            mr, h = v[f"{ch}_abbott_mr_pct"], v[f"{ch}_abbott_altura_mm"]
            assert mr[0] == 0 and mr[-1] == 100 and np.all(np.diff(h) <= 1e-6)    # curva decrescente
            assert v[f"{ch}_hist_contagem"].sum() > 0
        assert v["A_perfil_limpo"].shape == v["A_perfil_passa_baixa"].shape
        assert v["SL5_previa"].ndim == 2


def test_recipe_decides_recalculation(project, tmp_path):
    f = str(sorted(project["laz"].glob("*.laz"))[0])
    out = project["out"]
    assert tb.is_current(out, f, tb.recipe(f, project["cfg"]))
    cfg2 = dict(project["cfg"], ETD_factor=1.2)
    assert not tb.is_current(out, f, tb.recipe(f, cfg2))                # configuração mudou
    copy = tmp_path / Path(f).name
    copy.write_bytes(Path(f).read_bytes()[:-10] + b"0123456789")
    assert tb.recipe(str(copy), project["cfg"])["hash"] != tb.recipe(f, project["cfg"])["hash"]  # LAZ mudou


def test_load_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"ETD_fator": 1.2}', encoding="utf-8")
    with pytest.raises(SystemExit):
        tb.load_config(str(p))
    p.write_text('{"ETD_factor": 1.2}', encoding="utf-8")
    assert tb.load_config(str(p))["ETD_factor"] == 1.2
    assert tb.CFG["ETD_factor"] == 1.1                                 # padrão intacto


def test_project_folder_and_zip_read_the_same(project):
    out = project["out"]
    meta = json.loads((out / "projeto.json").read_text(encoding="utf-8"))
    assert meta["n_arquivos"] == 2 and meta["n_ok"] == 2 and all(i["visualizacao"] for i in meta["arquivos"])
    pkg = tb.pack_project(out)
    a, b, c = Project(out), Project(pkg), Project(pkg.read_bytes(), name=pkg.name)
    for p in (a, b, c):
        assert p.warnings == []
        assert list(p.index["arquivo"]) == list(a.index["arquivo"])
    arq = a.index["arquivo"][0]
    assert a.summary().equals(b.summary()) and b.summary().equals(c.summary())
    va, vb = a.view(arq), b.view(arq)
    assert va.keys() == vb.keys() and all(np.array_equal(va[k], vb[k]) for k in va)
    assert a.table(arq, "espectro_terco_oitava.csv").equals(b.table(arq, "espectro_terco_oitava.csv"))
    z, step = b.preview(arq)
    assert z.ndim == 2 and step > 0


def test_old_results_without_index(project, tmp_path):
    """Pasta antiga (sem projeto.json nem visualizacao.npz) abre com aviso, só leitura."""
    src = project["out"] / "T1" / "T1xx_AC_MP1_3DT_NR01_20240101"
    dst = tmp_path / "old" / "T1" / src.name
    dst.mkdir(parents=True)
    r = json.loads((src / "resumo.json").read_text(encoding="utf-8"))
    r["versao_script"] = "2.0.0"
    (dst / "resumo.json").write_text(json.dumps(r), encoding="utf-8")
    p = Project(tmp_path / "old")
    assert len(p.index) == 1 and not p.index["visualizacao"][0]
    assert any("2.0.0" in w for w in p.warnings)
    assert p.view(p.index["arquivo"][0]) == {}
    assert not (tmp_path / "old" / "projeto.json").exists()             # o app não grava nada


def test_reader_never_unpickles(project, tmp_path):
    out = project["out"]
    arq = Project(out).index["arquivo"][0]
    bad = tmp_path / "bad"
    import shutil
    shutil.copytree(out, bad)
    folder = bad / Project(bad).index["pasta"][0]
    np.savez(folder / "visualizacao.npz", x=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):
        Project(bad).view(arq)


@pytest.mark.skipif(not (os.environ.get("TEXTURELAB_REF_LAZ") and os.environ.get("TEXTURELAB_REF_DIR")),
                    reason="defina TEXTURELAB_REF_LAZ e TEXTURELAB_REF_DIR para o teste com arquivo real")
def test_real_file_matches_reference(tmp_path):
    laz, ref_dir = os.environ["TEXTURELAB_REF_LAZ"], Path(os.environ["TEXTURELAB_REF_DIR"])
    r = tb.process_file(laz, str(tmp_path), tb.load_config(None))
    ref = json.loads((ref_dir / "resumo.json").read_text(encoding="utf-8"))
    assert r["status"] == "ok"
    for k, v in ref.items():
        if isinstance(v, float) and not k.endswith("_s") and k != "pico_memoria_processo_gb":
            assert r[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k

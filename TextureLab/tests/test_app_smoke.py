"""
test_app_smoke.py – Roda o visualizador (app.py) sem navegador sobre um projeto real e confere que nenhuma
tela quebra, em EN/DE/PT.

O projeto é grande demais para o repositório, então o teste só roda com TEXTURELAB_TEST_PROJECT apontando
para uma pasta de resultados ou um .tlproj, por exemplo:

    TEXTURELAB_TEST_PROJECT=/data/.../Resultados_v3.4.1.tlproj pytest TextureLab/tests/test_app_smoke.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

PROJECT = os.environ.get("TEXTURELAB_TEST_PROJECT")
pytestmark = pytest.mark.skipif(not PROJECT, reason="defina TEXTURELAB_TEST_PROJECT para o teste do visualizador")

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def project():
    from components.project_reader import Project
    return Project(PROJECT)


def _run(project, view: str, lang: str) -> "AppTest":
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=300)
    at.session_state["projeto"] = project
    at.session_state["proj_token"] = 1
    at.session_state["view"] = view
    at.query_params["lang"] = lang
    at.run()
    return at


@pytest.mark.parametrize("lang", ["en", "de", "pt"])
@pytest.mark.parametrize("view", ["summary", "file", "compare", "stats", "server"])
def test_views_render_without_errors(project, view, lang):
    at = _run(project, view, lang)
    assert not at.exception, [e.value for e in at.exception]


def test_radial_psd_chart_matches_summary(project):
    """Com núcleo >= 3.3.0 o primeiro arquivo tem psd_radial.csv: o gráfico aparece e a legenda traz o
    mesmo H2D do resumo."""
    at = _run(project, "file", "en")
    first = at.selectbox(key="file_sel").value
    if project.table(first, "psd_radial.csv") is None:
        pytest.skip("projeto sem psd_radial.csv (núcleo anterior à 3.3.0)")
    specs = [json.loads(c.proto.spec) for c in at.get("plotly_chart")]
    radial = [s for s in specs if "PSD 2D" in json.dumps(s.get("layout", {}).get("yaxis", {}))]
    assert len(radial) == 1
    names = " ".join(str(tr.get("name")) for tr in radial[0]["data"])
    r = project.resumo(first)
    if r.get("H2D_macro") is not None:
        assert f"H={r['H2D_macro']:.3f}" in names

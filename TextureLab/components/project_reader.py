"""
project_reader.py – Leitura de projetos calculados pelo pipeline (pipeline/texturelab_batch.py).

O app não calcula: só lê o que o servidor gravou. Um projeto é a pasta de saída do lote
ou o mesmo conteúdo empacotado em um zip (.tlproj). Formato em docs/FORMATO_PROJETO.md.

Só JSON, CSV e NPZ (np.load com allow_pickle=False): nenhum código é executado ao abrir.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

SUPPORTED_FORMAT = 1
INDEX_COLUMNS = ["arquivo", "trecho", "revestimento", "mp", "nr", "data", "pasta", "status",
                 "versao_nucleo", "receita_hash", "visualizacao"]


class ProjectError(Exception):
    """Erro de abertura com código traduzível (components/i18n.py) e valores para o texto."""

    def __init__(self, key: str, **kw):
        super().__init__(key, kw)
        self.key, self.kw = key, kw


class Project:
    """Acesso somente leitura a um projeto (pasta ou zip)."""

    def __init__(self, source: str | Path | bytes, name: str | None = None):
        self._zip = None
        self._root = None
        if isinstance(source, (bytes, bytearray)):
            self._zip = zipfile.ZipFile(io.BytesIO(source))
            self.name = name or "projeto.tlproj"
        else:
            p = Path(source).expanduser()
            if p.is_file() and zipfile.is_zipfile(p):
                self._zip = zipfile.ZipFile(p)
            elif p.is_dir():
                self._root = p
            else:
                raise ProjectError("err_not_project", path=str(p))
            self.name = name or p.name
        if self._zip is not None:
            self._names = set(self._zip.namelist())
        self.meta = self._load_index()
        self.index = pd.DataFrame(self.meta["arquivos"], columns=INDEX_COLUMNS)
        self._summary = None

    # ── acesso a arquivos ──────────────────────────────────────────────
    def _exists(self, rel: str) -> bool:
        if self._zip is not None:
            return rel in self._names
        return (self._root / rel).is_file()

    def _read(self, rel: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(rel)
        return (self._root / rel).read_bytes()

    def _rglob_resumo(self) -> list[str]:
        if self._zip is not None:
            return sorted(n for n in self._names if PurePosixPath(n).name == "resumo.json")
        return sorted(p.relative_to(self._root).as_posix() for p in self._root.rglob("resumo.json"))

    def _load_index(self) -> dict:
        if self._exists("projeto.json"):
            meta = json.loads(self._read("projeto.json"))
            if meta.get("formato") != "texturelab-projeto":
                raise ProjectError("err_bad_index")
            if int(meta.get("versao_formato", 0)) > SUPPORTED_FORMAT:
                raise ProjectError("err_format_newer", fmt=meta["versao_formato"], sup=SUPPORTED_FORMAT)
            return meta
        # Resultados antigos sem projeto.json: monta o índice só para leitura.
        items = []
        for rel in self._rglob_resumo():
            try:
                d = json.loads(self._read(rel))
            except Exception:  # noqa: BLE001
                continue
            folder = str(PurePosixPath(rel).parent)
            items.append({"arquivo": d.get("arquivo", PurePosixPath(folder).name), "trecho": d.get("trecho", ""),
                          "revestimento": d.get("revestimento", ""), "mp": d.get("mp", ""), "nr": d.get("nr", ""),
                          "data": d.get("data", ""), "pasta": folder, "status": d.get("status", ""),
                          "versao_nucleo": d.get("versao_script", ""), "receita_hash": d.get("receita_hash", ""),
                          "visualizacao": self._exists(f"{folder}/visualizacao.npz")})
        if not items:
            raise ProjectError("err_no_results")
        versions = sorted({i["versao_nucleo"] for i in items if i["versao_nucleo"]})
        return {"formato": "texturelab-projeto", "versao_formato": 0, "versao_nucleo": "",
                "versoes_nos_resultados": versions, "config": {}, "n_arquivos": len(items),
                "n_ok": sum(i["status"] == "ok" for i in items), "arquivos": items,
                "sem_indice": True}

    def _folder(self, arquivo: str) -> str:
        row = self.index.loc[self.index["arquivo"] == arquivo]
        if row.empty:
            raise KeyError(arquivo)
        return row.iloc[0]["pasta"]

    # ── conteúdo ───────────────────────────────────────────────────────
    @property
    def warnings(self) -> list[tuple[str, dict]]:
        """Avisos como (código, valores), traduzidos pelo app (components/i18n.py)."""
        w = []
        if self.meta.get("sem_indice"):
            w.append(("warn_no_index", {}))
        vers = self.meta.get("versoes_nos_resultados", [])
        if len(vers) > 1:
            w.append(("warn_mixed", {}))
        if any(v and v < "3" for v in vers):
            w.append(("warn_old_core", {"vers": ", ".join(vers)}))
        n_err = int((self.index["status"] != "ok").sum())
        if n_err:
            w.append(("warn_errors", {"n": n_err}))
        return w

    def summary(self) -> pd.DataFrame:
        """Uma linha por arquivo com todos os parâmetros escalares (resumo.json)."""
        if self._summary is None:
            rows = []
            for arq in self.index["arquivo"]:
                d = self.resumo(arq)
                rows.append({k: v for k, v in d.items() if not isinstance(v, (dict, list))})
            df = pd.DataFrame(rows)
            if len(df):
                df = df.sort_values([c for c in ("trecho", "mp", "arquivo") if c in df]).reset_index(drop=True)
            self._summary = df
        return self._summary

    def resumo(self, arquivo: str) -> dict:
        return json.loads(self._read(f"{self._folder(arquivo)}/resumo.json"))

    def has(self, arquivo: str, filename: str) -> bool:
        return self._exists(f"{self._folder(arquivo)}/{filename}")

    def table(self, arquivo: str, filename: str) -> pd.DataFrame | None:
        rel = f"{self._folder(arquivo)}/{filename}"
        if not self._exists(rel):
            return None
        return pd.read_csv(io.BytesIO(self._read(rel)))

    def arrays(self, arquivo: str, filename: str) -> dict | None:
        rel = f"{self._folder(arquivo)}/{filename}"
        if not self._exists(rel):
            return None
        with np.load(io.BytesIO(self._read(rel)), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def preview(self, arquivo: str) -> tuple[np.ndarray, float] | None:
        a = self.arrays(arquivo, "previa.npz")
        if a is None:
            return None
        return a["z"].astype(np.float32), float(a["passo_mm"])

    def view(self, arquivo: str) -> dict:
        return self.arrays(arquivo, "visualizacao.npz") or {}

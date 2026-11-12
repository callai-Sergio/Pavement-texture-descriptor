# TextureLab — resumo do que foi feito e próximos passos

Levantamento de 2026-11-10 no servidor videolabor-01 (`/data/callai/workspace/tyron`).
Repositório: `callai-Sergio/Pavement-texture-descriptor`.

---

## 1. Estado atual em uma olhada

| Item | Estado |
|---|---|
| `master` no GitHub | **v3.4.1** (`ae1cb3e`, PR #7 mesclado em 2026-11-10) |
| Código mais novo | `master`, **core 3.4.1** |
| PRs | nenhum aberto. #7 mesclado; #5 e #6 fechados como incluídos no #7 |
| Visualizador no servidor | `texturelab_repo` em detached HEAD no `origin/master` (`38f6a3f`, visualizador 3.4.2, núcleo 3.4.1). **O serviço `texturelab-app` não existe mais** (nenhuma unidade systemd de usuário, nada escutando na 8501); a visualização é feita no PC |
| Últimos resultados | **`Resultados_v3.4.1`** + `.tlproj`: 80/80 OK com o core 3.4.1, sem ML5 (2026-11-10) |
| Excel final | **`Resultados_v3.4.1_medians.xlsx`**: em inglês, com aba `All_data` |
| Testes do pipeline | **14 passaram, 1 pulado** (o pulado pede um LAZ de referência), rodando com o venv de `texturelab_next` |

---

## 2. O que já foi feito (cronologia)

### v3.0 – v3.2.2 (já no `master`)
- **3.0 / 3.1**: o cálculo passou a rodar só no servidor (`pipeline/servidor.sh`: 4 núcleos, limite de 25 GB, nice/ionice, unidade systemd `texturelab-lote`). O PC só visualiza. O resultado é empacotado em `.tlproj`.
- **Visualizador**: Streamlit em EN/DE/PT, Abbott com Smr1/Smr2, 3D em escala real, agrupamento por pavimento e PCA oculto (`SHOW_PCA = False`).
- **3.2.0**: g-factor conforme DIN ISO 10844:2024 Anexo B, ondaletas (db4), ETD removido, white paper em PDF e correção do nome do `.tlproj` quando a pasta tem ponto.
- **3.2.1 / 3.2.2**: visualização local no PC (`baixar_resultados`, `abrir_app`), usando o disco local em vez de `H:`.
- **Análise do Ssk negativo** (2026-10-06): o cálculo está correto. A1_WB e A1_MALA só mudam de sinal quando se cortam as caudas de 0,5–2 %. A opção de assimetria robusta ficou **adiada** pelo usuário.

### v3.3.0 (PR #5, entrou no `master` via #7)
- PSD radial 2D na SF (`psd_radial.csv`) e Hurst 2D (`H2D_micro`, `H2D_macro`).
- Percentis P50/P70/P90/P95 e a razão P90/P70 por cadeia areal.
- g-factor 3D (`g3D_pct`) em SF e SL5.
- Exportador de medianas para Excel (`tools/exportar_medianas.py`), com PSDs numa grade log comum de 60 pontos.

### v3.4.0 → 3.4.1 (PRs #6 e #7, mesclados via #7 em 2026-11-10)
- **Verificação dos filtros contra as normas**: `tools/verificar_filtros.py`, relatório em `docs/verificacao_filtros/` e testes. Passaram Gauss ISO 16610-61, Butterworth da cadeia A e bandas de 1/3 de oitava IEC 61260-1.
- A cadeia ML5 foi criada na 3.4.0 e **removida na 3.4.1**, porque é praticamente igual à SL5 (Ssk difere em até 0,01).
- Excel em inglês, com a aba `All_data`.

---

## 3. Pendências encontradas no código

### Alta prioridade
1. ~~**Nada da v3.3/3.4 está no `master`.**~~ **Resolvido em 2026-11-10:** #7 mesclado no `master` e #5/#6 fechados.
2. ~~**O visualizador não mostra o que é novo na v3.3.**~~ **Resolvido em 2026-11-10 (PR #9, visualizador 3.4.2).** `PARAM_GROUPS` em `TextureLab/app.py` (linha ~75) não inclui:
   - `*_P50/P70/P90/P95` e `*_P90_P70`;
   - `SF_g3D_pct` e `SL5_g3D_pct`;
   - `H2D_micro/macro`;
   - `psd_radial.csv` (nenhum gráfico).

   Esses valores existem nos resultados e no Excel, mas não aparecem na tela.
3. ~~**Os resultados v3.4 estão marcados como 3.4.0.**~~ **Resolvido em 2026-11-10:** use `Resultados_v3.4.1`. A pasta `Resultados_v3.4` antiga pode ser apagada (passo 6). Ainda têm 27 colunas `ML5_*` no `resumo_geral.csv` e no `.tlproj`. Os descritores são idênticos aos de agora, então a diferença é só de rótulo.

### Média prioridade (documentação e versão)
4. ~~**Versões desalinhadas.**~~ **Resolvido em 2026-11-10 (PR #8).** O `pyproject.toml` diz **3.2.2**, enquanto `pipeline/texturelab_batch.py` diz **3.4.1**. O `CHANGELOG.md` termina na 3.2.2, sem entradas para 3.3.0, 3.4.0 e 3.4.1.
5. ~~**O README está desatualizado.**~~ **Resolvido em 2026-11-10 (PR #10).** O título ainda é "v3.2.0", ainda lista **ETD** (removido na 3.2.0) e destaca o PCA, que está oculto.
6. ~~**O white paper (EN/PT + PDFs) para na 3.2.0.**~~ **Resolvido em 2026-11-10 (PR #10).** Falta descrever:
   - PSD radial e Hurst 2D;
   - percentis e P90/P70;
   - g3D;
   - a verificação dos filtros (dá para citar `docs/verificacao_filtros/`).

### Baixa prioridade (limpeza)
7. ~~**O venv do pipeline em `texturelab_repo` está velho.**~~ **Resolvido em 2026-11-10** (`servidor.sh instalar`). Falta o `pywt` (PyWavelets), e por isso 6 testes falham lá. Se alguém rodar `servidor.sh` dessa pasta, o lote falha. Hoje os lotes rodam de `texturelab_next`, que está OK.
8. **Pasta duplicada versionada `TextureLab/src/GIT LAB TYRON/`** — **resolvido em 2026-11-10 (PR #11).** É uma cópia antiga do `descriptors.py`, da v2.3.0.
9. **Os testes do visualizador (`TextureLab/tests`) não rodam no venv do viewer** — **resolvido em 2026-11-10 (PR #12).** Falta o `numba`, e eles testam o código legado (`descriptors.py`).
10. **Arquivos soltos** (resolvido em 2026-11-10; ficaram só as versões v3.2–v3.4.1):
    - `texturelab_repo/pipeline/Resultados_v3.tlproj` (48 MB, fora do git);
    - `Resultados_v3.3_medianas_PREVIA_PARCIAL.xlsx` (0 bytes);
    - `Resultados~`;
    - cinco `.tlproj` antigos de ~386 MB cada.
12. **O venv do pipeline não tem `openpyxl`** — **resolvido em 2026-11-10 (PR #11)**: `tools/exportar_medianas.py` só roda com o Python do visualizador (`texturelab_repo/TextureLab/.venv/bin/python`). Dá para incluir `openpyxl` em `pipeline/requirements.txt`.
11. ~~**`gh` sem login no servidor**~~ **Resolvido em 2026-11-10** (login refeito como `callai-Sergio`).

---

## 4. Passos sugeridos (em ordem)

| # | Passo | Esforço | Precisa recalcular? |
|---|---|---|---|
| 1 | ✅ **Feito em 2026-11-10**: v3.4.1 mesclada no `master` (#7); #5 e #6 fechados | — | Não |
| 2 | ✅ **Feito em 2026-11-10** (PR #8, `f838057`): versão 3.4.1 em pyproject, visualizador e desktop; CHANGELOG 3.3.0 / 3.4.0 / 3.4.1 | — | Não |
| 3 | ✅ **Feito em 2026-11-10** (PR #9, `38f6a3f`, visualizador 3.4.2): percentis e P90/P70 nos grupos areais, grupos "Hurst 2D" e "g-factor 3D", gráfico da PSD radial | — | Não |
| 4 | ✅ **Feito em 2026-11-10**: `Resultados_v3.4.1` (+ `.tlproj`), 80/80 OK em 37,5 min, idêntico à v3.4 sem as 27 colunas `ML5_*`; Excel `Resultados_v3.4.1_medians.xlsx` (todas as abas iguais às da v3.4, só muda o cabeçalho do Read_me) | — | Sim |
| 5 | ✅ **Feito em 2026-11-10** (PR #10): README v3.4, white paper EN/PT 3.4.1 (seções 3.7–3.9, percentis, verificação de filtros, novos arquivos, campos e equações) e PDFs regerados | — | Não |
| 6 | ✅ **Feito em 2026-11-10** (arquivos): apagados lixo e restos (xlsx 0 bytes, `Resultados`, `Resultados~`, `__pycache__`, `LAZ_lote2`, `LAZ_teste10`, `.tlproj` solto em `texturelab_repo/pipeline`) e as versões v2 e v3 (`Resultados_v2`, `Resultados_v3`, `Resultados_v3.tlproj`), ~1 GB; mantidas v3.2, v3.3, v3.4 e v3.4.1. Venv do pipeline em `texturelab_repo` refeito (14 testes ok). PR #11 mesclado (`e61f1fc`): `GIT LAB TYRON/` removida e `openpyxl` no pipeline; os dois venvs do pipeline foram refeitos | — | Não |
| 7 | ✅ **Feito em 2026-11-10** (PR #12): teste do visualizador sem navegador (`test_app_smoke.py`, todas as telas em EN/DE/PT, com `TEXTURELAB_TEST_PROJECT`) e testes legados pulados sem numba | — | Não |
| 8 | ✅ **Assimetria robusta** (Ssk_trim05, Ssk_trim2, Ssk_bowley; Spk/Svk já existiam): núcleo 3.5.0, PR #13, `Resultados_v3.5` (80/80, 0 erros, demais valores idênticos à v3.4.1) | feito 2026-11-11 | Sim |

O passo 4 só tira as colunas `ML5_*` e acerta o rótulo de versão; o visualizador 3.4.2 já mostra tudo com os resultados atuais.

---

## 5. Fora do escopo deste levantamento
- `/data/callai/workspace/3dpave` e `3dpave_stress` (app Marshall digital) são outro projeto e não foram revisados aqui.

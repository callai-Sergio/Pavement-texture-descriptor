# TextureLab Batch (v3.1.0)

Cálculo em lote, sem interface, dos descritores de textura de pavimento a partir de
varreduras 3dT (LAZ). Núcleo conferido contra as normas ISO 13473-1:2019, 13473-4:2024,
25178-2/-3 e 13565-2, mais Hurst/dimensão fractal (descritivo).

É o **único** núcleo de cálculo. O app (`TextureLab/app.py`) só abre o projeto gerado aqui e desenha
(formato em [docs/FORMATO_PROJETO.md](../docs/FORMATO_PROJETO.md)).

## Fluxo: calcular no servidor, ver no PC

### No servidor (uma vez)

```bash
cd /data/callai/workspace/tyron
git clone -b feature/server-compute https://github.com/callai-Sergio/Pavement-texture-descriptor.git texturelab_repo
cd texturelab_repo/pipeline
./servidor.sh instalar          # venv próprio em pipeline/.venv, versões gravadas em .venv-versoes.txt
loginctl enable-linger sergio   # o cálculo continua depois de sair do SSH
```

O venv é só do pipeline (numpy, scipy, pandas, laspy). Os outros scripts da pasta não mexem nele, e
os números não mudam por atualização de biblioteca feita para outra coisa. Para atualizar o código: `git pull`.

### Calcular, com uso do servidor controlado

```bash
./servidor.sh iniciar /data/callai/workspace/tyron/LAZ /data/callai/workspace/tyron/Resultados_v3
./servidor.sh status            # n/total, RAM e CPU em uso, estimativa de término, erros
./servidor.sh pausar            # congela (libera CPU); ./servidor.sh continuar retoma
./servidor.sh parar             # interrompe; iniciar de novo pula o que já terminou
./servidor.sh log               # acompanha o log
```

Limites padrão (topo do script ou variável de ambiente, ex.: `CPU_NUCLEOS=8 ./servidor.sh iniciar`):

| Limite | Padrão | Efeito |
|---|---|---|
| `CPU_NUCLEOS` | 4 | teto de CPU (`CPUQuota`) e número de processos; prioridade mínima (`nice 19`) |
| `RAM_MAX` | 25G | acima disso o sistema encerra só este cálculo, nunca o de outros usuários |
| `RAM_LIVRE_MIN_GB` | 30 | não inicia se o servidor tiver menos RAM disponível |
| `DISCO_LIVRE_MIN_GB` | 20 | não inicia sem espaço em disco |
| disco | `idle` | leitura e escrita só quando ninguém mais está usando o disco |

São ~1,9 GB de pico e ~2 min por arquivo com o servidor livre. Com o servidor carregado demora mais.

### No PC

```bash
python -m venv .venv
.venv\Scripts\activate                        # Linux/macOS: source .venv/bin/activate
pip install -r TextureLab/requirements-viewer.txt
scp sergio@137.226.169.235:/data/callai/workspace/tyron/Resultados_v3.tlproj .
streamlit run TextureLab/app.py                # abra o Resultados_v3.tlproj na barra lateral
```

## Opções

| Opção | Efeito |
|---|---|
| `--skip-done` | pula arquivos já calculados com a **mesma receita** (versão do núcleo + configuração + LAZ). Só o que mudou é recalculado. |
| `--config ajustes.json` | muda chaves da configuração, ex. `{"A_lp_design_mm": 2.4, "S_L_macro_mm": 5.0}`. Chaves desconhecidas são erro. Use uma pasta de saída nova para cada configuração que quiser manter. |
| `--zip` | ao final, empacota o projeto em `<saída>.tlproj` (zip só com os arquivos do projeto) |
| `--index-only` | não calcula, só regrava `projeto.json` e `resumo_geral.csv` (ex.: sobre `Resultados_v2`) |
| `--workers N`, `--mem-gb G` | processos paralelos; o automático usa a RAM *disponível* e o tamanho dos arquivos |
| `--pattern "*B6*MP1*.laz"` | filtro de nomes |

## Testes

```bash
pip install pytest
pytest pipeline/tests
```

O teste com arquivo real (opcional) confere um LAZ contra resultados já calculados:

```bash
TEXTURELAB_REF_LAZ=LAZ/A1_WB_MP2_3DT_0-011_NR01_20240918.laz \
TEXTURELAB_REF_DIR=Resultados_v2/A1/A1_WB_MP2_3DT_0-011_NR01_20240918 pytest pipeline/tests -k real_file
```

Documentação completa:
- [White paper (PT)](../docs/WHITE_PAPER_PT.md) · [White paper (EN)](../docs/WHITE_PAPER_EN.md)
- [Diagnóstico e correções](../docs/DIAGNOSTICO.md)
- [Formato do projeto](../docs/FORMATO_PROJETO.md)

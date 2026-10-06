# TextureLab Batch (v3.1.0)

Cálculo em lote, sem interface, dos descritores de textura de pavimento a partir de
varreduras 3dT (LAZ). Núcleo conferido contra as normas ISO 13473-1:2019, 13473-4:2024,
25178-2/-3 e 13565-2, mais Hurst/dimensão fractal (descritivo).

É o **único** núcleo de cálculo. O app (`TextureLab/app.py`) só abre o projeto gerado aqui e desenha
(formato em [docs/FORMATO_PROJETO.md](../docs/FORMATO_PROJETO.md)).

## Fluxo: calcular no servidor, ver no PC

1. Copie os LAZ para o servidor (ex.: `/data/callai/workspace/tyron/LAZ`).
2. Rode o lote com a trava de memória (servidor compartilhado: limite de 25 GB, prioridade baixa
   de CPU e disco):

   ```bash
   cd /data/callai/workspace/tyron
   nohup systemd-run --user --scope -p MemoryMax=25G -p MemorySwapMax=0 nice -n 19 ionice -c3 \
     .venv/bin/python pipeline/texturelab_batch.py --input LAZ --output Resultados_v3 \
     --workers 4 --skip-done --zip > processamento.log 2>&1 &
   tail -f processamento.log
   ```

   São ~1,9 GB de pico e ~2 min por arquivo com o servidor livre (80 arquivos ≈ 45 min com 4 processos).
   Com o servidor carregado demora mais.
3. Copie o pacote para o PC e abra no app:

   ```bash
   scp sergio@137.226.169.235:/data/callai/workspace/tyron/Resultados_v3.tlproj .
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

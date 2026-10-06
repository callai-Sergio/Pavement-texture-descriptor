# Formato do projeto TextureLab (versão de formato 1)

O cálculo é feito **uma vez, no servidor**, por `pipeline/texturelab_batch.py`. O resultado é um
**projeto**: a pasta de saída do lote, ou o mesmo conteúdo empacotado em um zip `.tlproj`.
O app (`TextureLab/app.py`) só lê o projeto e desenha. Não recalcula nada e não usa `pickle`.

## Estrutura

```
Resultados_v3/                          (ou Resultados_v3.tlproj = zip com o mesmo conteúdo)
├── projeto.json                        índice do projeto
├── resumo_geral.csv                    uma linha por arquivo, todos os parâmetros escalares
├── execucao.json                       última execução: config, máquina, processos, duração
└── <trecho>/<arquivo>/
    ├── resumo.json                     parâmetros escalares + receita
    ├── cadeiaA_segmentos.csv           MSD e estatísticas de perfil por faixa × segmento de 100 mm
    ├── espectro_terco_oitava.csv       L_tx média, desvio e n por banda (ISO 13473-4)
    ├── espectro_por_perfil.csv         L_tx por perfil e banda
    ├── psd_media.csv                   PSD 1D média no sentido da via (base do Hurst)
    ├── previa.npz                      z (float16) a cada 8 pontos e passo_mm
    ├── visualizacao.npz                dados só para o app (abaixo)
    └── erro.txt                        só se o cálculo falhou
```

O `.tlproj` só contém esses arquivos. Outras pastas de análise que estiverem dentro de
`Resultados_v3` (`graficos/`, `features/`…) ficam fora do pacote.

## projeto.json

| Campo | Conteúdo |
|---|---|
| `formato` | sempre `"texturelab-projeto"` |
| `versao_formato` | `1`. O app recusa formatos mais novos do que conhece. |
| `versao_nucleo` | versão do pipeline que gravou o índice |
| `versoes_nos_resultados` | versões do núcleo encontradas nos `resumo.json`. Se houver mais de uma, aparece `aviso`. |
| `config` | configuração padrão do núcleo |
| `n_arquivos`, `n_ok` | contagens |
| `arquivos[]` | `arquivo`, `trecho`, `revestimento`, `mp`, `nr`, `data`, `pasta` (relativa), `status`, `versao_nucleo`, `receita_hash`, `visualizacao` (há `visualizacao.npz`) |

## Receita e recálculo

Cada `resumo.json` guarda a `receita`:

```json
{"versao_nucleo": "3.2.0", "config": {...}, "laz": {"nome": "...", "tamanho_bytes": 164163335,
 "sha256_pontas": "..."}, "hash": "1a2b3c4d5e6f7a8b"}
```

- `sha256_pontas` é o SHA-256 do primeiro e do último bloco de 4 MB do LAZ. Assim um arquivo trocado
  ou regravado é detectado sem reler os GB inteiros e sem depender da data de modificação.
- Com `--skip-done`, um arquivo só é pulado se o `resumo.json` tiver `status: ok` **e** o mesmo
  `receita_hash`. Mudou o código (versão do núcleo), a configuração (`--config`) ou o LAZ, e ele é
  recalculado. O resto é reaproveitado.

## visualizacao.npz

Arrays NumPy simples (lidos com `allow_pickle=False`).

| Chave | Forma | Conteúdo |
|---|---|---|
| `versao_formato` | () | 1 |
| `nativo_perfil`, `nativo_passo_mm`, `nativo_y_mm`, `nativo_inicio_mm` | (n,) | 100 mm de uma linha central na resolução nativa |
| `A_perfil_limpo`, `A_perfil_passa_baixa` | (5, n) | perfis da cadeia A (0,5 mm) após spikes/interpolação e após o passa-baixa |
| `A_perfil_faixa`, `A_perfil_y_mm`, `A_perfil_passo_mm` | | faixas escolhidas (igualmente espaçadas) e passo |
| `A_segmento_inicio_mm`, `A_segmento_mm`, `A_n_segmentos` | () | posição dos segmentos de 100 mm |
| `{SF,SL5,MICRO}_abbott_mr_pct`, `…_abbott_altura_mm` | (201,) | curva de Abbott-Firestone (altura em relação à média) |
| `{SF,SL5,MICRO}_hist_bordas_mm`, `…_hist_contagem` | (121,), (120,) | histograma de alturas (PDF) |
| `SL5_previa` | (ny/8, nx/8) | superfície SL5 filtrada (float16), para o 3D |
| `g_seg_z`, `g_seg_zmid`, `g_seg_g`, `g_seg_faixa`, `g_seg_y_mm`, `g_seg_inicio_mm` | (n,), () | um segmento de 100 mm de exemplo para o g-factor (DIN ISO 10844 Anexo B): alturas ordenadas, z_mid, g (v3.2.0) |

São subprodutos do mesmo cálculo e não alteram nenhum parâmetro. Com `V_n_profiles`,
`V_abbott_points` etc. a quantidade gravada é ajustada (entram na receita).

## Compatibilidade

- O app abre também pastas antigas (`Resultados_v2`, núcleo 2.0.0, sem `projeto.json`). Mostra
  parâmetros, espectros e prévia, e avisa que Hurst, perfis e Abbott exigem recalcular.
- `python pipeline/texturelab_batch.py --output <pasta> --index-only` grava `projeto.json` e
  `resumo_geral.csv` de uma pasta existente sem recalcular nada.
- Uma mudança incompatível no formato aumenta `versao_formato`. Campos novos e opcionais não aumentam.

# Histórico do TextureLab

Linha do tempo resumida. Detalhes técnicos de cada versão: [`CHANGELOG.md`](../../CHANGELOG.md).
Documentos antigos desta pasta: changelogs datados de 2026-07 e o resumo de próximos passos de 2026-11-10 (todos concluídos).

| Data | Versão | O que mudou | PR |
|---|---|---|---|
| 2026-02-24 | 1.x | Primeira versão: app Streamlit com vista 3D e métricas | — |
| 2026-07-08 | 1.3 / 2.1 Desktop | App desktop | — |
| 2026-08-05/07 | 2.3, 2.4 | Novos descritores e lote | — |
| 2026-10-05/06 | 3.0–3.2.0 | Cálculo no servidor, visualizador EN/DE/PT, g-factor ISO 10844, ondaletas | #1, #2 |
| 2026-10-22 | 3.2.1, 3.2.2 | Ver resultados no PC sem o servidor; scripts no disco local | #3, #4 |
| 2026-10-26 | 3.3.0 | PSD radial, Hurst 2D, percentis, g-factor 3D, exportador de medianas | #5 |
| 2026-10-26 | 3.4.0 | Verificação dos filtros (ISO 16610-61, IEC 61260-1) e cadeia ML5 | #6 (incluído no #7) |
| 2026-10-27 | 3.4.1 | ML5 removida (≈ SL5); Excel em inglês com aba All_data | #7, #8 |
| 2026-11-10 | 3.4.2 | Visualizador mostra os campos 3.3; README e white paper EN/PT; testes do visualizador; limpeza | #9–#12 |
| 2026-11-11 | 3.5.0 | Assimetria robusta (Ssk aparado 0,5/2 %, Bowley); Resultados_v3.5 (80/80) | #13 |
| 2026-11-12 | 3.5.0 | Exportador de superfícies para o simulador; dados de trabalho em `dados/` | #14 |
| 2026-11-12 | — | Organização: histórico, roadmap; resultados v3.2–v3.4 apagados (manifesto em `arquivo/`) | este |

## Decisões que valem lembrar

- **Cálculo só no servidor**, com limite de CPU/RAM (`pipeline/servidor.sh`); o PC só visualiza.
- **ML5 removida** em 3.4.1: a banda MICRO já está dentro da SL5, então ML5 ≈ SL5.
- **Ssk negativo é real** em todas as superfícies SL5; WB e MALA mudam de sinal quando se cortam 0,5–2 % das caudas. Por isso existe a assimetria robusta (3.5.0).
- **O simulador é outro projeto** (`pavement-texture-simulator`). O único ponto de contato é `tools/exportar_superficies.py`.
- **Resultados não vão para o Git**: são reproduzíveis a partir dos LAZ e da versão do core.

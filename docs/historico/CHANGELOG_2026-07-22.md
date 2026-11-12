# TextureLab Improvements Changelog (2026-07-22)

## 🚀 Version 2.2.0

### Novas Funcionalidades
- **Texture Level Spectrum (Espectro de Nível de Textura)**:
  - Adicionada a extração e agregação de dados espectrais (bandas de 1/3 de oitava) para cada tipo de pavimento.
  - Implementado o novo gráfico interativo Plotly "Texture Level Spectrum", convertendo frequências espaciais em comprimento de onda ($\lambda$).
  - O gráfico agora plota o $L_{TX}$ (Texture Level) em decibéis ($20 \log_{10}(RMS \times 10^3 + 1e^{-12})$).
  - Incluídas marcações visuais com sombreamento para identificar claramente o domínio de Microtextura ($\lambda < 0.5$ mm) e Macrotextura ($\lambda \ge 0.5$ mm).
  - O gráfico é renderizado dinamicamente na aba de "Batch Comparison" sob a secção "By Pavement Type".

### Correções e Melhorias
- Corrigido um erro (`AttributeError`) na iteração dos resultados de lote (`batch_agg`) ao renderizar os novos gráficos espectrais agrupados por tipo de pavimento.
- **Documentação de Parâmetros**:
  - Criada uma documentação/cartão de referência detalhado contendo 47 parâmetros de textura da ISO 4287, ISO 13473, ISO 13565 e ISO 25178.
  - Disponibilização das equações matemáticas exatas e representação gráfica detalhada de cada parâmetro no formato correto (renderizadas utilizando blocos de matemática MathJax).

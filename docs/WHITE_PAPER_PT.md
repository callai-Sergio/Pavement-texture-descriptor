# TextureLab Batch — White Paper

Cálculo em lote de descritores de textura de pavimento a partir de varreduras 3dT

Versão 3.0.0 · 5 out. 2026 · Sergio Callai

> Os três diagramas da versão online (fluxo de processamento, esquema do MSD e curva de material) não são exportáveis para Markdown; aqui estão descritos em texto.

## Resumo

O `pipeline/texturelab_batch.py` lê varreduras 3D de pavimento do equipamento 3dT (arquivos LAZ) e calcula, sem interface e em lote, os descritores de textura definidos nas normas ISO 13473-1, 13473-4, 25178 e 13565-2. Cada arquivo gera uma pasta de resultados, e o lote inteiro gera uma tabela-resumo com uma linha por ponto de medição.

- **Macrotextura por perfil:** MPD, MSD e ETD conforme a ISO 13473-1:2019.
- **Espectro:** níveis de textura em terços de oitava conforme a ISO 13473-4:2024.
- **Parâmetros areais:** alturas, inclinações, curva de material e autocorrelação (ISO 25178), em três bandas de comprimento de onda.
- **Família Rk:** núcleo, picos e vales da curva de Abbott (ISO 13565-2), em perfil e em área.
- **Expoente de Hurst e dimensão fractal:** descritivos, pela inclinação do espectro de potência, ajustados separadamente para micro e macrotextura.

O script foi verificado com sinais sintéticos de resposta conhecida e com o arquivo real B6 MP1 (SMA 11): MPD = 0,819 mm em 70 s por arquivo, numa única CPU. A microtextura sai marcada como provisória até que a resolução óptica do 3dT seja conhecida.

## 1. Escopo e objetivo

O script substitui o cálculo do aplicativo Streamlit TextureLab por um núcleo único, conferido contra os textos das normas, que roda em servidor sem interface. A visualização é feita depois, a partir dos arquivos de saída.

| Grandeza | Norma | Situação |
| --- | --- | --- |
| MPD, MSD, ETD | ISO 13473-1:2019 | Conforme (medição pontual) |
| Nível de textura em terços de oitava | ISO 13473-4:2024, método 1 | Conforme |
| Parâmetros areais S\* (bandas SF e SL5) | ISO 25178-2 e -3, filtro ISO 16610-61 | Conforme, com a escolha de bandas declarada |
| Parâmetros areais da microtextura | ISO 25178-2 e -3 | Provisório |
| Rk, Rpk, Rvk, Mr1, Mr2 e Sk, Spk, Svk | ISO 13565-2 | Conforme no algoritmo; sem o filtro especial da ISO 13565-1 |
| Rq, Rsk, Rku por segmento | — | Descritivo, não é ISO 21920 |
| Expoente de Hurst e dimensão fractal (micro e macro) | — | Descritivo, sem norma; micro provisório |

**Fora do escopo, de propósito:** g-factor e ENDT (ISO 10844) e os parâmetros de perfil da ISO 21920-2. Ainda não foram conferidos contra as normas, e o script não os calcula para não produzir números sem base normativa.

## 2. Dados de entrada

A entrada é uma pasta com arquivos LAZ do 3dT, lida de forma recursiva. Cada arquivo é a grade completa de um ponto de medição (MP).

### Equipamento

| Característica | Valor |
| --- | --- |
| Equipamento | 3dT-MM (BASt), triangulação por linha de laser, sensor duplo de 405 nm; avaliação original no BATex |
| Campo de medição | 45 × 300 mm (253 mm após a avaliação) |
| Resolução horizontal usada | 0,011 mm (ajustável de 0,010 a 1,0 mm) |
| Resolução vertical | 0,001 mm |
| Ângulo de triangulação | 39° (a ISO 13473-1 §5.5 recomenda ≤ 30°) |
| Resolução óptica lateral | desconhecida |

### Formato do arquivo

| Item | Valor nos arquivos verificados (B6 SMA 11 e B248 DSHV) |
| --- | --- |
| Formato | LAS 1.2, formato de ponto 3, comprimido (LAZ), cerca de 165–170 MB |
| Pontos | 89–94 milhões, grade completa e sem furos |
| Passo | exatamente 0,011 mm nos dois eixos |
| Unidade | mm (não declarada no LAZ; o TXT original declara mm) |
| Sentido da via | eixo Y do arquivo: sempre 23 333 pontos, 256,65 mm |
| Largura | eixo X do arquivo: 3 826 a 4 032 pontos, 42,1 a 44,4 mm |
| Campos extras | intensidade, classificação, tempo e RGB são todos zero |

### Como o script lê

1. Lê os pontos com `laspy` (cerca de 2 a 7 s por arquivo).
2. Verifica se a grade está completa e gravada linha a linha. Se estiver, monta a matriz por `reshape`, sem interpolação. Caso contrário, monta a grade pela média dos pontos em cada célula.
3. Assume mm. Se o passo for menor que 0,001, interpreta como metros e converte.
4. Define o sentido da via como o **eixo maior** da grade, independentemente do nome do eixo no arquivo.
5. Tira trecho, revestimento, MP, repetição e data do nome do arquivo. Por exemplo, `B6xxx_SMA11_MP1_3DT_0-011_NR01_20240807` vira trecho B6, revestimento SMA11, MP1, NR01, 20240807.

**Outros formatos:** os TXT originais (cerca de 2,7 GB cada) têm o mesmo conteúdo do LAZ e não são necessários. Os STL **não servem** para cálculo: são malhas simplificadas, com cerca de 2,4 % dos pontos e picos rebaixados.

**Drop-outs:** os arquivos não trazem marcação de leituras inválidas. O script trata valores ausentes, se existirem, e registra a fração encontrada.

## 3. Processamento

A mesma grade alimenta três cadeias independentes, cada uma com os filtros que a sua norma exige. Elas não se misturam: a cadeia do MPD descarta a microtextura de propósito, e a cadeia areal não aplica a normalização de agudez do MPD.

*Diagrama (versão online): Arquivo LAZ → Leitura (mm, via = eixo maior) → três cadeias em paralelo (Cadeia A · MPD, ISO 13473-1; Espectro, ISO 13473-4; Areal, ISO 25178 e 13565-2) → Resultados (pasta por MP, resumo geral CSV).*

### 3.1 Cadeia A — MPD, MSD e ETD (ISO 13473-1:2019)

Mede a profundidade da macrotextura. Como o campo do 3dT tem no máximo 300 mm, aplicam-se as regras de medição pontual.

1. **Faixas:** agrupa 46 linhas vizinhas (0,506 mm de largura) e tira a média, formando um perfil por faixa no sentido da via. São cerca de 85 perfis por arquivo (Tab. D.1, nota b: 0,5 a 1 mm).
2. **Reamostragem para 0,5 mm**, pela média das amostras em cada intervalo (§7.4). Um perfil de 256,65 mm vira 513 amostras.
3. **Drop-outs:** interpolação linear, extrapolação de até 5 mm nas pontas e descarte do segmento com mais de 10 % inválido (§7.3).
4. **Spikes (Anexo E):** marca a amostra quando zᵢ − zᵢ₋₁ ≥ 3 · Δx, ou seja, um salto de 1,5 mm entre amostras, nos dois sentidos. As amostras marcadas são interpoladas. Segmentos com mais de 5 % de spikes são descartados.
5. **Passa-baixa:** Butterworth de 2ª ordem projetado com −3 dB em 2,40 mm, aplicado ida e volta (corte efetivo de 3 mm). Os coeficientes são idênticos aos da Tabela D.2.
6. **Segmentos:** dois segmentos de 100 mm por perfil, centralizados, com cerca de 28 mm de margem em cada ponta para afastar os transientes do filtro.
7. **Supressão de rampa:** subtrai a reta de regressão de cada segmento, o método da norma para medição pontual (§7.6).
8. **MSD:** média dos picos de cada metade do segmento menos o nível médio. **MPD:** média dos MSD válidos, com desvio-padrão e número de segmentos (§7.10). **ETD = 1,1 · MPD**, a fórmula da edição de 2019; a fórmula antiga (0,2 + 0,8 · MPD) não é usada.

### 3.2 Espectro de textura (ISO 13473-4:2024, método 1)

Mede quanta amplitude existe em cada faixa de comprimento de onda, em terços de oitava.

1. Usa as mesmas faixas de 0,5 mm, mas **sem** os filtros do MPD (ISO 13473-1, §7.1).
2. Aplica um anti-aliasing (Butterworth de 10ª ordem, ida e volta) e reduz o passo para 0,099 mm.
3. Detecta spikes pelo Anexo D, com o mesmo critério do Anexo E.
4. Remove a reta de regressão (Anexo F.1) e espelha o início do perfil para estabilizar os filtros (Anexo F.2).
5. Filtra cada banda de terço de oitava (Butterworth, bordas conforme IEC 61260-1) e calcula o RMS.
6. Converte para nível em dB com referência de 1 µm.

**Bandas calculadas:** de 0,4 mm até 20 mm. O limite superior vem da regra l ≥ 12 · λmax: com perfis de 256,65 mm, a maior banda válida é a de 20 mm.

### 3.3 Parâmetros areais (ISO 25178-2 e -3)

Analisa a superfície inteira com filtros gaussianos (ISO 16610-61), em três bandas:

| Superfície | S-filter | F-operation | L-filter | O que representa |
| --- | --- | --- | --- | --- |
| SF | 0,05 mm | plano | — | Toda a textura acima de 0,05 mm |
| SL5 | 0,05 mm | plano | 5 mm | Macrotextura fina e microtextura grossa (0,05 a 5 mm) |
| MICRO | nenhum (instrumento) | plano | 0,5 mm | Microtextura (abaixo de 0,5 mm), provisório |

**Por que 0,05 mm:** pela Tabela 3 da ISO 25178-3 (superfícies ópticas, razão 3:1), um passo de 0,011 mm só é compatível com S-filter a partir de 0,05 mm. O par 0,05 / 5 mm é uma combinação 100:1 da Tabela 1.

**Por que MICRO é provisório:** a razão L/S fica perto de 10:1, fora das combinações padrão da Tabela 1, e a resolução óptica do 3dT é desconhecida.

**Bordas:** os parâmetros desconsideram uma margem igual ao L-filter em cada lado (5 mm em SF e SL5; 0,5 mm em MICRO).

Para cada superfície são calculados:

- **Alturas:** Sa, Sq, Ssk, Sku, Sp, Sv, Sz.
- **Inclinação e área:** Sdq (gradiente quadrático médio) e Sdr (aumento de área, em %).
- **Curva de material:** Sk, Spk, Svk, Smr1, Smr2, pelo método da ISO 13565-2.
- **Volumes:** Vmp, Vmc, Vvc e Vvv, com p = 10 % e q = 80 %.
- **Autocorrelação:** Sal (distância em que a autocorrelação cai a 0,2) e Str (razão de isotropia). Em SF e SL5 são calculados sobre a superfície reduzida; em MICRO, sobre uma janela central de 22,5 × 22,5 mm.

### 3.4 Família Rk (ISO 13565-2)

O mesmo algoritmo serve para perfil (Rk) e área (Sk):

1. Monta a curva de material (curva de Abbott) com 10 001 pontos.
2. Acha a janela de 40 % com a menor inclinação da secante.
3. Ajusta uma reta de mínimos quadrados nessa janela e a prolonga até 0 % e 100 %, o que define Rk, Mr1 e Mr2.
4. Calcula Rpk e Rvk como as alturas de triângulos com a mesma área dos picos e dos vales.

O filtro especial da ISO 13565-1 (supressão de vales) não é aplicado.

### 3.5 Parâmetros de perfil por segmento (descritivos)

Para cada segmento de 100 mm da cadeia A são calculados Ra, Rq, Rsk, Rku, Rp, Rv, Rz e a família Rk. O cálculo usa o perfil de 0,5 mm limpo e com a rampa suprimida, mas **sem** o passa-baixa. Esses valores servem para comparação entre amostras, mas não seguem a ISO 21920.

### 3.6 Expoente de Hurst e dimensão fractal (descritivos)

Nenhuma das normas usadas define esses parâmetros para pavimento. Por isso o script os calcula como descritivos, a partir do espectro de potência (PSD) no sentido da via, e ajusta as duas faixas **separadamente**: a textura de pavimento raramente é fractal numa faixa larga.

1. Toma uma linha a cada 0,506 mm de largura (cerca de 85 linhas), na resolução nativa de 0,011 mm. Não faz média entre linhas, que atenuaria a microtextura.
2. Calcula a PSD de cada linha pelo método de Welch: janela de Hann de 90 mm (8 192 amostras), 50 % de sobreposição e tendência linear removida por segmento. Tira a média entre as linhas.
3. Em cada faixa, agrupa a PSD em 20 intervalos igualmente espaçados em escala logarítmica e ajusta uma reta em log-log. A inclinação dá o expoente β.
4. Converte β em H, e H em dimensão fractal de perfil e de superfície.

| Faixa | Comprimento de onda | Frequência espacial |
| --- | --- | --- |
| Micro | 0,05 a 0,5 mm | 2 a 20 mm⁻¹ |
| Macro | 0,5 a 20 mm | 0,05 a 2 mm⁻¹ |

**Como interpretar:** H fica entre 0 e 1 para uma superfície auto-afim. Valores fora desse intervalo indicam que a PSD não segue uma lei de potência na faixa; o script marca esses casos e a dimensão fractal correspondente não tem significado físico. O R² do ajuste mostra o quanto a PSD é de fato uma reta em log-log.

**Limitações:** a dimensão de superfície (D = 3 − H) supõe textura isotrópica. A faixa micro é provisória pelo mesmo motivo da superfície MICRO: depende da resolução óptica do 3dT. O ruído do sensor (desvio-padrão de 9,6 µm segundo o fabricante) tende a reduzir H_micro.

## 4. Saídas

Cada arquivo LAZ gera uma pasta própria. No fim do lote, o script junta todos os resumos numa tabela única, com uma linha por MP. Os dados brutos nunca são alterados.

```
resultados/
  execucao.json                 configuração, versões, data, duração
  resumo_geral.csv              uma linha por arquivo, todos os parâmetros
  B6/
    B6xxx_SMA11_MP1_3DT_0-011_NR01_20240807/
      resumo.json
      cadeiaA_segmentos.csv
      espectro_terco_oitava.csv
      espectro_por_perfil.csv
      psd_media.csv
      previa.npz
      erro.txt                  só se o arquivo falhar
  B248/
    ...
```

### 4.1 Arquivos

| Arquivo | Conteúdo | Uso típico |
| --- | --- | --- |
| `resumo_geral.csv` | Uma linha por MP, com todos os campos do `resumo.json` | Comparar trechos e revestimentos; base para gráficos |
| `resumo.json` | Todos os resultados escalares de um MP, metadados e tempos | Consultar um MP; auditoria |
| `cadeiaA_segmentos.csv` | Uma linha por segmento de 100 mm: faixa, posição, MSD, validade, fração de spikes e drop-outs, parâmetros de perfil | Variabilidade do MSD; mapa do MSD na largura |
| `espectro_terco_oitava.csv` | Uma linha por banda: comprimento de onda central, nível médio em dB, desvio-padrão, número de perfis | Gráfico do espectro do MP |
| `espectro_por_perfil.csv` | Nível de cada banda (coluna) em cada perfil (linha) | Dispersão do espectro; incerteza |
| `psd_media.csv` | PSD média no sentido da via: frequência (mm⁻¹), comprimento de onda (mm) e PSD (mm³) | Conferir o ajuste de Hurst; gráfico log-log |
| `previa.npz` | Mapa de alturas reduzido 8× em cada eixo (cerca de 490 × 2 900 células, float16) e o passo em mm | Visualização 2D/3D leve |
| `execucao.json` | Configuração completa, versão do script, Python, numpy, plataforma, data, duração e número de processos | Reprodutibilidade |

### 4.2 Campos do resumo

Alturas e volumes estão em mm (volumes em mm³/mm², que equivale a mm; multiplique por 1 000 para obter ml/m²). Comprimentos estão em mm, e material ratio em %.

| Grupo | Campos | Significado |
| --- | --- | --- |
| Identificação | `arquivo`, `trecho`, `revestimento`, `mp`, `nr`, `data`, `caminho` | Tirados do nome e do caminho do arquivo |
| Leitura | `n_pontos`, `n_largura`, `n_via`, `dx_mm`, `largura_mm`, `comprimento_mm`, `leitura`, `frac_invalidos`, avisos | Geometria lida; `leitura` = `reshape` (rápida) ou `generica` |
| Cadeia A | `MPD`, `MPD_desvio`, `ETD` | Profundidade média de perfil, seu desvio-padrão entre segmentos e a profundidade de textura estimada (mm) |
| Cadeia A, controle | `A_n_faixas`, `A_n_segmentos_total`, `A_n_segmentos_validos`, `A_largura_faixa_mm`, `A_inicio_primeiro_segmento_mm`, `A_spike_frac_media`, `A_dropout_frac_media` | Quantos perfis e segmentos entraram no MPD e quanto foi corrigido |
| Perfil (descritivo) | `perfil_Rq_media`, `perfil_Rsk_media`, `perfil_Rku_media`, `perfil_Rk_media`, `perfil_Rpk_media`, `perfil_Rvk_media`, `perfil_Rmr1_media`, `perfil_Rmr2_media` | Média dos segmentos válidos |
| Espectro | `E_passo_mm`, `E_comprimento_avaliacao_mm`, `E_n_perfis`, `E_lambda_min_mm`, `E_lambda_max_mm`, `E_spike_frac_media` | Condições da análise; os níveis ficam nos CSV de espectro |
| Areal, por superfície (`SF_`, `SL5_`, `MICRO_`) | `Sa`, `Sq`, `Ssk`, `Sku`, `Sp`, `Sv`, `Sz` | Alturas: média absoluta, RMS, assimetria, curtose, maior pico, vale mais profundo e amplitude total |
| | `Sdq`, `Sdr_pct` | Inclinação RMS (adimensional) e aumento de área real sobre a projetada (%) |
| | `Sk`, `Spk`, `Svk`, `Smr1`, `Smr2` | Núcleo, picos reduzidos e vales reduzidos da curva de material |
| | `Vmp`, `Vmc`, `Vvc`, `Vvv` | Volume de material dos picos e do núcleo; volume de vazios do núcleo e dos vales |
| | `Sal`, `Str`, `acf_aviso` | Comprimento de autocorrelação, razão de isotropia (0 = direcional, 1 = isotrópico) e aviso se a janela limita o cálculo |
| | `MICRO_status` | Texto que marca a microtextura como provisória |
| Hurst e fractal (sufixo `_micro` ou `_macro`) | `H`, `D_perfil`, `D_superficie`, `H_beta`, `H_R2`, `H_faixa_mm`, `H_aviso`, `H_n_linhas`, `H_segmento_mm`, `H_micro_status` | Expoente de Hurst, dimensão fractal de perfil e de superfície, expoente da PSD, qualidade do ajuste, faixa usada e aviso se H sair de (0, 1) |
| Execução | `status`, `erro`, `versao_script`, `leitura_s`, `cadeia_A_s`, `espectro_s`, `hurst_s`, `areal_s`, `total_s` | Resultado e tempo de cada etapa (s) |

### 4.3 Como ler os resultados

- **MPD e ETD** são os números para comparar com outros equipamentos e com a mancha de areia. O MPD de um local é a média dos MPs, informada com o desvio-padrão e o número de segmentos.
- **O espectro** mostra em que comprimentos de onda está a energia da textura. É a base para estudos de ruído pneu-pavimento.
- **Ssk negativo** indica uma superfície dominada por vazios (textura negativa, típica de SMA). Ssk positivo indica uma superfície dominada por picos.
- **SF** descreve a textura inteira, **SL5** isola a escala dos agregados finos e **MICRO** descreve a aspereza dos agregados, ainda como valor provisório.

### 4.4 Equações dos parâmetros

As fórmulas abaixo são as que o script calcula, na ordem dos grupos da tabela 4.2. Nas fórmulas contínuas, o script usa a versão discreta: integrais viram médias sobre as amostras da grade.

**Reamostragem (cadeia A e espectro).** Cada amostra nova é a média das amostras originais dentro do intervalo Δ (0,5 mm na cadeia A):

```math
\bar{z}_k = \frac{1}{N_k} \sum_{i \in I_k} z_i, \qquad I_k = \{\, i : k\Delta \le x_i < (k+1)\Delta \,\}
```

**Spikes (ISO 13473-1 Anexo E; ISO 13473-4 Anexo D).** Uma amostra é marcada como inválida quando sobe mais que α · Δx em relação à vizinha, em qualquer dos dois sentidos, e depois é interpolada:

```math
z_i - z_{i-1} \ge \alpha \, \Delta x \quad \text{ou} \quad z_i - z_{i+1} \ge \alpha \, \Delta x, \qquad \alpha = 3
```

**Passa-baixa da cadeia A.** Butterworth de 2ª ordem aplicado ida e volta, com λc = 2,40 mm de projeto. A recursão é a Fórmula D.1 da norma; o módulo resultante é o quadrado do de uma passagem:

```math
\begin{aligned}
y_i &= \frac{x_i + 2x_{i-1} + x_{i-2}}{A_0} + A_1\, y_{i-2} + A_2\, y_{i-1} \\
|H(\lambda)| &= \frac{1}{1 + (\lambda_c / \lambda)^4}, \qquad \lambda_c = 2{,}40\ \text{mm}
\end{aligned}
```

**Supressão de rampa, MSD, MPD e ETD (ISO 13473-1).** Em cada segmento de 100 mm, subtrai-se a reta de mínimos quadrados; o MSD compara os picos das duas metades com o nível médio:

```math
\begin{aligned}
z'(x) &= z(x) - (a + b\,x) \\
MSD &= \frac{\max_{\text{1ª metade}} z' + \max_{\text{2ª metade}} z'}{2} - \overline{z'} \\
MPD &= \frac{1}{N}\sum_{j=1}^{N} MSD_j, \qquad ETD = 1{,}1 \cdot MPD
\end{aligned}
```

*Esquema do MSD (versão online): o pico mais alto é procurado separadamente em cada metade do segmento, já com a rampa suprimida; o MSD é a distância vertical entre a média desses dois picos e o nível médio do segmento.*

**Espectro (ISO 13473-4).** Cada banda de terço de oitava tem centro fₘ = 10^(n/10) m⁻¹ e bordas a ±1/6 de oitava (base 10). O perfil é estendido no início por espelhamento (Anexo F), filtrado na banda e convertido em nível:

```math
\begin{aligned}
f_1 &= f_m \cdot 10^{-1/20}, \qquad f_2 = f_m \cdot 10^{1/20} \\
z_{-k} &= 2 z_0 - z_k \\
a_\lambda &= \sqrt{\frac{1}{L}\int_0^L y_\lambda^2(x)\,dx}, \qquad L_{tx,\lambda} = 20 \log_{10} \frac{a_\lambda}{10^{-6}\ \text{m}}
\end{aligned}
```

**Filtro gaussiano areal (ISO 16610-61).** A função peso e a transmissão, que vale 50 % em λ = λc; o S-filter suaviza, o L-filter remove a parte suavizada:

```math
\begin{aligned}
s(x,y) &= \frac{1}{\alpha^2 \lambda_c^2} \exp\!\left(-\pi \frac{x^2 + y^2}{\alpha^2 \lambda_c^2}\right), \qquad \alpha = \sqrt{\ln 2 / \pi} \approx 0{,}4697 \\
H(\lambda) &= \exp\!\left[-\pi \left(\frac{\alpha \lambda_c}{\lambda}\right)^2\right] \\
Z_{SF} &= (s_{N_{is}} * Z) - \text{plano}, \qquad Z_{SL} = Z_{SF} - s_{L} * Z_{SF}
\end{aligned}
```

**Alturas (ISO 25178-2).** Sobre a superfície filtrada, com média nula e área de avaliação A:

```math
\begin{aligned}
S_a &= \frac{1}{A}\iint_A |z|\,dA, \qquad S_q = \sqrt{\frac{1}{A}\iint_A z^2\,dA} \\
S_{sk} &= \frac{1}{S_q^3}\,\frac{1}{A}\iint_A z^3\,dA, \qquad S_{ku} = \frac{1}{S_q^4}\,\frac{1}{A}\iint_A z^4\,dA \\
S_p &= \max z, \qquad S_v = |\min z|, \qquad S_z = S_p + S_v
\end{aligned}
```

**Inclinação e área desenvolvida.** As derivadas são diferenças centrais na grade:

```math
\begin{aligned}
S_{dq} &= \sqrt{\frac{1}{A}\iint_A \left[\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2\right] dA} \\
S_{dr} &= \frac{100\,\%}{A}\iint_A \left(\sqrt{1 + \left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2} - 1\right) dA
\end{aligned}
```

**Curva de material e família Rk / Sk (ISO 13565-2).** c(Mr) é a altura em que a fração Mr da superfície está acima. Na janela de 40 % com menor inclinação, ajusta-se a reta ℓ(Mr) = k · Mr + q:

```math
\begin{aligned}
R_k &= \ell(0) - \ell(100), \qquad c(Mr_1) = \ell(0), \qquad c(Mr_2) = \ell(100) \\
A_1 &= \int_0^{Mr_1} \left[c(Mr) - \ell(0)\right] dMr, \qquad R_{pk} = \frac{2 A_1}{Mr_1} \\
A_2 &= \int_{Mr_2}^{100} \left[\ell(100) - c(Mr)\right] dMr, \qquad R_{vk} = \frac{2 A_2}{100 - Mr_2}
\end{aligned}
```

*Esquema da curva de material (versão online): a reta ajustada na janela de 40 % de menor inclinação, prolongada até 0 % e 100 %, delimita o núcleo Rk; os triângulos têm a mesma área dos picos acima e dos vales abaixo do núcleo, e as suas alturas são Rpk e Rvk.*

**Volumes (ISO 25178-2),** com p = 10 % e q = 80 %:

```math
\begin{aligned}
V_m(p) &= \frac{1}{100}\int_0^{p} \left[c(Mr) - c(p)\right] dMr, \qquad V_v(p) = \frac{1}{100}\int_p^{100} \left[c(p) - c(Mr)\right] dMr \\
V_{mp} &= V_m(10), \quad V_{mc} = V_m(80) - V_m(10), \quad V_{vc} = V_v(10) - V_v(80), \quad V_{vv} = V_v(80)
\end{aligned}
```

**Autocorrelação, Sal e Str (ISO 25178-2).** A autocorrelação é calculada por FFT com zero-padding. Sal é o menor deslocamento em que ela cai a 0,2; Str divide esse valor pelo maior:

```math
\begin{aligned}
f_{ACF}(\tau_x, \tau_y) &= \frac{\iint z(x,y)\, z(x+\tau_x, y+\tau_y)\,dx\,dy}{\iint z^2(x,y)\,dx\,dy} \\
S_{al} &= \min_{f_{ACF}(\tau) \le 0{,}2} \lVert \tau \rVert, \qquad S_{tr} = \frac{S_{al}}{\max_{\theta} \, \tau_{0,2}(\theta)}
\end{aligned}
```

**Expoente de Hurst e dimensão fractal (descritivos).** Em cada faixa, a PSD média de Welch é ajustada por uma lei de potência em log-log:

```math
\begin{aligned}
PSD(f) &\propto f^{-\beta}, \qquad \beta = 1 + 2H \quad \Rightarrow \quad H = \frac{\beta - 1}{2} \\
D_{\text{perfil}} &= 2 - H, \qquad D_{\text{superfície}} = 3 - H
\end{aligned}
```

## 5. Como executar

O script é um arquivo Python único. Roda sem interface e sem GPU.

**Requisitos:** Python 3.10 ou mais novo e cerca de 4 GB de RAM por arquivo processado em paralelo.

```bash
pip install -r pipeline/requirements.txt
```

**Teste com um arquivo** (recomendado antes do lote):

```bash
python pipeline/texturelab_batch.py --input /caminho/dos/laz --output /caminho/resultados --pattern "*B6*MP1*.laz"
```

**Lote completo**, dentro de uma sessão `tmux` para continuar se a conexão SSH cair:

```bash
tmux new -s textura
python pipeline/texturelab_batch.py --input /caminho/dos/laz --output /caminho/resultados
```

| Opção | Efeito |
| --- | --- |
| `--input` | Pasta com os LAZ, lida de forma recursiva (obrigatória) |
| `--output` | Pasta de resultados, criada se não existir (obrigatória) |
| `--pattern` | Filtro de nomes, por exemplo `"*B248*.laz"` (padrão: `*.laz`) |
| `--workers` | Número de arquivos em paralelo (padrão: automático, pelos núcleos e pela RAM) |
| `--skip-done` | Pula arquivos que já têm resultado com status `ok`, para retomar um lote interrompido |

Durante a execução, o script imprime uma linha por arquivo concluído, com o MPD e o tempo. Um arquivo com erro não interrompe o lote: o erro fica em `erro.txt` na pasta do arquivo e no campo `erro` do resumo.

Os parâmetros das normas (largura de faixa, filtros, bandas) ficam no dicionário `CFG`, no início do script, e são gravados em `execucao.json` a cada rodada.

## 6. Referências normativas

| Norma | Título resumido | Usada para |
| --- | --- | --- |
| ISO 13473-1:2019 (DIN EN ISO 13473-1:2021-11) | Profundidade média de perfil | Cadeia A: MPD, MSD, ETD; Anexos D e E |
| ISO 13473-2:2002 (DIN ISO 13473-2:2004-07) | Terminologia e requisitos da análise de perfis | Bandas de terço de oitava; faixas micro e macro |
| ISO 13473-4:2024 | Análise espectral de perfis | Espectro: método 1, Anexos D e F |
| ISO 13473-5 (DIN EN ISO 13473-5:2024-05) | Megatextura | Definição de microtextura (< 0,5 mm) |
| ISO 25178-2 | Parâmetros areais | Definição dos parâmetros S\* e V\* |
| ISO 25178-3:2012 (DIN EN ISO 25178-3:2012-11) | Operadores de especificação | Ordem S-F-L; Tabelas 1 e 3 (escolha de S e L) |
| ISO 16610-61 | Filtro gaussiano areal | Filtros S e L |
| ISO 13565-2:1996 (DIN EN ISO 13565-2:1998-04) | Curva de material: Rk | Família Rk e Sk |
| IEC 61260-1 | Filtros de banda de oitava | Bordas das bandas de terço de oitava |
| ISO 10844, ISO 21920-2, ISO 13565-1 | — | Ainda não aplicadas |

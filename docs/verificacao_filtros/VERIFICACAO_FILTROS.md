# Verificação dos filtros do pipeline

Gerado por `tools/verificar_filtros.py` com o núcleo 3.3.0; as funções medidas são as mesmas do cálculo. Senoides de comprimento de onda λ passam pelo filtro e a amplitude de saída é comparada com a teoria da norma.

## 1. Gaussiano areal – ISO 16610-61 (filtros S e L da ISO 25178-3)

Transmissão teórica T(λ) = exp(−π(αλc/λ)²), α = √(ln2/π) = 0,4697. O filtro S é o próprio gaussiano (passa-baixa); o filtro L é a superfície menos o gaussiano (passa-alta), T_L = 1 − T. Passo dx = 0.011 mm (3dT).

**S 0,05 mm** — T(λc) medido = 50.02 % (teoria 50 %), mesmo valor na outra direção: 50.02 %; maior desvio na tabela = 1.12 p.p. → OK. O desvio maior aparece só em λ = 0,5·λc = 0,025 mm (2,3 amostras por período), no limite de amostragem do 3dT: o núcleo gaussiano tem σ < 1 célula.

| λ/λc | λ [mm] | T medido (passa-baixa) | T teoria | 1 − T medido (filtro L) |
|---|---|---|---|---|
| 0.5 | 0.025 | 7.37 % | 6.25 % | 92.63 % |
| 0.7 | 0.035 | 24.42 % | 24.30 % | 75.58 % |
| 1.0 | 0.05 | 50.02 % | 50.00 % | 49.98 % |
| 1.4 | 0.07 | 70.22 % | 70.21 % | 29.78 % |
| 2.0 | 0.1 | 84.09 % | 84.09 % | 15.91 % |
| 3.0 | 0.15 | 92.59 % | 92.59 % | 7.41 % |

**L 0,5 mm (MICRO)** — T(λc) medido = 50.00 % (teoria 50 %), mesmo valor na outra direção: 50.00 %; maior desvio na tabela = 0.01 p.p. → OK.

| λ/λc | λ [mm] | T medido (passa-baixa) | T teoria | 1 − T medido (filtro L) |
|---|---|---|---|---|
| 0.5 | 0.25 | 6.25 % | 6.25 % | 93.75 % |
| 0.7 | 0.35 | 24.30 % | 24.30 % | 75.70 % |
| 1.0 | 0.5 | 50.00 % | 50.00 % | 50.00 % |
| 1.4 | 0.7 | 70.22 % | 70.21 % | 29.78 % |
| 2.0 | 1 | 84.10 % | 84.09 % | 15.90 % |
| 3.0 | 1.5 | 92.59 % | 92.59 % | 7.41 % |

**L 5 mm (SL5 / ML5)** — T(λc) medido = 50.00 % (teoria 50 %), mesmo valor na outra direção: 50.00 %; maior desvio na tabela = 0.01 p.p. → OK.

| λ/λc | λ [mm] | T medido (passa-baixa) | T teoria | 1 − T medido (filtro L) |
|---|---|---|---|---|
| 0.5 | 2.5 | 6.26 % | 6.25 % | 93.74 % |
| 0.7 | 3.5 | 24.30 % | 24.30 % | 75.70 % |
| 1.0 | 5 | 50.00 % | 50.00 % | 50.00 % |
| 1.4 | 7 | 70.22 % | 70.21 % | 29.78 % |
| 2.0 | 10 | 84.10 % | 84.09 % | 15.90 % |
| 3.0 | 15 | 92.59 % | 92.59 % | 7.41 % |

**SL5 = SF − G₅(SF)** (subtração no lugar, como no cálculo): idêntico à fórmula direta → OK.

## 2. Passa-baixa da cadeia A – ISO 13473-1, Anexo D (MPD, g-factor de perfil)

Butterworth de 2ª ordem (bilinear), projetado em 2,40 mm, aplicado ida e volta (Fórmula D.1): a transmissão é o quadrado da de uma passagem. Ponto de −3 dB do conjunto (teoria): λ = 2.85 mm. Passo 0,5 mm. Maior desvio medido × teoria: 0.00 p.p. → OK

| λ [mm] | T medido | T teoria |
|---|---|---|
| 1.25 | 0.4 % | 0.4 % |
| 1.5 | 3.7 % | 3.7 % |
| 2 | 25.7 % | 25.7 % |
| 2.4 | 50.0 % | 50.0 % |
| 3 | 75.7 % | 75.7 % |
| 4 | 92.2 % | 92.2 % |
| 6 | 98.5 % | 98.5 % |
| 10 | 99.8 % | 99.8 % |
| 20 | 100.0 % | 100.0 % |

A supressão de inclinação (passa-alta) é feita por reta de regressão em cada segmento de 100 mm, uma das duas opções da ISO 13473-1. Comprimentos de onda até 2 passos (1 mm) estão no limite de Nyquist da reamostragem de 0,5 mm e não são medidos.

## 3. Passa-bandas de terço de oitava – ISO 13473-4 / IEC 61260-1

Butterworth de 3ª ordem por banda (bordas f_m·10^(±1/20)). Ganho medido no centro e nas bordas; o passo depende da banda (0,1 mm após o anti-aliasing).

| λ central [mm] | ganho no centro [dB] | ganho nas bordas [dB] |
|---|---|---|
| 0.5 | -0.00 | -3.01 / -3.01 |
| 1 | -0.00 | -3.01 / -3.01 |
| 2 | -0.00 | -3.01 / -3.01 |
| 5 | -0.00 | -3.01 / -3.01 |
| 10 | -0.00 | -3.01 / -3.01 |
| 20 | -0.00 | -3.01 / -3.01 |

## Conclusão: todos os filtros conferem com a teoria das normas

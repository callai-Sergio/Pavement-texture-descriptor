# Diagnóstico do TextureLab v2.4.0 e correções da v3.0.0

Avaliação feita em outubro de 2026 sobre o código das branches `master` (v1.3.0) e `beta` (v2.4.0), conferida contra os textos das normas (DIN EN ISO 13473-1:2021-11, ISO 13473-4:2024, DIN EN ISO 25178-3:2012-11, DIN EN ISO 13565-2:1998-04) e contra arquivos reais do 3dT (trechos B6 SMA 11 e B248 DSHV).

**Resultado:** o aplicativo Streamlit calcula vários parâmetros fora das normas que cita, e a leitura dos arquivos 3dT troca o sentido da via com as configurações padrão. A v3.0.0 não altera o aplicativo: acrescenta um núcleo de cálculo novo e conferido (`pipeline/texturelab_batch.py`), que passa a ser a referência para os resultados. O aplicativo continua na versão anterior para visualização, e seus números não devem ser usados em relatórios até ser migrado para o novo núcleo.

## 1. Problemas de cálculo encontrados

Todos confirmados na branch `beta` (v2.4.0).

| # | Onde | Problema | Efeito | Correção na v3.0.0 |
| --- | --- | --- | --- | --- |
| 1 | `src/descriptors.py`, `calc_etd` | ETD = 0,2 + 0,8·MPD (fórmula de 1997) | ETD diferente do da norma vigente | ETD = 1,1·MPD (ISO 13473-1:2019 §7.11) |
| 2 | `calc_mpd` | Sem reamostragem para 0,5 mm, sem detecção de spikes do Anexo E, sem passa-baixa nem supressão de rampa por segmento | MPD não conforme e dependente da resolução do equipamento | Cadeia A completa conforme §7 e Anexos D e E |
| 3 | `calc_mpd` | Perfil menor que 100 mm cai num caso especial e é tratado como um segmento inteiro, sem aviso | MPD inválido em perfis transversais (43 mm) | Cadeia A recusa perfis menores que o segmento |
| 4 | `calc_rk_params` | Rk = diferença de alturas na janela de 40 %; Rpk e Rvk = pico e vale absolutos | Rk, Rpk e Rvk fora da ISO 13565-2 | Reta de mínimos quadrados na janela, prolongada a 0 % e 100 %; Rpk e Rvk por triângulos de área equivalente |
| 5 | `calc_psd_welch` | `nperseg = 256` fixo; com dx = 0,011 mm o maior comprimento de onda é de cerca de 2,8 mm | Espectro, BandRMS, TextureLevel e ENDT não enxergam a macrotextura (0,5–50 mm) | Método 1 da ISO 13473-4 (filtros de terço de oitava), bandas de 0,4 a 20 mm |
| 6 | `calc_spectral` | TextureLevel = 20·log10(rms) sem a referência de 1 µm | Valor deslocado em 60 dB com dados em mm | L = 20·log10(a/10⁻⁶ m) |
| 7 | `calc_endt` | Ponderação inventada (peso = f/fmax), apresentada como ISO 10844 | Número sem base normativa | Removido até a conferência da ISO 10844 |
| 8 | `calc_mean_spacing` | Sm mede a distância entre cruzamentos consecutivos do zero (meio período) | Sm pela metade | Não calculado na v3.0.0 |
| 9 | `calc_sdr` | NaN ignorado no numerador, mas contado na área projetada | Sdr subestimado com drop-outs | Sdr como média de (√(1+∇z²) − 1) sobre as células válidas |
| 10 | `calc_peak_density` | `find_peaks` sem proeminência mínima | Conta ruído como pico | Não calculado na v3.0.0 |
| 11 | `calc_acl` | Autocorrelação por `np.correlate`, O(n²) | Lento em perfis longos | Sal e Str por FFT com zero-padding (ISO 25178-2) |
| 12 | `calc_fractal_dimension` | Contagem de caixas com 6 escalas sobre perfil normalizado | Resultado dependente da resolução | Hurst e dimensão fractal pela inclinação da PSD, micro e macro separadas (descritivos) |

## 2. Problemas de leitura e pré-processamento

| # | Onde | Problema | Efeito | Correção na v3.0.0 |
| --- | --- | --- | --- | --- |
| 13 | `src/data_io.py`, `read_laz` + `app.py` | A grade é montada como (Y, X) e "longitudinal", o padrão, extrai `z[i, :]` | Nos arquivos 3dT o perfil "longitudinal" é o de **43 mm na largura** | Sentido da via = eixo maior da grade, detectado automaticamente |
| 14 | `read_laz` | Vários pontos na mesma célula: o último sobrescreve | Ruído e viés na grade | Leitura direta por `reshape` (grade completa verificada) ou média por célula |
| 15 | `read_laz` | Unidade não verificada | Arquivos em metros dariam parâmetros 1 000× errados | Passo < 0,001 → interpretado como metros e convertido |
| 16 | `read_csv_xyz` | Grade montada por `unique()` de coordenadas float, `dx` informado ignorado | Grades enormes e esparsas | Pipeline usa LAZ; TXT/CSV continuam no aplicativo |
| 17 | `_sniff_csv_format` | Matriz com exatamente 3 colunas confundida com formato xyz | Erro silencioso | — (aplicativo) |
| 18 | `src/preprocessing.py`, `hampel_filter_1d` | Laço Python ponto a ponto, nas linhas e nas colunas | Cerca de 185 milhões de iterações por arquivo (estimativa: da ordem de uma hora) | Operações vetorizadas sobre a matriz; cerca de 70 s por arquivo |
| 19 | `app.py` (padrão Hampel, K = 3, janela 7) | O MAD da varredura fica no piso de digitalização (cerca de 1,9 µm) | O filtro padrão **apaga arestas de agregado e paredes de vazio** | Hampel removido; spikes pelo critério normativo (Anexo E) |
| 20 | Três cópias do pré-processamento (`TextureLab/preprocessing.py`, `TextureLab/src/`, `TextureLabDesktop/engine/`) | Versões divergentes | Resultados diferentes entre web e desktop | Núcleo único no pipeline |

## 3. Segurança e repositório

| # | Problema | Correção na v3.0.0 |
| --- | --- | --- |
| 21 | Projetos `.tlp` lidos com `pickle.load`: um arquivo malicioso executa código | Pipeline não usa pickle (JSON e CSV). O aplicativo ainda usa; corrigir antes de rodar o app num servidor |
| 22 | `__pycache__`, `*.egg-info`, `pytest_out.txt`, `test_output.txt` e `scratch_test_tk.py` versionados | Removidos; `.gitignore` criado |
| 23 | Entrada `Pavement-texture-descriptor` era um submódulo quebrado (aponta para o próprio repositório, sem `.gitmodules`) | Removida |
| 24 | `tests/test_llm_utils.py` importava o pacote `texturelab_llm`, já removido | Removido |
| 25 | Teste de MPD usava seno e só verificava `mpd > 0` | Pipeline validado com cosseno sobre número inteiro de períodos (seno carrega viés de fase da supressão de rampa) |

## 4. Conferência do `FILTROS.md` (pavfric_batch) contra as normas

O documento de filtros de outro projeto foi usado como referência inicial. A matemática dos filtros está correta, mas três escolhas contrariam o texto das normas, e a v3.0.0 segue a norma.

| Ponto | `FILTROS.md` | Norma | Decisão na v3.0.0 |
| --- | --- | --- | --- |
| Reamostragem | Filtra em 11 µm | 0,5 mm (preferido) ou 1,0 mm; perfil 3D = faixa de 0,5 a 1 mm (ISO 13473-1 §7.4, Tab. D.1) | 0,5 mm |
| Spikes | Amplitude > 0,135·h_rms e isolamento ≤ 2 células; substituição pela mediana | Anexo E: zᵢ − zᵢ₋₁ ≥ 3·Δx, ida e volta, interpolação linear | Anexo E |
| Passa-baixa | Butterworth projetado em 2,5 mm; especificação "≥ 3 dB em 2,5 mm, ≤ 1 dB em 5 mm" | Projeto em 2,40 mm, ida e volta; Tabela D.2; "no filter other than the one required here shall be used" | 2,40 mm; coeficientes iguais aos da Tab. D.2 |
| Onda longa | Passa-alta de 100 mm; trata a supressão de rampa como "método de 1997" | Medição pontual: supressão de rampa por segmento é o método preferido; passa-alta de 140 mm (projeto 174,2 mm) só para perfis contínuos ≥ 1 m (§7.6) | Supressão de rampa |
| Espectro | Feito sobre a cadeia do MPD (com passa-baixa) | Perfil limpo antes dos filtros do §7.6 (§7.1); ISO 13473-4 método 1 | Cadeia própria, sem os filtros do MPD |
| S-filter areal | 25 µm, pulado quando σ < 0,25 célula | Tabela 3 da ISO 25178-3 (óptico, 3:1): 25 µm exige passo ≤ 8 µm | 0,05 mm (compatível com 11 µm) |
| Corte micro/macro | L = 2,5 mm "separa micro de macro" | Macrotextura = 0,5 a 50 mm (ISO 13473-1 §3.4) | L = 0,5 mm para microtextura |

## 5. Características dos dados 3dT verificadas

- **LAZ = TXT original:** 91 768 689 pontos no B6 MP1, idênticos nos dois formatos; o TXT declara `X [mm], Y [mm], Z [mm]`.
- **Geometria:** grade completa, passo exato de 0,011 mm, 43,25 × 256,65 mm (campo do equipamento: 45 × 300 mm, 253 mm após a avaliação). Os grãos do SMA 11 medem 5 a 12 mm na imagem, o que confirma a unidade mm.
- **STL:** malha simplificada (cerca de 2,4 % dos pontos, vértices fora da grade, picos rebaixados, triângulos de até 203 mm). Não serve para cálculo.
- **Equipamento:** 3dT-MM, triangulação por linha de laser, sensor duplo de 405 nm, ângulo de triangulação de 39° (a ISO 13473-1 §5.5 recomenda ≤ 30°), exatidão declarada com desvio-padrão de 9,6 µm. A resolução óptica lateral não está na ficha técnica; por isso a microtextura e o H_micro saem como provisórios.

## 6. Validação do pipeline v3.0.0

| Teste | Esperado | Obtido |
| --- | --- | --- |
| Coeficientes do passa-baixa (passo de 0,5 mm) | Tabela D.2 da ISO 13473-1 | Idênticos até a 14ª casa decimal |
| MPD de cosseno, A = 0,5 mm, λ = 10 mm | cerca de 0,50 mm | 0,507 mm |
| Nível de senoide, λ = 8 mm, A = 0,1 mm | 36,99 dB na banda de 8 mm | 36,98 dB |
| Transmissão do gaussiano em λ = λc | 0,500 | 0,500 |
| Rk de distribuição uniforme [0, 1] | Rk = 1, Rpk = Rvk = 0 | 1,000; 0,000; 0,001 |
| B6 MP1 (SMA 11), arquivo real | — | MPD 0,819 mm (σ 0,144), 170/170 segmentos válidos, Sq 0,659 mm, Ssk −1,85, 70 s |

O cálculo de Hurst e dimensão fractal (acrescentado na v3.0.0) ainda não foi executado em dados reais.

## 7. Pendências

1. Migrar o aplicativo Streamlit para o núcleo do pipeline e remover o `pickle`.
2. Conferir a ISO 10844 (g-factor, ENDT) e a ISO 21920-2 antes de reintroduzir esses parâmetros.
3. Obter a resolução óptica do sensor do 3dT, ou estimá-la pela PSD medida, para tirar a microtextura do status provisório.
4. Comparar os resultados com os do BATex 3dT-MM nos mesmos MPs.
5. Aceleração em GPU (CuPy), validada contra a versão em CPU.

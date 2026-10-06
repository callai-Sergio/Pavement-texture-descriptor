# TextureLab Batch — White Paper

Batch computation of pavement texture descriptors from 3dT scans

Version 3.0.0 · 5 Oct 2026 · Sergio Callai

> The three diagrams of the online version (processing flow, MSD schematic and material ratio curve) cannot be exported to Markdown; they are described in text here. File and field names are in Portuguese, as written by the script.

## Summary

`pipeline/texturelab_batch.py` reads 3D pavement scans from the 3dT device (LAZ files) and computes, headless and in batch, the texture descriptors defined in ISO 13473-1, 13473-4, 25178 and 13565-2. Each file produces its own result folder, and the whole batch produces one summary table with one row per measurement point.

- **Profile macrotexture:** MPD, MSD and ETD per ISO 13473-1:2019.
- **Spectrum:** one-third-octave texture levels per ISO 13473-4:2024.
- **Areal parameters:** heights, slopes, material ratio curve and autocorrelation (ISO 25178), in three wavelength bands.
- **Rk family:** core, peaks and valleys of the Abbott curve (ISO 13565-2), for profiles and areas.
- **Hurst exponent and fractal dimension:** descriptive, from the slope of the power spectrum, fitted separately for micro- and macrotexture.

The script was checked against synthetic signals with known answers and against the real file B6 MP1 (SMA 11): MPD = 0.819 mm, in 70 s per file on a single CPU. Microtexture results are flagged as provisional until the optical resolution of the 3dT is known.

## 1. Scope and objective

The script replaces the calculations of the Streamlit TextureLab app with a single core, checked against the standard texts, that runs headless on a server. Visualisation is done afterwards, from the output files.

| Quantity | Standard | Status |
| --- | --- | --- |
| MPD, MSD, ETD | ISO 13473-1:2019 | Conforming (spot measurement) |
| One-third-octave texture level | ISO 13473-4:2024, method 1 | Conforming |
| Areal S parameters (SF and SL5 bands) | ISO 25178-2 and -3, ISO 16610-61 filter | Conforming, with the band choice stated |
| Areal microtexture parameters | ISO 25178-2 and -3 | Provisional |
| Rk, Rpk, Rvk, Mr1, Mr2 and Sk, Spk, Svk | ISO 13565-2 | Conforming algorithm; without the special ISO 13565-1 filter |
| Rq, Rsk, Rku per segment | — | Descriptive, not ISO 21920 |
| Hurst exponent and fractal dimension (micro and macro) | — | Descriptive, no standard; micro provisional |

**g-factor and skewness (v3.2.0):** per DIN ISO 10844:2024-11, clause 5.3.2 and Annex B, per 100 mm segment of the MPD profile (ISO 13473-1 processing without low-pass, zero mean); z_mid = (z_max + z_min)/2 and g = cumulative distribution where z_mid crosses the bearing area curve; mean of the segments per measuring point (`g_factor_media`). The skewness of 5.3.2 is the per-segment Rsk (ISO 13473-2), mean in `perfil_Rsk_media`. Clause 5.3.3 asks for bands from 5 to 100 mm; with ~257 mm scans and l ≥ 12·λ (ISO 13473-4) only bands up to 20 mm are reachable.

**ETD:** no longer written from v3.2.0 (it is just 1.1 · MPD).

**Deliberately out of scope:** ENDT (removed from ISO 10844 in the 2021 edition) and the ISO 21920-2 profile parameters.

## 2. Input data

The input is a folder of 3dT LAZ files, searched recursively. Each file is the complete grid of one measurement point (MP).

### Device

| Property | Value |
| --- | --- |
| Device | 3dT-MM (BASt), laser line triangulation, dual 405 nm sensor; originally evaluated with BATex |
| Measuring field | 45 × 300 mm (253 mm after evaluation) |
| Horizontal resolution used | 0.011 mm (adjustable from 0.010 to 1.0 mm) |
| Vertical resolution | 0.001 mm |
| Triangulation angle | 39° (ISO 13473-1 §5.5 recommends ≤ 30°) |
| Lateral optical resolution | unknown |

### File format

| Item | Value in the files checked (B6 SMA 11 and B248 DSHV) |
| --- | --- |
| Format | LAS 1.2, point format 3, compressed (LAZ), about 165–170 MB |
| Points | 89–94 million, complete grid with no gaps |
| Spacing | exactly 0.011 mm on both axes |
| Unit | mm (not stated in the LAZ; the original TXT states mm) |
| Driving direction | file Y axis: always 23,333 points, 256.65 mm |
| Width | file X axis: 3,826 to 4,032 points, 42.1 to 44.4 mm |
| Extra fields | intensity, classification, time and RGB are all zero |

### How the script reads it

1. Reads the points with `laspy` (about 2 to 7 s per file).
2. Checks whether the grid is complete and stored row by row. If so, it builds the matrix by `reshape`, with no interpolation. Otherwise it grids the points by averaging each cell.
3. Assumes mm. If the spacing is below 0.001, it treats it as metres and converts.
4. Takes the driving direction as the **longer axis** of the grid, whatever the axis is called in the file.
5. Parses section, surface type, MP, repeat and date from the file name. For example, `B6xxx_SMA11_MP1_3DT_0-011_NR01_20240807` becomes section B6, surface SMA11, MP1, NR01, 20240807.

**Other formats:** the original TXT files (about 2.7 GB each) hold the same content as the LAZ and are not needed. The STL files are **not suitable** for computation: they are simplified meshes with about 2.4 % of the points and lowered peaks.

**Drop-outs:** the files carry no invalid-reading flags. The script handles missing values if any exist and records the fraction found.

## 3. Processing

The same grid feeds three independent chains, each with the filters its standard requires. They do not mix: the MPD chain discards microtexture on purpose, and the areal chain does not apply the MPD sharpness normalisation.

*Diagram (online version): LAZ file → Reading (mm, road = longer axis) → three parallel chains (Chain A · MPD, ISO 13473-1; Spectrum, ISO 13473-4; Areal, ISO 25178 and 13565-2) → Results (folder per MP, summary CSV).*

### 3.1 Chain A — MPD, MSD and ETD (ISO 13473-1:2019)

Measures macrotexture depth. Since the 3dT field is at most 300 mm long, the rules for spot measurements apply.

1. **Strips:** groups 46 neighbouring lines (0.506 mm wide) and averages them into one profile per strip along the driving direction, about 85 profiles per file (Table D.1, note b: 0.5 to 1 mm).
2. **Resampling to 0.5 mm** by averaging the samples in each interval (§7.4). A 256.65 mm profile becomes 513 samples.
3. **Drop-outs:** linear interpolation, extrapolation of up to 5 mm at the ends, and segments with more than 10 % invalid data are discarded (§7.3).
4. **Spikes (Annex E):** flags a sample when zᵢ − zᵢ₋₁ ≥ 3 · Δx, i.e. a 1.5 mm jump between samples, in both directions. Flagged samples are interpolated. Segments with more than 5 % spikes are discarded.
5. **Low-pass:** 2nd-order Butterworth designed with −3 dB at 2.40 mm, applied forward and backward (effective cut-off 3 mm). The coefficients are identical to Table D.2.
6. **Segments:** two 100 mm segments per profile, centred, with about 28 mm margin at each end to keep away from filter transients.
7. **Slope suppression:** the regression line of each segment is subtracted, the standard's method for spot measurements (§7.6).
8. **MSD:** mean of the peaks of each half of the segment minus the mean level. **MPD:** mean of the valid MSD values, with standard deviation and number of segments (§7.10). **ETD = 1.1 · MPD**, the 2019 formula; the old 0.2 + 0.8 · MPD is not used.

### 3.2 Texture spectrum (ISO 13473-4:2024, method 1)

Measures how much amplitude lies in each wavelength range, in one-third octaves.

1. Uses the same 0.5 mm strips, but **without** the MPD filters (ISO 13473-1, §7.1).
2. Applies anti-aliasing (10th-order Butterworth, forward and backward) and reduces the spacing to 0.099 mm.
3. Detects spikes per Annex D, with the same criterion as Annex E.
4. Removes the regression line (Annex F.1) and mirrors the start of the profile to settle the filters (Annex F.2).
5. Filters each one-third-octave band (Butterworth, band edges per IEC 61260-1) and computes the RMS.
6. Converts to a level in dB re 1 µm.

**Bands computed:** 0.4 mm to 20 mm. The upper limit comes from the rule l ≥ 12 · λmax: with 256.65 mm profiles, the largest valid band is 20 mm.

### 3.3 Areal parameters (ISO 25178-2 and -3)

Analyses the whole surface with Gaussian filters (ISO 16610-61), in three bands:

| Surface | S-filter | F-operation | L-filter | What it represents |
| --- | --- | --- | --- | --- |
| SF | 0.05 mm | plane | — | All texture above 0.05 mm |
| SL5 | 0.05 mm | plane | 5 mm | Fine macrotexture and coarse microtexture (0.05 to 5 mm) |
| MICRO | none (instrument) | plane | 0.5 mm | Microtexture (below 0.5 mm), provisional |

**Why 0.05 mm:** under Table 3 of ISO 25178-3 (optical surfaces, 3:1 ratio), a 0.011 mm spacing is only compatible with S-filters from 0.05 mm upwards. The 0.05 / 5 mm pair is a 100:1 combination from Table 1.

**Why MICRO is provisional:** the L/S ratio is close to 10:1, outside the standard combinations of Table 1, and the optical resolution of the 3dT is unknown.

**Edges:** the parameters exclude a margin equal to the L-filter on each side (5 mm for SF and SL5; 0.5 mm for MICRO).

For each surface the script computes:

- **Heights:** Sa, Sq, Ssk, Sku, Sp, Sv, Sz.
- **Slope and area:** Sdq (root mean square gradient) and Sdr (developed area increase, in %).
- **Material ratio curve:** Sk, Spk, Svk, Smr1, Smr2, by the ISO 13565-2 method.
- **Volumes:** Vmp, Vmc, Vvc and Vvv, with p = 10 % and q = 80 %.
- **Autocorrelation:** Sal (distance at which the autocorrelation drops to 0.2) and Str (isotropy ratio). For SF and SL5 they are computed on a reduced surface; for MICRO, on a central 22.5 × 22.5 mm window.

### 3.4 Rk family (ISO 13565-2)

The same algorithm serves profiles (Rk) and areas (Sk):

1. Builds the material ratio (Abbott) curve with 10,001 points.
2. Finds the 40 % window with the smallest secant slope.
3. Fits a least-squares line in that window and extends it to 0 % and 100 %, which defines Rk, Mr1 and Mr2.
4. Computes Rpk and Rvk as the heights of triangles with the same area as the peaks and the valleys.

The special ISO 13565-1 filter (valley suppression) is not applied.

### 3.5 Profile parameters per segment (descriptive)

For each 100 mm segment of chain A the script computes Ra, Rq, Rsk, Rku, Rp, Rv, Rz and the Rk family. It uses the cleaned 0.5 mm profile with the slope suppressed, but **without** the low-pass. These values support comparisons between samples, but do not follow ISO 21920.

### 3.6 Hurst exponent and fractal dimension (descriptive)

None of the standards used defines these parameters for pavements. The script therefore computes them as descriptive values from the power spectral density (PSD) along the driving direction, and fits the two ranges **separately**: pavement texture is rarely fractal over a wide range.

1. Takes one line every 0.506 mm of width (about 85 lines) at the native 0.011 mm resolution. It does not average lines together, which would attenuate microtexture.
2. Computes each line's PSD by Welch's method: 90 mm Hann window (8,192 samples), 50 % overlap, linear trend removed per segment. Averages the PSD over the lines.
3. In each range, groups the PSD into 20 log-spaced bins and fits a straight line in log-log. The slope gives the exponent β.
4. Converts β to H, and H to the profile and surface fractal dimensions.

| Range | Wavelength | Spatial frequency |
| --- | --- | --- |
| Micro | 0.05 to 0.5 mm | 2 to 20 mm⁻¹ |
| Macro | 0.5 to 20 mm | 0.05 to 2 mm⁻¹ |

**How to read it:** H lies between 0 and 1 for a self-affine surface. Values outside that interval mean the PSD does not follow a power law in the range; the script flags them, and the corresponding fractal dimension has no physical meaning. The R² of the fit shows how straight the PSD really is in log-log.

**Limitations:** the surface dimension (D = 3 − H) assumes isotropic texture. The micro range is provisional for the same reason as the MICRO surface: it depends on the optical resolution of the 3dT. Sensor noise (9.6 µm standard deviation per the manufacturer) tends to lower H_micro.

## 4. Outputs

Each LAZ file gets its own folder. At the end of the batch, the script merges all summaries into one table with one row per MP. Raw data are never modified.

```
results/
  execucao.json                 configuration, versions, date, duration
  resumo_geral.csv              one row per file, all parameters
  B6/
    B6xxx_SMA11_MP1_3DT_0-011_NR01_20240807/
      resumo.json
      cadeiaA_segmentos.csv
      espectro_terco_oitava.csv
      espectro_por_perfil.csv
      psd_media.csv
      previa.npz
      erro.txt                  only if the file fails
  B248/
    ...
```

### 4.1 Files

| File | Content | Typical use |
| --- | --- | --- |
| `resumo_geral.csv` | One row per MP, with every field of `resumo.json` | Compare sections and surfaces; basis for charts |
| `resumo.json` | All scalar results of one MP, metadata and timings | Inspect one MP; audit |
| `cadeiaA_segmentos.csv` | One row per 100 mm segment: strip, position, MSD, validity, spike and drop-out fractions, profile parameters | MSD variability; MSD across the width |
| `espectro_terco_oitava.csv` | One row per band: centre wavelength, mean level in dB, standard deviation, number of profiles | Spectrum chart of the MP |
| `espectro_por_perfil.csv` | Level of each band (column) for each profile (row) | Spectrum scatter; uncertainty |
| `psd_media.csv` | Mean PSD along the driving direction: frequency (mm⁻¹), wavelength (mm) and PSD (mm³) | Visual check of the Hurst fit; log-log chart |
| `previa.npz` | Height map reduced 8× on each axis (about 490 × 2,900 cells, float16) and its spacing in mm | Light 2D/3D visualisation |
| `execucao.json` | Full configuration, script version, Python, numpy, platform, date, duration and number of processes | Reproducibility |

### 4.2 Summary fields

Heights and volumes are in mm (volumes in mm³/mm², which equals mm; multiply by 1,000 for ml/m²). Lengths are in mm and material ratios in %.

| Group | Fields | Meaning |
| --- | --- | --- |
| Identification | `arquivo`, `trecho`, `revestimento`, `mp`, `nr`, `data`, `caminho` | File name, section, surface, MP, repeat, date and path |
| Reading | `n_pontos`, `n_largura`, `n_via`, `dx_mm`, `largura_mm`, `comprimento_mm`, `leitura`, `frac_invalidos`, warnings | Geometry read; `leitura` = `reshape` (fast) or `generica` |
| Chain A | `MPD`, `MPD_desvio` | Mean profile depth and its standard deviation across segments (mm); ETD = 1.1·MPD up to v3.1 |
| ISO 10844 | `g_factor_media`, `g_factor_desvio`, `g_factor_n_segmentos`; `g_pct` per segment | Shape factor (%) per 100 mm segment, mean and std per measuring point (v3.2.0) |
| Chain A, control | `A_n_faixas`, `A_n_segmentos_total`, `A_n_segmentos_validos`, `A_largura_faixa_mm`, `A_inicio_primeiro_segmento_mm`, `A_spike_frac_media`, `A_dropout_frac_media` | How many profiles and segments entered the MPD and how much was corrected |
| Profile (descriptive) | `perfil_Rq_media`, `perfil_Rsk_media`, `perfil_Rku_media`, `perfil_Rk_media`, `perfil_Rpk_media`, `perfil_Rvk_media`, `perfil_Rmr1_media`, `perfil_Rmr2_media` | Mean over valid segments |
| Spectrum | `E_passo_mm`, `E_comprimento_avaliacao_mm`, `E_n_perfis`, `E_lambda_min_mm`, `E_lambda_max_mm`, `E_spike_frac_media` | Analysis conditions; the levels are in the spectrum CSV files |
| Areal, per surface (`SF_`, `SL5_`, `MICRO_`) | `Sa`, `Sq`, `Ssk`, `Sku`, `Sp`, `Sv`, `Sz` | Heights: mean absolute, RMS, skewness, kurtosis, highest peak, deepest valley and total range |
| | `Sdq`, `Sdr_pct` | RMS slope (dimensionless) and real area increase over the projected area (%) |
| | `Sk`, `Spk`, `Svk`, `Smr1`, `Smr2` | Core, reduced peaks and reduced valleys of the material ratio curve |
| | `Vmp`, `Vmc`, `Vvc`, `Vvv` | Material volume of peaks and core; void volume of core and valleys |
| | `Sal`, `Str`, `acf_aviso` | Autocorrelation length, isotropy ratio (0 = directional, 1 = isotropic) and a warning if the window limits the computation |
| | `MICRO_status` | Text marking the microtexture as provisional |
| Hurst and fractal (suffix `_micro` or `_macro`) | `H`, `D_perfil`, `D_superficie`, `H_beta`, `H_R2`, `H_faixa_mm`, `H_aviso`, `H_n_linhas`, `H_segmento_mm`, `H_micro_status` | Hurst exponent, profile and surface fractal dimension, PSD exponent, fit quality, range used and a warning if H falls outside (0, 1) |
| Run | `status`, `erro`, `versao_script`, `leitura_s`, `cadeia_A_s`, `espectro_s`, `hurst_s`, `areal_s`, `total_s` | Outcome and time of each step (s) |

### 4.3 Reading the results

- **MPD and ETD** are the numbers to compare with other devices and with the sand patch. The MPD of a site is the mean of its MPs, reported with the standard deviation and the number of segments.
- **The spectrum** shows at which wavelengths the texture energy lies. It is the basis for tyre/road noise studies.
- **Negative Ssk** means a surface dominated by voids (negative texture, typical of SMA). Positive Ssk means a surface dominated by peaks.
- **SF** describes the whole texture, **SL5** isolates the scale of fine aggregate and **MICRO** describes aggregate harshness, still as a provisional value.

### 4.4 Parameter equations

These are the formulas the script computes, in the order of the groups in table 4.2. For continuous formulas the script uses the discrete form: integrals become means over the grid samples.

**Resampling (chain A and spectrum).** Each new sample is the mean of the original samples within the interval Δ (0.5 mm in chain A):

```math
\bar{z}_k = \frac{1}{N_k} \sum_{i \in I_k} z_i, \qquad I_k = \{\, i : k\Delta \le x_i < (k+1)\Delta \,\}
```

**Spikes (ISO 13473-1 Annex E; ISO 13473-4 Annex D).** A sample is flagged as invalid when it rises more than α · Δx above its neighbour, in either direction, and is then interpolated:

```math
z_i - z_{i-1} \ge \alpha \, \Delta x \quad \text{or} \quad z_i - z_{i+1} \ge \alpha \, \Delta x, \qquad \alpha = 3
```

**Chain A low-pass.** 2nd-order Butterworth applied forward and backward, with a 2.40 mm design λc. The recursion is Formula D.1 of the standard; the resulting magnitude is the square of a single pass:

```math
\begin{aligned}
y_i &= \frac{x_i + 2x_{i-1} + x_{i-2}}{A_0} + A_1\, y_{i-2} + A_2\, y_{i-1} \\
|H(\lambda)| &= \frac{1}{1 + (\lambda_c / \lambda)^4}, \qquad \lambda_c = 2.40\ \text{mm}
\end{aligned}
```

**Slope suppression, MSD, MPD and ETD (ISO 13473-1).** In each 100 mm segment the least-squares line is subtracted; MSD compares the peaks of both halves with the mean level:

```math
\begin{aligned}
z'(x) &= z(x) - (a + b\,x) \\
MSD &= \frac{\max_{\text{1st half}} z' + \max_{\text{2nd half}} z'}{2} - \overline{z'} \\
MPD &= \frac{1}{N}\sum_{j=1}^{N} MSD_j, \qquad ETD = 1.1 \cdot MPD
\end{aligned}
```

*MSD schematic (online version): the highest peak is searched separately in each half of the segment, after slope suppression; MSD is the vertical distance between the mean of those two peaks and the mean level of the segment.*

**Spectrum (ISO 13473-4).** Each one-third-octave band has centre fₘ = 10^(n/10) m⁻¹ and edges at ±1/6 octave (base 10). The profile is extended at the start by mirroring (Annex F), band-filtered and converted to a level:

```math
\begin{aligned}
f_1 &= f_m \cdot 10^{-1/20}, \qquad f_2 = f_m \cdot 10^{1/20} \\
z_{-k} &= 2 z_0 - z_k \\
a_\lambda &= \sqrt{\frac{1}{L}\int_0^L y_\lambda^2(x)\,dx}, \qquad L_{tx,\lambda} = 20 \log_{10} \frac{a_\lambda}{10^{-6}\ \text{m}}
\end{aligned}
```

**Areal Gaussian filter (ISO 16610-61).** Weighting function and transmission, which is 50 % at λ = λc; the S-filter smooths, the L-filter removes the smoothed part:

```math
\begin{aligned}
s(x,y) &= \frac{1}{\alpha^2 \lambda_c^2} \exp\!\left(-\pi \frac{x^2 + y^2}{\alpha^2 \lambda_c^2}\right), \qquad \alpha = \sqrt{\ln 2 / \pi} \approx 0.4697 \\
H(\lambda) &= \exp\!\left[-\pi \left(\frac{\alpha \lambda_c}{\lambda}\right)^2\right] \\
Z_{SF} &= (s_{N_{is}} * Z) - \text{plane}, \qquad Z_{SL} = Z_{SF} - s_{L} * Z_{SF}
\end{aligned}
```

**Heights (ISO 25178-2).** On the filtered surface, with zero mean and evaluation area A:

```math
\begin{aligned}
S_a &= \frac{1}{A}\iint_A |z|\,dA, \qquad S_q = \sqrt{\frac{1}{A}\iint_A z^2\,dA} \\
S_{sk} &= \frac{1}{S_q^3}\,\frac{1}{A}\iint_A z^3\,dA, \qquad S_{ku} = \frac{1}{S_q^4}\,\frac{1}{A}\iint_A z^4\,dA \\
S_p &= \max z, \qquad S_v = |\min z|, \qquad S_z = S_p + S_v
\end{aligned}
```

**Slope and developed area.** Derivatives are central differences on the grid:

```math
\begin{aligned}
S_{dq} &= \sqrt{\frac{1}{A}\iint_A \left[\left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2\right] dA} \\
S_{dr} &= \frac{100\,\%}{A}\iint_A \left(\sqrt{1 + \left(\frac{\partial z}{\partial x}\right)^2 + \left(\frac{\partial z}{\partial y}\right)^2} - 1\right) dA
\end{aligned}
```

**Material ratio curve and Rk / Sk family (ISO 13565-2).** c(Mr) is the height above which a fraction Mr of the surface lies. In the 40 % window with the smallest slope, the line ℓ(Mr) = k · Mr + q is fitted:

```math
\begin{aligned}
R_k &= \ell(0) - \ell(100), \qquad c(Mr_1) = \ell(0), \qquad c(Mr_2) = \ell(100) \\
A_1 &= \int_0^{Mr_1} \left[c(Mr) - \ell(0)\right] dMr, \qquad R_{pk} = \frac{2 A_1}{Mr_1} \\
A_2 &= \int_{Mr_2}^{100} \left[\ell(100) - c(Mr)\right] dMr, \qquad R_{vk} = \frac{2 A_2}{100 - Mr_2}
\end{aligned}
```

*Material ratio curve schematic (online version): the line fitted in the 40 % window with the smallest slope, extended to 0 % and 100 %, bounds the Rk core; the triangles have the same area as the peaks above and the valleys below the core, and their heights are Rpk and Rvk.*

**Volumes (ISO 25178-2),** with p = 10 % and q = 80 %:

```math
\begin{aligned}
V_m(p) &= \frac{1}{100}\int_0^{p} \left[c(Mr) - c(p)\right] dMr, \qquad V_v(p) = \frac{1}{100}\int_p^{100} \left[c(p) - c(Mr)\right] dMr \\
V_{mp} &= V_m(10), \quad V_{mc} = V_m(80) - V_m(10), \quad V_{vc} = V_v(10) - V_v(80), \quad V_{vv} = V_v(80)
\end{aligned}
```

**Autocorrelation, Sal and Str (ISO 25178-2).** The autocorrelation is computed by FFT with zero-padding. Sal is the shortest shift at which it drops to 0.2; Str divides that by the longest:

```math
\begin{aligned}
f_{ACF}(\tau_x, \tau_y) &= \frac{\iint z(x,y)\, z(x+\tau_x, y+\tau_y)\,dx\,dy}{\iint z^2(x,y)\,dx\,dy} \\
S_{al} &= \min_{f_{ACF}(\tau) \le 0.2} \lVert \tau \rVert, \qquad S_{tr} = \frac{S_{al}}{\max_{\theta} \, \tau_{0.2}(\theta)}
\end{aligned}
```

**Hurst exponent and fractal dimension (descriptive).** In each range, the mean Welch PSD is fitted by a power law in log-log:

```math
\begin{aligned}
PSD(f) &\propto f^{-\beta}, \qquad \beta = 1 + 2H \quad \Rightarrow \quad H = \frac{\beta - 1}{2} \\
D_{\text{profile}} &= 2 - H, \qquad D_{\text{surface}} = 3 - H
\end{aligned}
```

## 5. How to run

The script is a single Python file. It runs headless and without a GPU.

**Requirements:** Python 3.10 or later and about 4 GB of RAM per file processed in parallel.

```bash
pip install -r pipeline/requirements.txt
```

**Test with one file** (recommended before the batch):

```bash
python pipeline/texturelab_batch.py --input /path/to/laz --output /path/to/results --pattern "*B6*MP1*.laz"
```

**Full batch**, inside a `tmux` session so it keeps running if the SSH connection drops:

```bash
tmux new -s texture
python pipeline/texturelab_batch.py --input /path/to/laz --output /path/to/results
```

| Option | Effect |
| --- | --- |
| `--input` | Folder with the LAZ files, searched recursively (required) |
| `--output` | Results folder, created if missing (required) |
| `--pattern` | Name filter, e.g. `"*B248*.laz"` (default: `*.laz`) |
| `--workers` | Number of files in parallel (default: automatic, from cores and RAM) |
| `--skip-done` | Skips files that already have a result with status `ok`, to resume an interrupted batch |

While running, the script prints one line per finished file, with its MPD and time. A failing file does not stop the batch: the error goes to `erro.txt` in the file's folder and to the `erro` field of the summary.

The normative settings (strip width, filters, bands) live in the `CFG` dictionary at the top of the script and are saved to `execucao.json` on every run.

## 6. Normative references

| Standard | Short title | Used for |
| --- | --- | --- |
| ISO 13473-1:2019 (DIN EN ISO 13473-1:2021-11) | Mean profile depth | Chain A: MPD, MSD, ETD; Annexes D and E |
| ISO 13473-2:2002 (DIN ISO 13473-2:2004-07) | Terminology and requirements for profile analysis | One-third-octave bands; micro and macro ranges |
| ISO 13473-4:2024 | Spectral analysis of surface profiles | Spectrum: method 1, Annexes D and F |
| ISO 13473-5 (DIN EN ISO 13473-5:2024-05) | Megatexture | Definition of microtexture (< 0.5 mm) |
| ISO 25178-2 | Areal parameters | Definition of the S and V parameters |
| ISO 25178-3:2012 (DIN EN ISO 25178-3:2012-11) | Specification operators | S-F-L order; Tables 1 and 3 (choice of S and L) |
| ISO 16610-61 | Areal Gaussian filter | S and L filters |
| ISO 13565-2:1996 (DIN EN ISO 13565-2:1998-04) | Material ratio curve: Rk | Rk and Sk family |
| IEC 61260-1 | Octave-band filters | One-third-octave band edges |
| ISO 10844, ISO 21920-2, ISO 13565-1 | — | Not yet applied |

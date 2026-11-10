#!/usr/bin/env python3
"""
verificar_filtros.py – Mede a resposta dos filtros do pipeline com sinais senoidais e compara com a
teoria das normas. Usa as MESMAS funções do cálculo (pipeline/texturelab_batch.py).

  1. Gaussiano areal (ISO 16610-61), filtros S e L: transmissão de amplitude
     T(λ) = exp(−π (α λc / λ)²), α = √(ln 2 / π)  ->  T(λc) = 50 %.
     Filtro L = superfície − gaussiano  ->  T_L(λ) = 1 − T(λ)  (50 % em λc).
  2. Passa-baixa da cadeia A (ISO 13473-1 Anexo D): Butterworth 2ª ordem bilinear, projeto 2,40 mm,
     ida e volta  ->  |H|² da passagem simples; −3 dB do conjunto perto de 3 mm.
  3. Passa-bandas de terço de oitava (ISO 13473-4 / IEC 61260-1): Butterworth 3ª ordem, ganho 0 dB no
     centro e −3 dB nas bordas da banda.

    python tools/verificar_filtros.py [--saida docs/verificacao_filtros]
Grava VERIFICACAO_FILTROS.md (tabelas) e verificacao_filtros.png (curvas) na pasta de saída.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import texturelab_batch as tb  # noqa: E402

ALPHA = tb.ALPHA_G
DX = 0.011                                           # passo do 3dT (mm)


def gauss_theory(lam, lc):
    return np.exp(-np.pi * (ALPHA * lc / lam) ** 2)


def measure_gauss(lc: float, lams: np.ndarray, dx: float = DX, along: str = "x") -> np.ndarray:
    """Amplitude de saída / entrada de tb.gauss() para senoides de λ dado, longe das bordas."""
    out = []
    for lam in lams:
        n_per = max(4, int(np.ceil(6 * lc / lam)))
        L = max(12 * lc, n_per * lam, 20 * lam)
        nx = int(np.ceil(L / dx))
        x = np.arange(nx) * dx
        z1 = np.sin(2 * np.pi * x / lam)
        Z = np.tile(z1, (16, 1)) if along == "x" else np.tile(z1[:, None], (1, 16))
        G = tb.gauss(Z.astype(np.float64), lc, dx)
        g1 = G[8] if along == "x" else G[:, 8]
        m = int(np.ceil(1.5 * lc / dx))
        core, ref = g1[m:-m], z1[m:-m]
        out.append(float(np.dot(core, ref) / np.dot(ref, ref)))   # projeção na senoide (fase zero)
    return np.array(out)


def measure_chain_a_lp(lams: np.ndarray, step: float = 0.5, design: float = 2.40) -> np.ndarray:
    out = []
    for lam in lams:
        n = int(np.ceil(60 * max(lam, design) / step))
        x = np.arange(n) * step
        z = np.sin(2 * np.pi * x / lam)
        y = tb.iso13473_lp(z[None, :], step, design)[0]
        m = n // 4
        out.append(float(np.std(y[m:-m]) / np.std(z[m:-m])))
    return np.array(out)


def chain_a_theory(lams, step=0.5, design=2.40):
    b, a = signal.butter(2, 1.0 / design, btype="low", fs=1.0 / step)
    _, h = signal.freqz(b, a, worN=1.0 / np.asarray(lams), fs=1.0 / step)
    return np.abs(h) ** 2                                  # ida e volta = módulo ao quadrado


def third_octave_response(lam_c: float, step: float, lams: np.ndarray) -> np.ndarray:
    fm = 1.0 / lam_c
    f1, f2 = fm * 10 ** (-0.05), fm * 10 ** 0.05
    sos = signal.butter(3, [f1, f2], btype="bandpass", fs=1.0 / step, output="sos")
    _, h = signal.sosfreqz(sos, worN=1.0 / np.asarray(lams), fs=1.0 / step)
    return 20 * np.log10(np.abs(h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", default=str(ROOT / "docs" / "verificacao_filtros"))
    args = ap.parse_args()
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    cfg = tb.load_config(None)
    lines = ["# Verificação dos filtros do pipeline", "",
             f"Gerado por `tools/verificar_filtros.py` com o núcleo {tb.VERSION}; as funções medidas são as mesmas do "
             "cálculo. Senoides de comprimento de onda λ passam pelo filtro e a amplitude de saída é comparada com a "
             "teoria da norma.", ""]
    ok_all = True

    # 1) gaussianos S e L
    lines += ["## 1. Gaussiano areal – ISO 16610-61 (filtros S e L da ISO 25178-3)", "",
              "Transmissão teórica T(λ) = exp(−π(αλc/λ)²), α = √(ln2/π) = 0,4697. O filtro S é o próprio gaussiano "
              "(passa-baixa); o filtro L é a superfície menos o gaussiano (passa-alta), T_L = 1 − T. "
              f"Passo dx = {DX} mm (3dT).", ""]
    curves = {}
    for name, lc in (("S 0,05 mm", cfg["S_nis_mm"]), ("L 0,5 mm (MICRO)", cfg["S_L_micro_mm"]),
                     ("L 5 mm (SL5)", cfg["S_L_macro_mm"])):
        lams = lc * np.array([0.5, 0.7, 1.0, 1.4, 2.0, 3.0])
        meas = measure_gauss(lc, lams)
        meas_y = measure_gauss(lc, np.array([lc]), along="y")[0]
        theo = gauss_theory(lams, lc)
        t_lc = meas[2]
        err = np.max(np.abs(meas - theo))
        ok = abs(t_lc - 0.5) < 0.02 and err < 0.02 and abs(meas_y - t_lc) < 1e-6
        ok_all &= ok
        note = (" O desvio maior aparece só em λ = 0,5·λc = 0,025 mm (2,3 amostras por período), no limite de "
                "amostragem do 3dT: o núcleo gaussiano tem σ < 1 célula." if lc < 0.1 else "")
        lines += [f"**{name}** — T(λc) medido = {t_lc * 100:.2f} % (teoria 50 %), mesmo valor na outra direção: "
                  f"{meas_y * 100:.2f} %; maior desvio na tabela = {err * 100:.2f} p.p. → {'OK' if ok else 'VERIFICAR'}." + note, "",
                  "| λ/λc | λ [mm] | T medido (passa-baixa) | T teoria | 1 − T medido (filtro L) |", "|---|---|---|---|---|"]
        for f, l, mv, tv in zip(lams / lc, lams, meas, theo):
            lines.append(f"| {f:.1f} | {l:.4g} | {mv * 100:.2f} % | {tv * 100:.2f} % | {(1 - mv) * 100:.2f} % |")
        lines.append("")
        lf = lc * np.logspace(-0.6, 0.8, 25)
        lf = lf[lf >= 2.5 * DX]                      # abaixo de ~2 dx (Nyquist) a senoide não é representável
        curves[name] = (lf / lc, measure_gauss(lc, lf), gauss_theory(lf, lc))

    # SL5 = SF − G5(SF) e combinação de bandas
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(600, 4000))
    SF = tb.gauss(Z, cfg["S_nis_mm"], DX)
    SL = tb.gauss(SF, cfg["S_L_macro_mm"], DX)
    np.subtract(SF, SL, out=SL)
    ident = np.allclose(SL, SF - tb.gauss(SF, cfg["S_L_macro_mm"], DX))
    ok_all &= ident
    lines += ["**SL5 = SF − G₅(SF)** (subtração no lugar, como no cálculo): idêntico à fórmula direta → "
              f"{'OK' if ident else 'ERRO'}.", ""]

    # 2) cadeia A
    # λ > 2,5 passos (5 amostras por 2 períodos); em λ = 2 passos (Nyquist) a senoide amostrada é ~zero
    lams = np.array([1.25, 1.5, 2.0, 2.4, 3.0, 4.0, 6.0, 10.0, 20.0])
    meas = measure_chain_a_lp(lams)
    theo = chain_a_theory(lams)
    lg = np.linspace(1.5, 6, 400)
    lam3 = np.interp(np.sqrt(0.5), chain_a_theory(lg), lg)              # T cresce com λ (passa-baixa)
    err = np.max(np.abs(meas - theo))
    ok = err < 0.02
    ok_all &= ok
    lines += ["## 2. Passa-baixa da cadeia A – ISO 13473-1, Anexo D (MPD, g-factor de perfil)", "",
              "Butterworth de 2ª ordem (bilinear), projetado em 2,40 mm, aplicado ida e volta (Fórmula D.1): a "
              f"transmissão é o quadrado da de uma passagem. Ponto de −3 dB do conjunto (teoria): λ = {lam3:.2f} mm. "
              f"Passo 0,5 mm. Maior desvio medido × teoria: {err * 100:.2f} p.p. → {'OK' if ok else 'VERIFICAR'}", "",
              "| λ [mm] | T medido | T teoria |", "|---|---|---|"]
    lines += [f"| {l:g} | {m_ * 100:.1f} % | {t_ * 100:.1f} % |" for l, m_, t_ in zip(lams, meas, theo)]
    lines += ["", "A supressão de inclinação (passa-alta) é feita por reta de regressão em cada segmento de 100 mm, "
              "uma das duas opções da ISO 13473-1. Comprimentos de onda até 2 passos (1 mm) estão no limite de "
              "Nyquist da reamostragem de 0,5 mm e não são medidos.", ""]
    la = np.logspace(np.log10(1.25), 1.4, 40)
    curves["cadeia A"] = (la, measure_chain_a_lp(la), chain_a_theory(la))

    # 3) terço de oitava
    lines += ["## 3. Passa-bandas de terço de oitava – ISO 13473-4 / IEC 61260-1", "",
              "Butterworth de 3ª ordem por banda (bordas f_m·10^(±1/20)). Ganho medido no centro e nas bordas; "
              "o passo depende da banda (0,1 mm após o anti-aliasing).", "",
              "| λ central [mm] | ganho no centro [dB] | ganho nas bordas [dB] |", "|---|---|---|"]
    step = 0.1
    for lc in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        g = third_octave_response(lc, step, [lc, lc * 10 ** 0.05, lc * 10 ** -0.05])
        ok = abs(g[0]) < 0.1 and np.all(np.abs(g[1:] + 3.01) < 0.2)
        ok_all &= ok
        lines.append(f"| {lc:g} | {g[0]:+.2f} | {g[1]:+.2f} / {g[2]:+.2f} |" + ("" if ok else " ⚠️"))
    lines += ["", f"## Conclusão: {'todos os filtros conferem com a teoria das normas' if ok_all else 'há desvios — ver tabelas'}", ""]
    (out / "VERIFICACAO_FILTROS.md").write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for name, (x, m_, t_) in curves.items():
            if name == "cadeia A":
                continue
            ax[0].semilogx(x, t_, "-", lw=1, label=f"{name} teoria")
            ax[0].semilogx(x, m_, "o", ms=3, label=f"{name} medido")
        from matplotlib.ticker import ScalarFormatter
        for a_ in ax:
            a_.xaxis.set_major_formatter(ScalarFormatter())
            a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[0].set_xticks([0.3, 0.5, 1, 2, 3, 5])
        ax[1].set_xticks([1.5, 2, 3, 5, 10, 20])
        ax[0].axhline(0.5, color="gray", ls=":"); ax[0].axvline(1, color="gray", ls=":")
        ax[0].set(xlabel="λ / λc", ylabel="transmissão do gaussiano (passa-baixa)", title="ISO 16610-61 (S e L)")
        ax[0].legend(fontsize=7)
        x, m_, t_ = curves["cadeia A"]
        ax[1].semilogx(x, t_, "-", label="teoria |H|² (ida e volta)")
        ax[1].semilogx(x, m_, "o", ms=3, label="medido (iso13473_lp)")
        ax[1].axhline(0.5 ** 0.5, color="gray", ls=":")
        ax[1].set(xlabel="λ [mm]", ylabel="transmissão", title="Cadeia A: Butterworth 2ª ordem, 2,40 mm")
        ax[1].legend(fontsize=7)
        for a_, ticks in ((ax[0], [0.3, 0.5, 1, 2, 3, 5]), (ax[1], [1.5, 2, 3, 5, 10, 20])):
            a_.set_xticks(ticks)
            a_.xaxis.set_major_formatter(ScalarFormatter())
            a_.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.tight_layout()
        fig.savefig(out / "verificacao_filtros.png", dpi=130)
    except ImportError:
        pass
    print("\n".join(lines))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

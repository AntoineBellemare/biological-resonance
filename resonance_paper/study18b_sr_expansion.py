"""Study 18b — Stochastic-resonance go/no-go: is noise-induced harmonic resonance
strong enough to carry its own paper?

Three tests, two oscillator models (van der Pol + Stuart-Landau) for universality:
  (1) INVERTED-U: per ratio, is H(noise) non-monotonic with an INTERIOR peak (the
      stochastic-resonance hallmark)? Quantify the fraction of complex ratios with
      an interior optimum and the optimal noise sigma*.
  (2) SIMPLE vs COMPLEX: is the noise-induced H gain larger for complex (hard-to-
      lock) ratios than simple ones?
  (3) TONGUE-FILLING: at complex ratios (no deterministic Arnold tongue), does the
      best-noise H rise materially above the noiseless H?

Decision rule (printed): the SR story stands alone if BOTH models show the
inverted-U in a majority of complex ratios AND complex > simple SR gain.

Outputs: results/study18b_sr_expansion.json, figures/study18b_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from fractions import Fraction

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from resonance_paper.study18_stochastic_resonance import forced_vdp, _sf, OMEGA0, DT, TARGET_F0
from biotuner.resonance import compute_resonance


def forced_stuart_landau(omega_drive, amp=1.0, noise=0.0, mu=1.0, dur=320.0, seed=0):
    """Generic limit-cycle oscillator (normal form) forced at omega_drive."""
    rng = np.random.default_rng(seed)
    n = int(dur / DT); z = 0.1 + 0j; out = np.empty(n); sdt = np.sqrt(DT)
    for t in range(n):
        drive = amp * np.exp(1j * omega_drive * t * DT)
        z = z + DT * ((mu + 1j * OMEGA0 - abs(z) ** 2) * z + drive) \
            + noise * sdt * (rng.standard_normal() + 1j * rng.standard_normal())
        out[t] = z.real
    return out


MODELS = {"van_der_Pol": forced_vdp, "Stuart_Landau": forced_stuart_landau}


def _complexity(omega):
    fr = Fraction(float(omega)).limit_denominator(9)
    return float(np.log2(fr.numerator * fr.denominator)), f"{fr.numerator}:{fr.denominator}"


def _interior_peak(noises, H):
    """Return (has_interior_peak, sigma_star, gain) for an H(noise) curve."""
    H = np.asarray(H, float)
    if not np.all(np.isfinite(H)) or H.max() <= 0:
        return False, float("nan"), float("nan")
    i = int(np.argmax(H))
    interior = 0 < i < len(H) - 1 and H[i] > H[0] and H[i] >= H[-1]
    return bool(interior), float(noises[i]), float(H[i] - H[0])


def run(quick=True):
    sf = _sf()
    omegas = np.round(np.linspace(0.5, 3.0, 10 if quick else 16), 3)
    noises = [0.0, 0.05, 0.1, 0.2, 0.35, 0.55, 0.8, 1.1] if quick else \
             [0.0, 0.04, 0.08, 0.15, 0.25, 0.4, 0.6, 0.85, 1.15, 1.5]
    seeds = range(2) if quick else range(4)
    warm = int(60.0 / DT)
    cfg = C.default_config(fmin=2, fmax=60, precision_hz=0.5)

    out = {}
    for mname, mfun in MODELS.items():
        rows = []
        for om in omegas:
            comp, rstr = _complexity(om)
            Hs = []
            for nz in noises:
                hh = []
                for seed in seeds:
                    y = mfun(om, noise=nz, seed=seed)[warm:]
                    r = compute_resonance(_norm(y).astype(np.float64), sf=sf, config=cfg)
                    hh.append(float(r.summaries["H"]["max"]))
                Hs.append(float(np.mean(hh)))
            interior, sstar, gain = _interior_peak(noises, Hs)
            rows.append(dict(omega=float(om), ratio=rstr, complexity=comp,
                             H_by_noise=Hs, interior_peak=interior,
                             sigma_star=sstar, sr_gain=gain))
            print(f"  [{mname}] omega={om:.2f} ({rstr}) interior={interior} gain={gain:+.3f}", flush=True)

        comp = np.array([r["complexity"] for r in rows])
        med = np.median(comp)
        simple = comp <= med; complex_ = comp > med
        out[mname] = dict(
            noises=noises, rows=rows,
            frac_interior_complex=float(np.mean([r["interior_peak"] for r, m in zip(rows, complex_) if m])),
            frac_interior_simple=float(np.mean([r["interior_peak"] for r, m in zip(rows, simple) if m])),
            sr_gain_complex=float(np.nanmean([r["sr_gain"] for r, m in zip(rows, complex_) if m])),
            sr_gain_simple=float(np.nanmean([r["sr_gain"] for r, m in zip(rows, simple) if m])))

    verdict = _verdict(out)
    result = dict(quick=quick, models=out, verdict=verdict)
    C.save_json(result, "study18b_sr_expansion.json")
    _figures(result)
    _headline(result)
    return result


def _verdict(out):
    both_invU = all(m["frac_interior_complex"] >= 0.5 for m in out.values())
    both_cgs = all(m["sr_gain_complex"] > m["sr_gain_simple"] for m in out.values())
    return dict(both_models_inverted_U=bool(both_invU),
                both_models_complex_gt_simple=bool(both_cgs),
                stands_alone=bool(both_invU and both_cgs))


def _headline(result):
    print("\n  --- Study 18b headline (stochastic-resonance go/no-go) ---")
    for m, d in result["models"].items():
        print(f"  [{m}] interior-peak fraction: complex={d['frac_interior_complex']:.2f} "
              f"simple={d['frac_interior_simple']:.2f} | SR gain: complex={d['sr_gain_complex']:+.3f} "
              f"simple={d['sr_gain_simple']:+.3f}")
    v = result["verdict"]
    print(f"  inverted-U in both models (>=50% complex): {v['both_models_inverted_U']}")
    print(f"  complex>simple SR gain in both models: {v['both_models_complex_gt_simple']}")
    print(f"  => SR STANDS ALONE AS A PAPER: {v['stands_alone']}")


def _figures(result):
    plt = C.setup_mpl()
    models = result["models"]
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(6 * len(models) + 5, 4.4))
    for ax, (mname, d) in zip(axes[:len(models)], models.items()):
        noises = d["noises"]
        comps = [r["complexity"] for r in d["rows"]]
        cmin, cmax = min(comps), max(comps)
        for r in d["rows"]:
            frac = (r["complexity"] - cmin) / (cmax - cmin + 1e-9)
            col = plt.cm.viridis(frac)
            H = np.array(r["H_by_noise"]); H = H - H[0]            # gain over noiseless
            ax.plot(noises, H, "-", color=col, alpha=0.8, lw=1.3)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("noise level"); ax.set_ylabel("H gain over noiseless")
        ax.set_title(f"{mname}\n(color = ratio complexity)", fontsize=10)
    # summary bars
    ax = axes[-1]
    labels = list(models.keys()); x = np.arange(len(labels))
    ax.bar(x - 0.18, [models[m]["sr_gain_simple"] for m in labels], width=0.36, label="simple", color="#1565c0")
    ax.bar(x + 0.18, [models[m]["sr_gain_complex"] for m in labels], width=0.36, label="complex", color="#b71c1c")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean SR gain in H"); ax.set_title("Complex > simple?\n(noise-induced gain)", fontsize=10); ax.legend(fontsize=8)
    fig.suptitle("Study 18b — Stochastic-resonance go/no-go: inverted-U + complex>simple across two models",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study18b_sr_expansion")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

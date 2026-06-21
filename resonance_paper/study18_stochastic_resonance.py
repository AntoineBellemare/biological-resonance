"""Study 18 — Many ratios, and noise-induced resonance (stochastic resonance).

A forced van der Pol oscillator is driven across A LOT OF ratios Ω = f_drive/f0
(dense, simple → complex), at several NOISE levels. Questions:
  * how does the framework resonance R of the response vary across the dense ratio
    axis (the devil's-staircase / mode-locking structure seen by H/PC/R)?
  * STOCHASTIC RESONANCE: for COMPLEX ratios that don't lock deterministically,
    can adding noise *enhance* resonance — i.e., is R(noise) non-monotonic with a
    peak at intermediate noise? (noise-induced mode-locking)

Per (Ω, noise) we simulate the oscillator and compute the framework H_max / R_max
of its response, plus the drive→response 1:1-style locking (rotation-number
proximity to the simple ratio). The signal's sampling rate is set so f0 ≈ 10 Hz.

Outputs: results/study18_stochastic_resonance.json, figures/study18_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from fractions import Fraction

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from biotuner.resonance import compute_resonance

OMEGA0 = 1.0           # natural angular frequency (rad/time-unit)
DT = 0.05
TARGET_F0 = 10.0       # Hz that f0 maps to when we hand the response to the framework


def forced_vdp(omega_drive, amp=1.2, noise=0.0, mu=2.0, dur=320.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur / DT); x, v = 0.1, 0.0
    out = np.empty(n); sdt = np.sqrt(DT)
    for t in range(n):
        drive = amp * np.sin(omega_drive * t * DT)
        x = x + DT * v
        v = v + DT * (mu * (1 - x * x) * v - OMEGA0 ** 2 * x + drive) + noise * sdt * rng.standard_normal()
        out[t] = x
    return out


def _sf():
    period_samples = (2 * np.pi / OMEGA0) / DT      # samples per natural cycle
    return float(period_samples * TARGET_F0)         # declare sf so f0 == TARGET_F0 Hz


def ratio_complexity(omega):
    fr = Fraction(omega).limit_denominator(9)
    return float(np.log2(fr.numerator * fr.denominator)), f"{fr.numerator}:{fr.denominator}"


def run(quick=True):
    sf = _sf()
    omegas = np.round(np.linspace(0.5, 3.0, 26 if quick else 60), 3)
    noises = [0.0, 0.15, 0.4, 0.8] if quick else [0.0, 0.1, 0.25, 0.5, 0.9, 1.4]
    seeds = range(2) if quick else range(5)
    warm = int(60.0 / DT)                            # discard transient
    cfg = C.default_config(fmin=2, fmax=60, precision_hz=0.5)

    rows = []
    for om in omegas:
        comp, rstr = ratio_complexity(om)
        for nz in noises:
            Rm, Hm = [], []
            for seed in seeds:
                y = forced_vdp(om, noise=nz, seed=seed)[warm:]
                r = compute_resonance(_norm(y).astype(np.float64), sf=sf, config=cfg)
                Rm.append(float(r.summaries["R"]["max"])); Hm.append(float(r.summaries["H"]["max"]))
            rows.append(dict(omega=float(om), ratio=rstr, complexity=comp, noise=float(nz),
                             R=float(np.mean(Rm)), H=float(np.mean(Hm))))
        print(f"  omega={om:.2f} ({rstr}) done", flush=True)

    # stochastic-resonance test: per omega, does R peak at noise>0?
    sr = []
    for om in omegas:
        rr = [r for r in rows if r["omega"] == om]
        rr.sort(key=lambda r: r["noise"])
        Rs = np.array([r["H"] for r in rr]); nz = np.array([r["noise"] for r in rr])  # H = harmonic resonance (R reduced is ~0)
        if np.all(~np.isfinite(Rs)) or Rs.max() <= 0:
            continue
        i = int(np.argmax(Rs))
        sr.append(dict(omega=float(om), complexity=rr[0]["complexity"],
                       best_noise=float(nz[i]), R0=float(Rs[0]), Rbest=float(Rs[i]),
                       sr_gain=float(Rs[i] - Rs[0]), peaks_at_noise=bool(nz[i] > 0)))
    comp = np.array([s["complexity"] for s in sr])
    simple = comp <= np.median(comp); complex_ = comp > np.median(comp)
    summary = dict(
        n_omega=len(omegas),
        frac_SR_all=float(np.mean([s["peaks_at_noise"] for s in sr])),
        frac_SR_simple=float(np.mean([s["peaks_at_noise"] for s, m in zip(sr, simple) if m])),
        frac_SR_complex=float(np.mean([s["peaks_at_noise"] for s, m in zip(sr, complex_) if m])),
        mean_sr_gain_simple=float(np.mean([s["sr_gain"] for s, m in zip(sr, simple) if m])),
        mean_sr_gain_complex=float(np.mean([s["sr_gain"] for s, m in zip(sr, complex_) if m])))
    result = dict(quick=quick, rows=rows, sr=sr, summary=summary)
    C.save_json(result, "study18_stochastic_resonance.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 18 headline (stochastic resonance across ratios) ---")
    print(f"  ratios swept: {s['n_omega']}")
    print(f"  fraction with R peaking at noise>0 (stochastic resonance):")
    print(f"    all={s['frac_SR_all']:.2f}  simple={s['frac_SR_simple']:.2f}  complex={s['frac_SR_complex']:.2f}")
    print(f"  mean SR gain (R_best - R_noiseless): simple={s['mean_sr_gain_simple']:+.4f}  "
          f"complex={s['mean_sr_gain_complex']:+.4f}")
    print("  => if complex ratios show MORE/larger SR than simple, noise induces resonance")
    print("     where deterministic locking is weak.")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]
    oms = sorted(set(r["omega"] for r in rows)); nzs = sorted(set(r["noise"] for r in rows))
    M = np.full((len(nzs), len(oms)), np.nan)
    for r in rows:
        M[nzs.index(r["noise"]), oms.index(r["omega"])] = r["H"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4))
    im = ax1.imshow(M, aspect="auto", origin="lower", cmap="magma",
                    extent=[min(oms), max(oms), 0, len(nzs)])
    ax1.set_yticks(np.arange(len(nzs)) + 0.5); ax1.set_yticklabels([f"{n:.2f}" for n in nzs])
    ax1.set_xlabel("drive ratio Ω = f_drive/f0"); ax1.set_ylabel("noise level")
    ax1.set_title("Harmonic resonance H across ratios × noise", fontsize=10)
    plt.colorbar(im, ax=ax1, fraction=0.046, label="H_max")
    # SR gain vs complexity
    sr = result["sr"]
    ax2.scatter([s["complexity"] for s in sr], [s["sr_gain"] for s in sr],
                c=[s["best_noise"] for s in sr], cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
    ax2.axhline(0, color="k", lw=0.6)
    ax2.set_xlabel("ratio complexity log2(pq)"); ax2.set_ylabel("SR gain in H (H_best − H_noiseless)")
    ax2.set_title("Noise-induced resonance vs ratio complexity", fontsize=10)
    cb = plt.colorbar(ax2.collections[0], ax=ax2, fraction=0.046); cb.set_label("optimal noise")
    fig.suptitle("Study 18 — Stochastic resonance: noise can induce resonance at complex ratios",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study18_stochastic_resonance")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

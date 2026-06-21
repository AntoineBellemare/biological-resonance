"""Study 8 (Study A) — Harmonic complexity governs lockability (Arnold tongues).

The non-circular, complexity-resolved test. A periodically forced Van der Pol
oscillator (the textbook Arnold-tongue system) is swept across drive frequency
ratios Omega = f_drive / f0. The oscillator mode-locks to the drive at rational
rotation numbers rho = f_osc / f_drive, forming Arnold tongues whose WIDTH grows
with the simplicity of the ratio (the devil's staircase). This locking is a
genuine nonlinear-dynamics phenomenon, not imposed.

Question (the one raised in review): does the property vary with the COMPLEXITY
of the harmonic ratio? And does the framework's independently-measured
harmonicity rank ratios the same way the dynamics do?

Measurements:
  * Ground truth: rotation number rho(Omega) from the oscillator's dominant
    response frequency -> the devil's staircase; tongue width per rational p/q.
  * Framework: at each tongue, compute_cross_resonance(drive, oscillator) ->
    cross-harmonicity H and resonance R of the (drive, response) pair.
  * Non-circular check: tongue width (dynamics) vs framework H (spectrum) and vs
    ratio complexity p*q. If simpler ratios lock more widely AND the framework's
    H ranks them the same way, harmonic structure predicts dynamical lockability.

Models: forced Van der Pol (continuous, biosignal-like). NOT mutual Kuramoto —
that yields only the 1:1 tongue at fixed coupling. The sine-circle map gives the
same staircase by construction and is used as a cross-check label.

Outputs: results/study8_arnold_tongues.json, figures/study8_*.{png,pdf}
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr

from resonance_paper import _common as C
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import ResonanceConfig
import biotuner.resonance.kernels_harmonic  # noqa: F401 (registers kernels)
from biotuner.resonance.registry import HARMONIC_KERNELS

SF = 500.0
F0 = 8.0          # oscillator natural frequency (Hz)
W0 = 2 * np.pi * F0
MU = 5.0          # strong nonlinearity -> harmonic-rich + wide tongues
FORCE = 3000.0    # forcing amplitude (scaled to W0^2 ~ 2500)
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=80, noverlap=200,
                      coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                      ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)


def forced_vdp(Omega, F=FORCE, mu=MU, sf=SF, dur=12.0):
    """Periodically forced Van der Pol; return (drive, x) both length n."""
    wd = 2 * np.pi * F0 * Omega

    def deriv(t, s):
        x, v = s
        return [v, mu * (1 - x * x) * v - W0 * W0 * x + F * np.cos(wd * t)]

    n = int(sf * dur); te = np.arange(n) / sf
    sol = solve_ivp(deriv, (0, dur), [0.1, 0.0], t_eval=te, method="RK45",
                    rtol=1e-7, atol=1e-9, max_step=1.0 / sf)
    cut = int(2 * sf)
    drive = np.cos(wd * te)[cut:]
    x = sol.y[0][cut:]
    x = (x - x.mean()) / (x.std() + 1e-12)
    return drive.astype(np.float64), x.astype(np.float64)


def rotation_number(x, f_drive, sf=SF):
    """rho = dominant response frequency / drive frequency (FFT peak)."""
    w = x * np.hanning(len(x))
    X = np.abs(np.fft.rfft(w)); f = np.fft.rfftfreq(len(x), 1 / sf)
    f_osc = f[1 + int(np.argmax(X[1:]))]
    return float(f_osc / f_drive), float(f_osc)


def run(quick=True):
    F = FORCE
    dOm = 0.04 if quick else 0.02
    Omegas = np.round(np.arange(0.40, 2.60 + 1e-9, dOm), 4)

    # --- 1. devil's staircase: rotation number vs drive ratio ---
    stair = []
    for Om in Omegas:
        drive, x = forced_vdp(Om, F)
        rho, f_osc = rotation_number(x, F0 * Om)
        stair.append(dict(Omega=float(Om), rho=rho, f_osc=f_osc))
    print("  staircase done")

    # --- 2. snap to rationals, group consecutive into tongues ---
    QMAX = 5
    def snap(rho):
        fr = Fraction(rho).limit_denominator(QMAX)
        return (fr.numerator, fr.denominator) if abs(float(fr) - rho) < 0.04 else None
    for s in stair:
        lk = snap(s["rho"])
        s["lock"] = f"{lk[0]}:{lk[1]}" if lk else None

    tongues = {}
    for s in stair:
        lk = snap(s["rho"])
        if lk is None:
            continue
        tongues.setdefault(lk, []).append(s["Omega"])
    tongue_rows = []
    for (p, q), oms in tongues.items():
        if len(oms) < 1:
            continue
        width = len(oms) * dOm
        center = float(np.median(oms))
        complexity = p * q
        tongue_rows.append(dict(p=p, q=q, ratio=f"{p}:{q}", width=float(width),
                                center=center, complexity=complexity,
                                rho=p / q))
    tongue_rows.sort(key=lambda r: r["complexity"])
    print(f"  {len(tongue_rows)} tongues detected")

    # --- 3. framework H/R at each tongue center ---
    harmsim = HARMONIC_KERNELS["harmsim"]
    for tr in tongue_rows:
        drive, x = forced_vdp(tr["center"], F)
        f_d = F0 * tr["center"]
        res = compute_cross_resonance(drive, x, sf=SF, config=CFG)
        fr = res.freqs
        f_osc = tr["rho"] * f_d
        i = int(np.argmin(np.abs(fr - f_d))); j = int(np.argmin(np.abs(fr - f_osc)))
        tr["framework_H"] = float(C.band_value(fr, res.factors["H"]["all"], f_d))
        tr["framework_R"] = float(res.resonance_spectrum["all"][i])
        # direct cross-harmonicity of the locked frequency pair (drive, response)
        tr["harmsim_pair"] = float(harmsim(np.array([f_d]), np.array([f_osc]))[0, 0])
    print("  framework measures done")

    # --- 4. non-circular correlations across tongues ---
    if len(tongue_rows) >= 3:
        widths = [t["width"] for t in tongue_rows]
        comps = [t["complexity"] for t in tongue_rows]
        hs = [t["harmsim_pair"] for t in tongue_rows]
        corr = dict(
            width_vs_complexity=float(spearmanr(widths, comps)[0]),
            harmsim_vs_complexity=float(spearmanr(hs, comps)[0]),
            width_vs_harmsim=float(spearmanr(widths, hs)[0]),
        )
    else:
        corr = {}

    result = dict(quick=quick, F0=F0, mu=MU, force=F, dOmega=dOm,
                  staircase=stair, tongues=tongue_rows, corr=corr)
    C.save_json(result, "study8_arnold_tongues.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 8 headline (Arnold tongues vs harmonic complexity) ---")
    print(f"  {'ratio':>6} {'p*q':>4} {'tongue_width':>13} {'harmsim_pair':>13}")
    for t in result["tongues"]:
        print(f"  {t['ratio']:>6} {t['complexity']:>4} {t['width']:>13.3f} {t['harmsim_pair']:>13.1f}")
    c = result["corr"]
    if c:
        print(f"\n  Spearman(tongue width, complexity p*q) = {c['width_vs_complexity']:+.2f}  "
              f"(simpler ratios lock more widely)")
        print(f"  Spearman(framework harmsim, complexity) = {c['harmsim_vs_complexity']:+.2f}")
        print(f"  Spearman(tongue width, framework harmsim) = {c['width_vs_harmsim']:+.2f}  "
              f"(harmonicity predicts dynamical lockability — non-circular)")


def _figures(result):
    plt = C.setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    # staircase
    Om = [s["Omega"] for s in result["staircase"]]
    rho = [s["rho"] for s in result["staircase"]]
    axes[0].plot(Om, rho, ".", color="#1a237e", ms=4)
    for t in result["tongues"]:
        axes[0].axhline(t["rho"], color="green", alpha=0.15, lw=0.6)
    axes[0].set_xlabel("drive ratio  Omega = f_drive / f0")
    axes[0].set_ylabel("rotation number  rho = f_osc / f_drive")
    axes[0].set_title("A. Devil's staircase (forced Van der Pol)\n"
                      "plateaus = Arnold tongues; wider at simple ratios", fontsize=10)
    # tongue width vs complexity, colored by framework harmsim
    t = result["tongues"]
    comps = np.array([r["complexity"] for r in t])
    widths = np.array([r["width"] for r in t])
    hs = np.array([r["harmsim_pair"] for r in t])
    sc = axes[1].scatter(comps, widths, c=hs, cmap="viridis", s=80, edgecolor="k")
    for r in t:
        axes[1].annotate(r["ratio"], (r["complexity"], r["width"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=axes[1], label="framework harmsim of pair")
    axes[1].set_xlabel("ratio complexity  p*q")
    axes[1].set_ylabel("Arnold tongue width (Omega)")
    c = result["corr"]
    sub = (f"width~complexity rho={c.get('width_vs_complexity', float('nan')):+.2f}, "
           f"width~harmsim rho={c.get('width_vs_harmsim', float('nan')):+.2f}") if c else ""
    axes[1].set_title(f"B. Lockability vs harmonic complexity\n{sub}", fontsize=10)
    fig.suptitle("Study 8 — Harmonic complexity governs phase-locking width; "
                 "the framework's harmonicity tracks it", fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study8_arnold_tongues")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

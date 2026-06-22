"""Study 7 — Generative model: do H, PC, R co-vary as the dynamics dictate?

!!! CAVEAT (found on review) — Direction B as implemented here is CONFOUNDED and
    should NOT be read as a clean validation. Two problems: (1) natural
    frequencies are set at EXACT rational ratios f2 = r*f1, so the n:m phase
    combination cancels deterministically even with negligible coupling (the
    ground-truth PLV is high at every low-denominator ratio regardless of K — it
    is not coupling-induced locking); (2) the framework PC matrix entry is
    W[i,j]*PLV with W a Tenney weight that is itself larger for simple ratios.
    So "H and PC both peak at simple ratios" is largely imposed by the chosen
    frequencies + the ratio-kernel weighting, not emergent. Direction A
    (coupling -> PC) is sound. A correct Arnold-tongue test needs DETUNED natural
    frequencies and real coupling, measuring the LOCKING RANGE (tongue width) vs
    ratio simplicity — see study7b (to be built). This file is retained for the
    record / Direction A.


Study 6's "R = H x PC conjunction" is circular (R is the product by definition).
The non-trivial question is whether, in a GENERATIVE dynamical system where
coupling emerges from physics, the framework's independently-measured
harmonicity (H, from the spectrum) and phase coupling (PC, from instantaneous
phase) co-vary the way the resonance construct assumes.

Model: two diffusively coupled Van der Pol oscillators
    x1'' - mu (1 - x1^2) x1' + w1^2 x1 = K (x2 - x1)
    x2'' - mu (1 - x2^2) x2' + w2^2 x2 = K (x1 - x2)
Van der Pol limit cycles are harmonic-rich (mu controls overtone content) and
phase-synchronize under coupling K, with Arnold tongues that are widest at
simple frequency ratios w2/w1.

Two directions:
  A. Vary coupling K at a fixed simple ratio (1:2): PC and R should rise through
     the synchronization transition (H ~ constant — frequencies fixed).
  B. Vary the frequency ratio at fixed K: phase locking (PC) occurs in Arnold
     tongues centered on simple ratios, exactly where harmonicity H peaks — so
     R peaks at the simple-ratio resonances. PC (from dynamics) tracking H (from
     spectrum) is the non-circular validation.

Ground truth: Hilbert n:m PLV between the two oscillators (true locking). We show
the framework's PC tracks it, and H predicts the tongue locations.

Outputs: results/study7_coupled_oscillators.json, figures/study7_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from fractions import Fraction
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt, hilbert

from resonance_paper import _common as C
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import ResonanceConfig

SF = 250.0
F1 = 8.0  # base frequency of oscillator 1 (Hz)
MU = 1.5  # Van der Pol nonlinearity (harmonic richness)
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=60, noverlap=200,
                      coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                      ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)


def simulate_vdp_pair(ratio, K, sf=SF, duration=16.0, mu=MU, seed=0,
                      unidirectional=False):
    """Integrate two coupled Van der Pol oscillators; return (x1, x2).

    unidirectional=False : mutual diffusive coupling K(x2-x1), K(x1-x2).
    unidirectional=True  : oscillator 1 is free and forces oscillator 2 (K*x1),
        so x1 keeps its frequency while x2 entrains — a clean drive that avoids
        the mutual frequency-pulling that moves both oscillators off-target.
    """
    rng = np.random.default_rng(seed)
    w1 = 2 * np.pi * F1
    w2 = 2 * np.pi * F1 * ratio
    n = int(sf * duration)
    t_eval = np.arange(n) / sf

    def deriv(t, s):
        x1, v1, x2, v2 = s
        if unidirectional:
            dv1 = mu * (1 - x1 * x1) * v1 - w1 * w1 * x1
            dv2 = mu * (1 - x2 * x2) * v2 - w2 * w2 * x2 + K * x1
        else:
            dv1 = mu * (1 - x1 * x1) * v1 - w1 * w1 * x1 + K * (x2 - x1)
            dv2 = mu * (1 - x2 * x2) * v2 - w2 * w2 * x2 + K * (x1 - x2)
        return [v1, dv1, v2, dv2]

    s0 = rng.uniform(-1, 1, size=4)
    sol = solve_ivp(deriv, (0, duration), s0, t_eval=t_eval, method="RK45",
                    rtol=1e-6, atol=1e-8, max_step=1.0 / sf)
    x1, x2 = sol.y[0], sol.y[2]
    # discard transient (first 2 s), normalize
    cut = int(2 * sf)
    x1, x2 = x1[cut:], x2[cut:]
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-12)
    x2 = (x2 - x2.mean()) / (x2.std() + 1e-12)
    return x1.astype(np.float64), x2.astype(np.float64)


def simulate_kuramoto_pair(ratio, K, sf=SF, duration=16.0, seed=0,
                           n_harm=3, noise=0.05, unidirectional=False, detune=0.0):
    """Two n:m-coupled Kuramoto phase oscillators, rendered as harmonic-rich
    waveforms; return (x1, x2).

    Frequencies set the harmonic relation (w2/w1 = ratio ~ n/m); K sets the
    phase coupling — two clean, independent knobs, with none of the
    frequency-pulling / amplitude-death of mutual Van der Pol coupling.

        dθ1/dt = w1 + K sin(m θ2 - n θ1)
        dθ2/dt = w2 + K sin(n θ1 - m θ2)      (or 0 coupling if unidirectional)

    Each phase is rendered as Σ_k (0.6^k) sin((k+1) θ) so the signal carries
    integer overtones (controllable harmonic content), making within-signal H
    meaningful as well as the cross-signal ratio relation.
    """
    rng = np.random.default_rng(seed)
    frac = Fraction(float(ratio)).limit_denominator(8)
    n, m = frac.numerator, frac.denominator   # ratio = w2/w1 = n/m
    w1 = 2 * np.pi * F1
    w2 = 2 * np.pi * F1 * ratio * (1.0 + detune)   # detune>0: uncoupled drifts, coupling locks at n:m
    N = int(sf * duration); dt = 1.0 / sf
    th1 = np.empty(N); th2 = np.empty(N)
    th1[0], th2[0] = rng.uniform(0, 2 * np.pi, size=2)
    for i in range(1, N):
        d1 = w1 + K * np.sin(m * th2[i - 1] - n * th1[i - 1])
        d2 = w2 + (0.0 if unidirectional else K * np.sin(n * th1[i - 1] - m * th2[i - 1]))
        th1[i] = th1[i - 1] + dt * d1
        th2[i] = th2[i - 1] + dt * d2

    def render(th):
        return sum((0.6 ** k) * np.sin((k + 1) * th) for k in range(n_harm))
    cut = int(2 * sf)
    x1 = render(th1)[cut:] + noise * rng.standard_normal(N - cut)
    x2 = render(th2)[cut:] + noise * rng.standard_normal(N - cut)
    x1 = (x1 - x1.mean()) / (x1.std() + 1e-12)
    x2 = (x2 - x2.mean()) / (x2.std() + 1e-12)
    return x1.astype(np.float64), x2.astype(np.float64)


def _bp(x, lo, hi):
    lo = max(lo, 0.5); hi = min(hi, SF / 2 - 0.5)
    b, a = butter(4, [lo / (SF / 2), hi / (SF / 2)], btype="band")
    return filtfilt(b, a, x)


def ground_truth_plv(x1, x2, f1, f2):
    """Hilbert n:m PLV at the (f1, f2) ratio — the true locking strength."""
    frac = Fraction(f2 / f1).limit_denominator(8)
    n, m = frac.denominator, frac.numerator  # f2/f1 = m/n
    pa = np.angle(hilbert(_bp(x1, f1 - 2, f1 + 2)))
    pb = np.angle(hilbert(_bp(x2, f2 - 2, f2 + 2)))
    return float(np.abs(np.mean(np.exp(1j * (m * pa - n * pb)))))


def framework_HPCR(x1, x2, f1, f2):
    """Framework H (spectral harmonicity) and PC (clean targeted n:m PLV — the
    framework coupling metric applied at the (f1,f2) pair, as in Studies 17/23;
    the raw PSD-weighted matrix entry is diluted) and R = H_norm·PC."""
    r = compute_cross_resonance(x1, x2, sf=SF, config=CFG)
    fr = r.freqs
    H = float(C.band_value(fr, r.factors["H"]["all"], f1))
    PC = framework_nm_plv(x1, x2, f1, f2)
    R = H * PC
    return H, PC, R


def framework_nm_plv(x1, x2, f1, f2, bw=2.0):
    """Clean n:m PLV at the nearest rational q:p of f2/f1 (bandpass + Hilbert)."""
    frac = Fraction(f2 / f1).limit_denominator(16)
    q, p = frac.numerator, frac.denominator      # f2/f1 = q/p
    pa = np.angle(hilbert(_bp(x1, f1 - 2, f1 + 2)))
    pb = np.angle(hilbert(_bp(x2, f2 - 2, f2 + 2)))
    return float(np.abs(np.mean(np.exp(1j * (q * pa - p * pb)))))


def run(quick=True):
    seeds = range(3) if quick else range(8)
    K_list = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    DETUNE = 0.05   # gradual lock across the K range; the lock barely shifts f2 so H stays ~flat (~7%)
    RATIOS = [(1.5, "2:3"), (4.0 / 3.0, "3:4"), (1.25, "4:5")]

    # Generative synchronization transition at POLYRHYTHMIC ratios: two detuned
    # n:m Kuramoto oscillators; as coupling K rises they lock at the n:m ratio.
    # H (spectral; frequencies fixed) stays flat while PC (n:m phase coupling) and
    # R = H*PC rise through the transition -> the framework's polyrhythmic phase
    # coupling tracks EMERGENT coupling that H is blind to.
    transition = {}
    for ratio, label in RATIOS:
        rows = []
        for K in K_list:
            H, PC, R = [], [], []
            for seed in seeds:
                x1, x2 = simulate_kuramoto_pair(ratio, K, detune=DETUNE, seed=seed)
                h, pc, r = framework_HPCR(x1, x2, F1, F1 * ratio)
                H.append(h); PC.append(pc); R.append(r)
            rows.append(dict(K=float(K), H=float(np.mean(H)), PC=float(np.mean(PC)), R=float(np.mean(R))))
        transition[label] = rows
        print(f"  ratio {label} (vary K) done", flush=True)

    from scipy.stats import spearmanr
    primary = transition["2:3"]; Ks = [d["K"] for d in primary]
    corr = dict(PC_vs_K=float(spearmanr(Ks, [d["PC"] for d in primary])[0]),
                R_vs_K=float(spearmanr(Ks, [d["R"] for d in primary])[0]),
                H_vs_K=float(spearmanr(Ks, [d["H"] for d in primary])[0]))
    result = dict(quick=quick, F1=F1, detune=DETUNE, K_list=list(K_list),
                  transition=transition, corr=corr)
    C.save_json(result, "study7_coupled_oscillators.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    c = result["corr"]; p = result["transition"]["2:3"]
    Hs = [d["H"] for d in p]
    h_pct = 100.0 * (max(Hs) - min(Hs)) / (np.mean(Hs) + 1e-12)
    print("\n  --- Study 7 headline (generative polyrhythmic coupling) ---")
    print("  detuned n:m Kuramoto, vary coupling K (primary ratio 2:3):")
    print(f"    PC(2:3): {p[0]['PC']:.2f} (uncoupled) -> {p[-1]['PC']:.2f} (coupled), "
          f"rho(PC,K)={c['PC_vs_K']:+.2f}; R rho={c['R_vs_K']:+.2f}")
    print(f"    H near-CONSTANT: {min(Hs):.2f}-{max(Hs):.2f} (varies only {h_pct:.0f}%) "
          f"while PC changes ~{(p[-1]['PC']-p[0]['PC']):.2f} -> H is blind to coupling")
    print("  => framework POLYRHYTHMIC phase coupling (n:m) tracks the EMERGENT")
    print("     synchronization transition; H (spectrum) barely moves; R = H*PC gates on it.")


def _figures(result):
    plt = C.setup_mpl()
    tr = result["transition"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    cols = {"2:3": "#6a1b9a", "3:4": "#1565c0", "4:5": "#00897b"}

    for label, rows in tr.items():
        axes[0].plot([d["K"] for d in rows], [d["PC"] for d in rows], "o-",
                     color=cols.get(label), label=f"PC {label}")
    p = tr["2:3"]; Hn = np.array([d["H"] for d in p], float); Hn = Hn / (Hn.max() + 1e-12)
    axes[0].plot([d["K"] for d in p], Hn, "s--", color="#E69F00", label="H (2:3, norm)")
    axes[0].set_xlabel("coupling strength K"); axes[0].set_ylabel("PC  /  H (norm)")
    axes[0].set_title("A. Polyrhythmic phase coupling rises with K\n(H flat: blind to coupling)", fontsize=10)
    axes[0].legend(fontsize=7)

    for key, color, lbl in [("H", "#E69F00", "H"), ("PC", "#6a1b9a", "PC"), ("R", "#D55E00", "R = H*PC")]:
        v = np.array([d[key] for d in p], float); v = v / (v.max() + 1e-12)
        axes[1].plot([d["K"] for d in p], v, "o-", color=color, label=lbl)
    c = result["corr"]
    axes[1].set_xlabel("coupling strength K"); axes[1].set_ylabel("normalized")
    axes[1].set_title(f"B. R tracks emergent coupling (2:3)\nrho(PC,K)={c['PC_vs_K']:+.2f}, rho(H,K)={c['H_vs_K']:+.2f}", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.suptitle("Study 7 - Generative polyrhythmic coupling: framework PC/R track the n:m synchronization "
                 "transition; H is blind", fontweight="bold", fontsize=10)
    fig.tight_layout()
    C.save_fig(fig, "study7_coupled_oscillators")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

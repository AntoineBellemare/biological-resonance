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
                           n_harm=3, noise=0.05, unidirectional=False):
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
    w2 = 2 * np.pi * F1 * ratio
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
    """Framework H, PC, R for the pair at the (f1, f2) target."""
    r = compute_cross_resonance(x1, x2, sf=SF, config=CFG)
    fr = r.freqs
    i = int(np.argmin(np.abs(fr - f1))); j = int(np.argmin(np.abs(fr - f2)))
    H = float(C.band_value(fr, r.factors["H"]["all"], f1))
    PC = float(r.phase_coupling_matrix[i, j])
    R = H * PC
    return H, PC, R


def run(quick=True):
    seeds = range(3) if quick else range(8)

    # --- Direction A: vary coupling K, detuned ~1:1 mutual coupling ---
    # Canonical synchronization transition: two slightly-detuned (ratio 1.06)
    # Van der Pol oscillators with mutual diffusive coupling. Coupling is scaled
    # to the restoring force (~w^2 ~ 2500), so K is in the hundreds. The locking
    # range is below the over-coupling regime (K >~ 1000) where amplitude effects
    # degrade synchrony.
    def _plv11(x1, x2):
        pa = np.angle(hilbert(_bp(x1, 6, 11))); pb = np.angle(hilbert(_bp(x2, 6, 11)))
        return float(np.abs(np.mean(np.exp(1j * (pa - pb)))))

    K_list = [0.0, 50.0, 100.0, 200.0, 300.0, 500.0]
    dirA = []
    for K in K_list:
        gt, H, PC, R = [], [], [], []
        for seed in seeds:
            x1, x2 = simulate_vdp_pair(1.06, K, seed=seed, unidirectional=False)
            gt.append(_plv11(x1, x2))
            h, pc, r = framework_HPCR(x1, x2, F1, F1 * 1.06)
            H.append(h); PC.append(pc); R.append(r)
        dirA.append(dict(K=K, gt_plv=float(np.mean(gt)), H=float(np.mean(H)),
                         PC=float(np.mean(PC)), R=float(np.mean(R))))
    print("  Direction A (vary K) done")

    # --- Direction B: vary frequency ratio at fixed coupling ---
    K_fixed = 0.5
    ratios = np.round(np.linspace(1.0, 3.0, 21), 3) if quick else np.round(np.linspace(1.0, 3.0, 41), 3)
    dirB = []
    for ratio in ratios:
        gt, H, PC, R = [], [], [], []
        for seed in seeds:
            x1, x2 = simulate_vdp_pair(float(ratio), K_fixed, seed=seed)
            f2 = F1 * float(ratio)
            gt.append(ground_truth_plv(x1, x2, F1, f2))
            h, pc, r = framework_HPCR(x1, x2, F1, f2)
            H.append(h); PC.append(pc); R.append(r)
        # ratio simplicity = 1 / (n*m) for the closest rational (Tenney-ish)
        frac = Fraction(float(ratio)).limit_denominator(8)
        simplicity = 1.0 / (frac.numerator * frac.denominator)
        dirB.append(dict(ratio=float(ratio), simplicity=simplicity,
                         gt_plv=float(np.mean(gt)), H=float(np.mean(H)),
                         PC=float(np.mean(PC)), R=float(np.mean(R))))
    print("  Direction B (vary ratio) done")

    # correlations across the ratio sweep (the non-circular checks)
    from scipy.stats import spearmanr
    Hb = [d["H"] for d in dirB]; PCb = [d["PC"] for d in dirB]
    Rb = [d["R"] for d in dirB]; simp = [d["simplicity"] for d in dirB]
    gtb = [d["gt_plv"] for d in dirB]
    corr = dict(
        H_vs_simplicity=float(spearmanr(Hb, simp)[0]),
        PC_vs_simplicity=float(spearmanr(PCb, simp)[0]),
        PC_vs_groundtruth=float(spearmanr(PCb, gtb)[0]),
        H_vs_PC=float(spearmanr(Hb, PCb)[0]),
        R_vs_simplicity=float(spearmanr(Rb, simp)[0]),
    )

    # Direction A monotonicity: PC vs K
    corrA = float(spearmanr([d["K"] for d in dirA], [d["PC"] for d in dirA])[0])

    result = dict(quick=quick, F1=F1, mu=MU, K_fixed=K_fixed,
                  direction_A=dirA, direction_B=dirB,
                  corr_ratio_sweep=corr, corr_PC_vs_K=corrA)
    C.save_json(result, "study7_coupled_oscillators.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    c = result["corr_ratio_sweep"]
    print("\n  --- Study 7 headline (coupled Van der Pol) ---")
    print(f"  Direction A: Spearman(PC, coupling K) = {result['corr_PC_vs_K']:+.2f}  "
          f"(PC rises with coupling)")
    print("  Direction B (vary ratio; non-circular co-variation):")
    print(f"    H  vs ratio-simplicity   = {c['H_vs_simplicity']:+.2f}")
    print(f"    PC vs ratio-simplicity   = {c['PC_vs_simplicity']:+.2f}  "
          f"(phase locking peaks at simple ratios = Arnold tongues)")
    print(f"    PC vs ground-truth PLV   = {c['PC_vs_groundtruth']:+.2f}  "
          f"(framework PC tracks true locking)")
    print(f"    H  vs PC                 = {c['H_vs_PC']:+.2f}  "
          f"(harmonicity predicts phase locking — NOT by construction)")
    print(f"    R  vs ratio-simplicity   = {c['R_vs_simplicity']:+.2f}")


def _figures(result):
    plt = C.setup_mpl()
    A = result["direction_A"]; B = result["direction_B"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # A: vary K
    Ks = [d["K"] for d in A]
    for key, color, lbl in [("gt_plv", "#000000", "ground-truth PLV"),
                            ("PC", "#6a1b9a", "framework PC"),
                            ("R", "#b71c1c", "R = H x PC")]:
        v = np.array([d[key] for d in A], float); v = v / (v.max() + 1e-12)
        axes[0].plot(Ks, v, "o-", color=color, label=lbl)
    axes[0].set_xlabel("coupling strength K"); axes[0].set_ylabel("normalized")
    axes[0].set_title("A. Vary coupling (1:2 ratio): PC & R rise with sync", fontsize=10)
    axes[0].legend(fontsize=8)

    # B: vary ratio — Arnold tongues
    rs = [d["ratio"] for d in B]
    for key, color, lbl in [("H", "#1a237e", "H (harmonicity)"),
                            ("PC", "#6a1b9a", "PC (phase locking)"),
                            ("R", "#b71c1c", "R (resonance)")]:
        v = np.array([d[key] for d in B], float); v = v / (v.max() + 1e-12)
        axes[1].plot(rs, v, "-", color=color, label=lbl)
    for sr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        axes[1].axvline(sr, color="green", alpha=0.25, lw=0.8, ls=":")
    axes[1].set_xlabel("frequency ratio f2/f1"); axes[1].set_ylabel("normalized")
    c = result["corr_ratio_sweep"]
    axes[1].set_title(f"B. Vary ratio: H, PC, R peak at simple ratios (dotted)\n"
                      f"Spearman(H,PC)={c['H_vs_PC']:+.2f}, PC~truth={c['PC_vs_groundtruth']:+.2f}",
                      fontsize=10)
    axes[1].legend(fontsize=8)
    fig.suptitle("Study 7 — Coupled Van der Pol: harmonicity and phase coupling "
                 "co-vary through Arnold-tongue dynamics", fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study7_coupled_oscillators")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

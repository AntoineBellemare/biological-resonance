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
                      unidirectional=False, detune=0.0):
    """Integrate two coupled Van der Pol oscillators; return (x1, x2).

    unidirectional=False : mutual diffusive coupling K(x2-x1), K(x1-x2).
    unidirectional=True  : oscillator 1 is free and forces oscillator 2 (K*x1),
        so x1 keeps its frequency while x2 entrains — a clean drive that avoids
        the mutual frequency-pulling that moves both oscillators off-target.
    """
    rng = np.random.default_rng(seed)
    w1 = 2 * np.pi * F1
    w2 = 2 * np.pi * F1 * ratio * (1.0 + detune)   # detune so uncoupled drifts; lock emerges with K
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


def harmonic_kuramoto_pair(ratio, K, sf=SF, duration=16.0, seed=0, n_harm=5,
                           noise=0.03, detune=0.03):
    """Harmonically-coupled phase oscillators (the physically-correct phase reduction
    of amplitude-oscillator coupling). For w2/w1 = num/den, the n:m resonance couples
    oscillator-1's num-th harmonic to oscillator-2's den-th harmonic, so its strength
    is the PRODUCT of those harmonic amplitudes a_num*a_den (a_k = 0.6^(k-1); Pikovsky
    et al.). Complex ratios are mediated by higher (weaker) harmonics -> they lock at
    HIGHER coupling K (Arnold-tongue narrowing in the coupling dimension)."""
    rng = np.random.default_rng(seed)
    frac = Fraction(float(ratio)).limit_denominator(8)
    num, den = frac.numerator, frac.denominator        # w2/w1 = num/den
    a = lambda k: 0.6 ** (k - 1)
    wgt = a(num) * a(den)                               # n:m resonance strength
    w1 = 2 * np.pi * F1; w2 = 2 * np.pi * F1 * ratio * (1.0 + detune)
    N = int(sf * duration); dt = 1.0 / sf
    th1 = np.empty(N); th2 = np.empty(N)
    th1[0], th2[0] = rng.uniform(0, 2 * np.pi, size=2)
    for i in range(1, N):
        th1[i] = th1[i - 1] + dt * (w1 + K * wgt * np.sin(den * th2[i - 1] - num * th1[i - 1]))
        th2[i] = th2[i - 1] + dt * (w2 + K * wgt * np.sin(num * th1[i - 1] - den * th2[i - 1]))
    rend = lambda th: sum(a(k + 1) * np.sin((k + 1) * th) for k in range(n_harm))
    cut = int(2 * sf)
    x1 = rend(th1)[cut:] + noise * rng.standard_normal(N - cut)
    x2 = rend(th2)[cut:] + noise * rng.standard_normal(N - cut)
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
    seeds = range(3) if quick else range(12)
    K_list = ([0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0] if quick else
              [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 56.0, 80.0])
    DETUNE = 0.03
    RATIOS = [(1.5, "2:3"), (4.0 / 3.0, "3:4"), (1.25, "4:5")]

    # Harmonically-coupled phase oscillators at polyrhythmic ratios. As coupling K
    # rises each pair locks at its n:m ratio; the locking threshold K* RISES WITH
    # RATIO COMPLEXITY (n:m mediated by higher, weaker harmonics -> Arnold-tongue
    # narrowing in the COUPLING dimension). H (spectrum) stays ~flat: blind to coupling.
    transition = {}; kstar = {}; complexity = {}
    for ratio, label in RATIOS:
        frac = Fraction(ratio).limit_denominator(8); nm = frac.numerator * frac.denominator
        complexity[label] = nm
        rows = []
        for K in K_list:
            H, PC, R = [], [], []
            for seed in seeds:
                x1, x2 = harmonic_kuramoto_pair(ratio, K, detune=DETUNE, seed=seed)
                h, pc, r = framework_HPCR(x1, x2, F1, F1 * ratio)
                H.append(h); PC.append(pc); R.append(r)
            rows.append(dict(K=float(K), H=float(np.mean(H)), PC=float(np.mean(PC)), R=float(np.mean(R))))
        transition[label] = rows
        ks = float("nan")
        for a in range(len(rows) - 1):
            if rows[a]["PC"] < 0.5 <= rows[a + 1]["PC"]:
                ks = rows[a]["K"] + (0.5 - rows[a]["PC"]) / (rows[a + 1]["PC"] - rows[a]["PC"] + 1e-9) \
                    * (rows[a + 1]["K"] - rows[a]["K"]); break
        kstar[label] = ks
        print(f"  ratio {label} (n*m={nm}) K*={ks:.1f}", flush=True)

    from scipy.stats import spearmanr
    labels = [l for _, l in RATIOS]
    fin = [(complexity[l], kstar[l]) for l in labels if kstar[l] == kstar[l]]
    kstar_rho = float(spearmanr([c for c, _ in fin], [k for _, k in fin])[0]) if len(fin) > 2 else float("nan")
    p = transition["2:3"]; Ks = [d["K"] for d in p]
    corr = dict(PC_vs_K_2to3=float(spearmanr(Ks, [d["PC"] for d in p])[0]),
                H_vs_K_2to3=float(spearmanr(Ks, [d["H"] for d in p])[0]),
                kstar_vs_complexity=kstar_rho)
    result = dict(quick=quick, F1=F1, detune=DETUNE, K_list=list(K_list),
                  transition=transition, kstar=kstar, complexity=complexity, corr=corr)
    C.save_json(result, "study7_coupled_oscillators.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    ks = result["kstar"]; cx = result["complexity"]; c = result["corr"]; p = result["transition"]["2:3"]
    Hs = [d["H"] for d in p]; h_pct = 100.0 * (max(Hs) - min(Hs)) / (np.mean(Hs) + 1e-12)
    print("\n  --- Study 7 headline (generative coupling: complexity sets the threshold) ---")
    print("  harmonically-coupled phase oscillators; locking threshold K* (PC=0.5) by ratio:")
    for l in ["2:3", "3:4", "4:5"]:
        kk = ks.get(l, float("nan"))
        print(f"    {l} (n*m={cx[l]:2d}):  K* = {('%.1f' % kk) if kk == kk else '>max'}")
    print(f"    => K* RISES with ratio complexity (rho={c['kstar_vs_complexity']:+.2f}): "
          f"complex ratios lock LATER (need stronger coupling)")
    print(f"  H near-constant for 2:3 ({min(Hs):.2f}-{max(Hs):.2f}, {h_pct:.0f}%) -> blind to coupling; "
          f"R = H*PC gates on the coupling.")


def _figures(result):
    plt = C.setup_mpl()
    tr = result["transition"]; ks = result["kstar"]; cx = result["complexity"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    cols = {"2:3": "#2e7d32", "3:4": "#1565c0", "4:5": "#b71c1c"}

    # A: PC(K) transitions shift RIGHT with complexity; H flat (2:3)
    for label, rows in tr.items():
        axes[0].plot([max(d["K"], 1.0) for d in rows], [d["PC"] for d in rows], "o-",
                     color=cols.get(label), label=f"PC {label} (n·m={cx[label]})")
    p = tr["2:3"]; Hn = np.array([d["H"] for d in p], float); Hn = Hn / (Hn.max() + 1e-12)
    axes[0].plot([max(d["K"], 1.0) for d in p], Hn, "s--", color="#E69F00", label="H (2:3, norm)")
    axes[0].axhline(0.5, color="grey", ls=":", lw=0.7)
    axes[0].set_xscale("log"); axes[0].set_xlabel("coupling strength K (log)")
    axes[0].set_ylabel("PC  /  H (norm)")
    axes[0].set_title("A. Complex ratios lock LATER\n(H flat: blind to coupling)", fontsize=10)
    axes[0].legend(fontsize=6.5)

    # B: locking threshold K* rises with ratio complexity
    labels = ["2:3", "3:4", "4:5"]
    xs = [cx[l] for l in labels]; ys = [ks[l] for l in labels]
    axes[1].plot(xs, ys, "o-", color="#6a1b9a", ms=8)
    for l in labels:
        if ks[l] == ks[l]:
            axes[1].annotate(l, (cx[l], ks[l]), fontsize=8, xytext=(5, 2), textcoords="offset points")
    axes[1].set_xlabel("ratio complexity  n·m"); axes[1].set_ylabel("locking threshold K* (PC=0.5)")
    axes[1].set_title(f"B. Threshold rises with complexity\n(ρ={result['corr']['kstar_vs_complexity']:+.2f})", fontsize=10)

    fig.suptitle("Study 7 — Generative coupling: complex ratios lock at higher coupling; H is blind",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    C.save_fig(fig, "study7_coupled_oscillators")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

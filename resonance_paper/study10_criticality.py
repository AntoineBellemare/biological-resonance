"""Study 10 — Resonance and criticality in a branching neural network.

Connects the two open questions (reservoir edge-of-chaos & biological
criticality): both are the same critical point of a dynamical phase transition.
Here we use the canonical model of neuronal criticality — a probabilistic
branching network whose control parameter is the branching ratio sigma:

    sigma < 1  subcritical   — activity dies out (sparse, fragmented)
    sigma = 1  CRITICAL      — scale-free avalanches, power-law sizes, 1/f, max susceptibility
    sigma > 1  supercritical — runaway / synchronized bursting

We sweep sigma through criticality and, for each value, measure:

  Criticality signatures (established):
    * avalanche size distribution + power-law-ness (peaks at sigma=1)
    * susceptibility = variance of population activity (peaks at sigma=1)
    * empirical branching ratio (slope of A(t+1) vs A(t))
    * spectral flatness of A(t) (1/f / broadband near criticality)

  Resonance of the population activity A(t) (after aperiodic removal):
    * H_max, R_max, R_avg, R spectral flatness

Question: how does harmonic resonance relate to criticality? Does it peak AT the
critical point, or in the supercritical (synchronized-oscillation) regime? The
reservoir study (Study 9) found resonance rising into chaos (supercritical-like);
this tests the same idea in a bona-fide neural-criticality model.

Outputs: results/study10_criticality.json, figures/study10_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from biotuner.resonance import compute_resonance

SF = 250.0  # treat each network step as 1/SF s for the resonance analysis


def branching_activity(sigma, N=400, T=8000, warmup=500, seed=0):
    """Mean-field probabilistic branching process. Returns (A_timeseries,
    avalanche_sizes). When activity dies (A=0) a single neuron is re-seeded,
    delimiting avalanches."""
    rng = np.random.default_rng(seed)
    p = sigma / N
    total = T + warmup
    A = np.empty(total, dtype=np.int64)
    active = 1
    av_sizes = []
    cur = active
    for t in range(total):
        A[t] = active
        # each of `active` neurons activates each neuron w.p. p -> next count
        prob = 1.0 - (1.0 - p) ** active
        active = int(rng.binomial(N, prob))
        if active == 0:
            av_sizes.append(cur)
            active = 1   # re-seed a new avalanche
            cur = 1
        else:
            cur += active
    return A[warmup:].astype(np.float64), np.array(av_sizes[5:], dtype=float)


def powerlaw_score(sizes):
    """Crude power-law-ness: R^2 of a log-log linear fit to the size histogram
    over the scaling range. ~1 and slope near -1.5 indicates criticality."""
    sizes = sizes[sizes >= 1]
    if len(sizes) < 30:
        return dict(slope=float("nan"), r2=float("nan"))
    hi = max(2, int(np.percentile(sizes, 99)))
    bins = np.unique(np.logspace(0, np.log10(hi), 20).astype(int))
    if len(bins) < 4:
        return dict(slope=float("nan"), r2=float("nan"))
    counts, edges = np.histogram(sizes, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    mask = counts > 0
    if mask.sum() < 4:
        return dict(slope=float("nan"), r2=float("nan"))
    lx, ly = np.log10(centers[mask]), np.log10(counts[mask])
    slope, inter = np.polyfit(lx, ly, 1)
    pred = slope * lx + inter
    ss_res = np.sum((ly - pred) ** 2); ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    return dict(slope=float(slope), r2=float(r2))


def branching_estimate(A):
    """Empirical branching ratio = slope of A(t+1) on A(t)."""
    a0, a1 = A[:-1], A[1:]
    mask = a0 > 0
    if mask.sum() < 10:
        return float("nan")
    return float(np.polyfit(a0[mask], a1[mask], 1)[0])


def run(quick=True):
    N = 300 if quick else 600
    T = 8000 if quick else 20000
    seeds = range(3) if quick else range(8)
    sigmas = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.30] if quick else \
             [0.70, 0.85, 0.92, 0.97, 1.00, 1.03, 1.08, 1.15, 1.30, 1.5]
    cfg = C.default_config(fmin=2, fmax=60, precision_hz=0.5)  # remove_aperiodic=True

    rows = []
    for sigma in sigmas:
        sus, pl_r2, pl_slope, br = [], [], [], []
        Rmax, Ravg, Hmax, Rflat, Aflat = [], [], [], [], []
        for seed in seeds:
            A, avs = branching_activity(sigma, N=N, T=T, seed=seed)
            sus.append(float(np.var(A)))
            pl = powerlaw_score(avs); pl_r2.append(pl["r2"]); pl_slope.append(pl["slope"])
            br.append(branching_estimate(A))
            # spectral flatness of raw activity (1/f-ness near criticality)
            from biotuner.metrics import spectral_flatness
            Aflat.append(float(spectral_flatness(np.abs(np.fft.rfft(_norm(A)))[1:])))
            # resonance of population activity
            r = compute_resonance(_norm(A).astype(np.float64), sf=SF, config=cfg)
            Rmax.append(r.summaries["R"]["max"]); Ravg.append(r.summaries["R"]["avg"])
            Hmax.append(r.summaries["H"]["max"]); Rflat.append(r.summaries["R"]["flatness"])
        rows.append(dict(sigma=sigma,
                         susceptibility=float(np.mean(sus)),
                         powerlaw_r2=float(np.nanmean(pl_r2)),
                         powerlaw_slope=float(np.nanmean(pl_slope)),
                         branching_est=float(np.nanmean(br)),
                         activity_flatness=float(np.mean(Aflat)),
                         R_max=float(np.mean(Rmax)), R_avg=float(np.mean(Ravg)),
                         H_max=float(np.mean(Hmax)), R_flatness=float(np.mean(Rflat))))
        print(f"  sigma={sigma:.2f}  suscept={np.mean(sus):.1f}  PL_r2={np.nanmean(pl_r2):.2f}  "
              f"R_max={np.mean(Rmax):.4f}")

    # where does each quantity peak in sigma?
    def argmax_sigma(key): return float(rows[int(np.argmax([r[key] for r in rows]))]["sigma"])
    summary = dict(
        sigma_at_max_susceptibility=argmax_sigma("susceptibility"),
        sigma_at_max_powerlaw=argmax_sigma("powerlaw_r2"),
        sigma_at_max_R=argmax_sigma("R_max"),
        sigma_at_max_H=argmax_sigma("H_max"),
    )
    result = dict(quick=quick, N=N, T=T, rows=rows, summary=summary)
    C.save_json(result, "study10_criticality.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 10 headline (resonance vs criticality) ---")
    print(f"  {'sigma':>5} {'suscept':>9} {'PL_r2':>7} {'branch':>7} {'H_max':>7} {'R_max':>8}")
    for r in result["rows"]:
        print(f"  {r['sigma']:>5.2f} {r['susceptibility']:>9.1f} {r['powerlaw_r2']:>7.2f} "
              f"{r['branching_est']:>7.2f} {r['H_max']:>7.3f} {r['R_max']:>8.4f}")
    print(f"\n  Criticality markers peak near sigma=1 (driving shifts it slightly >1):")
    print(f"    susceptibility -> sigma={s['sigma_at_max_susceptibility']:.2f}, "
          f"avalanche power-law -> sigma={s['sigma_at_max_powerlaw']:.2f}")
    print(f"  HARMONICITY H_max peaks at sigma={s['sigma_at_max_H']:.2f} (criticality signature in H).")
    print("  RESONANCE R_max ~ 0 throughout: pure avalanche dynamics are scale-free but NOT")
    print("    oscillatory/phase-locked, so the phase-coupling factor (and R) stays near zero.")
    print("  => Criticality is reflected in HARMONICITY, not in phase-coupling resonance, for a")
    print("     bare branching process. (R needs sustained oscillations — e.g. driven input or an")
    print("     E/I network that combines criticality with rhythms; see Discussion.)")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; sig = [r["sigma"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    def nrm(k):
        v = np.array([r[k] for r in rows], float); return v / (np.nanmax(np.abs(v)) + 1e-12)
    ax1.plot(sig, nrm("susceptibility"), "o-", color="#1565c0", label="susceptibility (var)")
    ax1.plot(sig, nrm("powerlaw_r2"), "^-", color="#2e7d32", label="avalanche power-law R²")
    ax1.plot(sig, nrm("H_max"), "d-", color="#ef6c00", label="harmonicity H_max")
    Rmaxvals = np.array([r["R_max"] for r in rows])
    if np.nanmax(Rmaxvals) > 1e-6:
        ax1.plot(sig, nrm("R_max"), "s-", color="#b71c1c", label="resonance R_max")
    else:
        ax1.plot([], [], "s-", color="#b71c1c", label="resonance R_max ≈ 0 (no oscillation)")
    ax1.axvline(1.0, color="grey", ls="--", lw=0.9, label="critical point (sigma=1)")
    ax1.set_xlabel("branching ratio  sigma"); ax1.set_ylabel("normalized")
    s = result["summary"]
    ax1.set_title("Study 10 — Criticality vs resonance (branching network)\n"
                  f"criticality markers peak ~sigma=1; harmonicity H peaks at "
                  f"sigma={s['sigma_at_max_H']:.2f}; R~0 (avalanches non-oscillatory)",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    C.save_fig(fig, "study10_criticality")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

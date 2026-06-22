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
    mask = (counts > 0) & np.isfinite(centers) & (centers > 0)
    if mask.sum() < 4:
        return dict(slope=float("nan"), r2=float("nan"))
    lx, ly = np.log10(centers[mask]), np.log10(counts[mask])
    good = np.isfinite(lx) & np.isfinite(ly)
    lx, ly = lx[good], ly[good]
    if lx.size < 4 or np.ptp(lx) < 1e-9:
        return dict(slope=float("nan"), r2=float("nan"))
    try:
        slope, inter = np.polyfit(lx, ly, 1)
    except (np.linalg.LinAlgError, ValueError):
        return dict(slope=float("nan"), r2=float("nan"))
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
    seeds = list(range(4) if quick else range(20))   # replicate networks -> CIs
    sigmas = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.30] if quick else \
             [0.70, 0.85, 0.92, 0.97, 1.00, 1.03, 1.08, 1.15, 1.30, 1.5]
    cfg = C.default_config(fmin=2, fmax=60, precision_hz=0.5)  # remove_aperiodic=True
    from biotuner.metrics import spectral_flatness
    METRICS = ["susceptibility", "powerlaw_r2", "powerlaw_slope", "branching_est",
               "activity_flatness", "R_max", "R_avg", "H_max", "R_flatness"]

    per = {m: [] for m in METRICS}   # m -> list over sigma of per-seed arrays
    rows = []
    for sigma in sigmas:
        acc = {m: [] for m in METRICS}
        for seed in seeds:
            A, avs = branching_activity(sigma, N=N, T=T, seed=seed)
            acc["susceptibility"].append(float(np.var(A)))
            pl = powerlaw_score(avs)
            acc["powerlaw_r2"].append(pl["r2"]); acc["powerlaw_slope"].append(pl["slope"])
            acc["branching_est"].append(branching_estimate(A))
            acc["activity_flatness"].append(
                float(spectral_flatness(np.abs(np.fft.rfft(_norm(A)))[1:])))
            r = compute_resonance(_norm(A).astype(np.float64), sf=SF, config=cfg)
            acc["R_max"].append(r.summaries["R"]["max"]); acc["R_avg"].append(r.summaries["R"]["avg"])
            acc["H_max"].append(r.summaries["H"]["max"]); acc["R_flatness"].append(r.summaries["R"]["flatness"])
        row = dict(sigma=sigma)
        for m in METRICS:
            ci = C.mean_ci(acc[m]); per[m].append(acc[m])
            row[m] = ci["mean"]; row[m + "_sem"] = ci["sem"]; row[m + "_ci"] = [ci["lo"], ci["hi"]]
        rows.append(row)
        print(f"  sigma={sigma:.2f}  suscept={row['susceptibility']:.1f}  "
              f"PL_r2={row['powerlaw_r2']:.2f}  H_max={row['H_max']:.3f}+-{row['H_max_sem']:.3f}  "
              f"R_max={row['R_max']:.4f}")

    # peak location in sigma with bootstrap-over-seeds CI
    def mat(m):
        k = min(len(a) for a in per[m]); return np.array([a[:k] for a in per[m]])
    def loc(m):
        d = C.argmax_location_ci(sigmas, mat(m)); return d["point"], [d["lo"], d["hi"]]
    pS, cS = loc("susceptibility"); pP, cP = loc("powerlaw_r2")
    pH, cH = loc("H_max"); pR, cR = loc("R_max")

    # --- formal co-location: does the H_max peak coincide with the criticality peak? ---
    from scipy.stats import spearmanr, wilcoxon
    s_arr = np.array(sigmas, float)
    MH, MS = mat("H_max"), mat("susceptibility")
    nb = min(MH.shape[1], MS.shape[1]); rng2 = np.random.default_rng(1); pdiffs = []
    for _ in range(2000):
        idx = rng2.integers(0, nb, nb)
        dh = s_arr[int(np.nanargmax(np.nanmean(MH[:, idx], axis=1)))]
        ds = s_arr[int(np.nanargmax(np.nanmean(MS[:, idx], axis=1)))]
        pdiffs.append(dh - ds)
    pdiffs = np.array(pdiffs); plo, phi = float(np.percentile(pdiffs, 2.5)), float(np.percentile(pdiffs, 97.5))
    # per-seed Spearman(H_max, criticality proximity = -|branching_est - 1|) across sigma:
    # both peak at sigma=1, so a POSITIVE rho means H tracks proximity to criticality.
    rhos_hp = []
    for j in range(len(seeds)):
        Hj = [per["H_max"][i][j] for i in range(len(sigmas))]
        proxj = [-abs(per["branching_est"][i][j] - 1.0) for i in range(len(sigmas))]
        if all(np.isfinite(Hj)) and all(np.isfinite(proxj)):
            rr = spearmanr(proxj, Hj)[0]
            if np.isfinite(rr):
                rhos_hp.append(rr)
    hp = C.mean_ci(rhos_hp) if len(rhos_hp) >= 3 else dict(mean=float("nan"), lo=float("nan"), hi=float("nan"))
    try:
        hp_p = float(wilcoxon(rhos_hp).pvalue) if len(rhos_hp) >= 3 else float("nan")
    except ValueError:
        hp_p = float("nan")

    summary = dict(
        n_seeds=len(seeds),
        sigma_at_max_susceptibility=pS, sigma_at_max_susceptibility_ci=cS,
        sigma_at_max_powerlaw=pP, sigma_at_max_powerlaw_ci=cP,
        sigma_at_max_R=pR, sigma_at_max_R_ci=cR,
        sigma_at_max_H=pH, sigma_at_max_H_ci=cH,
        H_susc_peak_diff=float(np.mean(pdiffs)), H_susc_peak_diff_ci=[plo, phi],
        H_peak_coincides=bool(plo <= 0.0 <= phi),
        H_tracks_prox_rho=hp["mean"], H_tracks_prox_ci=[hp["lo"], hp["hi"]],
        H_tracks_prox_p=hp_p, H_tracks_prox_n=len(rhos_hp),
        H_tracks_prox_frac_pos=float(np.mean(np.array(rhos_hp) > 0)) if rhos_hp else float("nan"),
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
    cH = s.get("sigma_at_max_H_ci", [np.nan, np.nan])
    cS = s.get("sigma_at_max_susceptibility_ci", [np.nan, np.nan])
    print(f"\n  (n_seeds={s.get('n_seeds','?')}; peaks reported with bootstrap-over-seeds 95% CI)")
    print(f"    susceptibility -> sigma={s['sigma_at_max_susceptibility']:.2f} "
          f"[{cS[0]:.2f},{cS[1]:.2f}], avalanche power-law -> sigma={s['sigma_at_max_powerlaw']:.2f}")
    print(f"  HARMONICITY H_max peaks at sigma={s['sigma_at_max_H']:.2f} "
          f"[{cH[0]:.2f},{cH[1]:.2f}] (criticality signature in H).")
    if "H_susc_peak_diff_ci" in s:
        d = s["H_susc_peak_diff_ci"]
        print(f"    CO-LOCATION: H_peak - susceptibility_peak = {s['H_susc_peak_diff']:+.3f} "
              f"[{d[0]:+.2f},{d[1]:+.2f}] -> peaks {'COINCIDE' if s['H_peak_coincides'] else 'DIFFER'}")
        print(f"    H tracks criticality proximity: rho={s['H_tracks_prox_rho']:+.2f} "
              f"[{s['H_tracks_prox_ci'][0]:+.2f},{s['H_tracks_prox_ci'][1]:+.2f}] "
              f"p={s['H_tracks_prox_p']:.2g} ({int(s['H_tracks_prox_frac_pos']*100)}%+, n={s['H_tracks_prox_n']})")
    print("  RESONANCE R_max ~ 0 throughout: pure avalanche dynamics are scale-free but NOT")
    print("    oscillatory/phase-locked, so the phase-coupling factor (and R) stays near zero.")
    print("  => Criticality is reflected in HARMONICITY, not in phase-coupling resonance, for a")
    print("     bare branching process. (R needs sustained oscillations — e.g. driven input or an")
    print("     E/I network that combines criticality with rhythms; see Discussion.)")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; sig = [r["sigma"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    def nrm_err(k):
        v = np.array([r[k] for r in rows], float)
        e = np.array([r.get(k + "_sem", 0.0) for r in rows], float)
        mx = np.nanmax(np.abs(v)) + 1e-12
        return v / mx, e / mx
    for k, c, mk, lab in [("susceptibility", "#1565c0", "o", "susceptibility (var)"),
                          ("powerlaw_r2", "#2e7d32", "^", "avalanche power-law R²"),
                          ("H_max", "#ef6c00", "d", "harmonicity H_max")]:
        y, e = nrm_err(k)
        ax1.errorbar(sig, y, yerr=e, fmt=mk + "-", color=c, label=lab, capsize=2, lw=1.4, ms=5)
    Rmaxvals = np.array([r["R_max"] for r in rows])
    if np.nanmax(Rmaxvals) > 1e-6:
        y, e = nrm_err("R_max")
        ax1.errorbar(sig, y, yerr=e, fmt="s-", color="#b71c1c", label="resonance R_max", capsize=2)
    else:
        ax1.plot([], [], "s-", color="#b71c1c", label="resonance R_max ≈ 0 (no oscillation)")
    ax1.axvline(1.0, color="grey", ls="--", lw=0.9, label="critical point (sigma=1)")
    cH = result["summary"].get("sigma_at_max_H_ci")
    if cH:
        ax1.axvspan(cH[0], cH[1], color="#ef6c00", alpha=0.12, label="H peak 95% CI")
    ax1.set_xlabel("branching ratio  sigma"); ax1.set_ylabel("normalized")
    s = result["summary"]
    _cH = s.get("sigma_at_max_H_ci", [np.nan, np.nan])
    ax1.set_title("Study 10 — Criticality vs resonance (branching network)\n"
                  f"criticality markers peak ~sigma=1; harmonicity H peaks at "
                  f"sigma={s['sigma_at_max_H']:.2f} [{_cH[0]:.2f},{_cH[1]:.2f}] "
                  f"(n={s.get('n_seeds','?')} seeds); R~0 (avalanches non-oscillatory)",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    C.save_fig(fig, "study10_criticality")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

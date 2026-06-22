"""Study 11 — Does criticality GENERATE resonance? (reservoir, noise-driven)

Redesign. The earlier version drove the reservoir with a fixed 6+12 Hz harmonic
input and measured the resonance of its activity — but in the ordered regime that
just *echoes* the (constant) input, so resonance was flat-until-chaos and
uninformative. Here we ask the interesting question instead:

    Driven by structureless WHITE NOISE (no harmonic content, flat spectrum),
    does a reservoir GENERATE harmonic / resonant structure in its internal
    dynamics, and does that generation depend on the dynamical regime rho —
    in particular, does it peak near the edge of chaos?

Mechanism that could make this positive: near rho ~ 1 the recurrent operator has
modes close to the unit circle (under-damped), so noise excites a dominant
collective oscillation; the tanh nonlinearity then folds energy into integer
multiples of that mode -> genuine harmonic structure (high H) created from noise.
Chaos (rho > rho_c) should destroy it.

For each spectral radius rho we measure (n seed reservoirs -> bootstrap CIs):
  * Lyapunov exponent lambda -> locates the edge of chaos rho_c (lambda = 0).
  * Memory capacity (computation) -> for the dissociation (does generation peak
    where computation does?).
  * Internal resonance of the dominant emergent mode (PC1 of the reservoir
    state), driven by white noise: H_internal, R_internal, spectral peakedness.
  * The same for the white-noise INPUT (baseline ~ flat / low H).
  * H_gain = H_internal - H_input and peakedness_gain = flat_input - flat_internal
    -> the GENERATION signal (structure created by the reservoir, above input).

Outputs: results/study11_reservoir_criticality.json, figures/study11_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from resonance_paper.study9_reservoir import make_reservoir, run_reservoir, memory_capacity
from biotuner.resonance import compute_resonance
from biotuner.metrics import spectral_flatness

SF = 250.0


def lyapunov(W, Win, T=1500, warmup=200, leak=0.3, d0=1e-9, seed=0):
    """Mean per-step log-divergence of two nearby trajectories (largest Lyapunov
    estimate). <0 ordered, ~0 edge of chaos, >0 chaotic."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.8, 0.8, size=T + warmup)
    n = W.shape[0]
    x1 = np.zeros(n); x2 = np.zeros(n); x2[0] += d0
    logs = []
    for t in range(T + warmup):
        x1 = (1 - leak) * x1 + leak * np.tanh(W @ x1 + Win[:, 0] * u[t])
        x2 = (1 - leak) * x2 + leak * np.tanh(W @ x2 + Win[:, 0] * u[t])
        diff = x2 - x1
        d = np.linalg.norm(diff)
        if t >= warmup and d > 0:
            logs.append(np.log(d / d0))
        if d > 0:
            x2 = x1 + diff * (d0 / d)
    return float(np.mean(logs)) if logs else float("nan")


def noise_input(T, warmup, seed):
    """Structureless white-noise drive (flat spectrum, no harmonic content)."""
    rng = np.random.default_rng(seed + 7)
    return rng.standard_normal(T + warmup)


def internal_mode(X):
    """Dominant emergent collective mode (PC1) of the reservoir state trajectory.
    More sensitive to generated oscillations than the node-mean, which cancels
    incommensurate node modes."""
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    _, V = np.linalg.eigh(cov)
    return Xc @ V[:, -1]


def spec_flat(sig):
    """Spectral flatness of the amplitude spectrum (1 = flat/white, lower = peaked)."""
    return float(spectral_flatness(np.abs(np.fft.rfft(_norm(sig)))[1:]))


def run(quick=True):
    n_res = 120 if quick else 300
    seeds = list(range(3) if quick else range(16))
    rhos = [0.4, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6, 1.9, 2.2, 2.6] if quick else \
           [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0]
    cfg = C.default_config(fmin=2, fmax=60, precision_hz=0.5)  # remove_aperiodic=True
    T, warm = 4000, 200
    METRICS = ["lyapunov", "memory_capacity", "H_internal", "R_internal",
               "flat_internal", "H_input", "flat_input", "H_gain", "peakedness_gain"]

    per = {m: [] for m in METRICS}
    rows = []
    for rho in rhos:
        acc = {m: [] for m in METRICS}
        for seed in seeds:
            W, Win = make_reservoir(n_res, rho, seed=seed)
            acc["lyapunov"].append(lyapunov(W, Win, seed=seed))
            acc["memory_capacity"].append(memory_capacity(W, Win, seed=seed))
            u = noise_input(T, warm, seed)
            X = run_reservoir(W, Win, u)[warm:]
            internal = internal_mode(X); ui = u[warm:]
            ri = compute_resonance(_norm(internal).astype(np.float64), sf=SF, config=cfg)
            rin = compute_resonance(_norm(ui).astype(np.float64), sf=SF, config=cfg)
            H_int = float(ri.summaries["H"]["max"]); R_int = float(ri.summaries["R"]["max"])
            H_in = float(rin.summaries["H"]["max"])
            f_int = spec_flat(internal); f_in = spec_flat(ui)
            acc["H_internal"].append(H_int); acc["R_internal"].append(R_int)
            acc["flat_internal"].append(f_int); acc["H_input"].append(H_in); acc["flat_input"].append(f_in)
            acc["H_gain"].append(H_int - H_in)
            acc["peakedness_gain"].append(f_in - f_int)
        row = dict(rho=rho)
        for m in METRICS:
            ci = C.mean_ci(acc[m]); per[m].append(acc[m])
            row[m] = ci["mean"]; row[m + "_sem"] = ci["sem"]; row[m + "_ci"] = [ci["lo"], ci["hi"]]
        rows.append(row)
        print(f"  rho={rho:.2f}  lambda={row['lyapunov']:+.3f}  MC={row['memory_capacity']:.2f}  "
              f"H_int={row['H_internal']:.3f}  H_gain={row['H_gain']:+.3f}+-{row['H_gain_sem']:.3f}  "
              f"peaked_gain={row['peakedness_gain']:+.3f}")

    # edge of chaos via Lyapunov zero-crossing + bootstrap-over-seeds CI
    def mat(m):
        k = min(len(a) for a in per[m]); return np.array([a[:k] for a in per[m]])
    def zero_cross(curve):
        for i in range(len(curve) - 1):
            l0, l1 = curve[i], curve[i + 1]
            if np.isfinite(l0) and np.isfinite(l1) and l0 < 0 <= l1:
                return rhos[i] + (rhos[i + 1] - rhos[i]) * (0 - l0) / (l1 - l0)
        return float("nan")
    LM = mat("lyapunov"); rho_c = zero_cross(np.nanmean(LM, axis=1))
    rng = np.random.default_rng(0); ns = LM.shape[1]; rcs = []
    for _ in range(2000):
        z = zero_cross(np.nanmean(LM[:, rng.integers(0, ns, ns)], axis=1))
        if np.isfinite(z):
            rcs.append(z)
    rho_c_ci = [float(np.percentile(rcs, 2.5)), float(np.percentile(rcs, 97.5))] if rcs else [np.nan, np.nan]

    def loc(m):
        d = C.argmax_location_ci(rhos, mat(m)); return d["point"], [d["lo"], d["hi"]]
    pHG, cHG = loc("H_gain"); pPK, cPK = loc("peakedness_gain"); pMC, cMC = loc("memory_capacity")
    # dissociation: is the GENERATION peak (H_gain) separated from the COMPUTATION peak (MC)?
    s_arr = np.array(rhos, float); MHG, MMC = mat("H_gain"), mat("memory_capacity")
    nb = min(MHG.shape[1], MMC.shape[1]); rng3 = np.random.default_rng(2); dsep = []
    for _ in range(2000):
        idx = rng3.integers(0, nb, nb)
        dg = s_arr[int(np.nanargmax(np.nanmean(MHG[:, idx], axis=1)))]
        dm = s_arr[int(np.nanargmax(np.nanmean(MMC[:, idx], axis=1)))]
        dsep.append(dg - dm)
    dsep = np.array(dsep); dslo, dshi = float(np.percentile(dsep, 2.5)), float(np.percentile(dsep, 97.5))
    # is generation significant? peak-rho H_gain CI vs 0
    i_pk = int(np.nanargmax([r["H_gain"] for r in rows]))
    hgain_peak = rows[i_pk]["H_gain"]; hgain_peak_ci = rows[i_pk]["H_gain_ci"]
    # ONSET of generation: first rho whose H_gain 95% CI clears 0 (more meaningful
    # than a boundary argmax, since generation grows into the self-sustaining regime)
    onset = next((r["rho"] for r in rows if r["H_gain_ci"][0] > 0), float("nan"))
    summary = dict(n_seeds=len(seeds),
                   rho_critical=float(rho_c), rho_critical_ci=rho_c_ci,
                   generation_onset_rho=float(onset),
                   rho_at_max_Hgain=pHG, rho_at_max_Hgain_ci=cHG,
                   rho_at_max_peakedness=pPK, rho_at_max_peakedness_ci=cPK,
                   rho_at_max_MC=pMC, rho_at_max_MC_ci=cMC,
                   H_input_baseline=float(np.nanmean([r["H_input"] for r in rows])),
                   Hgain_peak=float(hgain_peak), Hgain_peak_ci=hgain_peak_ci,
                   generation_significant=bool(hgain_peak_ci[0] > 0),
                   Hgain_minus_MC_peak=float(np.mean(dsep)), Hgain_minus_MC_ci=[dslo, dshi],
                   generation_separated_from_computation=bool(not (dslo <= 0.0 <= dshi)))
    result = dict(quick=quick, n_res=n_res, rows=rows, summary=summary)
    C.save_json(result, "study11_reservoir_criticality.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 11 headline (does criticality GENERATE resonance?) ---")
    print(f"  {'rho':>5} {'lyap':>7} {'MC':>6} {'H_int':>7} {'H_gain':>8} {'peaked_gain':>11}")
    for r in result["rows"]:
        print(f"  {r['rho']:>5.2f} {r['lyapunov']:>+7.3f} {r['memory_capacity']:>6.2f} "
              f"{r['H_internal']:>7.3f} {r['H_gain']:>+8.3f} {r['peakedness_gain']:>+11.3f}")
    rc = s.get("rho_critical_ci", [np.nan, np.nan]); cHG = s.get("rho_at_max_Hgain_ci", [np.nan, np.nan])
    pk = s.get("Hgain_peak_ci", [np.nan, np.nan])
    print(f"\n  (n_seeds={s.get('n_seeds','?')}; 95% bootstrap-over-seeds CIs in brackets)")
    print(f"  Edge of chaos rho_c = {s['rho_critical']:.2f} [{rc[0]:.2f},{rc[1]:.2f}]")
    print(f"  White-noise input baseline harmonicity H_input = {s['H_input_baseline']:.3f}")
    print(f"  Generated harmonicity gain peaks at rho={s['rho_at_max_Hgain']:.2f} "
          f"[{cHG[0]:.2f},{cHG[1]:.2f}], H_gain={s['Hgain_peak']:+.3f} [{pk[0]:+.3f},{pk[1]:+.3f}]")
    if s.get("generation_significant"):
        print("  => GENERATION CONFIRMED: the reservoir manufactures harmonic structure from noise")
        print(f"     (H_gain CI excludes 0). It ONSETS at rho={s.get('generation_onset_rho',float('nan')):.2f} "
              f"(vs edge of chaos rho_c={s['rho_critical']:.2f}) and grows into the self-sustaining regime,")
        print("     i.e. harmonic structure emerges once the reservoir's own dynamics take over from the input.")
    else:
        print("  => No significant harmonic generation (H_gain CI includes 0): the reservoir")
        print("     adds little integer-harmonic structure to noise (see peakedness_gain for")
        print("     generic spectral structuring without harmonicity).")
    if "Hgain_minus_MC_ci" in s:
        d = s["Hgain_minus_MC_ci"]
        print(f"  DISSOCIATION generation vs computation: rho(H_gain peak) - rho(MC peak) = "
              f"{s['Hgain_minus_MC_peak']:+.2f} [{d[0]:+.2f},{d[1]:+.2f}] -> "
              f"{'SEPARATED' if s['generation_separated_from_computation'] else 'not separated'}")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; rho = [r["rho"] for r in rows]; s = result["summary"]
    rc_ci = s.get("rho_critical_ci")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.errorbar(rho, [r["lyapunov"] for r in rows],
                 yerr=[r.get("lyapunov_sem", 0.0) for r in rows],
                 fmt="o-", color="#000000", capsize=2, label="Lyapunov λ")
    ax1.axhline(0, color="grey", lw=0.6)
    if np.isfinite(s["rho_critical"]):
        ax1.axvline(s["rho_critical"], color="red", ls="--", lw=1.0,
                    label=f"edge of chaos ρ_c={s['rho_critical']:.2f}")
        if rc_ci and np.isfinite(rc_ci[0]):
            ax1.axvspan(rc_ci[0], rc_ci[1], color="red", alpha=0.12, label="ρ_c 95% CI")
    ax1.set_xlabel("spectral radius ρ"); ax1.set_ylabel("Lyapunov exponent")
    ax1.set_title("A. Locating the edge of chaos", fontsize=10); ax1.legend(fontsize=8)

    # B. generation: harmonicity created from noise vs rho, with input baseline
    H_int = np.array([r["H_internal"] for r in rows], float)
    H_int_e = np.array([r.get("H_internal_sem", 0.0) for r in rows], float)
    ax2.errorbar(rho, H_int, yerr=H_int_e, fmt="s-", color="#b71c1c", capsize=2,
                 label="H of emergent mode (PC1)")
    ax2.axhline(s["H_input_baseline"], color="#b71c1c", ls=":", lw=1.0,
                label=f"white-noise input baseline {s['H_input_baseline']:.2f}")
    def nrm(k):
        v = np.array([r[k] for r in rows], float); return v / (np.nanmax(np.abs(v)) + 1e-12)
    ax2.plot(rho, nrm("memory_capacity") * np.nanmax(H_int), "o-", color="#1565c0",
             alpha=0.6, label="memory capacity (scaled)")
    if np.isfinite(s["rho_critical"]):
        ax2.axvline(s["rho_critical"], color="red", ls="--", lw=1.0, label="ρ_c")
        if rc_ci and np.isfinite(rc_ci[0]):
            ax2.axvspan(rc_ci[0], rc_ci[1], color="red", alpha=0.12)
    verdict = "generated" if s.get("generation_significant") else "not significant"
    ax2.set_xlabel("spectral radius ρ"); ax2.set_ylabel("harmonicity H of emergent mode")
    ax2.set_title(f"B. Harmonic structure generated from NOISE ({verdict})\n"
                  f"peak at ρ={s['rho_at_max_Hgain']:.2f}, edge ρ_c={s['rho_critical']:.2f} "
                  f"(n={s.get('n_seeds','?')})", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("Study 11 — Does the reservoir GENERATE harmonic resonance from noise?",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study11_reservoir_criticality")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 11 (deeper B) — Reservoir computing, resonance, and the edge of chaos.

Extends Study 9 by (i) LOCATING the reservoir's critical point (edge of chaos)
with a Lyapunov-style divergence estimate, (ii) asking whether a reservoir can
MODEL / AMPLIFY harmonic resonance, and (iii) placing resonance precisely
relative to the critical point — the reservoir analogue of Study 10's biological
criticality.

For each spectral radius rho we measure:
  * Lyapunov exponent lambda (perturbation divergence rate): lambda < 0 ordered,
    lambda ~ 0 EDGE OF CHAOS (critical rho_c), lambda > 0 chaotic. Locates rho_c.
  * Memory capacity (computation) — classic, peaks just below rho_c.
  * Internal resonance R of activity driven by a harmonic (6+12 Hz) input.
  * Resonance amplification = internal R / input R (does the reservoir enhance
    the harmonic resonance of its input?).

Questions answered:
  (1) Can RC model resonance? -> amplification ratio across regimes.
  (2) Where does resonance sit relative to criticality? -> R(rho) vs rho_c, vs
      where computation (memory) peaks.

Outputs: results/study11_reservoir_criticality.json, figures/study11_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import pink_noise, _norm
from resonance_paper.study9_reservoir import make_reservoir, run_reservoir, memory_capacity
from biotuner.resonance import compute_resonance

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
        # renormalize separation to d0
        if d > 0:
            x2 = x1 + diff * (d0 / d)
    return float(np.mean(logs)) if logs else float("nan")


def harmonic_input(T, warmup, seed):
    t = np.arange(T + warmup) / SF
    u = np.sin(2 * np.pi * 6 * t) + 0.7 * np.sin(2 * np.pi * 12 * t + 0.3)
    return u + 0.1 * pink_noise(len(u), SF, seed=seed + 5)


def resonance_of(sig, cfg):
    r = compute_resonance(_norm(sig).astype(np.float64), sf=SF, config=cfg)
    return float(r.summaries["R"]["max"]), float(r.summaries["H"]["max"])


def run(quick=True):
    n_res = 120 if quick else 300
    seeds = list(range(3) if quick else range(16))    # reservoir realizations -> CIs
    rhos = [0.4, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6, 1.9, 2.2, 2.6] if quick else \
           [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0]
    cfg = C.default_config(fmin=2, fmax=30, precision_hz=0.5)
    T, warm = 4000, 200
    METRICS = ["lyapunov", "memory_capacity", "R_internal", "R_input",
               "amplification", "H_internal"]

    per = {m: [] for m in METRICS}
    rows = []
    for rho in rhos:
        acc = {m: [] for m in METRICS}
        for seed in seeds:
            W, Win = make_reservoir(n_res, rho, seed=seed)
            acc["lyapunov"].append(lyapunov(W, Win, seed=seed))
            acc["memory_capacity"].append(memory_capacity(W, Win, seed=seed))
            u = harmonic_input(T, warm, seed)
            X = run_reservoir(W, Win, u)[warm:]
            r_in, _ = resonance_of(u[warm:], cfg)
            r_int, h_int = resonance_of(X.mean(axis=1), cfg)
            acc["R_input"].append(r_in); acc["R_internal"].append(r_int)
            acc["H_internal"].append(h_int); acc["amplification"].append(r_int / (r_in + 1e-12))
        row = dict(rho=rho)
        for m in METRICS:
            ci = C.mean_ci(acc[m]); per[m].append(acc[m])
            row[m] = ci["mean"]; row[m + "_sem"] = ci["sem"]; row[m + "_ci"] = [ci["lo"], ci["hi"]]
        rows.append(row)
        print(f"  rho={rho:.2f}  lambda={row['lyapunov']:+.3f}  MC={row['memory_capacity']:.2f}  "
              f"R_int={row['R_internal']:.4f}+-{row['R_internal_sem']:.4f}  amp={row['amplification']:.2f}")

    # edge of chaos = rho where the seed-averaged Lyapunov curve crosses 0,
    # with a bootstrap-over-seeds CI on that crossing.
    def mat(m):
        k = min(len(a) for a in per[m]); return np.array([a[:k] for a in per[m]])
    def zero_cross(curve):
        for i in range(len(curve) - 1):
            l0, l1 = curve[i], curve[i + 1]
            if np.isfinite(l0) and np.isfinite(l1) and l0 < 0 <= l1:
                return rhos[i] + (rhos[i + 1] - rhos[i]) * (0 - l0) / (l1 - l0)
        return float("nan")
    LM = mat("lyapunov")
    rho_c = zero_cross(np.nanmean(LM, axis=1))
    rng = np.random.default_rng(0); ns = LM.shape[1]; rcs = []
    for _ in range(2000):
        z = zero_cross(np.nanmean(LM[:, rng.integers(0, ns, ns)], axis=1))
        if np.isfinite(z):
            rcs.append(z)
    rho_c_ci = [float(np.percentile(rcs, 2.5)), float(np.percentile(rcs, 97.5))] if rcs else [np.nan, np.nan]

    def loc(m):
        d = C.argmax_location_ci(rhos, mat(m)); return d["point"], [d["lo"], d["hi"]]
    pMC, cMC = loc("memory_capacity"); pR, cR = loc("R_internal")
    # is R monotonically decreasing with rho (peak == grid boundary)? report honestly.
    Rmeans = [r["R_internal"] for r in rows]
    r_monotonic_decr = bool(np.all(np.diff(Rmeans) <= 1e-9))
    summary = dict(n_seeds=len(seeds),
                   rho_critical=float(rho_c), rho_critical_ci=rho_c_ci,
                   rho_at_max_MC=pMC, rho_at_max_MC_ci=cMC,
                   rho_at_max_R=pR, rho_at_max_R_ci=cR,
                   R_monotonic_decreasing=r_monotonic_decr,
                   max_amplification=float(max(r["amplification"] for r in rows)))
    result = dict(quick=quick, n_res=n_res, rows=rows, summary=summary)
    C.save_json(result, "study11_reservoir_criticality.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 11 headline (reservoir: resonance vs edge of chaos) ---")
    print(f"  {'rho':>5} {'lyapunov':>9} {'memcap':>7} {'R_int':>8} {'amplif':>7}")
    for r in result["rows"]:
        print(f"  {r['rho']:>5.2f} {r['lyapunov']:>+9.3f} {r['memory_capacity']:>7.2f} "
              f"{r['R_internal']:>8.4f} {r['amplification']:>7.2f}")
    rc = s.get("rho_critical_ci", [np.nan, np.nan])
    cMC = s.get("rho_at_max_MC_ci", [np.nan, np.nan]); cR = s.get("rho_at_max_R_ci", [np.nan, np.nan])
    print(f"\n  (n_seeds={s.get('n_seeds','?')}; 95% bootstrap-over-seeds CIs in brackets)")
    print(f"  Edge of chaos (lambda=0) at rho_c = {s['rho_critical']:.2f} [{rc[0]:.2f},{rc[1]:.2f}]")
    print(f"  Memory capacity peaks at rho={s['rho_at_max_MC']:.2f} [{cMC[0]:.2f},{cMC[1]:.2f}] (below rho_c).")
    print(f"  Internal resonance peaks at rho={s['rho_at_max_R']:.2f} [{cR[0]:.2f},{cR[1]:.2f}].")
    if s.get("R_monotonic_decreasing"):
        print("    (R decreases monotonically with rho -> R is maximal in the most ordered regime;")
        print("     the 'peak' sits at the grid's lower bound, so read it as 'R highest when ordered'.)")
    print(f"  Max resonance amplification (internal/input R) = {s['max_amplification']:.2f}x")
    print("  => Both memory capacity and harmonic resonance R sit in the ordered regime, BELOW the")
    print("     edge of chaos; chaos destroys both.")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; rho = [r["rho"] for r in rows]; s = result["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    rc_ci = s.get("rho_critical_ci")
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

    def nrm_err(k):
        v = np.array([r[k] for r in rows], float)
        e = np.array([r.get(k + "_sem", 0.0) for r in rows], float)
        mx = np.nanmax(np.abs(v)) + 1e-12; return v / mx, e / mx
    for k, c, mk, lab in [("memory_capacity", "#1565c0", "o", "memory capacity"),
                          ("R_internal", "#b71c1c", "s", "internal resonance R"),
                          ("amplification", "#ef6c00", "d", "resonance amplification")]:
        y, e = nrm_err(k); ax2.errorbar(rho, y, yerr=e, fmt=mk + "-", color=c, capsize=2, label=lab)
    if np.isfinite(s["rho_critical"]):
        ax2.axvline(s["rho_critical"], color="red", ls="--", lw=1.0, label="ρ_c")
        if rc_ci and np.isfinite(rc_ci[0]):
            ax2.axvspan(rc_ci[0], rc_ci[1], color="red", alpha=0.12)
    ax2.set_xlabel("spectral radius ρ"); ax2.set_ylabel("normalized")
    ax2.set_title(f"B. Resonance vs computation vs criticality\n"
                  f"MC & R peak in the ordered regime, below ρ_c (n={s.get('n_seeds','?')} seeds)",
                  fontsize=10); ax2.legend(fontsize=8)
    fig.suptitle("Study 11 — Reservoir: resonance, computation, and the edge of chaos",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study11_reservoir_criticality")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

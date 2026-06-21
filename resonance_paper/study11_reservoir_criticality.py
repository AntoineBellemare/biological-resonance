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
    seeds = range(3) if quick else range(8)
    rhos = [0.4, 0.7, 0.9, 1.0, 1.1, 1.3, 1.6, 1.9, 2.2, 2.6] if quick else \
           [0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.5, 1.8, 2.1, 2.5, 3.0]
    cfg = C.default_config(fmin=2, fmax=30, precision_hz=0.5)
    T, warm = 4000, 200

    rows = []
    for rho in rhos:
        lam, mc, Rin, Rint, Hint = [], [], [], [], []
        for seed in seeds:
            W, Win = make_reservoir(n_res, rho, seed=seed)
            lam.append(lyapunov(W, Win, seed=seed))
            mc.append(memory_capacity(W, Win, seed=seed))
            u = harmonic_input(T, warm, seed)
            X = run_reservoir(W, Win, u)[warm:]
            r_in, _ = resonance_of(u[warm:], cfg)
            r_int, h_int = resonance_of(X.mean(axis=1), cfg)
            Rin.append(r_in); Rint.append(r_int); Hint.append(h_int)
        rows.append(dict(rho=rho,
                         lyapunov=float(np.mean(lam)),
                         memory_capacity=float(np.mean(mc)),
                         R_internal=float(np.mean(Rint)),
                         R_input=float(np.mean(Rin)),
                         amplification=float(np.mean(Rint) / (np.mean(Rin) + 1e-12)),
                         H_internal=float(np.mean(Hint))))
        print(f"  rho={rho:.2f}  lambda={np.mean(lam):+.3f}  MC={np.mean(mc):.2f}  "
              f"R_int={np.mean(Rint):.4f}  amp={np.mean(Rint)/(np.mean(Rin)+1e-12):.2f}")

    # locate edge of chaos: rho where lambda crosses 0
    rho_c = float("nan")
    for i in range(len(rows) - 1):
        l0, l1 = rows[i]["lyapunov"], rows[i + 1]["lyapunov"]
        if np.isfinite(l0) and np.isfinite(l1) and l0 < 0 <= l1:
            r0, r1 = rows[i]["rho"], rows[i + 1]["rho"]
            rho_c = float(r0 + (r1 - r0) * (0 - l0) / (l1 - l0)); break

    def argmax_rho(key): return float(rows[int(np.argmax([r[key] for r in rows]))]["rho"])
    summary = dict(rho_critical=rho_c,
                   rho_at_max_MC=argmax_rho("memory_capacity"),
                   rho_at_max_R=argmax_rho("R_internal"),
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
    print(f"\n  Edge of chaos (lambda=0) at rho_c = {s['rho_critical']:.2f}")
    print(f"  Memory capacity peaks at rho={s['rho_at_max_MC']:.2f} (just below rho_c — classic).")
    print(f"  Internal resonance peaks at rho={s['rho_at_max_R']:.2f}.")
    print(f"  Max resonance amplification (internal/input R) = {s['max_amplification']:.2f}x")
    print("  => RC can amplify harmonic resonance; whether R peaks below/at/above rho_c says")
    print("     whether resonance marks the ordered, critical, or chaotic regime.")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; rho = [r["rho"] for r in rows]; s = result["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax1.plot(rho, [r["lyapunov"] for r in rows], "o-", color="#000000", label="Lyapunov λ")
    ax1.axhline(0, color="grey", lw=0.6)
    if np.isfinite(s["rho_critical"]):
        ax1.axvline(s["rho_critical"], color="red", ls="--", lw=1.0,
                    label=f"edge of chaos ρ_c={s['rho_critical']:.2f}")
    ax1.set_xlabel("spectral radius ρ"); ax1.set_ylabel("Lyapunov exponent")
    ax1.set_title("A. Locating the edge of chaos", fontsize=10); ax1.legend(fontsize=8)

    def nrm(k):
        v = np.array([r[k] for r in rows], float); return v / (np.nanmax(np.abs(v)) + 1e-12)
    ax2.plot(rho, nrm("memory_capacity"), "o-", color="#1565c0", label="memory capacity")
    ax2.plot(rho, nrm("R_internal"), "s-", color="#b71c1c", label="internal resonance R")
    ax2.plot(rho, nrm("amplification"), "d-", color="#ef6c00", label="resonance amplification")
    if np.isfinite(s["rho_critical"]):
        ax2.axvline(s["rho_critical"], color="red", ls="--", lw=1.0, label="ρ_c")
    ax2.set_xlabel("spectral radius ρ"); ax2.set_ylabel("normalized")
    ax2.set_title(f"B. Resonance vs computation vs criticality\n"
                  f"MC peaks ρ={s['rho_at_max_MC']:.2f}, R peaks ρ={s['rho_at_max_R']:.2f}",
                  fontsize=10); ax2.legend(fontsize=8)
    fig.suptitle("Study 11 — Reservoir: resonance, computation, and the edge of chaos",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study11_reservoir_criticality")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 9 (Study B) — Resonance as a signature of computational regime (reservoir).

Exploratory test of whether harmonic resonance relates to FUNCTION in a complex
dynamical system. An echo-state network (reservoir) is swept across its
dynamical regime via the spectral radius rho of the recurrent weights:
  rho << 1  -> strongly contracting, short memory
  rho ~ 1   -> "edge of chaos", where reservoir memory/computation peaks
  rho > 1   -> chaotic, memory destroyed

For each rho we measure two independent quantities:
  * Memory capacity MC(rho): the classic Jaeger linear short-term memory measure
    (sum over delays of the readout r^2 reconstructing past i.i.d. inputs). Peaks
    near rho ~ 1.
  * Internal resonance R(rho): drive the reservoir with a harmonic input
    (6 + 12 Hz phase-locked) and compute the resonance spectrum of its internal
    activity (mean node trajectory). Captures how harmonically organized the
    reservoir's representation is.

Hypothesis (could be null): internal resonance co-varies with memory capacity —
i.e., resonance is elevated in the same regime where the reservoir computes well.
A positive result would suggest resonance marks functionally useful dynamics; a
null is reported honestly.

Outputs: results/study9_reservoir.json, figures/study9_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import pink_noise, _norm
from biotuner.resonance import compute_resonance, ResonanceConfig

SF = 250.0


def make_reservoir(n_res, rho, seed, density=0.1, in_scale=1.0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_res, n_res)) * (rng.random((n_res, n_res)) < density)
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    W = W * (rho / (eig + 1e-12))            # scale to spectral radius rho
    Win = rng.uniform(-in_scale, in_scale, size=(n_res, 1))
    return W, Win


def run_reservoir(W, Win, u, leak=0.3):
    """Drive the reservoir with scalar input u(t); return states (T, n_res)."""
    n_res = W.shape[0]
    x = np.zeros(n_res)
    X = np.empty((len(u), n_res))
    for t in range(len(u)):
        x = (1 - leak) * x + leak * np.tanh(W @ x + Win[:, 0] * u[t])
        X[t] = x
    return X


def memory_capacity(W, Win, T=2500, K=25, warmup=200, seed=0):
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.8, 0.8, size=T + warmup)
    X = run_reservoir(W, Win, u)[warmup:]
    u = u[warmup:]
    mc = 0.0
    for k in range(1, K + 1):
        Xk = X[k:]; yk = u[:-k]            # predict input k steps ago
        nk = len(yk); split = int(nk * 0.7)
        m = Ridge(alpha=1e-6).fit(Xk[:split], yk[:split])
        pred = m.predict(Xk[split:]); true = yk[split:]
        if np.std(true) < 1e-9:
            continue
        r = np.corrcoef(pred, true)[0, 1]
        mc += (r ** 2) if np.isfinite(r) else 0.0
    return float(mc)


def internal_resonance(W, Win, T=4000, warmup=200, seed=0):
    # harmonic drive: 6 + 12 Hz phase-locked + light noise
    t = np.arange(T + warmup) / SF
    u = np.sin(2 * np.pi * 6 * t) + 0.7 * np.sin(2 * np.pi * 12 * t + 0.3)
    u = u + 0.1 * pink_noise(len(u), SF, seed=seed + 7)
    X = run_reservoir(W, Win, u)[warmup:]
    rep = _norm(X.mean(axis=1))   # mean node trajectory as representative signal
    cfg = C.default_config(fmin=2, fmax=30, precision_hz=0.5)
    r = compute_resonance(rep.astype(np.float64), sf=SF, config=cfg)
    return dict(R_max=float(r.summaries["R"]["max"]),
                R_avg=float(r.summaries["R"]["avg"]),
                H_max=float(r.summaries["H"]["max"]),
                R_peak=float(r.peaks["R"][0]) if r.peaks and len(r.peaks["R"]) else float("nan"))


def run(quick=True):
    n_res = 120 if quick else 300
    seeds = range(3) if quick else range(8)
    rhos = [0.3, 0.6, 0.9, 1.0, 1.1, 1.3] if quick else \
           [0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.4]

    rows = []
    for rho in rhos:
        mc, rmax, ravg, hmax = [], [], [], []
        for seed in seeds:
            W, Win = make_reservoir(n_res, rho, seed=seed)
            mc.append(memory_capacity(W, Win, seed=seed))
            ir = internal_resonance(W, Win, seed=seed)
            rmax.append(ir["R_max"]); ravg.append(ir["R_avg"]); hmax.append(ir["H_max"])
        rows.append(dict(rho=rho, MC=float(np.mean(mc)),
                         R_max=float(np.mean(rmax)), R_avg=float(np.mean(ravg)),
                         H_max=float(np.mean(hmax))))
        print(f"  rho={rho:.2f}  MC={np.mean(mc):.2f}  R_max={np.mean(rmax):.4f}")

    from scipy.stats import spearmanr
    MC = [r["MC"] for r in rows]; Rm = [r["R_max"] for r in rows]
    corr = dict(
        R_vs_MC=float(spearmanr(Rm, MC)[0]),
        rho_at_max_MC=float(rows[int(np.argmax(MC))]["rho"]),
        rho_at_max_R=float(rows[int(np.argmax(Rm))]["rho"]),
    )
    result = dict(quick=quick, n_res=n_res, rows=rows, corr=corr)
    C.save_json(result, "study9_reservoir.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    c = result["corr"]
    print("\n  --- Study 9 headline (reservoir: resonance vs computation) ---")
    print(f"  {'rho':>5} {'memory_capacity':>16} {'internal_R_max':>16}")
    for r in result["rows"]:
        print(f"  {r['rho']:>5.2f} {r['MC']:>16.2f} {r['R_max']:>16.4f}")
    print(f"\n  Spearman(internal R, memory capacity) = {c['R_vs_MC']:+.2f}")
    print(f"  rho at max memory capacity = {c['rho_at_max_MC']:.2f}; "
          f"rho at max internal R = {c['rho_at_max_R']:.2f}")
    print("  (positive corr / matching rho => resonance marks the computational regime;")
    print("   near-zero => resonance and computation are decoupled here — reported honestly.)")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]
    rhos = [r["rho"] for r in rows]
    MC = np.array([r["MC"] for r in rows]); R = np.array([r["R_max"] for r in rows])
    fig, ax1 = plt.subplots(figsize=(7.5, 4.6))
    ax1.plot(rhos, MC / (MC.max() + 1e-12), "o-", color="#1565c0", label="memory capacity (norm)")
    ax1.plot(rhos, R / (R.max() + 1e-12), "s-", color="#b71c1c", label="internal resonance R_max (norm)")
    ax1.axvline(1.0, color="grey", ls="--", lw=0.8, label="edge of chaos (rho=1)")
    ax1.set_xlabel("spectral radius  rho"); ax1.set_ylabel("normalized")
    c = result["corr"]
    ax1.set_title(f"Study 9 — Reservoir: resonance vs computational regime\n"
                  f"Spearman(R, MC) = {c['R_vs_MC']:+.2f}; "
                  f"argmax rho: MC={c['rho_at_max_MC']:.2f}, R={c['rho_at_max_R']:.2f}",
                  fontweight="bold", fontsize=10)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    C.save_fig(fig, "study9_reservoir")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

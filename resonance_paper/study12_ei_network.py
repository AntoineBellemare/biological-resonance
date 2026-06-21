"""Study 12 — E/I network: resonance at the edge of synchronization.

The model that finally lets the PHASE-COUPLING resonance R (not just harmonicity
H) be tested against criticality, because it produces BOTH a critical transition
AND oscillations. A stochastic Wilson-Cowan E/I network is swept across its
synchronization onset (a Hopf-type transition). Per the "edge of synchronization"
criticality picture (di Santo et al. 2018, PNAS), this transition is where
scale-free fluctuations and oscillations coincide — the oscillation-capable
analogue of Study 10's avalanche criticality and Study 11's reservoir edge of
chaos.

Control parameter: global coupling gain g.
  g small  -> quiescent / noise-dominated (no oscillation)
  g ~ g_c  -> EDGE OF SYNCHRONIZATION: oscillations emerge; order-parameter
              fluctuations (susceptibility) and critical slowing peak
  g large  -> strongly synchronized limit-cycle oscillations (harmonic-rich)

Per g we measure:
  * order parameter  = mean Hilbert envelope of E (oscillation amplitude)
  * susceptibility   = variance of the envelope (order-parameter fluctuations) -> peaks at g_c
  * critical slowing = autocorrelation time of E -> peaks at g_c
  * resonance        = H, PC, R of E(t)   (R is now non-trivial: E oscillates)

Question: where does phase-coupling resonance R peak relative to the synchroniza-
tion onset g_c? At the edge, or deep in the synchronized regime?

Outputs: results/study12_ei_network.json, figures/study12_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from biotuner.resonance import compute_resonance, ResonanceConfig
from biotuner.harmonic_connectivity import compute_cross_resonance

SF = 500.0
# Cross config: the matrix-entry PC path that works (Study 5), not the single-
# signal reduced PC (which reads ~0 for within-signal harmonic coupling).
CROSS_CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=80, noverlap=400,
                            coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                            ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                            return_intermediates=True)


def wilson_cowan(g, dur=10.0, dt=2e-4, tauE=0.010, tauI=0.020, noise=0.02, seed=0):
    """Stochastic Wilson-Cowan E/I rate model; returns (E, I) downsampled to SF."""
    rng = np.random.default_rng(seed)
    wEE, wEI, wIE, wII = 16.0, 12.0, 15.0, 3.0
    PE, PI = 1.25, 0.0
    sig = lambda x: 1.0 / (1.0 + np.exp(-(x - 4.0)))
    n = int(dur / dt)
    E = 0.1; I = 0.1
    outE = np.empty(n); outI = np.empty(n)
    sdt = np.sqrt(dt)
    for t in range(n):
        E = E + dt / tauE * (-E + sig(g * (wEE * E - wEI * I) + PE)) + noise * sdt * rng.standard_normal()
        I = I + dt / tauI * (-I + sig(g * (wIE * E - wII * I) + PI)) + noise * sdt * rng.standard_normal()
        outE[t] = E; outI[t] = I
    step = int((1.0 / SF) / dt)
    return outE[::step], outI[::step]


def cross_EI_resonance(E, I):
    """Cross-resonance between E and I populations (1:1 PING phase locking).
    Returns (H, PC, R) read at the dominant oscillation frequency via the
    targeted matrix entry — the readout that works (Study 5)."""
    Ez = _norm(E).astype(np.float64); Iz = _norm(I).astype(np.float64)
    # dominant oscillation frequency from E
    w = Ez * np.hanning(len(Ez)); X = np.abs(np.fft.rfft(w))
    f = np.fft.rfftfreq(len(Ez), 1 / SF); band = (f >= 2) & (f <= 80)
    f_osc = f[band][int(np.argmax(X[band]))]
    res = compute_cross_resonance(Ez, Iz, sf=SF, config=CROSS_CFG)
    fr = res.freqs; i = int(np.argmin(np.abs(fr - f_osc)))
    H = float(C.band_value(fr, res.factors["H"]["all"], f_osc))
    PC = float(res.phase_coupling_matrix[i, i])   # 1:1 entry at the oscillation
    return H, PC, H * PC, float(f_osc)


def autocorr_time(x, max_lag=200):
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    # lag where autocorr first drops below 1/e
    below = np.where(ac[:max_lag] < np.exp(-1))[0]
    return float(below[0]) if len(below) else float(max_lag)


def run(quick=True):
    seeds = range(3) if quick else range(8)
    gs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4] if quick else \
         [0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5]
    rows = []
    for g in gs:
        order, suscept, act, crossR, crossPC, crossH = [], [], [], [], [], []
        for seed in seeds:
            E, I = wilson_cowan(g, seed=seed)
            cut = len(E) // 5
            E = E[cut:]; I = I[cut:]                    # discard transient
            env = np.abs(hilbert(E - E.mean()))         # order parameter
            order.append(float(env.mean()))
            suscept.append(float(np.var(env)))          # order-parameter fluctuations
            act.append(autocorr_time(E))
            H, PC, R, _ = cross_EI_resonance(E, I)      # E<->I cross-resonance (PING)
            crossH.append(H); crossPC.append(PC); crossR.append(R)
        rows.append(dict(g=g,
                         order_param=float(np.mean(order)),
                         susceptibility=float(np.mean(suscept)),
                         autocorr_time=float(np.mean(act)),
                         crossEI_H=float(np.mean(crossH)),
                         crossEI_PC=float(np.mean(crossPC)),
                         crossEI_R=float(np.mean(crossR))))
        print(f"  g={g:.2f}  order={np.mean(order):.3f}  suscept={np.mean(suscept):.4f}  "
              f"crossEI_PC={np.mean(crossPC):.3f}  crossEI_R={np.mean(crossR):.3f}")

    # locate synchronization onset g_c = where susceptibility (order-param
    # fluctuation) peaks; fall back to steepest rise of the order parameter.
    g_c = float(rows[int(np.argmax([r["susceptibility"] for r in rows]))]["g"])

    def argmax_g(key): return float(rows[int(np.argmax([r[key] for r in rows]))]["g"])
    summary = dict(g_critical=g_c,
                   g_at_max_susceptibility=argmax_g("susceptibility"),
                   g_at_max_autocorr=argmax_g("autocorr_time"),
                   g_at_max_R=argmax_g("crossEI_R"),
                   g_at_max_PC=argmax_g("crossEI_PC"),
                   g_at_max_H=argmax_g("crossEI_H"))
    result = dict(quick=quick, rows=rows, summary=summary)
    C.save_json(result, "study12_ei_network.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 12 headline (E/I network: resonance vs edge of synchronization) ---")
    print(f"  {'g':>5} {'order':>7} {'suscept':>9} {'crossEI_PC':>11} {'crossEI_R':>10}")
    for r in result["rows"]:
        print(f"  {r['g']:>5.2f} {r['order_param']:>7.3f} {r['susceptibility']:>9.4f} "
              f"{r['crossEI_PC']:>11.3f} {r['crossEI_R']:>10.3f}")
    print(f"\n  Edge of synchronization g_c (susceptibility peak) = {s['g_critical']:.2f}")
    print(f"  E<->I phase coupling PC peaks at g={s['g_at_max_PC']:.2f}; "
          f"cross-resonance R peaks at g={s['g_at_max_R']:.2f}")
    print("  => first model where R (not just H) is non-trivial AND placeable vs criticality:")
    print("     R near g_c => resonance marks the edge of synchronization; R >> g_c => synchronized regime.")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; g = [r["g"] for r in rows]; s = result["summary"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    def nrm(k):
        v = np.array([r[k] for r in rows], float); return v / (np.nanmax(np.abs(v)) + 1e-12)
    ax.plot(g, nrm("order_param"), "o-", color="#455a64", label="order parameter (osc amplitude)")
    ax.plot(g, nrm("susceptibility"), "^-", color="#1565c0", label="susceptibility (fluctuations)")
    ax.plot(g, nrm("crossEI_PC"), "v-", color="#6a1b9a", label="E<->I phase coupling PC")
    ax.plot(g, nrm("crossEI_R"), "s-", color="#b71c1c", label="E<->I resonance R")
    ax.plot(g, nrm("crossEI_H"), "d-", color="#ef6c00", label="E<->I harmonicity H")
    ax.axvline(s["g_critical"], color="red", ls="--", lw=1.0,
               label=f"edge of synchronization g_c={s['g_critical']:.2f}")
    ax.set_xlabel("coupling gain  g"); ax.set_ylabel("normalized")
    ax.set_title("Study 12 — E/I network: resonance vs the edge of synchronization\n"
                 f"g_c={s['g_critical']:.2f}; R peaks at g={s['g_at_max_R']:.2f}, "
                 f"PC at g={s['g_at_max_PC']:.2f}", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    C.save_fig(fig, "study12_ei_network")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

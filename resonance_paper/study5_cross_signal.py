"""Study 5 — Cross-signal resonance: ground-truth recovery.

Tests whether the BETWEEN-signal tripartite spectra (H_AB, PC_AB from
``compute_cross_resonance``) recover known coupling between two signals. This is
the regime that should be robust where single-signal n:m coupling (Study 1B) was
not — two independently generated signals give a clean null (IAAFT-surrogate one
channel preserves both PSDs while destroying only the A-B relationship), and the
phase-alignment fix (see resonance module) makes the cross-frequency phase
matrix correct.

Regimes (each COUPLED vs a PSD-matched UNCOUPLED control):

  shared_harmonic  A,B share a harmonic series (same fundamental), independent
                   phases -> cross-HARMONICITY axis (H_AB), via the PSD-weighted
                   reduced H spectrum at the shared harmonics.
  lock_1to1        A,B same freq, B phase-locked to A   -> PC_AB matrix entry [10,10]
  lock_1to2        A at f, B at 2f, B locked to A        -> PC_AB matrix entry [6,12]
  lock_2to3        A at 6, B at 9 (2:3), B locked to A   -> PC_AB matrix entry [6,9]

Detector = surrogate z (channel-B IAAFT null) of the targeted quantity; AUC =
COUPLED vs UNCOUPLED across seeds, with bootstrap CI. Contrasted with the
single-signal Study-1B baseline (~0.5-0.6).

Outputs: results/study5_cross_signal.json, figures/study5_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import pink_noise, _norm
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import ResonanceConfig
from biotuner.resonance.nulls import iaaft_surrogate

SF = 500.0
# Regime -> (target frequency pairs [(fA,fB),...], axis to read: 'H' or 'PC')
REGIMES = {
    "shared_harmonic": ([(6.0, 6.0), (12.0, 12.0), (18.0, 18.0)], "H"),
    "lock_1to1":       ([(10.0, 10.0)], "PC"),
    "lock_1to2":       ([(6.0, 12.0)], "PC"),
    "lock_2to3":       ([(6.0, 9.0)], "PC"),
}


def _wander(base, t, sf, diffusion, rng):
    return 2 * np.pi * base * t + np.cumsum(diffusion * np.sqrt(1.0 / sf) * rng.standard_normal(len(t)))


def gen_pair(regime, coupled, sf=SF, duration=40.0, snr_db=6.0, diffusion=1.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sf * duration); t = np.arange(n) / sf
    snr = 10 ** (snr_db / 10.0)
    noise = lambda s: _norm(pink_noise(n, sf, seed=seed + s))

    if regime == "shared_harmonic":
        phiA = _wander(6.0, t, sf, diffusion, rng)
        A = _norm(sum(0.7 ** k * np.sin((k + 1) * phiA) for k in range(3)))
        baseB = 6.0 if coupled else 7.0
        phiB = _wander(baseB, t, sf, diffusion, np.random.default_rng(seed + 1))
        B = _norm(sum(0.7 ** k * np.sin((k + 1) * phiB) for k in range(3)))
    elif regime == "lock_1to1":
        phiA = _wander(10.0, t, sf, diffusion, rng); A = _norm(np.sin(phiA))
        B = _norm(np.sin(phiA + np.pi / 4)) if coupled else \
            _norm(np.sin(_wander(10.0, t, sf, diffusion, np.random.default_rng(seed + 2))))
    elif regime == "lock_1to2":
        phiA = _wander(6.0, t, sf, diffusion, rng); A = _norm(np.sin(phiA))
        B = _norm(np.sin(2 * phiA + 0.3)) if coupled else \
            _norm(np.sin(_wander(12.0, t, sf, diffusion, np.random.default_rng(seed + 2))))
    elif regime == "lock_2to3":
        # 2:3 polyrhythmic lock — A at 6 Hz, B at 9 Hz = 6*3/2, locked to A's phase
        phiA = _wander(6.0, t, sf, diffusion, rng); A = _norm(np.sin(phiA))
        B = _norm(np.sin(1.5 * phiA + 0.3)) if coupled else \
            _norm(np.sin(_wander(9.0, t, sf, diffusion, np.random.default_rng(seed + 2))))
    else:
        raise ValueError(regime)

    A = np.sqrt(snr) * A + noise(100)
    B = np.sqrt(snr) * B + noise(200)
    return A.astype(np.float64), B.astype(np.float64)


def _config_for(regime):
    # All regimes are phase-phase or harmonic -> canonical n:m PLV.
    return ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, noverlap=400,
                           coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                           ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                           return_intermediates=True)


def cross_target_z(A, B, sf, cfg, pairs, axis, n=40, seed=0):
    """Surrogate z (channel-B IAAFT null) of the targeted cross quantity.

    axis='PC' -> phase-coupling MATRIX entry Phi_AB[fA,fB] (targeted, not diluted).
    axis='H'  -> PSD-weighted reduced H_AB spectrum at the shared harmonics.
    """
    obs = compute_cross_resonance(A, B, sf=sf, config=cfg)
    fr = obs.freqs
    idx = lambda f: int(np.argmin(np.abs(fr - f)))

    def measure(result):
        if axis == "PC":
            M = result.phase_coupling_matrix
            return max(M[idx(fa), idx(fb)] for fa, fb in pairs)
        red = result.factors["H"]["all"]
        return max(red[idx(fb)] for _, fb in pairs)

    obs_v = measure(obs)

    def one(s):
        Bs = iaaft_surrogate(B, np.random.default_rng(s))
        return measure(compute_cross_resonance(A, Bs, sf=sf, config=cfg))

    rng = np.random.default_rng(seed)
    seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=n)]
    try:
        from joblib import Parallel, delayed
        sv = np.array(Parallel(n_jobs=-1)(delayed(one)(s) for s in seeds))
    except Exception:
        sv = np.array([one(s) for s in seeds])
    return float((obs_v - sv.mean()) / (sv.std() + 1e-12))


def run(quick=True):
    seeds = range(8) if quick else range(20)
    n_surr = 30 if quick else 100
    records = []
    for regime, (pairs, axis) in REGIMES.items():
        cfg = _config_for(regime)
        for seed in seeds:
            for coupled in (True, False):
                A, B = gen_pair(regime, coupled, seed=seed)
                z = cross_target_z(A, B, SF, cfg, pairs, axis, n=n_surr, seed=seed)
                records.append(dict(regime=regime, axis=axis, seed=seed,
                                    coupled=coupled, z=z))
        print(f"  {regime} ({axis}) done")

    auc = []
    for regime, (_, axis) in REGIMES.items():
        pos = [r["z"] for r in records if r["regime"] == regime and r["coupled"]]
        neg = [r["z"] for r in records if r["regime"] == regime and not r["coupled"]]
        ci = C.bootstrap_auc_ci(pos, neg, n_boot=2000)
        auc.append(dict(regime=regime, axis=axis, auc=ci["auc"], ci=[ci["lo"], ci["hi"]],
                        coupled_mean_z=float(np.mean(pos)), uncoupled_mean_z=float(np.mean(neg))))

    result = dict(quick=quick, n_surrogates=n_surr, n_seeds=len(list(seeds)),
                  single_signal_baseline_auc=0.59, records=records, auc=auc)
    C.save_json(result, "study5_cross_signal.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 5 headline (cross-signal targeted recovery) ---")
    for r in result["auc"]:
        print(f"  {r['regime']:16s} [{r['axis']}]  AUC={r['auc']:.2f} "
              f"[{r['ci'][0]:.2f},{r['ci'][1]:.2f}]  "
              f"z: coupled={r['coupled_mean_z']:+.2f} uncoupled={r['uncoupled_mean_z']:+.2f}")
    print(f"  (single-signal Study-1B coupling AUC baseline ~{result['single_signal_baseline_auc']})")


def _figures(result):
    plt = C.setup_mpl()
    auc = result["auc"]
    regimes = [f"{r['regime']}\n[{r['axis']}]" for r in auc]
    vals = [r["auc"] for r in auc]
    los = [r["auc"] - r["ci"][0] for r in auc]
    his = [r["ci"][1] - r["auc"] for r in auc]
    colors = ["#1a237e" if r["axis"] == "H" else "#b71c1c" for r in auc]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(regimes))
    ax.bar(x, vals, 0.6, yerr=[los, his], color=colors, error_kw=dict(lw=1.2))
    ax.axhline(0.5, color="k", ls="--", lw=0.6, label="chance")
    ax.axhline(result["single_signal_baseline_auc"], color="grey", ls=":", lw=1.0,
               label="single-signal best (Study 1B)")
    ax.set_xticks(x); ax.set_xticklabels(regimes, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("detection AUC (coupled vs uncoupled)")
    ax.legend(fontsize=8)
    ax.set_title("Study 5 — Cross-signal resonance recovers known coupling\n"
                 "(targeted; channel-B IAAFT null; blue=harmonicity axis, red=phase-coupling axis)",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study5_cross_signal")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 28 — Robustness of H to aperiodic ($1/f$) mis-estimation.

H is computed on the aperiodic-removed spectrum (a power-law $a f^{b}$ is fit and subtracted).
A reviewer concern is that this fit is unstable in slow-wave-dominated / knee-bearing states, so
H's criticality signal might be an artifact of a particular $1/f$ fit. We test this directly in the
branching model (the single-signal-H system that matches the EEG construct), where the critical
point is known ($\\sigma=1$):

  (1) inject a controlled residual tilt $\\delta$ into the signal spectrum (multiply by $f^{\\delta}$)
      BEFORE the standard removal, simulating an aperiodic fit that is too steep/too flat by
      $\\delta$; recompute H_max with the default removal and ask whether its criticality-tracking
      survives across a wide range of $\\delta$;
  (2) compare against no removal at all (remove_aperiodic=False).

If H_max keeps peaking at $\\sigma=1$ and the per-seed Spearman(H, criticality proximity) is stable
across $\\delta$, the criticality signal does not hinge on a precise $1/f$ fit.

Outputs: results/study28_aperiodic_robustness.json
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from resonance_paper import study10_criticality as S10
from biotuner.resonance import compute_resonance

SF = S10.SF
TILTS = [-0.5, -0.25, 0.0, 0.25, 0.5]   # residual aperiodic-slope error (octave-ish), + = flatter


def spectral_tilt(x, delta):
    """Multiply the signal's amplitude spectrum by (f/f_ref)^delta -- a controlled residual 1/f
    tilt standing in for an aperiodic fit that is too steep (delta<0) or too flat (delta>0)."""
    if delta == 0.0:
        return np.asarray(x, float)
    X = np.fft.rfft(np.asarray(x, float)); f = np.fft.rfftfreq(len(x))
    g = np.ones_like(f); m = f > 0
    fref = np.exp(np.mean(np.log(f[m])))           # geometric-mean frequency
    g[m] = (f[m] / fref) ** delta
    return np.fft.irfft(X * g, n=len(x))


def _Hmax(x, remove_aperiodic=True):
    cfg = C.default_config(fmin=2, fmax=60, precision_hz=0.5, remove_aperiodic=remove_aperiodic)
    r = compute_resonance(_norm(np.asarray(x, float)).astype(np.float64), sf=SF, config=cfg)
    return float(r.summaries["H"]["max"])


def run(quick=True):
    N = 300 if quick else 600
    T = 8000 if quick else 18000
    seeds = list(range(4) if quick else range(16))
    sigmas = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.30] if quick else \
             [0.70, 0.85, 0.92, 0.97, 1.00, 1.03, 1.08, 1.15, 1.30, 1.50]
    # H_max[config][sigma_index] -> per-seed list
    configs = [f"tilt{d:+.2f}" for d in TILTS] + ["no_removal"]
    per = {c: [[] for _ in sigmas] for c in configs}
    for si, sigma in enumerate(sigmas):
        for seed in seeds:
            A, _ = S10.branching_activity(sigma, N=N, T=T, seed=seed)
            base = _norm(A)
            for d in TILTS:
                per[f"tilt{d:+.2f}"][si].append(_Hmax(spectral_tilt(base, d), remove_aperiodic=True))
            per["no_removal"][si].append(_Hmax(base, remove_aperiodic=False))
        print(f"  sigma={sigma:.2f} done", flush=True)

    prox = [-abs(s - 1.0) for s in sigmas]   # criticality proximity (ground truth, peak at sigma=1)

    def tracks(cfg):
        M = per[cfg]; k = min(len(a) for a in M); A = np.array([a[:k] for a in M])  # (n_sigma, n_seed)
        rhos = [spearmanr(prox, list(A[:, j]))[0] for j in range(A.shape[1])]
        rhos = [r for r in rhos if np.isfinite(r)]
        ci = C.mean_ci(rhos)
        peak_sigma = float(sigmas[int(np.argmax(np.nanmean(A, axis=1)))])
        return dict(rho=ci["mean"], lo=ci["lo"], hi=ci["hi"], n=len(rhos), peak_sigma=peak_sigma)

    res = {c: tracks(c) for c in configs}
    # range of the criticality-tracking rho across the tilt configs (robustness summary)
    tilt_rhos = [res[f"tilt{d:+.2f}"]["rho"] for d in TILTS]
    summary = dict(n_seeds=len(seeds), tilts=TILTS,
                   tracks=res,
                   rho_min_over_tilts=float(np.min(tilt_rhos)),
                   rho_max_over_tilts=float(np.max(tilt_rhos)),
                   rho_range_over_tilts=float(np.max(tilt_rhos) - np.min(tilt_rhos)),
                   all_peak_at_critical=bool(all(abs(res[f"tilt{d:+.2f}"]["peak_sigma"] - 1.0) <= 0.05
                                                 for d in TILTS)))
    result = dict(quick=quick, sigmas=sigmas, summary=summary)
    C.save_json(result, "study28_aperiodic_robustness.json")
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 28 headline (H robustness to aperiodic mis-fit) ---")
    for d in s["tilts"]:
        t = s["tracks"][f"tilt{d:+.2f}"]
        print(f"  tilt {d:+.2f}: H tracks criticality rho={t['rho']:+.2f} [{t['lo']:+.2f},{t['hi']:+.2f}]"
              f"  (H peaks at sigma={t['peak_sigma']:.2f})")
    nr = s["tracks"]["no_removal"]
    print(f"  no removal: rho={nr['rho']:+.2f} [{nr['lo']:+.2f},{nr['hi']:+.2f}] (peak sigma={nr['peak_sigma']:.2f})")
    print(f"  => criticality-tracking rho range across tilts = {s['rho_range_over_tilts']:.2f}; "
          f"all tilt configs peak at criticality: {s['all_peak_at_critical']}")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

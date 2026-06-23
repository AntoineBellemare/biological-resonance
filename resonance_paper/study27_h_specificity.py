"""Study 27 — Specificity of H across all three model systems: does harmonicity peak at
criticality because of genuine integer-ratio harmonic organization, or merely because of
spectral SHAPE (slope, power, peakiness)?  (Reviewer #2, #6, #8.)

H is phase-blind (computed from the power spectrum), so phase-randomization leaves it ~unchanged.
The informative nulls perturb the SPECTRAL SHAPE while preserving one nuisance property each:

  * phase_random   preserves the PSD exactly      -> H ~ unchanged (demonstrates phase-blindness).
  * aaft           preserves PSD + amplitude dist.
  * matched_slope  1/f colored noise, SAME slope, no peaks -> H must stay low (not just the slope).
  * peak_warp      keeps the 1/f background and the SAME peaks (count + heights) but relocates each
                   peak frequency by an independent multiplicative jitter -> destroys integer-ratio
                   RELATIONSHIPS while preserving peakiness/power. The fair test of "is it the
                   integer-ratio structure, or just having peaks?".

Decisive test: H(real) peaks at the critical point with a margin over peak_warp and matched_slope,
which stay flat/low across the control sweep. Run on the branching, reservoir, and Wilson-Cowan
(E-signal) models.

Outputs: results/study27_h_specificity.json
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from resonance_paper import study10_criticality as S10
from resonance_paper import study11_reservoir_criticality as S11
from resonance_paper import study12_ei_network as S12
from biotuner.resonance import compute_resonance


# ---------------------------------------------------------------------------
# spectrum-matched surrogates (each preserves a different nuisance property)
# ---------------------------------------------------------------------------
def phase_random(x, rng):
    X = np.fft.rfft(x); mag = np.abs(X)
    ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    return np.fft.irfft(mag * np.exp(1j * ph), n=len(x))


def aaft(x, rng):
    n = len(x)
    g = np.sort(rng.standard_normal(n))
    xg = g[np.argsort(np.argsort(x))]
    xs = phase_random(xg, rng)
    return np.sort(x)[np.argsort(np.argsort(xs))]


def matched_slope(x, rng):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x)); P = np.abs(X) ** 2
    m = f > 0
    slope, inter = np.polyfit(np.log(f[m]), np.log(P[m] + 1e-30), 1)
    amp = np.zeros(len(X)); amp[m] = np.sqrt(np.exp(inter) * f[m] ** slope)
    ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    return np.fft.irfft(amp * np.exp(1j * ph), n=len(x))


def peak_warp(x, rng):
    """Keep the 1/f background and the same peaks (count + heights) but relocate each peak
    frequency by an independent multiplicative jitter -> integer-ratio relationships destroyed,
    peakiness/power preserved. The fair 'is it the harmonic positions?' null."""
    X = np.fft.rfft(x); mag = np.abs(X); n = len(x)
    f = np.fft.rfftfreq(n); m = f > 0
    slope, inter = np.polyfit(np.log(f[m]), np.log(mag[m] + 1e-30), 1)
    bg = np.zeros_like(mag); bg[m] = np.exp(inter) * f[m] ** slope
    resid = np.clip(mag - bg, 0.0, None)
    thr = resid.max() * 0.05 if resid.max() > 0 else 0.0
    pk, _ = find_peaks(resid, height=thr, distance=2)
    new_resid = np.zeros_like(resid)
    for p in pk:
        q = int(round(p * rng.uniform(0.80, 1.22)))     # independent multiplicative jitter
        if 1 <= q < len(new_resid):
            new_resid[q] += resid[p]
    ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    return np.fft.irfft((bg + new_resid) * np.exp(1j * ph), n=n)


SURR = {"real": None, "phase_random": phase_random, "aaft": aaft,
        "matched_slope": matched_slope, "peak_warp": peak_warp}


def _Hmax(x, sf, cfg):
    r = compute_resonance(_norm(np.asarray(x, float)).astype(np.float64), sf=sf, config=cfg)
    return float(r.summaries["H"]["max"])


# ---------------------------------------------------------------------------
# model signal generators (each returns a 1-D activity series)
# ---------------------------------------------------------------------------
def _gen_branching(sigma, seed, quick):
    return S10.branching_activity(sigma, N=(300 if quick else 600),
                                  T=(8000 if quick else 18000), seed=seed)[0]


def _gen_reservoir(rho, seed, quick):
    from resonance_paper.study9_reservoir import make_reservoir, run_reservoir
    W, Win = make_reservoir(120 if quick else 280, rho, seed=seed)
    u = S11.noise_input(4000, 200, seed)
    return S11.internal_mode(run_reservoir(W, Win, u)[200:])


def _gen_wc(g, seed, quick):
    E, _I = S12.wilson_cowan(g, seed=seed)
    return E[len(E) // 5:]


def _model_specs(quick):
    return {
        "branching": dict(gen=_gen_branching, sf=S10.SF, crit=1.00,
                          params=[0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.30] if quick else
                                  [0.70, 0.85, 0.92, 0.97, 1.00, 1.03, 1.08, 1.15, 1.30, 1.50],
                          cfg=C.default_config(fmin=2, fmax=60, precision_hz=0.5)),
        "reservoir": dict(gen=_gen_reservoir, sf=S11.SF, crit=1.00,
                          params=[0.6, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6] if quick else
                                  [0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.6, 2.1],
                          cfg=C.default_config(fmin=2, fmax=60, precision_hz=0.5)),
        "wilson_cowan": dict(gen=_gen_wc, sf=S12.SF, crit=1.00,
                          params=[0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4] if quick else
                                  [0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.1, 1.25, 1.4, 1.6],
                          cfg=C.default_config(fmin=2, fmax=80, precision_hz=0.5)),
    }


def _run_model(name, spec, seeds, quick):
    params, kinds = spec["params"], list(SURR)
    per = {k: [] for k in kinds}; rows = []
    for p in params:
        acc = {k: [] for k in kinds}
        for seed in seeds:
            x = spec["gen"](p, seed, quick)
            rng = np.random.default_rng(1000 + seed)
            for k in kinds:
                acc[k].append(_Hmax(x if k == "real" else SURR[k](x, rng), spec["sf"], spec["cfg"]))
        row = dict(param=p)
        for k in kinds:
            ci = C.mean_ci(acc[k]); per[k].append(acc[k])
            row["H_" + k] = ci["mean"]; row["H_" + k + "_sem"] = ci["sem"]
        rows.append(row)
        print(f"    {name[:9]:>9} p={p:.2f}  H_real={row['H_real']:.3f}  ph={row['H_phase_random']:.3f}  "
              f"aaft={row['H_aaft']:.3f}  slope={row['H_matched_slope']:.3f}  warp={row['H_peak_warp']:.3f}",
              flush=True)

    c_arr = np.array(params, float)
    ic = int(np.argmin(np.abs(c_arr - spec["crit"])))

    def mat(k):
        kk = min(len(a) for a in per[k]); return np.array([a[:kk] for a in per[k]])

    def peak_param(k):
        M = mat(k); rng = np.random.default_rng(7); locs = []
        for _ in range(2000):
            idx = rng.integers(0, M.shape[1], M.shape[1])
            locs.append(c_arr[int(np.nanargmax(np.nanmean(M[:, idx], axis=1)))])
        return dict(mean=float(np.mean(locs)),
                    lo=float(np.percentile(locs, 2.5)), hi=float(np.percentile(locs, 97.5)))

    def margin(k):
        R = mat("real")[ic]; S = mat(k)[ic]; nn = min(len(R), len(S))
        d = R[:nn] - S[:nn]; ci = C.mean_ci(d)
        return dict(delta=ci["mean"], lo=ci["lo"], hi=ci["hi"])

    def tracks(k):
        M = mat(k); prox = [-abs(s - spec["crit"]) for s in params]; rhos = []
        for j in range(M.shape[1]):
            rr = spearmanr(prox, list(M[:, j]))[0]
            if np.isfinite(rr):
                rhos.append(rr)
        ci = C.mean_ci(rhos)
        return dict(rho=ci["mean"], lo=ci["lo"], hi=ci["hi"], n=len(rhos))

    return dict(params=params, crit=spec["crit"], rows=rows,
                peak_param={k: peak_param(k) for k in kinds},
                margin_at_crit={k: margin(k) for k in kinds if k != "real"},
                tracks={k: tracks(k) for k in kinds})


def run(quick=True):
    seeds = list(range(4) if quick else range(12))
    specs = _model_specs(quick)
    out = {}
    for name, spec in specs.items():
        print(f"  [{name}]")
        out[name] = _run_model(name, spec, seeds, quick)
    result = dict(quick=quick, n_seeds=len(seeds), models=out)
    C.save_json(result, "study27_h_specificity.json")
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 27 headline (H specificity vs spectrum-matched surrogates, 3 models) ---")
    for name, m in result["models"].items():
        pp = m["peak_param"]
        print(f"  [{name}] criticality~{m['crit']:.2f}; H_real peaks at "
              f"{pp['real']['mean']:.2f} [{pp['real']['lo']:.2f},{pp['real']['hi']:.2f}]")
        for k in ["peak_warp", "matched_slope"]:
            mg = m["margin_at_crit"][k]; tr = m["tracks"][k]
            print(f"      vs {k:<13} margin@crit={mg['delta']:+.3f} [{mg['lo']:+.3f},{mg['hi']:+.3f}]"
                  f"   (surrogate tracks rho={tr['rho']:+.2f})")
        tr = m["tracks"]["real"]
        print(f"      H_real tracks criticality: rho={tr['rho']:+.2f} [{tr['lo']:+.2f},{tr['hi']:+.2f}]")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

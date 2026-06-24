"""Study 35 — Measuring n:m coupling PROPERLY: peak-based + Hilbert + volume-conduction-robust metric.

The earlier real-data probe (study30) used compute_cross_resonance with the DEFAULT stft phase
estimator + nm_plv_canonical on a hand-imposed frequency grid -- the least-sensitive combination on
three axes at once:
  * stft phase MISSES n:m locking (biotuner's own docs: hilbert "recovers n:m that stft misses").
  * nm_plv is volume-conduction (0-lag) SENSITIVE -> inflates cross-channel 1:1 with instantaneous mixing.
  * a fixed (fa,fb) grid ignores the channels' ACTUAL spectral peaks.

This study re-tests the SAME real recordings with biotuner's purpose-built peak-based instrument,
``harmonic_connectivity.compute_peak_phase_coupling_connectivity``, which detects each channel's real
peaks, resolves the n:m ratio between them, bandpass+Hilbert-transforms, and dispatches a chosen metric.
We sweep the coupling metric to separate genuine coupling from volume conduction:

  nm_plv_canonical : 0-lag SENSITIVE  (baseline; what study30 effectively used, but now peak+Hilbert)
  nm_wpli          : 0-lag ROBUST     (rejects volume conduction, magnitude-weighted)
  nm_rrci          : 0-lag ROBUST     (imaginary part only)
  nm_pli           : 0-lag ROBUST     (sign only)

For each (contrast, metric): fraction of epochs whose peak-based coupling is surrogate-significant
(IAAFT-of-B null, rank p<=0.05), median z, and the rest/motor or EO/EC state-AUC. The decisive
comparisons:
  (a) Does the 1:1-dominated coupling SURVIVE wpli/rrci/pli?  If it collapses vs plv -> it was volume
      conduction; if it survives -> genuine zero-lag-free coupling band power cannot see.
  (b) Does the proper instrument surface n!=m coupling that the stft+plv+grid probe missed?

Outputs: results/study35_proper_nm_coupling.json
"""
from __future__ import annotations

import warnings

import numpy as np

from resonance_paper import _common as C
from resonance_paper.study30_realdata_coupling import _paired_epochs, _band_power
from resonance_paper import datasets as D
from biotuner.harmonic_connectivity import harmonic_connectivity
from biotuner.resonance.nulls import iaaft_surrogate

warnings.filterwarnings("ignore")

METRICS = ["nm_plv_canonical", "nm_wpli", "nm_rrci", "nm_pli"]
ROBUST = {"nm_wpli", "nm_rrci", "nm_pli"}
PEAKS_FN = "fixed"          # fast FFT-based spectral peak detection; detects each channel's real peaks
BANDWIDTH = 3.0            # >=3 Hz so Hilbert phase is stable on non-stationary rhythms (per API docs)


def _peak_coupling(A, B, sf, metric):
    """Off-diagonal peak-based n:m coupling scalar between channels A and B (proper instrument)."""
    hc = harmonic_connectivity(sf=sf, data=np.array([A, B]), peaks_function=PEAKS_FN,
                               precision=0.5, min_freq=2, max_freq=45, n_peaks=5)
    M = hc.compute_peak_phase_coupling_connectivity(
        coupling_metric=metric, ratio_kernel="binary",
        ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True},
        bandwidth=BANDWIDTH, aggregate="max", graph=False)
    return float(M[0, 1])


def _z_and_p(A, B, sf, metric, n_surr, seed):
    obs = _peak_coupling(A, B, sf, metric)
    if not np.isfinite(obs):
        return dict(z=np.nan, rank_p=np.nan, obs=np.nan)
    rng = np.random.default_rng(seed)
    sv = []
    for s in rng.integers(0, 2 ** 31 - 1, n_surr):
        Bs = iaaft_surrogate(B, np.random.default_rng(int(s)))
        v = _peak_coupling(A, Bs, sf, metric)
        if np.isfinite(v):
            sv.append(v)
    sv = np.array(sv)
    if len(sv) < 5:
        return dict(z=np.nan, rank_p=np.nan, obs=obs)
    z = float((obs - sv.mean()) / (sv.std() + 1e-12))
    rank_p = float((1 + np.sum(sv >= obs)) / (len(sv) + 1))
    return dict(z=z, rank_p=rank_p, obs=obs)


def _probe(name, epochs, pos, neg, n_surr):
    res = {}
    for metric in METRICS:
        recs = []
        for ep in epochs:
            r = _z_and_p(ep["A"], ep["B"], ep["sf"], metric, n_surr,
                         seed=hash((ep["subj"], ep["cond"], metric)) % (2 ** 31))
            r["cond"] = ep["cond"]
            recs.append(r)
        sig = [r for r in recs if np.isfinite(r["rank_p"])]
        pz = [r["z"] for r in sig if r["cond"] == pos]
        nz = [r["z"] for r in sig if r["cond"] == neg]
        res[metric] = dict(
            n=len(sig),
            frac_sig=float(np.mean([r["rank_p"] <= 0.05 for r in sig])) if sig else float("nan"),
            median_z=float(np.median([r["z"] for r in sig])) if sig else float("nan"),
            state_auc=C.bootstrap_auc_ci(pz, nz)["auc"] if pz and nz else float("nan"))
        print(f"      {metric:18s} frac_sig={res[metric]['frac_sig']:.2f}  "
              f"median_z={res[metric]['median_z']:+.2f}  state-AUC={res[metric]['state_auc']:.2f}  "
              f"(n={res[metric]['n']})", flush=True)
    return res


def run(quick=True):
    subjects = (1, 2, 3, 4, 5) if quick else tuple(range(1, 11))
    n_surr = 24 if quick else 49
    out = {}

    motor = _paired_epochs(subjects, (("REST", D.REST_RUN), ("MOTOR", D.MOTOR_RUN)), ["C3", "C4"])
    if motor:
        print(f"  motor C3-C4 epochs: {len(motor)} (proper peak-based instrument)", flush=True)
        bp = {b: C.bootstrap_auc_ci([_band_power(e["A"], e["sf"], lo, hi) for e in motor if e["cond"] == "MOTOR"],
                                    [_band_power(e["A"], e["sf"], lo, hi) for e in motor if e["cond"] == "REST"])["auc"]
              for b, (lo, hi) in [("mu", (8, 12)), ("beta", (18, 22))]}
        out["motor"] = dict(pc=_probe("motor", motor, "MOTOR", "REST", n_surr), band_power_auc=bp)

    occ = _paired_epochs(subjects, (("EO", D.EO_RUN), ("EC", D.EC_RUN)), ["O1", "O2"])
    if occ:
        print(f"  occipital O1-O2 epochs: {len(occ)} (proper peak-based instrument)", flush=True)
        bp = {"alpha": C.bootstrap_auc_ci([_band_power(e["A"], e["sf"], 8, 12) for e in occ if e["cond"] == "EC"],
                                          [_band_power(e["A"], e["sf"], 8, 12) for e in occ if e["cond"] == "EO"])["auc"]}
        out["occipital"] = dict(pc=_probe("occ", occ, "EC", "EO", n_surr), band_power_auc=bp)

    C.save_json(dict(quick=quick, peaks_function=PEAKS_FN, bandwidth=BANDWIDTH, out=out),
                "study35_proper_nm_coupling.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 35 headline (proper peak-based n:m instrument) ---")
    for ds, blk in out.items():
        plv = blk["pc"].get("nm_plv_canonical", {})
        wpli = blk["pc"].get("nm_wpli", {})
        print(f"  [{ds}] band-power AUC: " + "  ".join(f"{k}={v:.2f}" for k, v in blk["band_power_auc"].items()))
        print(f"     volume-conduction test: PLV frac_sig={plv.get('frac_sig', float('nan')):.2f} "
              f"-> wPLI frac_sig={wpli.get('frac_sig', float('nan')):.2f} "
              f"({'survives -> genuine non-zero-lag coupling' if wpli.get('frac_sig', 0) > 0.3 else 'collapses -> was volume conduction'})")
    print("  => PLV high + wPLI low  => the cross-channel coupling was volume conduction (artifact).")
    print("     wPLI/rrci frac_sig clearly > 0.05 => genuine coupling band power cannot measure.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

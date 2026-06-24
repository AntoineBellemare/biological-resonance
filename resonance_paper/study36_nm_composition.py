"""Study 36 — Is the genuine cross-channel coupling 1:1 (lagged coherence) or true n!=m?

Study 35 showed the framework's peak-based + Hilbert + volume-conduction-robust coupling (wpli/rrci/pli)
detects genuine, surrogate-significant cross-channel phase coupling in real EEG that survives 0-lag
rejection. But ``aggregate="max"`` could be dominated by 1:1 (same-frequency) peak pairs -- which is
ordinary lagged coherence (real, band-power-blind, but NOT a novel n:m claim). This study resolves the
ratio composition honestly.

Per epoch we extract each channel's real peaks (fixed FFT peaks, n_peaks=5), resolve the n:m ratio of
every cross-channel peak pair with the binary kernel (max_nm=3, tolerance=0.05, fallback_to_1_1=FALSE so
non-matching pairs are excluded, not coerced to 1:1), and split pairs into:
  * genuine 1:1  : the two peaks are within tolerance of the same frequency
  * genuine n!=m : the two peaks sit at a low-integer ratio (1:2, 2:3, 1:3, ...)
For the volume-conduction-robust metric (wpli) we take the max coupling within each class and test it
against an IAAFT-of-B null (rank p). We report, per dataset:
  - fraction of epochs that even HAVE a genuine n!=m peak pair,
  - frac_sig of the n!=m-max vs the 1:1-max (both wpli),
  - the observed n!=m ratios.
Decisive: if the n!=m class is surrogate-significant well above 5%, the framework surfaces TRUE n:m
phase coupling band power cannot see; if only the 1:1 class fires, the honest claim is lagged coherence.

Outputs: results/study36_nm_composition.json
"""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np

from resonance_paper import _common as C
from resonance_paper.study30_realdata_coupling import _paired_epochs
from resonance_paper import datasets as D
from biotuner.harmonic_connectivity import harmonic_connectivity
from biotuner.resonance.nulls import iaaft_surrogate
from biotuner.resonance.registry import RATIO_KERNELS

warnings.filterwarnings("ignore")

METRIC = "nm_wpli"          # volume-conduction-robust
BANDWIDTH = 3.0
RK_PARAMS = {"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": False}


def _hc(A, B, sf):
    return harmonic_connectivity(sf=sf, data=np.array([A, B]), peaks_function="fixed",
                                 precision=0.5, min_freq=2, max_freq=45, n_peaks=5)


def _class_max(hc, A, B):
    """Return (max wpli over genuine 1:1 pairs, max over genuine n!=m pairs, list of n!=m ratio strings)."""
    peaks1, peaks2 = hc._extract_peaks_for_pair(A, B)
    rk = RATIO_KERNELS["binary"]
    one_vals, nm_vals, nm_ratios = [], [], []
    for p1 in peaks1:
        for p2 in peaks2:
            if p1 <= 0 or p2 <= 0:
                continue
            W, N, M = rk(np.array([float(p1)]), np.array([float(p2)]), **RK_PARAMS)
            if W[0, 0] <= 0:
                continue
            n, m = int(N[0, 0]), int(M[0, 0])
            val = hc._peak_pair_coupling(A, B, float(p1), float(p2), METRIC, {},
                                         rk, RK_PARAMS, BANDWIDTH)
            if not np.isfinite(val):
                continue
            if n == m:
                one_vals.append(val)
            else:
                nm_vals.append(val)
                nm_ratios.append(f"{min(n, m)}:{max(n, m)} ({p1:.1f},{p2:.1f})")
    return (max(one_vals) if one_vals else np.nan,
            max(nm_vals) if nm_vals else np.nan, nm_ratios)


def _sig(A, B, sf, which, n_surr, seed):
    """rank_p of the class-max (which='one' or 'nm') vs IAAFT-of-B null."""
    hc = _hc(A, B, sf)
    o1, onm, ratios = _class_max(hc, A, B)
    obs = o1 if which == "one" else onm
    if not np.isfinite(obs):
        return dict(rank_p=np.nan, obs=np.nan, ratios=ratios)
    rng = np.random.default_rng(seed)
    sv = []
    for s in rng.integers(0, 2 ** 31 - 1, n_surr):
        Bs = iaaft_surrogate(B, np.random.default_rng(int(s)))
        h = _hc(A, Bs, sf)
        s1, snm, _ = _class_max(h, A, Bs)
        v = s1 if which == "one" else snm
        if np.isfinite(v):
            sv.append(v)
    if len(sv) < 5:
        return dict(rank_p=np.nan, obs=obs, ratios=ratios)
    return dict(rank_p=float((1 + np.sum(np.array(sv) >= obs)) / (len(sv) + 1)), obs=obs, ratios=ratios)


def _probe(epochs, n_surr):
    res = {}
    for which in ("one", "nm"):
        recs = [_sig(ep["A"], ep["B"], ep["sf"], which, n_surr,
                     seed=hash((ep["subj"], ep["cond"], which)) % (2 ** 31)) for ep in epochs]
        have = [r for r in recs if np.isfinite(r["obs"])]
        sig = [r for r in have if np.isfinite(r["rank_p"])]
        res[which] = dict(
            frac_epochs_with_pair=float(len(have) / len(recs)) if recs else 0.0,
            frac_sig=float(np.mean([r["rank_p"] <= 0.05 for r in sig])) if sig else float("nan"),
            n=len(sig))
        if which == "nm":
            allr = [r for rec in recs for r in rec["ratios"]]
            from collections import Counter
            res["nm"]["top_ratios"] = Counter(r.split(" ")[0] for r in allr).most_common(6)
        print(f"    {which:4s}: epochs_with_pair={res[which]['frac_epochs_with_pair']:.2f}  "
              f"frac_sig={res[which]['frac_sig']:.2f}  (n={res[which]['n']})", flush=True)
    return res


def run(quick=True):
    subjects = (1, 2, 3, 4, 5) if quick else tuple(range(1, 11))
    n_surr = 24 if quick else 49
    out = {}
    motor = _paired_epochs(subjects, (("REST", D.REST_RUN), ("MOTOR", D.MOTOR_RUN)), ["C3", "C4"])
    if motor:
        print(f"  motor C3-C4 epochs: {len(motor)} (n:m composition, wpli)", flush=True)
        out["motor"] = _probe(motor, n_surr)
    occ = _paired_epochs(subjects, (("EO", D.EO_RUN), ("EC", D.EC_RUN)), ["O1", "O2"])
    if occ:
        print(f"  occipital O1-O2 epochs: {len(occ)} (n:m composition, wpli)", flush=True)
        out["occipital"] = _probe(occ, n_surr)
    C.save_json(dict(quick=quick, metric=METRIC, out=out), "study36_nm_composition.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 36 headline (n:m composition of the genuine coupling) ---")
    for ds, blk in out.items():
        nm = blk["nm"]; one = blk["one"]
        print(f"  [{ds}] 1:1 wpli frac_sig={one['frac_sig']:.2f} | "
              f"n!=m present in {nm['frac_epochs_with_pair']:.2f} of epochs, wpli frac_sig={nm['frac_sig']:.2f}  "
              f"top n!=m ratios={nm.get('top_ratios')}")
    print("  => n!=m frac_sig >> 0.05 => TRUE n:m coupling band power misses (novel).")
    print("     n!=m at ~0.05 / rare => the genuine coupling is 1:1 lagged coherence (value-add by kind, not n:m).")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

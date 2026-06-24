"""Study 40 — Polyrhythm soundness battery: specificity (ratio identification) + robustness.

Builds on study39 (detection) with the two deeper soundness axes, across a wide n:m range and every
technique (metrics called at the CORRECT multipliers, isolating the metric from kernel/estimator):

  PART A — SPECIFICITY / ratio identification. For a signal genuinely locked at a:b, scan a grid of
    candidate coprime multipliers (N,M) and ask whether the metric PEAKS at the true (N=b, M=a). A
    sound technique gives a sharp peak at the true ratio and low values elsewhere. Reports per-metric
    identification accuracy + a confusion matrix.

  PART B — ROBUSTNESS.
    B1 dose-response: mean metric vs coupling strength kappa (0=independent .. 1=perfect lock) — should
       rise monotonically.
    B2 SNR sweep: detection AUC (kappa=1 vs independent) vs observation SNR — sensitivity floor.

All metrics computed on bandpass+Hilbert phase at the two component frequencies, with the CORRECT n:m
multipliers (N=b, M=a). Raw metric fns are called with explicit correct multipliers (so this measures
each metric's intrinsic soundness, not the kernel-convention bug, which study39 already isolated).

Outputs: results/study40_polyrhythm_battery.json
"""
from __future__ import annotations

import warnings
from math import gcd

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from resonance_paper import _common as C
from biotuner.resonance.coupling import nm_plv, nm_pli, nm_wpli, nm_rrci, nm_wpli_complex

warnings.filterwarnings("ignore")

SF = 500.0
DUR = 24.0
N = int(SF * DUR)
T = np.arange(N) / SF

RATIOS = [(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (5, 7)]
# candidate coprime multipliers (N on phi_a, M on phi_b), 1..7 (covers 7:5 for the 5:7 ratio)
CANDIDATES = sorted({(n, m) for n in range(1, 8) for m in range(1, 8) if gcd(n, m) == 1})
OFFSET = np.pi / 3
METRICS = ["plv", "pli", "wpli", "rrci", "wpli_complex"]


def _bp(x, f, bw=3.0):
    lo = max(f - bw / 2, 0.5); hi = min(f + bw / 2, SF / 2 - 1)
    sos = butter(4, [lo / (SF / 2), hi / (SF / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def gen(a, b, kappa, seed, snr_db=3.0):
    """f_a=u*a, f_b=u*b. B mixes a phase-LOCKED b/a component with an INDEPENDENT one at f_b,
    weighted by kappa in [0,1] (graded coupling strength: 0=independent, 1=perfect lock)."""
    r = np.random.default_rng(seed)
    u = 30.0 / b; fa, fb = u * a, u * b
    drift = np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    phi_a = 2 * np.pi * fa * T + drift
    A = np.sin(phi_a)
    locked = np.sin((b / a) * phi_a + OFFSET)
    indep = np.sin(2 * np.pi * fb * T + np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N)))
    B = kappa * locked + (1.0 - kappa) * indep        # graded coupling via amplitude mixing
    snr = 10 ** (snr_db / 10.0)
    A = np.sqrt(snr) * A + r.standard_normal(N)
    B = np.sqrt(snr) * B + r.standard_normal(N)
    return _norm(A), _norm(B), fa, fb


def _metric_at(metric, ana_a, ana_b, Nn, Mm):
    pa, pb = np.angle(ana_a), np.angle(ana_b)
    if metric == "plv":          return nm_plv(pa, pb, Nn, Mm)
    if metric == "pli":          return nm_pli(pa, pb, Nn, Mm)
    if metric == "wpli":         return nm_wpli(pa, pb, Nn, Mm)
    if metric == "rrci":         return nm_rrci(pa, pb, Nn, Mm)
    if metric == "wpli_complex": return nm_wpli_complex(ana_a, ana_b, Nn, Mm)
    raise ValueError(metric)


def part_a_specificity(seeds):
    """Per metric: identification accuracy (argmax candidate == true) + confusion (true ratio x picked)."""
    ident = {m: [] for m in METRICS}
    confusion = {m: {} for m in METRICS}
    for (a, b) in RATIOS:
        true_nm = (b, a)                                  # correct multipliers for f_b/f_a = b/a
        picked = {m: [] for m in METRICS}
        for s in seeds:
            A, B, fa, fb = gen(a, b, 1.0, s)
            ana_a, ana_b = hilbert(_bp(A, fa)), hilbert(_bp(B, fb))
            for m in METRICS:
                vals = {cand: _metric_at(m, ana_a, ana_b, cand[0], cand[1]) for cand in CANDIDATES}
                best = max(vals, key=vals.get)
                picked[m].append(best)
                ident[m].append(1.0 if best == true_nm else 0.0)
        for m in METRICS:
            from collections import Counter
            confusion[m][f"{a}:{b}"] = Counter(f"{p[0]}:{p[1]}" for p in picked[m]).most_common(1)[0]
    return {"identification_accuracy": {m: float(np.mean(ident[m])) for m in METRICS},
            "confusion_top_pick": confusion}


def part_b_robustness(seeds):
    kappas = [0.0, 0.25, 0.5, 0.75, 1.0]
    snrs = [-9, -6, -3, 0, 3, 6]
    dose = {m: {} for m in METRICS}
    for k in kappas:
        for m in METRICS:
            vals = []
            for (a, b) in RATIOS:
                for s in seeds:
                    A, B, fa, fb = gen(a, b, k, s)
                    vals.append(_metric_at(m, hilbert(_bp(A, fa)), hilbert(_bp(B, fb)), b, a))
            dose[m][f"k={k}"] = float(np.mean(vals))
    snr_auc = {m: {} for m in METRICS}
    for snr in snrs:
        for m in METRICS:
            pos, neg = [], []
            for (a, b) in RATIOS:
                for s in seeds:
                    Ac, Bc, fa, fb = gen(a, b, 1.0, s, snr_db=snr)
                    Au, Bu, fa2, fb2 = gen(a, b, 0.0, s + 777, snr_db=snr)
                    pos.append(_metric_at(m, hilbert(_bp(Ac, fa)), hilbert(_bp(Bc, fb)), b, a))
                    neg.append(_metric_at(m, hilbert(_bp(Au, fa2)), hilbert(_bp(Bu, fb2)), b, a))
            snr_auc[m][f"{snr}dB"] = C.bootstrap_auc_ci(pos, neg)["auc"]
    return {"dose_response_kappa": dose, "snr_auc": snr_auc}


def run(quick=True):
    seeds = list(range(6 if quick else 15))
    out = {"part_a_specificity": part_a_specificity(seeds),
           "part_b_robustness": part_b_robustness(seeds)}
    C.save_json(dict(quick=quick, ratios=[f"{a}:{b}" for a, b in RATIOS],
                     n_candidates=len(CANDIDATES), out=out), "study40_polyrhythm_battery.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 40 headline (polyrhythm specificity + robustness) ---")
    ia = out["part_a_specificity"]["identification_accuracy"]
    print("  ratio-IDENTIFICATION accuracy (argmax candidate == true ratio):")
    for m in METRICS:
        print(f"      {m:14s} {ia[m]:.2f}")
    print("  dose-response (mean metric at kappa=0 -> 1):")
    for m in METRICS:
        d = out["part_b_robustness"]["dose_response_kappa"][m]
        print(f"      {m:14s} " + "  ".join(f"{k.split('=')[1]}:{v:.2f}" for k, v in d.items()))
    print("  SNR detection AUC (kappa=1 vs independent):")
    for m in METRICS:
        s = out["part_b_robustness"]["snr_auc"][m]
        print(f"      {m:14s} " + "  ".join(f"{k}:{v:.2f}" for k, v in s.items()))
    print("  => sound technique: high identification accuracy, monotone dose-response, AUC->1 as SNR rises.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

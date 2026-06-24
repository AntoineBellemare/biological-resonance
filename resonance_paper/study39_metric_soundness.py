"""Study 39 — Are the n:m coupling TECHNIQUES sound? Big-ratio × all-metrics × offset × volume-conduction.

Audit (not assertion). We feed each registered biotuner coupling metric GROUND-TRUTH n:m phase locks
across a wide ratio range and ask, per technique: does it retrieve the lock (detection AUC coupled vs
uncoupled), and under what conditions does it fail? Three axes that the code audit flagged:

  (1) CONVENTION. Metrics apply ``n*phi_i - m*phi_j`` literally. For a frequency ratio f_b/f_a = b/a
      the STATIONARY combination is ``b*phi_a - a*phi_b`` (=const). So the CORRECT call is (n=b, m=a).
      The biotuner ratio kernels instead return (n,m) with ratio=f_b/f_a≈m/n, i.e. they hand the metric
      (n=a, m=b) -> ``a*phi_a - b*phi_b``, which is NON-stationary for a true lock. Only
      nm_plv_canonical swaps internally; nm_pli/nm_wpli/nm_rrci do NOT. We test both the CORRECT call
      and the SWAPPED (kernel-convention) call to quantify the bug per metric.
  (2) 0-LAG BLIND SPOT. PLI/wPLI/RRCi are 0 by construction at 0/pi relative phase. We sweep the lock
      offset (0, pi/2) to expose it: PLV detects at any offset; the robust metrics need a lag.
  (3) VOLUME CONDUCTION. Adding a shared instantaneous component should INFLATE plv (false positive)
      but be rejected by pli/wpli/rrci. We add a 0-lag shared term to the uncoupled control and check
      whether each metric stays specific.

Each metric is computed on bandpass+Hilbert phase at the two known frequencies (isolating the metric
from the pipeline's estimator/kernel/weighting). Reports per (ratio, metric, offset) detection AUC.

Outputs: results/study39_metric_soundness.json
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from resonance_paper import _common as C
from biotuner.resonance.coupling import nm_plv, nm_pli, nm_wpli, nm_rrci, nm_plv_canonical, nm_wpli_complex

warnings.filterwarnings("ignore")

SF = 500.0
DUR = 24.0
N = int(SF * DUR)
T = np.arange(N) / SF

# wide low-integer ratio set (a:b, coprime, a<b), placed so both freqs land in [5,40] Hz
RATIOS = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (2, 7),
          (3, 4), (3, 5), (3, 7), (4, 5), (4, 7), (5, 6), (5, 7), (6, 7)]
OFFSETS = {"0": 0.0, "pi/2": np.pi / 2}
SNR_DB = 3.0


def _bp(x, f, bw=3.0):
    lo = max(f - bw / 2, 0.5); hi = min(f + bw / 2, SF / 2 - 1)
    sos = butter(4, [lo / (SF / 2), hi / (SF / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def gen(a, b, coupled, offset, seed, vc=0.0):
    """f_a = u*a, f_b = u*b (u set so f_b≈30). coupled: phi_b = (b/a) phi_a + offset (genuine b/a lock)."""
    r = np.random.default_rng(seed)
    u = 30.0 / b
    fa, fb = u * a, u * b
    drift = np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    phi_a = 2 * np.pi * fa * T + drift
    A = np.sin(phi_a)
    if coupled:
        B = np.sin((b / a) * phi_a + offset)
    else:
        B = np.sin(2 * np.pi * fb * T + np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N)))
    snr = 10 ** (SNR_DB / 10.0)
    A = np.sqrt(snr) * A + r.standard_normal(N)
    B = np.sqrt(snr) * B + r.standard_normal(N)
    if vc > 0:                       # volume conduction: shared 0-lag copy of A leaks into B
        B = B + vc * A
    return _norm(A), _norm(B), fa, fb


def _metrics(A, B, fa, fb, a, b):
    """Compute every technique with the CORRECT convention (n=b, m=a) on bandpass+Hilbert phase.
    Also the SWAPPED (kernel-convention, n=a,m=b) PLV to quantify the convention bug."""
    ana_a = hilbert(_bp(A, fa)); ana_b = hilbert(_bp(B, fb))
    pa, pb = np.angle(ana_a), np.angle(ana_b)
    n, m = b, a                      # correct: b*phi_a - a*phi_b is stationary for f_b/f_a=b/a
    return {
        "plv":          nm_plv(pa, pb, n, m),
        "plv_canonical": nm_plv_canonical(pa, pb, a, b),   # swaps internally -> b*phi_a - a*phi_b
        "pli":          nm_pli(pa, pb, n, m),
        "wpli":         nm_wpli(pa, pb, n, m),
        "rrci":         nm_rrci(pa, pb, n, m),
        "wpli_complex": nm_wpli_complex(ana_a, ana_b, n, m),
        "plv_SWAPPED":  nm_plv(pa, pb, a, b),              # kernel-convention (the bug)
    }


METRIC_KEYS = ["plv", "plv_canonical", "pli", "wpli", "rrci", "wpli_complex", "plv_SWAPPED"]


def run(quick=True):
    seeds = range(6 if quick else 15)
    out = {"detection": {}, "specificity": {}, "volume_conduction": {}}

    # --- detection AUC per (offset, metric), pooled across ratios; + per-ratio for offset pi/2 ---
    for okey, off in OFFSETS.items():
        per_metric = {k: {"pos": [], "neg": []} for k in METRIC_KEYS}
        per_ratio = {}
        for (a, b) in RATIOS:
            rp = {k: {"pos": [], "neg": []} for k in METRIC_KEYS}
            for s in seeds:
                gc = gen(a, b, True, off, s); cp = _metrics(gc[0], gc[1], gc[2], gc[3], a, b)
                gu = gen(a, b, False, off, s + 999); un = _metrics(gu[0], gu[1], gu[2], gu[3], a, b)
                for k in METRIC_KEYS:
                    per_metric[k]["pos"].append(cp[k]); per_metric[k]["neg"].append(un[k])
                    rp[k]["pos"].append(cp[k]); rp[k]["neg"].append(un[k])
            per_ratio[f"{a}:{b}"] = {k: C.bootstrap_auc_ci(rp[k]["pos"], rp[k]["neg"])["auc"] for k in METRIC_KEYS}
        out["detection"][okey] = {
            "pooled_auc": {k: C.bootstrap_auc_ci(per_metric[k]["pos"], per_metric[k]["neg"])["auc"] for k in METRIC_KEYS},
            "per_ratio": per_ratio}

    # --- volume-conduction specificity: uncoupled control gets a 0-lag shared component ---
    off = np.pi / 2
    vc_metric = {k: {"pos": [], "neg": []} for k in METRIC_KEYS}
    for (a, b) in RATIOS:
        for s in seeds:
            gc = gen(a, b, True, off, s); cp = _metrics(gc[0], gc[1], gc[2], gc[3], a, b)
            gv = gen(a, b, False, off, s + 555, vc=0.8); vc = _metrics(gv[0], gv[1], gv[2], gv[3], a, b)
            for k in METRIC_KEYS:
                vc_metric[k]["pos"].append(cp[k]); vc_metric[k]["neg"].append(vc[k])
    out["volume_conduction"] = {
        "auc_coupled_vs_vc_uncoupled": {k: C.bootstrap_auc_ci(vc_metric[k]["pos"], vc_metric[k]["neg"])["auc"] for k in METRIC_KEYS},
        "note": "AUC<0.5 => the metric scores the 0-lag-contaminated UNCOUPLED control above the true lock (volume-conduction false positive)."}

    C.save_json(dict(quick=quick, ratios=[f"{a}:{b}" for a, b in RATIOS], snr_db=SNR_DB, out=out),
                "study39_metric_soundness.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 39 headline (technique soundness on n:m ground truth) ---")
    print(f"  {'metric':14s} {'AUC@0lag':>9s} {'AUC@pi/2':>9s} {'AUC vs VC':>10s}")
    for k in METRIC_KEYS:
        a0 = out["detection"]["0"]["pooled_auc"][k]
        a9 = out["detection"]["pi/2"]["pooled_auc"][k]
        vc = out["volume_conduction"]["auc_coupled_vs_vc_uncoupled"][k]
        print(f"  {k:14s} {a0:9.2f} {a9:9.2f} {vc:10.2f}")
    print("  guide: sound technique -> high AUC at BOTH offsets + AUC>=0.5 vs VC (rejects volume conduction).")
    print("         plv: detects any offset but inflates on VC. pli/wpli/rrci: ~0.5 at 0-lag (blind spot),")
    print("         should work at pi/2 + reject VC. plv_SWAPPED: the kernel-convention bug (expect ~0.5).")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 46 — Does n:m (polyrhythmic) phase coupling CHANGE across sleep stages?

A state contrast is more robust than an absolute coupling claim: sleep stages have very different
oscillatory regimes (Wake alpha; N2 spindles + SO; N3 slow-wave/delta; REM theta), so a SYSTEMATIC change
in n:m coupling across stages cannot come from a fixed within-signal harmonic artifact. We sweep the FULL
complex-ratio space (coprime n:m, denominator <= 16) realized at sleep-band frequencies, measure
within-channel cross-frequency PHASE coupling in Fpz-Cz per 30 s epoch with the ENTIRE metric panel
(plv / pli / wpli / rrci vs the all-moment rho_entropy / conditional_prob / phase_mi), normalize each
epoch by a WAVEFORM-PRESERVING time-shuffle surrogate (controls the harmonic baseline; Study 43), and
test whether the surrogate-z differs across stages (Kruskal-Wallis), Benjamini-Hochberg FDR-corrected.

Outputs: results/study46_sleep_stages.json
"""
from __future__ import annotations

import warnings
from math import gcd

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kruskal

from resonance_paper import _common as C
from resonance_paper.study14_sleep import load_sleep
from biotuner.resonance.coupling import (nm_plv, nm_pli, nm_wpli, nm_rrci,
                                         nm_rho_entropy, nm_conditional_prob, nm_phase_mi)
from biotuner.resonance.nulls import time_shuffle_surrogate

warnings.filterwarnings("ignore")

PANEL = {"plv": nm_plv, "pli": nm_pli, "wpli": nm_wpli, "rrci": nm_rrci,
         "rho_entropy": nm_rho_entropy, "conditional_prob": nm_conditional_prob, "phase_mi": nm_phase_mi}
ALL_MOMENT = {"rho_entropy", "conditional_prob", "phase_mi"}
FIRST_MOMENT = {"plv", "pli", "wpli", "rrci"}
STAGES = ["Wake", "N2", "N3", "REM"]
CENTER = 4.0  # geometric centre (Hz) of each ratio's frequency pair (keeps both in the sleep band)
# full complex-ratio space: coprime (p,q), q<=16, ratio q/p in [1.4, 6]
RATIOS = sorted({(p, q) for q in range(2, 17) for p in range(1, q)
                 if gcd(p, q) == 1 and 1.4 <= q / p <= 6.0}, key=lambda pq: pq[1] / pq[0])


def _phase(x, sf, f):
    nyq = sf / 2.0; bw = max(0.6, 0.4 * f)
    sos = butter(3, [max(f - bw / 2, 0.3) / nyq, min(f + bw / 2, nyq - 0.1) / nyq], btype="band", output="sos")
    return np.angle(hilbert(sosfiltfilt(sos, x)))


def _epoch_z(x, sf, fa, fb, n, m, n_surr, rng):
    pa, pb = _phase(x, sf, fa), _phase(x, sf, fb)
    sp = [(lambda xs: (_phase(xs, sf, fa), _phase(xs, sf, fb)))(time_shuffle_surrogate(x, rng))
          for _ in range(n_surr)]
    out = {}
    for mn, fn in PANEL.items():
        obs = fn(pa, pb, n, m)
        sv = np.array([fn(a, b, n, m) for a, b in sp])
        out[mn] = float((obs - sv.mean()) / (sv.std() + 1e-12))
    return out


def _bh_fdr(pvals):
    p = np.asarray(pvals); order = np.argsort(p); ranks = np.arange(1, len(p) + 1)
    q = np.empty(len(p)); q[order] = (p[order] * len(p) / ranks)
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]; q[order] = q_sorted
    return np.clip(q, 0, 1)


def run(quick=True):
    n_sub = 8 if quick else 12
    n_surr = 15 if quick else 30
    items = load_sleep(n_subjects=n_sub, max_epochs_per_stage=8 if quick else 14)
    by_stage = {s: [it for it in items if it.get("state") == s] for s in STAGES}
    print("  epochs/stage:", {s: len(v) for s, v in by_stage.items()}, "| ratios:", len(RATIOS), flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for (p, q) in RATIOS:
        # anchor the SLOW component in the SO/delta band (where sleep stages differ most),
        # keeping >=1.5 Hz separation so the two bands don't overlap.
        r = q / p; fa = max(1.0, 1.5 / (r - 1)); fb = fa * r; n, m = q, p   # n*fa = m*fb stationary
        per = {mn: {s: [] for s in STAGES} for mn in PANEL}
        for s in STAGES:
            for it in by_stage[s]:
                z = _epoch_z(np.asarray(it["X"])[0].astype(np.float64), it["sf"], fa, fb, n, m, n_surr, rng)
                for mn in PANEL:
                    per[mn][s].append(z[mn])
        for mn in PANEL:
            groups = [per[mn][s] for s in STAGES if len(per[mn][s]) >= 3]
            try:
                H, pv = kruskal(*groups)
            except Exception:
                H, pv = float("nan"), 1.0
            means = {s: float(np.mean(per[mn][s])) if per[mn][s] else float("nan") for s in STAGES}
            rows.append(dict(ratio=f"{n}:{m}", fa=round(fa, 2), fb=round(fb, 2), metric=mn,
                             stage_mean_z=means, kruskal_p=float(pv),
                             peak_stage=max(means, key=lambda s: means[s] if np.isfinite(means[s]) else -9)))
    qv = _bh_fdr([r["kruskal_p"] if np.isfinite(r["kruskal_p"]) else 1.0 for r in rows])
    for r, q_ in zip(rows, qv):
        r["fdr_q"] = float(q_)
    C.save_json(dict(quick=quick, n_subjects=n_sub, n_surr=n_surr, n_ratios=len(RATIOS),
                     epochs_per_stage={s: len(v) for s, v in by_stage.items()}, rows=rows),
                "study46_sleep_stages.json")
    _headline(rows)
    return rows


def _headline(rows):
    sig = sorted([r for r in rows if r["fdr_q"] <= 0.05], key=lambda r: r["fdr_q"])
    print("\n  --- Study 46 headline (n:m coupling changes across sleep stages) ---")
    print(f"  {len(sig)}/{len(rows)} (ratio x metric) tests survive FDR(0.05) for stage-dependence.")
    for r in sig[:14]:
        z = r["stage_mean_z"]
        print(f"    {r['ratio']:6s} ({r['fa']}:{r['fb']}Hz) {r['metric']:16s} peak {r['peak_stage']:4s} q={r['fdr_q']:.3f}  "
              + " ".join(f"{s}={z[s]:+.1f}" for s in STAGES))
    fm = {r["ratio"] for r in sig if r["metric"] in FIRST_MOMENT}
    am = {r["ratio"] for r in sig if r["metric"] in ALL_MOMENT}
    print(f"  first-moment (plv/pli/wpli/rrci) flag {len(fm)} ratios; all-moment (rho/cond/MI) flag {len(am)};")
    print(f"  all-moment indices reveal stage-dependent coupling at {len(am - fm)} ratios the first-moment family MISSES.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 48 — Sleep n:m phase-coupling spectrum: WITHIN-channel vs BETWEEN-channel, surrogate-controlled.

Two questions left open by study47:
 (1) Does the within-channel stage-dependence survive a WAVEFORM-PRESERVING surrogate (time-shuffle)?
     If yes -> the n:m coupling changes BEYOND the channel's own non-sinusoidal waveform/harmonic baseline
     (genuine); if it collapses -> the stage effect was the waveform shape changing (still real physiology,
     but harmonic structure, not inter-oscillator coupling).
 (2) Is there a BETWEEN-channel (Fpz-Cz <-> Pz-Oz) stage-dependent n:m phase-coupling spectrum? This is the
     cleaner test of coupling between two (more independent) regional oscillators; null = IAAFT-of-channel-B.

We build the whole-spectrum n:m PC (all pairs, fraction-kernel ratios, correct multipliers, Hilbert phase,
power-independent), surrogate-z per epoch, then test stage-dependence (Kruskal + BH-FDR) for WITHIN and
BETWEEN. Metric panel: PLV (first moment) vs rho_entropy (all-moment).

Outputs: results/study48_sleep_within_cross.json
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import kruskal

from resonance_paper import _common as C
from resonance_paper.study14_sleep import load_sleep
from biotuner.resonance.coupling import nm_plv, nm_rho_entropy
from biotuner.resonance.nm_detect import nm_multipliers
from biotuner.resonance.nulls import time_shuffle_surrogate, iaaft_surrogate

warnings.filterwarnings("ignore")

STAGES = ["Wake", "N2", "N3", "REM"]
METRICS = {"plv": nm_plv, "rho_entropy": nm_rho_entropy}
FREQS = np.arange(1.5, 15.1, 1.5)                      # 10 frequency bins (1.5 .. 15 Hz)
NM = [[nm_multipliers(fa, fb) for fb in FREQS] for fa in FREQS]


def _phases(x, sf):
    nyq = sf / 2.0; out = []
    for f in FREQS:
        bw = max(0.8, 0.4 * f)
        sos = butter(3, [max(f - bw / 2, 0.3) / nyq, min(f + bw / 2, nyq - 0.1) / nyq], btype="band", output="sos")
        out.append(np.angle(hilbert(sosfiltfilt(sos, x))))
    return out


def _pc_spectrum(pa, pb, fn):
    nf = len(FREQS); pc = np.zeros(nf)
    for i in range(nf):
        acc = 0.0
        for j in range(nf):
            if i == j:
                continue
            n, m = NM[i][j]
            acc += fn(pa[i], pb[j], n, m)
        pc[i] = acc / (nf - 1)
    return pc


def _z_spectrum(pa_obs, pb_obs, surr_pb_list, fn, surr_pa_list=None):
    obs = _pc_spectrum(pa_obs, pb_obs, fn)
    sv = []
    for k in range(len(surr_pb_list)):
        spa = surr_pa_list[k] if surr_pa_list is not None else pa_obs
        sv.append(_pc_spectrum(spa, surr_pb_list[k], fn))
    sv = np.array(sv)
    return (obs - sv.mean(0)) / (sv.std(0) + 1e-12)


def run(quick=True):
    n_sub = 5 if quick else 8
    n_surr = 10 if quick else 20
    items = load_sleep(n_subjects=n_sub, max_epochs_per_stage=5 if quick else 10)
    by_stage = {s: [it for it in items if it.get("state") == s and np.asarray(it["X"]).shape[0] >= 2]
                for s in STAGES}
    print("  epochs/stage:", {s: len(v) for s, v in by_stage.items()}, "| freqs:", len(FREQS), flush=True)

    rng = np.random.default_rng(0)
    Z = {reg: {mn: {s: [] for s in STAGES} for mn in METRICS} for reg in ("within", "between")}
    for s in STAGES:
        for it in by_stage[s]:
            A = np.asarray(it["X"])[0].astype(np.float64)      # Fpz-Cz
            B = np.asarray(it["X"])[1].astype(np.float64)      # Pz-Oz
            sf = it["sf"]
            pA, pB = _phases(A, sf), _phases(B, sf)
            # within (Fpz-Cz): waveform-preserving time-shuffle null
            surr_within = [_phases(time_shuffle_surrogate(A, rng), sf) for _ in range(n_surr)]
            # between (A<->B): IAAFT-of-B null (preserves B spectrum, breaks A-B relationship)
            surr_between = [_phases(iaaft_surrogate(B, rng), sf) for _ in range(n_surr)]
            for mn, fn in METRICS.items():
                Z["within"][mn][s].append(_z_spectrum(pA, pA, surr_within, fn))
                Z["between"][mn][s].append(_z_spectrum(pA, pB, surr_between, fn))
        print(f"    {s} done ({len(by_stage[s])} epochs)", flush=True)

    out = {"freqs": FREQS.tolist(), "stages": STAGES, "within": {}, "between": {}}
    for reg in ("within", "between"):
        for mn in METRICS:
            mean_z = {s: (np.mean(np.stack(Z[reg][mn][s]), 0).tolist() if Z[reg][mn][s] else []) for s in STAGES}
            pf = []
            for k in range(len(FREQS)):
                groups = [[zz[k] for zz in Z[reg][mn][s]] for s in STAGES if len(Z[reg][mn][s]) >= 3]
                try:
                    pf.append(kruskal(*groups)[1])
                except Exception:
                    pf.append(1.0)
            q = _bh_fdr(pf)
            out[reg][mn] = dict(mean_z=mean_z, kruskal_p=pf, fdr_q=q.tolist(),
                                n_sig=int(np.sum(np.asarray(q) <= 0.05)), min_p=float(np.min(pf)),
                                peak_freq=float(FREQS[int(np.argmin(pf))]))
    C.save_json(dict(quick=quick, n_subjects=n_sub, n_surr=n_surr,
                     epochs_per_stage={s: len(v) for s, v in by_stage.items()}, **out),
                "study48_sleep_within_cross.json")
    _headline(out)
    return out


def _bh_fdr(p):
    p = np.asarray(p, float); order = np.argsort(p); ranks = np.arange(1, len(p) + 1)
    q = np.empty(len(p)); q[order] = np.minimum.accumulate((p[order] * len(p) / ranks)[::-1])[::-1]
    return np.clip(q, 0, 1)


def _headline(out):
    print("\n  --- Study 48 headline (within- vs between-channel, surrogate-controlled) ---")
    for reg in ("within", "between"):
        for mn in METRICS:
            d = out[reg][mn]
            print(f"  {reg:8s} {mn:12s}: {d['n_sig']}/{len(out['freqs'])} freq bins of the surrogate-z PC "
                  f"spectrum stage-dependent (FDR<=0.05); strongest {d['peak_freq']:.1f} Hz (p={d['min_p']:.1e}).")
    print("  WITHIN surviving the time-shuffle null => coupling beyond the channel's own waveform/harmonics.")
    print("  BETWEEN significant => inter-regional n:m coupling beyond the IAAFT (chance) null.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

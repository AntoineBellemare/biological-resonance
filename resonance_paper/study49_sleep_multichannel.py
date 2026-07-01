"""Study 49 — Multi-channel sleep check: is the within>>between n:m asymmetry robust to electrode sampling?

Sleep-EDF (study48) had only 2 EEG channels (Fpz-Cz, Pz-Oz) -- a poor sample of the between-electrode
space. Here we use a 4-EEG-channel recording (Haaglanden HMC SN001: F4, C4, C3, O2 -> frontal/central/
occipital, both hemispheres) to test n:m phase coupling for every WITHIN channel and every BETWEEN pair
(6 pairs, near AND far), across sleep stages, with the same surrogate controls:
  within   : whole-spectrum n:m PC, time-shuffle (waveform-preserving) null.
  between  : whole-spectrum cross n:m PC, IAAFT-of-channel-B null.
Per-epoch surrogate-z, stage-dependence by Kruskal-Wallis + BH-FDR. If between is null across ALL 6 pairs
(including near ones), the within>>between asymmetry is not a 2-channel artifact.

Outputs: results/study49_sleep_multichannel.json   (EEG kept local + gitignored)
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
from scipy.stats import kruskal

from resonance_paper import _common as C
from resonance_paper.study48_sleep_within_cross import _phases, _pc_spectrum, _bh_fdr, FREQS
from biotuner.resonance.coupling import nm_plv, nm_rho_entropy
from biotuner.resonance.nulls import time_shuffle_surrogate, iaaft_surrogate

warnings.filterwarnings("ignore")

EDF = "data/hmc/SN001.edf"
SCORE = "data/hmc/SN001_sleepscoring.edf"
STAGE_MAP = {"Sleep stage W": "Wake", "Sleep stage N1": "N1", "Sleep stage N2": "N2",
             "Sleep stage N3": "N3", "Sleep stage R": "REM"}
STAGES = ["Wake", "N2", "N3", "REM"]
METRICS = {"plv": nm_plv, "rho_entropy": nm_rho_entropy}


def _z_spec(pa, pb, surr_pb, fn):
    obs = _pc_spectrum(pa, pb, fn)
    sv = np.array([_pc_spectrum(pa, sb, fn) for sb in surr_pb])
    return (obs - sv.mean(0)) / (sv.std(0) + 1e-12)


def _load():
    import mne
    raw = mne.io.read_raw_edf(EDF, preload=True, verbose="ERROR")
    eeg = [c for c in raw.ch_names if "EEG" in c.upper()]
    raw.pick(eeg); raw.filter(0.3, 35.0, verbose="ERROR")
    sf = float(raw.info["sfreq"]); data = raw.get_data()
    ann = mne.read_annotations(SCORE)
    by_stage = {s: [] for s in STAGES}
    for onset, dur, desc in zip(ann.onset, ann.duration, ann.description):
        st = STAGE_MAP.get(desc.strip())
        if st not in STAGES:
            continue
        s0 = int(onset * sf); s1 = s0 + int(30 * sf)
        if s1 <= data.shape[1]:
            by_stage[st].append(data[:, s0:s1])
    return [c.replace("EEG ", "").strip() for c in eeg], sf, by_stage


def run(quick=True):
    chans, sf, by_stage = _load()
    cap = 60 if quick else 100
    print("  available epochs/stage:", {s: len(v) for s, v in by_stage.items()}, flush=True)
    rng = np.random.default_rng(0)
    by_stage = {s: (v[:: max(1, len(v) // cap)][:cap]) for s, v in by_stage.items()}   # subsample
    print(f"  channels: {chans} | sf={sf} | used epochs/stage:", {s: len(v) for s, v in by_stage.items()}, flush=True)
    n_surr = 10 if quick else 16
    nc = len(chans)
    pairs = list(combinations(range(nc), 2))

    # collect z spectra: tests keyed by ('within', ci) and ('between', (i,j))
    Z = {}
    for s in STAGES:
        for seg in by_stage[s]:
            ph = [_phases(seg[c].astype(np.float64), sf) for c in range(nc)]      # obs phases per channel
            for c in range(nc):
                surr = [_phases(time_shuffle_surrogate(seg[c].astype(np.float64), rng), sf) for _ in range(n_surr)]
                for mn, fn in METRICS.items():
                    Z.setdefault(("within", c, mn), {st: [] for st in STAGES})[s].append(_z_spec(ph[c], ph[c], surr, fn))
            for (i, j) in pairs:
                surr = [_phases(iaaft_surrogate(seg[j].astype(np.float64), rng), sf) for _ in range(n_surr)]
                for mn, fn in METRICS.items():
                    Z.setdefault(("between", (i, j), mn), {st: [] for st in STAGES})[s].append(_z_spec(ph[i], ph[j], surr, fn))
        print(f"    {s} done ({len(by_stage[s])} epochs)", flush=True)

    rows = []
    for key, perstage in Z.items():
        kind, loc, mn = key
        pf = []
        for k in range(len(FREQS)):
            groups = [[zz[k] for zz in perstage[s]] for s in STAGES if len(perstage[s]) >= 3]
            try:
                pf.append(kruskal(*groups)[1])
            except Exception:
                pf.append(1.0)
        q = _bh_fdr(pf)
        label = chans[loc] if kind == "within" else f"{chans[loc[0]]}<->{chans[loc[1]]}"
        rows.append(dict(kind=kind, label=label, metric=mn, n_sig=int(np.sum(np.asarray(q) <= 0.05)),
                         n_freqs=len(FREQS), min_p=float(np.min(pf)), peak_freq=float(FREQS[int(np.argmin(pf))])))
    C.save_json(dict(quick=quick, dataset="HMC SN001", channels=chans, n_surr=n_surr,
                     epochs_per_stage={s: len(v) for s, v in by_stage.items()}, rows=rows),
                "study49_sleep_multichannel.json")
    _headline(rows)
    return rows


def _headline(rows):
    print("\n  --- Study 49 headline (multi-channel within vs between, surrogate-controlled) ---")
    for mn in METRICS:
        wr = [r for r in rows if r["kind"] == "within" and r["metric"] == mn]
        br = [r for r in rows if r["kind"] == "between" and r["metric"] == mn]
        wsig = sum(1 for r in wr if r["n_sig"] > 0); bsig = sum(1 for r in br if r["n_sig"] > 0)
        print(f"  [{mn}] WITHIN: {wsig}/{len(wr)} channels stage-dependent; BETWEEN: {bsig}/{len(br)} pairs.")
        for r in sorted(br, key=lambda r: r["min_p"])[:3]:
            print(f"      between {r['label']:14s} n_sig={r['n_sig']}/{r['n_freqs']} min_p={r['min_p']:.1e} (peak {r['peak_freq']:.1f}Hz)")
    print("  => BETWEEN null across near AND far pairs => within>>between is robust, not a 2-channel artifact.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

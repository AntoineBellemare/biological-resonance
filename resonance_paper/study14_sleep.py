"""Study 14 — Does harmonicity track criticality across sleep? (Sleep-EDF)

The cleanest natural traversal of criticality: wake (near-critical) -> deep NREM
(N3, moves away from criticality). Tests the same prediction as Study 13/Study 10
— is harmonicity H maximized when the brain is closest to criticality? — on a
large public dataset (PhysioNet Sleep-EDF, auto-fetched by MNE).

Criticality marker here is DFA/LRTC of the alpha envelope (the canonical
human-EEG long-range-temporal-correlation measure, and the classic sleep finding:
LRTC is highest in wake and falls into deep sleep). Sleep-EDF has only 2 EEG
channels, so the avalanche branching ratio m_hat is under-powered and reported
but not relied on; DFA is the primary marker.

Outputs: results/study14_sleep.json, figures/study14_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import crit_resonance as CR

STAGE_EVENT = {"Sleep stage W": 1, "Sleep stage 1": 2, "Sleep stage 2": 3,
               "Sleep stage 3": 4, "Sleep stage 4": 4, "Sleep stage R": 5}
EVENT_STATE = {1: "Wake", 2: "N1", 3: "N2", 4: "N3", 5: "REM"}


def load_sleep(n_subjects=6, channels=("EEG Fpz-Cz", "EEG Pz-Oz"),
               max_epochs_per_stage=12, bandpass=(0.3, 45.0), cohort="age"):
    """Multichannel 30 s sleep epochs labelled by stage (Sleep-EDF, via MNE).

    cohort='age'       -> Sleep Cassette (natural sleep; the primary cohort);
    cohort='temazepam' -> Sleep Telemetry, an INDEPENDENT set of subjects recorded with a
                          different system in a separate PhysioNet study (a second traversing
                          cohort; note these are a temazepam/placebo protocol, a mild benzo)."""
    import re
    import mne
    rng = np.random.default_rng(0)
    items = []
    if cohort == "temazepam":
        from mne.datasets.sleep_physionet.temazepam import fetch_data
        paths = fetch_data(subjects=list(range(n_subjects)))
    else:
        from mne.datasets.sleep_physionet.age import fetch_data
        paths = fetch_data(subjects=list(range(n_subjects)), recording=[1], on_missing="warn")
    for psg, hyp in paths:
        raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
        annot = mne.read_annotations(hyp)
        raw.set_annotations(annot, verbose="ERROR")
        present = [c for c in channels if c in raw.ch_names]
        if not present:
            continue
        raw.pick(present)
        if bandpass is not None:
            raw.filter(bandpass[0], bandpass[1], verbose="ERROR")
        sf = float(raw.info["sfreq"])
        try:
            events, _ = mne.events_from_annotations(
                raw, event_id=STAGE_EVENT, chunk_duration=30.0, verbose="ERROR")
        except ValueError:
            continue
        epo = mne.Epochs(raw, events, event_id={v: k for k, v in EVENT_STATE.items()},
                         tmin=0.0, tmax=30.0 - 1.0 / sf, baseline=None,
                         preload=True, verbose="ERROR", on_missing="ignore")
        _m = re.search(r"S[TC]\d+", str(psg)); subj = _m.group(0) if _m else str(psg)[-8:]
        data = epo.get_data(copy=True)             # (n_epochs, n_ch, n_times)
        codes = epo.events[:, 2]
        h_idx = [0]                                  # Fpz-Cz (frontal) for H/DFA
        for st_code in set(codes):
            idx = np.where(codes == st_code)[0]
            if len(idx) > max_epochs_per_stage:
                idx = rng.choice(idx, max_epochs_per_stage, replace=False)
            for k in idx:
                items.append(dict(X=data[k].astype(np.float64), h_idx=h_idx, sf=sf,
                                  subject=subj, state=EVENT_STATE.get(st_code, "?")))
    return items


def run(quick=True):
    n_sub = 3 if quick else 8
    mxep = 6 if quick else 15
    cfg = CR.state_config()
    print(f"Study 14 — fetching {n_sub} Sleep-EDF nights ...")
    items = load_sleep(n_subjects=n_sub, max_epochs_per_stage=mxep)
    print(f"  {len(items)} epochs; computing features ...")
    rows = []
    for i, it in enumerate(items):
        rows.append(CR.window_features(it["X"], it["sf"], it["subject"], it["state"],
                                       cfg, h_idx=it["h_idx"]))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(items)}", flush=True)
    result = CR.analyze(rows, "sleep (Sleep-EDF)")
    result["quick"] = quick
    C.save_json(result, "study14_sleep.json")
    CR.figure(result, "study14_sleep")
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 14 headline (sleep: H vs criticality) ---")
    pr = result["primary"]
    print(f"  H maximal near criticality? per-subject ρ(H, proximity), n={result['n_subjects']}:")
    for k in ["H_max_vs_prox_dfa", "H_max_vs_prox_m", "R_max_vs_prox_dfa"]:
        v = pr.get(k, {})
        if np.isfinite(v.get("mean_rho", np.nan)):
            print(f"    {k:20s} ρ={v['mean_rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] "
                  f"p={v['wilcoxon_p']:.3g} ({int(v['frac_pos']*100)}% +)")
    print("  marker traversal across sleep stages (mean):")
    for st in ["Wake", "N1", "N2", "N3", "REM"]:
        if st in result["by_state"]:
            b = result["by_state"][st]
            print(f"    {st:5s} dfa={b['dfa']:.3f} m_hat={b['m_hat']:.3f} H_max={b['H_max']:.3f} "
                  f"(n={b['n_windows']})")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

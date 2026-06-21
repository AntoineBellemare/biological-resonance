"""Deep general-anesthesia EEG (OpenNeuro ds004541): loader for Study 15.

Human scalp EEG (58 ch, 1000 Hz) traversing actual loss of consciousness: each
recording runs baseline(awake) -> anesthesia start -> LOC -> ... -> ROC. We slice
each EDF on the events.tsv onsets into clean WAKE ([baseline, start]) and
UNCONSCIOUS ([loc, roc]) segments — a far stronger criticality traversal than
titrated sedation. (Anesthetic agent is not stated in the BIDS metadata; we say
"general-anesthesia LOC, agent unconfirmed".)

58 channels make the avalanche branching ratio m_hat well-powered here (unlike
2-channel Sleep-EDF). Downloaded via direct S3 HTTPS with the resumable
downloader; resampled to 250 Hz to match the other studies.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from resonance_paper.anesthesia import _robust_download

BASE = "https://s3.amazonaws.com/openneuro.org/ds004541"
SUBJECTS = ["sub-02", "sub-03", "sub-04", "sub-07", "sub-08", "sub-09", "sub-10", "sub-11"]
FRONTAL_H = ["Fz", "F3", "F4", "Fp1", "Fp2", "Fpz", "AF3", "AF4"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ds004541"
TARGET_SF = 250.0


def _events(path):
    lines = Path(path).read_text(encoding="utf-8-sig").strip().splitlines()  # strip BOM
    hdr = lines[0].split("\t")
    onset = {}
    for ln in lines[1:]:
        d = dict(zip(hdr, ln.split("\t")))
        tt = d.get("trial_type", "").strip()
        if tt and tt not in onset:                  # first occurrence of each label
            try:
                onset[tt] = float(d["onset"])
            except (KeyError, ValueError):
                pass
    return onset


def download_deepGA(subjects=SUBJECTS, dest=DATA_DIR, verbose=True):
    dest = Path(dest)
    for i, sub in enumerate(subjects):
        stem = f"{sub}/ses-01/eeg/{sub}_ses-01_task-anesthesia"
        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {sub}", flush=True)
        _robust_download(f"{BASE}/{stem}_events.tsv", dest / f"{stem}_events.tsv", verbose=verbose)
        _robust_download(f"{BASE}/{stem}_eeg.edf", dest / f"{stem}_eeg.edf", verbose=verbose)
    return dest


def load_deepGA_multichannel(subjects=SUBJECTS, dest=DATA_DIR, h_channels=FRONTAL_H,
                             win_len=30.0, max_win=8, bandpass=(0.3, 45.0),
                             download=True, verbose=True):
    """WAKE and UNCONSCIOUS multichannel 30 s windows, sliced on LOC/ROC."""
    import mne
    dest = Path(dest)
    if download:
        download_deepGA(subjects, dest, verbose=verbose)
    items = []
    for sub in subjects:
        stem = f"{sub}/ses-01/eeg/{sub}_ses-01_task-anesthesia"
        edf = dest / f"{stem}_eeg.edf"; ev = dest / f"{stem}_events.tsv"
        if not edf.exists() or not ev.exists():
            continue
        onset = _events(ev)
        segs = []
        if "baseline" in onset and "start" in onset and onset["start"] > onset["baseline"]:
            segs.append(("wake", onset["baseline"], onset["start"]))
        if "loc" in onset and "roc" in onset and onset["roc"] > onset["loc"]:
            segs.append(("unconscious", onset["loc"], onset["roc"]))
        if not segs:
            continue
        raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
        # keep EEG-like channels (drop trigger/status)
        drop = [c for c in raw.ch_names if any(t in c.lower() for t in ("trig", "status", "stim", "ecg", "eog"))]
        if drop:
            raw.drop_channels(drop)
        if bandpass is not None:
            raw.filter(bandpass[0], bandpass[1], verbose="ERROR")
        if raw.info["sfreq"] > TARGET_SF:
            raw.resample(TARGET_SF, verbose="ERROR")
        sf = float(raw.info["sfreq"])
        chn = raw.ch_names
        h_idx = [i for i, c in enumerate(chn) if c in h_channels]
        if not h_idx:
            h_idx = list(range(min(5, len(chn))))
        data = raw.get_data()
        seglen = int(win_len * sf)
        for state, t0, t1 in segs:
            i0, i1 = int(t0 * sf), int(t1 * sf)
            seg = data[:, i0:i1]
            n_win = min(max_win, seg.shape[1] // seglen)
            for w in range(n_win):
                items.append(dict(X=seg[:, w * seglen:(w + 1) * seglen].astype(np.float64),
                                  h_idx=h_idx, sf=sf, subject=sub, state=state))
    return items


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(SUBJECTS)
    print(f"Downloading {n} ds004541 subjects to {DATA_DIR} ...")
    download_deepGA(SUBJECTS[:n])
    print("done.")

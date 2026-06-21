"""Chennu graded-propofol EEG: robust downloader + loader (Study 13).

Data: Chennu et al. 2016 "Brain connectivity during propofol sedation", BIDS
mirror on the FieldTrip server (open HTTP, no login). 20 subjects, 91-ch EEG at
250 Hz, four runs per subject = four graded states (baseline / mild / moderate /
recovery), each with the propofol plasma concentration recorded in
``sub-XX_scans.tsv``.

The downloader streams with a socket timeout, retries with backoff, and RESUMES
partial files via HTTP Range (the FieldTrip server supports 206) — so it survives
the stalls that hang a naive ``urlretrieve``.
"""
from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://download.fieldtriptoolbox.org/workshop/madrid2019/extra/complete_resting_data"

# 20 subjects present in the BIDS mirror (verified)
SUBJECTS = ["sub-02", "sub-03", "sub-05", "sub-06", "sub-07", "sub-08", "sub-09",
            "sub-10", "sub-13", "sub-14", "sub-18", "sub-20", "sub-22", "sub-23",
            "sub-24", "sub-25", "sub-26", "sub-27", "sub-28", "sub-29"]
RUNS = [1, 2, 3, 4]
# frontal channels — where propofol's slow + alpha and their structure live
FRONTAL = ["Fz", "F3", "F4", "Fp1", "Fp2", "AFz", "F7", "F8"]

# default local data dir (gitignored)
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "chennu"


def _robust_download(url, dest, timeout=60, retries=8, chunk=1 << 20, verbose=True):
    """Download ``url`` -> ``dest`` with timeout, retry+backoff, and resume.

    Writes to ``dest.part`` and atomically renames on completion, so a present
    ``dest`` is always a complete file (and is skipped on re-run).
    """
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resumed = getattr(resp, "status", resp.getcode()) == 206
                mode = "ab" if (have and resumed) else "wb"
                with open(tmp, mode) as f:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        f.write(block)
            tmp.rename(dest)
            return dest
        except (socket.timeout, urllib.error.URLError, ConnectionError,
                TimeoutError, OSError) as exc:
            wait = min(30, 2 * (attempt + 1))
            if verbose:
                print(f"    [retry {attempt+1}/{retries}] {dest.name}: "
                      f"{type(exc).__name__}; resume from {have/1e6:.0f}MB in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url} after {retries} attempts")


def download_chennu(subjects=SUBJECTS, dest=DATA_DIR, verbose=True):
    """Fetch the BrainVision triplets + scans.tsv for each subject (resumable)."""
    dest = Path(dest)
    for i, sub in enumerate(subjects):
        if verbose:
            print(f"  [{i+1}/{len(subjects)}] {sub}", flush=True)
        _robust_download(f"{BASE}/{sub}/{sub}_scans.tsv", dest / sub / f"{sub}_scans.tsv", verbose=verbose)
        for run in RUNS:
            stem = f"{sub}/eeg/{sub}_task-rest_run-{run}_eeg"
            for ext in ("vhdr", "eeg", "vmrk"):
                _robust_download(f"{BASE}/{stem}.{ext}", dest / f"{stem}.{ext}", verbose=verbose)
    return dest


def _scan_labels(scans_path):
    """run-number (str) -> (sedation_level, propofol_concentration_float)."""
    lines = Path(scans_path).read_text().strip().splitlines()
    hdr = lines[0].split("\t")
    out = {}
    for line in lines[1:]:
        d = dict(zip(hdr, line.split("\t")))
        run = d["filename"].split("run-")[1][0]
        try:
            conc = float(d.get("concentration", "nan"))
        except ValueError:
            conc = float("nan")
        out[run] = (d.get("sedation", "?"), conc)
    return out


def load_chennu_sedation(subjects=SUBJECTS, dest=DATA_DIR, channels=FRONTAL,
                         epoch_len=10.0, max_epochs=12, bandpass=(1.0, 45.0),
                         download=True, verbose=True):
    """Load frontal epochs labelled by sedation level + propofol concentration.

    Returns a list of dicts: signal, sf, subject, run, sedation, concentration,
    channel, epoch_idx — one per (subject, run, channel, epoch).
    """
    import mne
    dest = Path(dest)
    if download:
        download_chennu(subjects, dest, verbose=verbose)
    out = []
    for sub in subjects:
        labels = _scan_labels(dest / sub / f"{sub}_scans.tsv")
        for run in RUNS:
            vhdr = dest / f"{sub}/eeg/{sub}_task-rest_run-{run}_eeg.vhdr"
            if not vhdr.exists():
                if verbose:
                    print(f"  [skip] missing {vhdr.name}")
                continue
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            if bandpass is not None:
                raw.filter(bandpass[0], bandpass[1], verbose="ERROR")
            sf = float(raw.info["sfreq"])
            present = [c for c in channels if c in raw.ch_names]
            if not present:
                continue
            data = raw.get_data(picks=present)
            seg = int(epoch_len * sf)
            n_ep = min(max_epochs, data.shape[1] // seg)
            sed, conc = labels.get(str(run), ("?", float("nan")))
            for ci, ch in enumerate(present):
                for e in range(n_ep):
                    out.append(dict(
                        signal=data[ci, e * seg:(e + 1) * seg].astype(np.float64),
                        sf=sf, subject=sub, run=run, sedation=sed,
                        concentration=conc, channel=ch, epoch_idx=int(e)))
    return out


def load_chennu_multichannel(subjects=SUBJECTS, dest=DATA_DIR, h_channels=FRONTAL,
                             win_len=30.0, max_win=4, bandpass=(0.3, 45.0),
                             download=True, verbose=True):
    """Multichannel windows for the criticality<->resonance analysis.

    Returns list of dicts: X (all EEG channels, n_ch x n_times) for the branching
    ratio; h_idx (indices of frontal channels within X) for H/DFA/band-power;
    sf, subject, state (sedation level), concentration, run.
    """
    import mne
    dest = Path(dest)
    if download:
        download_chennu(subjects, dest, verbose=verbose)
    items = []
    for sub in subjects:
        labels = _scan_labels(dest / sub / f"{sub}_scans.tsv")
        for run in RUNS:
            vhdr = dest / f"{sub}/eeg/{sub}_task-rest_run-{run}_eeg.vhdr"
            if not vhdr.exists():
                continue
            raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
            if bandpass is not None:
                raw.filter(bandpass[0], bandpass[1], verbose="ERROR")
            sf = float(raw.info["sfreq"])
            data = raw.get_data()                       # (n_ch, n_times), all channels
            chn = raw.ch_names
            h_idx = [i for i, c in enumerate(chn) if c in h_channels]
            if not h_idx:
                continue
            seg = int(win_len * sf); n_win = min(max_win, data.shape[1] // seg)
            sed, conc = labels.get(str(run), ("?", float("nan")))
            for w in range(n_win):
                items.append(dict(X=data[:, w * seg:(w + 1) * seg].astype(np.float64),
                                  h_idx=h_idx, sf=sf, subject=sub, state=sed,
                                  concentration=conc, run=run))
    return items


if __name__ == "__main__":
    # CLI: pre-download N subjects, e.g. `python -m resonance_paper.anesthesia 10`
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(SUBJECTS)
    print(f"Downloading {n} Chennu subjects to {DATA_DIR} ...")
    download_chennu(SUBJECTS[:n])
    print("done.")

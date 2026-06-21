"""Real / semi-real biosignal loaders for the validation suite.

  * EEG  — PhysioNet EEG Motor Movement/Imagery (``mne.datasets.eegbci``).
           Run 1 = eyes-open baseline, Run 2 = eyes-closed baseline. The
           eyes-closed condition is the textbook alpha-enhancement manipulation,
           giving a clean, well-understood state contrast.
  * ECG  — synthesized with neurokit2 at a KNOWN heart rate, so the harmonic
           series (HR and its integer multiples from the QRS complex) is ground
           truth. Real-ECG-like morphology, controllable, reproducible.

All loaders return plain float64 numpy arrays + metadata, ready for
``compute_resonance``.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# EEG — eyes-open / eyes-closed
# --------------------------------------------------------------------------
# eegbci baseline runs: 1 = eyes open, 2 = eyes closed.
EO_RUN = 1
EC_RUN = 2
# occipital channels carry the strongest alpha
OCCIPITAL = ["O1", "O2", "Oz"]


def _load_eegbci_raw(subject, run):
    import mne
    from mne.io import read_raw_edf
    from mne.datasets import eegbci

    fname = eegbci.load_data(subject, [run], update_path=True)[0]
    raw = read_raw_edf(fname, preload=True, verbose=False)
    eegbci.standardize(raw)  # canonical 10-05 channel names
    raw.rename_channels(lambda s: s.strip("."))
    return raw


def load_eeg_states(
    subjects=(1, 2, 3, 4, 5),
    channels=OCCIPITAL,
    epoch_len=4.0,
    max_epochs_per_run=12,
    target_sf=160.0,
    bandpass=(1.0, 45.0),
):
    """Load eyes-open / eyes-closed occipital EEG epochs.

    Returns
    -------
    epochs : list of dict
        one entry per (subject, condition, channel, epoch) with keys
        ``signal, sf, subject, condition (EO|EC), channel, epoch_idx``.
    """
    out = []
    for subj in subjects:
        for cond, run in (("EO", EO_RUN), ("EC", EC_RUN)):
            try:
                raw = _load_eegbci_raw(subj, run)
            except Exception as exc:  # pragma: no cover - network/IO
                print(f"  [skip] subject {subj} run {run}: {type(exc).__name__}: {exc}")
                continue
            if bandpass is not None:
                raw.filter(bandpass[0], bandpass[1], verbose=False)
            sf = float(raw.info["sfreq"])
            present = [ch for ch in channels if ch in raw.ch_names]
            if not present:
                continue
            data = raw.get_data(picks=present)  # (n_ch, n_times)
            seg = int(epoch_len * sf)
            n_ep = min(max_epochs_per_run, data.shape[1] // seg)
            for ci, ch in enumerate(present):
                for e in range(n_ep):
                    s = data[ci, e * seg:(e + 1) * seg]
                    out.append(dict(
                        signal=s.astype(np.float64), sf=sf,
                        subject=int(subj), condition=cond, channel=ch,
                        epoch_idx=int(e),
                    ))
    return out


# --------------------------------------------------------------------------
# ECG — neurokit2, known heart rate
# --------------------------------------------------------------------------
def make_ecg(heart_rate=60.0, sf=500.0, duration=30.0, noise=0.05, seed=0):
    """Synthesize an ECG with a KNOWN heart rate.

    The QRS complex is a sharp, near-impulse-train feature, so its spectrum is a
    rich integer-harmonic series at the heart-rate fundamental f0 = HR/60 Hz and
    its multiples. Ground-truth resonance should be high and harmonically dense.
    """
    import neurokit2 as nk

    # neurokit2's 'ecgsyn' method requires an INTEGER duration (it slices with
    # it internally); passing a float raises "slice indices must be integers".
    dur_int = int(round(duration))
    try:
        ecg = nk.ecg_simulate(
            duration=dur_int, sampling_rate=int(sf), heart_rate=float(heart_rate),
            noise=noise, random_state=int(seed), method="ecgsyn",
        )
    except Exception:
        # robust fallback: 'simple'/'daubechies' template synthesis
        ecg = nk.ecg_simulate(
            duration=dur_int, sampling_rate=int(sf), heart_rate=float(heart_rate),
            noise=noise, random_state=int(seed), method="daubechies",
        )
    ecg = np.asarray(ecg, dtype=np.float64)
    f0 = heart_rate / 60.0
    meta = dict(kind="ecg", heart_rate=float(heart_rate), f0_hz=float(f0),
                harmonics=[float(f0 * k) for k in range(1, 8)], sf=float(sf))
    return np.asarray(ecg, dtype=np.float64), meta


# --------------------------------------------------------------------------
# Real ECG + extra physiological modalities
# --------------------------------------------------------------------------
def load_real_ecg(target_sf=500.0, segment_s=30.0, n_segments=6):
    """Real recorded ECG (neurokit2 bundled 'ecg_3000hz'), resampled + segmented.

    Returns list of (signal, sf). Falls back to [] if unavailable.
    """
    import neurokit2 as nk
    from scipy.signal import resample
    try:
        raw = np.asarray(nk.data("ecg_3000hz"), dtype=np.float64)
    except Exception as exc:  # pragma: no cover
        print(f"  [real ecg skip] {exc}")
        return []
    src_sf = 3000.0
    n_tgt = int(len(raw) * target_sf / src_sf)
    sig = resample(raw, n_tgt)
    seg = int(segment_s * target_sf)
    needed = seg * n_segments
    # The bundled recording is short (~17 s); tile it to provide n_segments
    # matched-duration segments rather than dropping the modality entirely.
    if len(sig) < needed:
        reps = int(np.ceil(needed / len(sig)))
        sig = np.tile(sig, reps)
    out = []
    for i in range(n_segments):
        out.append((sig[i * seg:(i + 1) * seg].astype(np.float64), target_sf))
    return out


def make_ppg(heart_rate=70.0, sf=500.0, duration=30.0, seed=0):
    """Photoplethysmogram (neurokit2) — cardiovascular, harmonic at the HR."""
    import neurokit2 as nk
    x = nk.ppg_simulate(duration=int(round(duration)), sampling_rate=int(sf),
                        heart_rate=float(heart_rate), random_state=int(seed))
    return np.asarray(x, dtype=np.float64), dict(kind="ppg", heart_rate=float(heart_rate))


def make_rsp(rate=15.0, sf=500.0, duration=30.0, seed=0):
    """Respiration (neurokit2) — slow, near-sinusoidal oscillator."""
    import neurokit2 as nk
    x = nk.rsp_simulate(duration=int(round(duration)), sampling_rate=int(sf),
                        respiratory_rate=float(rate), random_state=int(seed))
    return np.asarray(x, dtype=np.float64), dict(kind="rsp", rate=float(rate))


# --------------------------------------------------------------------------
# EEG motor-task contrast (second state contrast)
# --------------------------------------------------------------------------
# eegbci run 1 = eyes-open baseline (rest); runs 3/7/11 = motor execution
# (open/close fists). Motor cortex channels show task-related mu/beta change.
REST_RUN = 1
MOTOR_RUN = 3
MOTOR_CHANNELS = ["C3", "Cz", "C4"]


def load_eeg_motor_contrast(subjects=(1, 2, 3, 4, 5), channels=MOTOR_CHANNELS,
                            epoch_len=8.0, max_epochs_per_run=7, bandpass=(1.0, 45.0)):
    """Rest (run 1) vs motor-execution (run 3) epochs over motor channels.

    Same return shape as ``load_eeg_states`` with condition in {REST, MOTOR}.
    """
    out = []
    for subj in subjects:
        for cond, run in (("REST", REST_RUN), ("MOTOR", MOTOR_RUN)):
            try:
                raw = _load_eegbci_raw(subj, run)
            except Exception as exc:  # pragma: no cover
                print(f"  [skip] subj {subj} run {run}: {exc}")
                continue
            if bandpass is not None:
                raw.filter(bandpass[0], bandpass[1], verbose=False)
            sf = float(raw.info["sfreq"])
            present = [ch for ch in channels if ch in raw.ch_names]
            if not present:
                continue
            data = raw.get_data(picks=present)
            seg = int(epoch_len * sf)
            n_ep = min(max_epochs_per_run, data.shape[1] // seg)
            for ci, ch in enumerate(present):
                for e in range(n_ep):
                    out.append(dict(
                        signal=data[ci, e * seg:(e + 1) * seg].astype(np.float64),
                        sf=sf, subject=int(subj), condition=cond, channel=ch,
                        epoch_idx=int(e)))
    return out


# --------------------------------------------------------------------------
# Cross-modality bundle
# --------------------------------------------------------------------------
def modality_bundle(sf=500.0, duration=30.0, n_per_class=6, seed=0):
    """A labeled set of signals spanning seven modality classes for the cross-modality study.

    Returns list of dict(signal, sf, modality, label, meta).

      * ECG_synth     — neurokit2, HR jittered around 60 bpm (highly harmonic)
      * ECG_real      — real recorded ECG segments (tiled to the analysis duration)
      * PPG           — neurokit2 photoplethysmogram, harmonic at the heart rate
      * RSP           — neurokit2 respiration, slow oscillator
      * EEG_alpha     — eyes-closed-like: 1/f + bursty 10 Hz alpha + 20 Hz beta
      * harmonic_stack— clean phase-locked 6/12/18/24 stack (max resonance)
      * pink_noise    — pink noise (min resonance, negative control)

    All classes share the same sf and duration so resonance configs are directly
    comparable.
    """
    from resonance_paper.signals import pink_noise, harmonic_stack, _norm
    out = []
    N = int(sf * duration)
    t = np.arange(N) / sf

    # Real recorded ECG segments (shared across the i loop)
    real_ecg_segs = load_real_ecg(target_sf=sf, segment_s=duration, n_segments=n_per_class)

    for i in range(n_per_class):
        rng = np.random.default_rng(seed + i)
        # Synthetic ECG with slight HR variation
        hr = 60.0 + rng.uniform(-6, 6)
        try:
            ecg, meta = make_ecg(heart_rate=hr, sf=sf, duration=duration,
                                 noise=0.05, seed=seed + i)
            out.append(dict(signal=ecg, sf=sf, modality="ECG", label="ECG_synth", meta=meta))
        except Exception as exc:  # pragma: no cover
            print(f"  [ecg skip] {exc}")

        # Real recorded ECG
        if i < len(real_ecg_segs):
            rs, rsf = real_ecg_segs[i]
            out.append(dict(signal=_norm(rs).astype(np.float64), sf=rsf,
                            modality="ECG", label="ECG_real", meta=dict(kind="ecg_real")))

        # PPG (cardiovascular, harmonic at HR)
        try:
            ppg, pmeta = make_ppg(heart_rate=70.0 + rng.uniform(-8, 8), sf=sf,
                                  duration=duration, seed=seed + 400 + i)
            out.append(dict(signal=_norm(ppg).astype(np.float64), sf=sf,
                            modality="PPG", label="PPG", meta=pmeta))
        except Exception as exc:  # pragma: no cover
            print(f"  [ppg skip] {exc}")

        # Respiration (slow oscillator)
        try:
            rsp, rmeta = make_rsp(rate=15.0 + rng.uniform(-3, 3), sf=sf,
                                  duration=duration, seed=seed + 500 + i)
            out.append(dict(signal=_norm(rsp).astype(np.float64), sf=sf,
                            modality="RSP", label="RSP", meta=rmeta))
        except Exception as exc:  # pragma: no cover
            print(f"  [rsp skip] {exc}")

        # EEG-alpha-like: 1/f + intermittent alpha bursts + weak beta harmonic
        bg = 1.2 * pink_noise(N, sf, seed=seed + 100 + i)
        sig = bg.copy()
        n_bursts = 10
        for bt in np.linspace(1.0, duration - 2.0, n_bursts) + rng.uniform(-0.2, 0.2, n_bursts):
            idx = (t >= bt) & (t < bt + 0.8)
            loc = t[idx] - bt
            win = np.sin(np.pi * loc / 0.8) ** 2
            ph = rng.uniform(0, 2 * np.pi)
            sig[idx] += 2.2 * win * np.sin(2 * np.pi * 10 * loc + ph)
            sig[idx] += 0.8 * win * np.sin(2 * np.pi * 20 * loc + 2 * ph)  # beta harmonic
        out.append(dict(signal=_norm(sig).astype(np.float64), sf=sf,
                        modality="EEG", label="EEG_alpha",
                        meta=dict(kind="eeg_alpha", alpha=10.0, beta=20.0)))

        # Clean harmonic stack
        hs, hmeta = harmonic_stack(base_freq=6.0, n_harmonics=4, sf=sf,
                                   duration=duration, snr_db=10.0,
                                   phase_lock=True, seed=seed + 200 + i)
        out.append(dict(signal=hs, sf=sf, modality="SYNTH",
                        label="harmonic_stack", meta=hmeta))

        # Pink noise control
        out.append(dict(signal=_norm(pink_noise(N, sf, seed=seed + 300 + i)).astype(np.float64),
                        sf=sf, modality="SYNTH", label="pink_noise",
                        meta=dict(kind="pink")))
    return out

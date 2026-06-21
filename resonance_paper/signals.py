"""Synthetic signal generators with KNOWN ground-truth harmonic structure.

These drive the ground-truth-recovery study: each generator either embeds a
genuine n:m phase lock at a known frequency pair, or produces a PSD-matched
control whose phase structure has been destroyed. A correct resonance method
must light up on the former and stay quiet on the latter.

All generators return a float64 1-D array sampled at ``sf`` for ``duration`` s.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Background noise
# --------------------------------------------------------------------------
def pink_noise(n, sf, seed=0):
    """1/f (pink) noise via spectral shaping of white noise."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1.0 / sf)
    f[0] = f[1]
    spec = np.fft.rfft(w) / np.sqrt(f)
    return np.fft.irfft(spec, n=n)


def _norm(x):
    x = x - np.mean(x)
    s = np.std(x)
    return x / s if s > 0 else x


# --------------------------------------------------------------------------
# Ground-truth coupling
# --------------------------------------------------------------------------
def nm_phase_locked(
    n_ratio=1, m_ratio=2, base_freq=6.0, sf=500.0, duration=30.0,
    snr_db=6.0, seed=0,
):
    """A genuine n:m phase-locked pair on a pink-noise background.

    Two oscillators at ``f1 = base_freq`` and ``f2 = base_freq * m_ratio / n_ratio``
    are driven from a COMMON phase, so ``n_ratio * phi(f2) - m_ratio * phi(f1)``
    is constant in time — a true n:m mode lock. Ground truth: a resonance peak
    is expected at f1 and f2.

    Parameters
    ----------
    n_ratio, m_ratio : int
        The coupling ratio. f2/f1 = m_ratio/n_ratio.
    snr_db : float
        Oscillation-to-noise power ratio in dB. Lower = harder detection.

    Returns
    -------
    signal : ndarray
    meta : dict with f1, f2, n_ratio, m_ratio, coupled_freqs
    """
    rng = np.random.default_rng(seed)
    N = int(sf * duration)
    t = np.arange(N) / sf
    f1 = base_freq
    f2 = base_freq * m_ratio / n_ratio
    # common base phase at the fundamental of the ratio
    base = base_freq / m_ratio  # so that f1 = base*m_ratio, f2 = base*?  -- keep simple:
    # Lock directly: phi1 = 2*pi*f1*t + c1 ; phi2 = (m/n)*... share a clock
    clock = 2 * np.pi * (base_freq / 1.0) * t + rng.uniform(0, 2 * np.pi)
    osc1 = np.sin(clock)                                  # at f1
    osc2 = np.sin((m_ratio / n_ratio) * clock + 0.3)      # at f2, phase-locked to osc1
    osc = _norm(osc1 + 0.8 * osc2)

    noise = _norm(pink_noise(N, sf, seed=seed + 1000))
    # scale noise to hit target SNR
    snr_lin = 10 ** (snr_db / 10.0)
    sig = np.sqrt(snr_lin) * osc + noise
    meta = dict(f1=float(f1), f2=float(f2), n_ratio=n_ratio, m_ratio=m_ratio,
                coupled_freqs=[float(f1), float(f2)], snr_db=float(snr_db),
                kind="nm_phase_locked")
    return sig.astype(np.float64), meta


def psd_matched_control(signal, sf, seed=0):
    """PSD-matched phase-randomized control of ``signal`` (AAFT-style).

    Same power spectrum and amplitude distribution, phase relations destroyed.
    Ground truth: NO genuine resonance should survive surrogate normalization.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    # amplitude-adjusted Fourier transform surrogate
    # 1) rank-map gaussian, 2) phase randomize, 3) rank-map back
    gauss = np.sort(rng.standard_normal(n))
    ranks = x.argsort().argsort()
    y = gauss[ranks]
    Y = np.fft.rfft(y)
    phases = rng.uniform(0, 2 * np.pi, size=Y.shape)
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    Y2 = np.abs(Y) * np.exp(1j * phases)
    y2 = np.fft.irfft(Y2, n=n)
    # rank-map back to original amplitude distribution
    out = np.empty(n)
    out[y2.argsort()] = np.sort(x)
    return out.astype(np.float64)


def harmonic_stack(base_freq=8.0, n_harmonics=4, sf=500.0, duration=30.0,
                   amp_decay=0.7, snr_db=6.0, phase_lock=True, seed=0):
    """A phase-locked (or scrambled) harmonic stack: base_freq, 2x, 3x, ...

    With ``phase_lock=True`` all partials share a common clock (true harmonic
    resonance). With ``phase_lock=False`` each partial gets an independent random
    phase (same PSD, no cross-partial coupling) — a built-in negative control.
    """
    rng = np.random.default_rng(seed)
    N = int(sf * duration)
    t = np.arange(N) / sf
    clock = 2 * np.pi * base_freq * t + rng.uniform(0, 2 * np.pi)
    osc = np.zeros(N)
    freqs = []
    for k in range(1, n_harmonics + 1):
        amp = amp_decay ** (k - 1)
        if phase_lock:
            osc += amp * np.sin(k * clock)
        else:
            osc += amp * np.sin(2 * np.pi * base_freq * k * t + rng.uniform(0, 2 * np.pi))
        freqs.append(base_freq * k)
    osc = _norm(osc)
    noise = _norm(pink_noise(N, sf, seed=seed + 2000))
    snr_lin = 10 ** (snr_db / 10.0)
    sig = np.sqrt(snr_lin) * osc + noise
    meta = dict(base_freq=float(base_freq), harmonics=freqs, phase_lock=phase_lock,
                snr_db=float(snr_db), coupled_freqs=[float(f) for f in freqs],
                kind="harmonic_stack")
    return sig.astype(np.float64), meta


def nonstationary_coupling(
    locked=True, base_freq=6.0, m_ratio=2, n_ratio=1, sf=500.0, duration=40.0,
    snr_db=6.0, diffusion=0.8, seed=0,
):
    """Non-stationary n:m coupling — the design where phase coupling is DETECTABLE.

    The fundamental's phase performs a random walk (diffusion). A harmonic
    partner either FOLLOWS that wandering phase (``locked=True`` — a genuine n:m
    mode lock that persists despite the drifting marginal phase) or oscillates at
    a fixed frequency with its OWN independent phase walk (``locked=False`` —
    same power spectrum, no coupling).

    This is the regime a PSD-preserving surrogate (AAFT) can test: AAFT destroys
    the cross-frequency *following* relationship, so a phase-coupling z-score is
    high for ``locked`` and ~0 for ``unlocked``. (For a *stationary* generator the
    across-window phase relationship is trivially constant in both cases, so no
    PSD-preserving null can distinguish them — see the paper's operating-regime
    analysis.)

    Returns (signal, meta).
    """
    rng = np.random.default_rng(seed)
    N = int(sf * duration)
    t = np.arange(N) / sf
    f1 = base_freq
    f2 = base_freq * m_ratio / n_ratio
    step = diffusion * np.sqrt(1.0 / sf)
    phi = 2 * np.pi * f1 * t + np.cumsum(step * rng.standard_normal(N))
    o1 = np.sin(phi)
    if locked:
        o2 = np.sin((m_ratio / n_ratio) * phi + 0.3)
    else:
        phi2 = 2 * np.pi * f2 * t + np.cumsum(step * rng.standard_normal(N))
        o2 = np.sin(phi2)
    osc = _norm(o1 + 0.8 * o2)
    noise = _norm(pink_noise(N, sf, seed=seed + 7000))
    snr_lin = 10 ** (snr_db / 10.0)
    sig = np.sqrt(snr_lin) * osc + noise
    meta = dict(f1=float(f1), f2=float(f2), n_ratio=n_ratio, m_ratio=m_ratio,
                coupled_freqs=[float(f1), float(f2)], snr_db=float(snr_db),
                locked=bool(locked), diffusion=float(diffusion),
                kind="nonstationary_coupling")
    return sig.astype(np.float64), meta


def pink_only(sf=500.0, duration=30.0, seed=0):
    """Pure pink noise — the absolute negative control (no oscillation at all)."""
    N = int(sf * duration)
    sig = _norm(pink_noise(N, sf, seed=seed))
    return sig.astype(np.float64), dict(kind="pink_only", coupled_freqs=[])


# Convenience registry for sweeps
RATIO_BANK = {
    "octave_1:2": (1, 2),
    "fifth_2:3": (2, 3),
    "fourth_3:4": (3, 4),
    "twelfth_1:3": (1, 3),
    "major_third_4:5": (4, 5),
}

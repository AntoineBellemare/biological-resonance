"""Study 18c — Stochastic-resonance RESCUE: the classic sub-threshold regime.

The Study-18/18b framing (noise helps COMPLEX ratios mode-lock) failed its
go/no-go. This is the textbook SR setting instead: a bistable double-well driven
by a WEAK, SUB-THRESHOLD periodic signal that on its own cannot push the system
over the barrier. Noise lets the switching synchronize to the weak drive, and the
output SNR is NON-MONOTONIC in noise — the canonical inverted-U.

    dx/dt = x - x^3 + A*s(t) + sqrt(2D) xi(t)      (barrier at x=0, wells at +-1)

Two tests:
  (1) CLASSIC SR: s = sin(2pi f0 t), A sub-threshold. Spectral SNR at f0 vs noise D
      -> expect a clear interior peak (the SR hallmark). Also input->output PLV.
  (2) HARMONIC SR (the framework angle): s = sin(f0)+1/2 sin(2f0)+1/3 sin(3f0),
      all sub-threshold. Does the OUTPUT harmonicity H peak at intermediate noise
      (noise helps the system TRANSMIT/RECONSTRUCT the harmonic structure)?

Decision: SR is rescuable as a framework story if (1) shows a clear inverted-U
(SNR peak at D>0) AND (2) H is non-monotonic with an interior optimum.

Outputs: results/study18c_sr_rescue.json, figures/study18c_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from resonance_paper import _common as C
from resonance_paper.signals import _norm
from biotuner.resonance import compute_resonance

DT = 0.02
F0 = 0.05               # cycles per model time-unit (slow vs intra-well relaxation ~1)
A_SUB = 0.12            # sub-threshold drive amplitude (static barrier removal needs ~0.385)
DUR = 2000.0            # model time units
DECIM = 50              # decimate the raw integration before analysis (f0 -> 0.05 cyc/sample)
F0_HZ = 5.0             # f0 maps to 5 Hz after decimation; harmonics 10/15 Hz
SF_DECL = F0_HZ / (F0 * DT * DECIM)   # = 100 Hz (well-conditioned for filtering/PSD)


def bistable(drive, D, seed=0):
    rng = np.random.default_rng(seed)
    n = len(drive); x = -1.0; out = np.empty(n); s = np.sqrt(2.0 * D * DT)
    for t in range(n):
        x = x + DT * (x - x ** 3 + drive[t]) + s * rng.standard_normal()
        out[t] = x
    return out


def _prep(y):
    """Decimate the raw model output to the declared analysis rate."""
    from scipy.signal import resample_poly
    return resample_poly(np.asarray(y, float), 1, DECIM)


def _drive(harmonic=False):
    n = int(DUR / DT); t = np.arange(n) * DT
    if harmonic:
        s = (np.sin(2 * np.pi * F0 * t) + 0.5 * np.sin(2 * np.pi * 2 * F0 * t)
             + (1 / 3) * np.sin(2 * np.pi * 3 * F0 * t))
        s = s / np.max(np.abs(s))           # keep peak sub-threshold
    else:
        s = np.sin(2 * np.pi * F0 * t)
    return A_SUB * s, t


def _resp_amp(x, f_hz, sf=SF_DECL):
    """Coherent response AMPLITUDE at the drive frequency (classic SR 'spectral
    amplification' numerator): tiny for the weak sub-threshold drive, large when
    noise-aided well-switching synchronizes to it, falling again at high noise."""
    tt = np.arange(len(x)) / sf
    return float(np.abs(np.mean(x * np.exp(-1j * 2 * np.pi * f_hz * tt))) * 2.0)


def _plv(inp, out, f_hz, sf=SF_DECL, bw=1.0):
    from scipy.signal import butter, filtfilt, hilbert
    b, a = butter(4, [(f_hz - bw) / (sf / 2), (f_hz + bw) / (sf / 2)], btype="band")
    pi = np.angle(hilbert(filtfilt(b, a, inp))); po = np.angle(hilbert(filtfilt(b, a, out)))
    return float(np.abs(np.mean(np.exp(1j * (po - pi)))))


def _interior(noises, vals):
    v = np.asarray(vals, float)
    if not np.all(np.isfinite(v)) or v.max() <= v[0]:
        i = int(np.nanargmax(v)) if np.any(np.isfinite(v)) else 0
        return bool(0 < i < len(v) - 1), float(noises[i])
    i = int(np.argmax(v))
    return bool(0 < i < len(v) - 1 and v[i] > v[0] and v[i] >= v[-1]), float(noises[i])


def run(quick=True):
    noises = [0.0, 0.02, 0.04, 0.07, 0.11, 0.17, 0.25, 0.35, 0.5] if quick else \
             [0.0, 0.01, 0.02, 0.035, 0.055, 0.08, 0.12, 0.17, 0.24, 0.33, 0.45, 0.6, 0.8]
    seeds = range(3) if quick else range(6)
    cfg = C.default_config(fmin=2, fmax=30, precision_hz=0.5)

    s1, _ = _drive(harmonic=False)
    sh, _ = _drive(harmonic=True)
    s1_d = _prep(s1)                                   # decimated input for input->output PLV

    classic, harmonic = [], []
    for D in noises:
        pwrs, plvs = [], []
        Hs = []
        for sd in seeds:
            y1 = _prep(bistable(s1, D, seed=sd))
            pwrs.append(_resp_amp(y1, F0_HZ)); plvs.append(_plv(s1_d, y1, F0_HZ))
            yh = _prep(bistable(sh, D, seed=sd + 100))
            Hs.append(float(compute_resonance(_norm(yh).astype(np.float64), sf=SF_DECL,
                                              config=cfg).summaries["H"]["max"]))
        classic.append(dict(D=float(D), power=float(np.mean(pwrs)), plv=float(np.mean(plvs))))
        harmonic.append(dict(D=float(D), H=float(np.mean(Hs))))
        print(f"  D={D:.3f}  amp(f0)={classic[-1]['power']:.3f} PLV={classic[-1]['plv']:.2f} "
              f"H={harmonic[-1]['H']:.3f}", flush=True)

    snr_int, snr_D = _interior(noises, [r["power"] for r in classic])
    plv_int, plv_D = _interior(noises, [r["plv"] for r in classic])
    H_int, H_D = _interior(noises, [r["H"] for r in harmonic])
    verdict = dict(classic_SR_inverted_U=bool(snr_int), classic_SR_optimal_D=snr_D,
                   plv_inverted_U=bool(plv_int), plv_optimal_D=plv_D,
                   harmonic_SR_inverted_U=bool(H_int), harmonic_SR_optimal_D=H_D,
                   rescued=bool(snr_int and H_int))
    result = dict(quick=quick, noises=noises, classic=classic, harmonic=harmonic, verdict=verdict)
    C.save_json(result, "study18c_sr_rescue.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    v = result["verdict"]
    print("\n  --- Study 18c headline (classic SR rescue) ---")
    print(f"  (1) classic SR: SNR inverted-U = {v['classic_SR_inverted_U']} "
          f"(optimal D={v['classic_SR_optimal_D']:.3f}); input->output PLV inverted-U = {v['plv_inverted_U']}")
    print(f"  (2) harmonic SR: output-H inverted-U = {v['harmonic_SR_inverted_U']} "
          f"(optimal D={v['harmonic_SR_optimal_D']:.3f})")
    print(f"  => SR RESCUED as a framework story: {v['rescued']}")


def _figures(result):
    plt = C.setup_mpl()
    noises = result["noises"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    axes[0].plot(noises, [r["power"] for r in result["classic"]], "o-", color="#1565c0")
    axes[0].set_xlabel("noise D"); axes[0].set_ylabel("response amplitude at f0")
    axes[0].set_title("A. Classic SR (sub-threshold)\nresponse at f0 peaks at optimal noise", fontsize=10)
    axes[1].plot(noises, [r["plv"] for r in result["classic"]], "o-", color="#6a1b9a")
    axes[1].set_xlabel("noise D"); axes[1].set_ylabel("input→output PLV at f0")
    axes[1].set_title("B. Input→output phase locking", fontsize=10)
    axes[2].plot(noises, [r["H"] for r in result["harmonic"]], "o-", color="#b71c1c")
    axes[2].set_xlabel("noise D"); axes[2].set_ylabel("output harmonicity H")
    axes[2].set_title("C. Harmonic SR\n(noise transmits harmonic structure?)", fontsize=10)
    v = result["verdict"]
    fig.suptitle(f"Study 18c — Classic stochastic resonance (rescued={v['rescued']})",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study18c_sr_rescue")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

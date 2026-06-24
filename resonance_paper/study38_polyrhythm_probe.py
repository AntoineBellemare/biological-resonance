"""Study 38 — Polyrhythm probe: the default pipeline anti-detects a clean 2:3 lock; the oracle nails it.

The seed result that motivates the whole soundness study. On a genuine 2:3 phase lock (f_a=10, f_b=15),
we compare the framework's targeted cross-resonance PC matrix entry under two configs against an oracle
(direct n:m PLV on bandpass+Hilbert phase at the known ratio):

  default config  : coupling_metric='nm_plv', ratio_kernel='binary', phase_estimator='stft'
  "sound" config  : coupling_metric='nm_plv_canonical', ratio_kernel='fraction', phase_estimator='hilbert'
  ORACLE          : |<exp(i(3*phi_a - 2*phi_b))>| on bandpass(3 Hz)+Hilbert phase

Result: the default config ANTI-detects (coupled < uncoupled); even the sound config recovers only a
sliver vs the oracle (the cross-resonance phase path + the Tenney weight on the matrix entry attenuate
it). This is why the polyrhythm soundness study exists, and motivates the cross-resonance Hilbert-path
fix (R4) + the targeted detector API (R1). See PLAN_polyrhythm_soundness.md.

Outputs: results/study38_polyrhythm_probe.json
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from resonance_paper import _common as C
from biotuner.resonance import ResonanceConfig
from biotuner.harmonic_connectivity import compute_cross_resonance

warnings.filterwarnings("ignore")

SF = 500.0
DUR = 24.0
N = int(SF * DUR)
T = np.arange(N) / SF


def _bp(x, f, bw=3.0):
    sos = butter(4, [(f - bw / 2) / (SF / 2), (f + bw / 2) / (SF / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def make(coupled, seed):
    """2:3 lock: f_a=10, f_b=15 (3*f_a = 2*f_b). coupled -> phi_b = 1.5*phi_a + pi/4."""
    r = np.random.default_rng(seed)
    drift = np.cumsum(0.6 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    phi_a = 2 * np.pi * 10 * T + drift
    A = np.sin(phi_a)
    if coupled:
        B = np.sin(1.5 * phi_a + np.pi / 4)
    else:
        B = np.sin(2 * np.pi * 15 * T + np.cumsum(0.6 * np.sqrt(1.0 / SF) * r.standard_normal(N)))
    return _norm(A + 0.3 * r.standard_normal(N)), _norm(B + 0.3 * r.standard_normal(N))


def _cfg(metric, kernel, est):
    kp = {"max_denom": 16, "beta": 1.0} if kernel == "fraction" else {"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True}
    return ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, coupling_metric=metric, ratio_kernel=kernel,
                           ratio_kernel_params=kp, phase_estimator=est,
                           phase_estimator_params=({"bandwidth": 3.0} if est == "hilbert" else {}),
                           return_intermediates=True)


def _pc_entry(A, B, cfg):
    r = compute_cross_resonance(A, B, sf=SF, config=cfg)
    fr = r.freqs
    i = int(np.argmin(np.abs(fr - 10))); j = int(np.argmin(np.abs(fr - 15)))
    return float(r.phase_coupling_matrix[i, j])


def _oracle(A, B):
    pa = np.angle(hilbert(_bp(A, 10))); pb = np.angle(hilbert(_bp(B, 15)))
    return float(np.abs(np.mean(np.exp(1j * (3 * pa - 2 * pb)))))


def run(quick=True):
    seeds = list(range(5 if quick else 12))
    configs = {"default (nm_plv+binary+stft)": _cfg("nm_plv", "binary", "stft"),
               "sound (canonical+fraction+hilbert)": _cfg("nm_plv_canonical", "fraction", "hilbert")}
    rows = []
    for name, cfg in configs.items():
        cp = float(np.mean([_pc_entry(*make(True, s), cfg) for s in seeds]))
        un = float(np.mean([_pc_entry(*make(False, s), cfg) for s in seeds]))
        rows.append(dict(method=name, coupled=cp, uncoupled=un, delta=cp - un))
    ocp = float(np.mean([_oracle(*make(True, s)) for s in seeds]))
    oun = float(np.mean([_oracle(*make(False, s)) for s in seeds]))
    rows.append(dict(method="ORACLE (handed 2:3, hilbert PLV)", coupled=ocp, uncoupled=oun, delta=ocp - oun))
    C.save_json(dict(quick=quick, ratio="2:3", rows=rows), "study38_polyrhythm_probe.json")
    print("\n  --- Study 38 headline (2:3 polyrhythm: pipeline vs oracle) ---")
    print(f"  {'method':40s} {'coupled':>9s} {'uncoupled':>10s} {'delta':>7s}")
    for r in rows:
        print(f"  {r['method']:40s} {r['coupled']:9.3f} {r['uncoupled']:10.3f} {r['delta']:7.3f}")
    print("  => default ANTI-detects (delta<0); even 'sound' << oracle -> pipeline under-detects (motivates R4).")
    return rows


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

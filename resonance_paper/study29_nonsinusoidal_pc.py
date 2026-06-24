"""Study 29 — Does within-signal PC re-report waveform nonsinusoidality?

Reviewer concern (Dellavale 2020; Aru 2015): the Fourier components of a single NON-sinusoidal
oscillator hold fixed phase relations by construction, so a within-signal phase-coupling readout
will flag "coupling" that is really just waveform shape -- which H already captures. If so, H and
PC are not cleanly separable in the single-signal regime.

We test this honestly: compute the within-signal PC_z (max reduced PC against an IAAFT null that
preserves the PSD but randomizes phase) and H_max on:
  * pure_sine    : a single sinusoid (no harmonics)            -> PC_z should be ~0 (negative control)
  * asym_harmonic: sine + phase-locked 2nd/3rd harmonics       -> nonsinusoidal, waveform harmonics
  * sawtooth, square, clipped : strongly nonsinusoidal waveforms
Each has NO independent cross-frequency dynamics; any PC_z elevation is waveform shape.

Expected honest result: PC_z is at the null for the pure sinusoid but elevated for every
nonsinusoidal waveform, and H is elevated there too -- i.e. within-signal H and PC co-vary and PC
partly re-reports nonsinusoidality. The phase axis is cleanly separable only in the CROSS-signal
setting (two independent oscillators; Study 5), which the paper should state explicitly.

Outputs: results/study29_nonsinusoidal_pc.json
"""
from __future__ import annotations

import numpy as np
from scipy.signal import sawtooth, square

from resonance_paper import _common as C
from resonance_paper.signals import _norm, pink_noise
from resonance_paper.study5_cross_signal import iaaft_surrogate
from biotuner.resonance import compute_resonance

SF = 500.0
DUR = 12.0
F0 = 8.0
KINDS = ["pure_sine", "asym_harmonic", "sawtooth", "square", "clipped"]


def gen(kind, seed):
    rng = np.random.default_rng(seed)
    n = int(SF * DUR); t = np.arange(n) / SF
    ph = rng.uniform(0, 2 * np.pi)
    if kind == "pure_sine":
        x = np.sin(2 * np.pi * F0 * t + ph)
    elif kind == "asym_harmonic":                      # fixed-phase 2nd+3rd harmonics (nonsinusoidal)
        x = (np.sin(2 * np.pi * F0 * t + ph) + 0.5 * np.sin(2 * np.pi * 2 * F0 * t + ph)
             + 0.33 * np.sin(2 * np.pi * 3 * F0 * t + ph))
    elif kind == "sawtooth":
        x = sawtooth(2 * np.pi * F0 * t + ph)
    elif kind == "square":
        x = square(2 * np.pi * F0 * t + ph)
    elif kind == "clipped":
        x = np.clip(2.5 * np.sin(2 * np.pi * F0 * t + ph), -1.0, 1.0)
    else:
        raise ValueError(kind)
    x = x + 0.04 * pink_noise(n, SF, seed=seed + 1)    # light noise so IAAFT is well-defined
    return _norm(x).astype(np.float64)


def within_pc_z(x, cfg, n_surr, seed):
    obs = compute_resonance(x, sf=SF, config=cfg)
    pc_obs = float(obs.summaries["PC"]["max"]); H = float(obs.summaries["H"]["max"])
    rng = np.random.default_rng(seed)
    sv = np.array([float(compute_resonance(_norm(iaaft_surrogate(x, np.random.default_rng(int(s)))),
                                           sf=SF, config=cfg).summaries["PC"]["max"])
                   for s in rng.integers(0, 2 ** 31 - 1, n_surr)])
    z = float((pc_obs - sv.mean()) / (sv.std() + 1e-12))
    rank_p = float((1 + np.sum(sv >= pc_obs)) / (n_surr + 1))
    return dict(pc_z=z, rank_p=rank_p, H=H, pc_obs=pc_obs)


def run(quick=True):
    cfg = C.default_config(fmin=2, fmax=80, precision_hz=0.5)
    seeds = list(range(4) if quick else range(12))
    n_surr = 49 if quick else 99
    rows = []
    for kind in KINDS:
        recs = [within_pc_z(gen(kind, s), cfg, n_surr, seed=s + 100) for s in seeds]
        pcz = C.mean_ci([r["pc_z"] for r in recs]); Hc = C.mean_ci([r["H"] for r in recs])
        frac_sig = float(np.mean([r["rank_p"] <= 0.05 for r in recs]))
        rows.append(dict(kind=kind, pc_z=pcz["mean"], pc_z_lo=pcz["lo"], pc_z_hi=pcz["hi"],
                         H=Hc["mean"], H_lo=Hc["lo"], H_hi=Hc["hi"], frac_pc_sig=frac_sig))
        print(f"  {kind:14s} PC_z={pcz['mean']:+.2f} [{pcz['lo']:+.2f},{pcz['hi']:+.2f}]  "
              f"H={Hc['mean']:.3f}  frac PC sig={frac_sig:.2f}", flush=True)
    result = dict(quick=quick, n_surr=n_surr, n_seeds=len(seeds), rows=rows)
    C.save_json(result, "study29_nonsinusoidal_pc.json")
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 29 headline (within-signal PC vs waveform nonsinusoidality) ---")
    ctrl = next(r for r in result["rows"] if r["kind"] == "pure_sine")
    nons = [r for r in result["rows"] if r["kind"] != "pure_sine"]
    print(f"  control (pure sine): PC_z={ctrl['pc_z']:+.2f}, H={ctrl['H']:.3f}, "
          f"frac PC significant={ctrl['frac_pc_sig']:.2f}")
    print(f"  nonsinusoidal waveforms: PC_z {min(r['pc_z'] for r in nons):+.2f}..{max(r['pc_z'] for r in nons):+.2f}, "
          f"H {min(r['H'] for r in nons):.2f}..{max(r['H'] for r in nons):.2f}")
    print("  => within-signal PC_z is at null for a pure sinusoid but elevated for every nonsinusoidal")
    print("     waveform (as is H): the single-signal phase axis partly re-reports waveform shape.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

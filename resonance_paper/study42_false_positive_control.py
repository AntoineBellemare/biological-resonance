"""Study 42 — The soundness make-or-break: harmonic-artifact false-positive control + surrogate-z.

The Scheffer-Teixeira & Tort (2016) trap: a SINGLE non-sinusoidal oscillator has harmonics at f0,2f0,3f0
that are phase-locked BY CONSTRUCTION (they are one rhythm's Fourier components). A naive n:m metric will
report strong "1:2 coupling" between f0 and 2f0 here -- a FALSE POSITIVE: there is no second oscillator.
A SOUND metric+surrogate must return NULL on this, while still detecting genuine coupling between two
independent oscillators locked at 1:2.

We test the surrogate-z of each technique (IAAFT null, which preserves the waveform's harmonic phase
structure) on:
  NEGATIVE (harmonic artifact): within ONE non-sinusoidal signal, "1:2" between f0 and 2f0.
                                 sound -> z ~ 0 (the IAAFT surrogate also carries the locked harmonics).
  POSITIVE (genuine coupling) : two oscillators A@f0, B@2f0 with B phase-locked to A, IAAFT-B null.
                                 sound -> z >> 0.
Techniques: PLV (nm_plv), rho_entropy, phase_mi, conditional_prob (the canonical convention is applied
via the correct multipliers n=2,m=1 for the 1:2 ratio).

Outputs: results/study42_false_positive_control.json
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from resonance_paper import _common as C
from biotuner.resonance.coupling import nm_plv
from biotuner.resonance.nulls import iaaft_surrogate
from resonance_paper.nm_techniques import rho_entropy, conditional_prob, phase_mi

warnings.filterwarnings("ignore")

SF = 500.0
DUR = 24.0
N = int(SF * DUR)
T = np.arange(N) / SF
F0 = 10.0           # harmonics at 10, 20 -> test 1:2 (n=2, m=1)


def _bp(x, f, bw=3.0):
    sos = butter(4, [(f - bw / 2) / (SF / 2), (f + bw / 2) / (SF / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def gen_negative(seed):
    """ONE non-sinusoidal oscillator: f0 + its harmonics (no second oscillator)."""
    r = np.random.default_rng(seed)
    phi = 2 * np.pi * F0 * T + np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    x = np.sin(phi) + 0.5 * np.sin(2 * phi) + 0.33 * np.sin(3 * phi)
    x = _norm(x + 0.1 * r.standard_normal(N))
    return x, x            # both "channels" are the SAME signal -> within-signal harmonic test


def gen_positive(seed):
    """Two oscillators: A@f0, B@2f0 genuinely phase-locked 1:2."""
    r = np.random.default_rng(seed)
    phi = 2 * np.pi * F0 * T + np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    A = np.sin(phi)
    B = np.sin(2 * phi + np.pi / 4)
    A = _norm(A + 0.3 * r.standard_normal(N)); B = _norm(B + 0.3 * r.standard_normal(N))
    return A, B


METRICS = {"plv": (lambda aa, ab: nm_plv(np.angle(aa), np.angle(ab), 2, 1)),
           "rho_entropy": (lambda aa, ab: rho_entropy(np.angle(aa), np.angle(ab), 2, 1)),
           "conditional_prob": (lambda aa, ab: conditional_prob(np.angle(aa), np.angle(ab), 2, 1)),
           "phase_mi": (lambda aa, ab: phase_mi(np.angle(aa), np.angle(ab), 2, 1))}


def _measure(A, B, mfn):
    return mfn(hilbert(_bp(A, F0)), hilbert(_bp(B, 2 * F0)))


def _surrogate_z(A, B, which, mfn, n_surr, seed):
    """z of the 1:2 measure vs an IAAFT null. NEGATIVE: surrogate the single signal (A==B).
    POSITIVE: surrogate channel B only (destroys the genuine cross-lock, preserves PSDs)."""
    obs = _measure(A, B, mfn)
    rng = np.random.default_rng(seed)
    sv = []
    for s in rng.integers(0, 2 ** 31 - 1, n_surr):
        if which == "negative":
            xs = iaaft_surrogate(A, np.random.default_rng(int(s)))   # surrogate the one oscillator
            v = _measure(xs, xs, mfn)
        else:
            Bs = iaaft_surrogate(B, np.random.default_rng(int(s)))
            v = _measure(A, Bs, mfn)
        if np.isfinite(v):
            sv.append(v)
    sv = np.array(sv)
    return dict(obs=float(obs), z=float((obs - sv.mean()) / (sv.std() + 1e-12)),
                rank_p=float((1 + np.sum(sv >= obs)) / (len(sv) + 1)))


def run(quick=True):
    seeds = list(range(6 if quick else 15))
    n_surr = 49 if quick else 99
    out = {"negative": {}, "positive": {}}
    for which, gen in (("negative", gen_negative), ("positive", gen_positive)):
        for mname, mfn in METRICS.items():
            zs, ps, obs = [], [], []
            for s in seeds:
                A, B = gen(s + (0 if which == "negative" else 500))
                r = _surrogate_z(A, B, which, mfn, n_surr, seed=s + 7)
                zs.append(r["z"]); ps.append(r["rank_p"]); obs.append(r["obs"])
            out[which][mname] = dict(mean_obs=float(np.mean(obs)), mean_z=float(np.mean(zs)),
                                     frac_sig=float(np.mean([p <= 0.05 for p in ps])))
    C.save_json(dict(quick=quick, n_surr=n_surr, out=out), "study42_false_positive_control.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 42 headline (harmonic-artifact false-positive control) ---")
    print(f"  {'metric':16s} | {'NEG raw':>8s} {'NEG z':>7s} {'NEG %sig':>8s} | {'POS raw':>8s} {'POS z':>7s} {'POS %sig':>8s}")
    for m in METRICS:
        ng = out["negative"][m]; ps = out["positive"][m]
        print(f"  {m:16s} | {ng['mean_obs']:8.2f} {ng['mean_z']:7.2f} {ng['frac_sig']:8.2f} | "
              f"{ps['mean_obs']:8.2f} {ps['mean_z']:7.2f} {ps['frac_sig']:8.2f}")
    print("  SOUND = NEGATIVE z~0 / %sig~0.05 (no false positive on harmonics) AND POSITIVE z>>0 (detects real coupling).")
    print("  raw obs is HIGH in both (harmonics look coupled); only the surrogate separates artifact from genuine.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

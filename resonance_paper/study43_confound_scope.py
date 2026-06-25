"""Study 43 — Is the harmonic-CFC false positive fixable by the null, or fundamental (scope to cross-signal)?

Study 42 showed every n:m technique false-positives on a single non-sinusoidal oscillator's harmonics
with an IAAFT null. Here we ask which NULL (if any) separates the harmonic artifact from genuine coupling:

  IAAFT          preserves PSD+amplitude dist, RANDOMIZES cross-frequency phase -> destroys the harmonic
                 lock -> false positive (study42).
  phase_randomize same (randomizes phases).
  time_shuffle   block-reorders the time series -> PRESERVES within-block waveform (harmonics stay locked
                 inside each block) while destroying long-range coupling -> should NULL the harmonic
                 artifact yet still detect genuine coupling. Candidate fix.

Scenarios:
  NEGATIVE within_harmonic : ONE non-sinusoidal oscillator, test 1:2 between f0 and 2f0. SOUND -> z ~ 0.
  POSITIVE cross_genuine   : two independent oscillators locked 2:3 (A@10, B@15). SOUND -> z >> 0.
A null PASSES iff it gives NEGATIVE z ~ 0 AND POSITIVE z >> 0 for all metrics. If none passes, within-
signal harmonic n:m is fundamentally confounded and the sound scope is CROSS-SIGNAL independent sources.

Outputs: results/study43_confound_scope.json
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from resonance_paper import _common as C
from biotuner.resonance.coupling import nm_plv
from biotuner.resonance.nulls import iaaft_surrogate, phase_randomize_surrogate, time_shuffle_surrogate
from resonance_paper.nm_techniques import rho_entropy, phase_mi

warnings.filterwarnings("ignore")

SF = 500.0
DUR = 24.0
N = int(SF * DUR)
T = np.arange(N) / SF

NULLS = {"iaaft": iaaft_surrogate, "phase_randomize": phase_randomize_surrogate,
         "time_shuffle": time_shuffle_surrogate}


def _bp(x, f, bw=3.0):
    sos = butter(4, [(f - bw / 2) / (SF / 2), (f + bw / 2) / (SF / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def gen_within_harmonic(seed):
    r = np.random.default_rng(seed)
    phi = 2 * np.pi * 10 * T + np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    x = np.sin(phi) + 0.5 * np.sin(2 * phi) + 0.33 * np.sin(3 * phi)
    x = _norm(x + 0.1 * r.standard_normal(N))
    return x, x, (10.0, 20.0), (2, 1)        # 1:2 between f0 and 2f0


def gen_cross_genuine(seed):
    r = np.random.default_rng(seed)
    phi = 2 * np.pi * 10 * T + np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    A = _norm(np.sin(phi) + 0.3 * r.standard_normal(N))
    B = _norm(np.sin(1.5 * phi + np.pi / 4) + 0.3 * r.standard_normal(N))
    return A, B, (10.0, 15.0), (3, 2)        # genuine 2:3 (n=3,m=2)


METRICS = {"plv": lambda pa, pb, n, m: nm_plv(pa, pb, n, m),
           "rho_entropy": lambda pa, pb, n, m: rho_entropy(pa, pb, n, m),
           "phase_mi": lambda pa, pb, n, m: phase_mi(pa, pb, n, m)}


def _measure(A, B, fpair, nm, mfn):
    pa = np.angle(hilbert(_bp(A, fpair[0]))); pb = np.angle(hilbert(_bp(B, fpair[1])))
    return mfn(pa, pb, nm[0], nm[1])


def _z(A, B, fpair, nm, mfn, nullfn, within, n_surr, seed):
    obs = _measure(A, B, fpair, nm, mfn)
    rng = np.random.default_rng(seed); sv = []
    for s in rng.integers(0, 2 ** 31 - 1, n_surr):
        if within:
            xs = nullfn(A, np.random.default_rng(int(s))); v = _measure(xs, xs, fpair, nm, mfn)
        else:
            Bs = nullfn(B, np.random.default_rng(int(s))); v = _measure(A, Bs, fpair, nm, mfn)
        if np.isfinite(v):
            sv.append(v)
    sv = np.array(sv)
    return float((obs - sv.mean()) / (sv.std() + 1e-12))


def run(quick=True):
    seeds = list(range(5 if quick else 12))
    n_surr = 49 if quick else 99
    out = {"within_harmonic_NEG": {}, "cross_genuine_POS": {}}
    for scen, gen, within in (("within_harmonic_NEG", gen_within_harmonic, True),
                              ("cross_genuine_POS", gen_cross_genuine, False)):
        for nname, nullfn in NULLS.items():
            for mname, mfn in METRICS.items():
                zs = []
                for s in seeds:
                    A, B, fpair, nm = gen(s + (0 if within else 300))
                    zs.append(_z(A, B, fpair, nm, mfn, nullfn, within, n_surr, seed=s + 11))
                out[scen][f"{nname}/{mname}"] = float(np.mean(zs))
    C.save_json(dict(quick=quick, n_surr=n_surr, out=out), "study43_confound_scope.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 43 headline (which null separates harmonic artifact from genuine coupling) ---")
    print(f"  {'null/metric':28s} {'NEG z (want ~0)':>16s} {'POS z (want >>0)':>17s}  verdict")
    for nname in NULLS:
        for mname in METRICS:
            k = f"{nname}/{mname}"
            neg = out["within_harmonic_NEG"][k]; pos = out["cross_genuine_POS"][k]
            ok = "PASS" if (neg < 2.0 and pos > 3.0) else "fails (false-pos)" if neg >= 2.0 else "fails (no detect)"
            print(f"  {k:28s} {neg:16.2f} {pos:17.2f}  {ok}")
    print("  PASS = nulls the within-signal harmonic artifact (NEG z<2) AND detects genuine coupling (POS z>3).")
    print("  If NO null passes within-signal -> n:m is soundly interpretable only CROSS-signal (independent sources).")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

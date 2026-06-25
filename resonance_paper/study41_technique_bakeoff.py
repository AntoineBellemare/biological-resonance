"""Study 41 — The comprehensive n:m technique bake-off: WHICH-WINS-WHEN across regimes.

Detection AUC (coupled vs independent) for every n:m phase-phase technique, pooled across a wide
coprime ratio set, computed SEPARATELY for each of eight dynamical regimes. This maps the
regime-dependent winner: PLV-canonical dominates clean unimodal locks but is BLIND to multimodal /
antipodal locks, where the Tass entropy / MI measures take over.

Techniques (all called at the CORRECT multipliers for a frequency ratio f_b/f_a = b/a):
  PLV-FAMILY (biotuner.resonance.coupling, canonical Tass-convention wrappers). The canonical wrapper
    takes the legacy ratio-kernel (n,m) (ratio = m/n) and swaps internally, so to test a genuine b/a
    lock we hand it (n=a, m=b). Phase-input: fn(angle_a, angle_b, a, b); the wpli_complex variant
    takes the analytic signals: fn(analytic_a, analytic_b, a, b).
      nm_plv_canonical, nm_pli_canonical, nm_wpli_canonical, nm_rrci_canonical, nm_wpli_complex_canonical
  PURPOSE-BUILT (resonance_paper.nm_techniques). Phase-input with multipliers (n=b, m=a):
      rho_entropy, conditional_prob, phase_mi

Regimes:
  clean            : phi_b = (b/a) phi_a + pi/4                          (unimodal lock)
  bimodal          : phi_b = (b/a) phi_a + theta(t); theta telegraph in {0, pi/a}   (antipodal)
  multistable3     : theta(t) telegraph over {0, 2pi/3, 4pi/3}/a-scaled (3 stable phases)
  nonsinusoidal_cf : phi_b = (b/a) phi_a + 0.9*sin(phi_a)               (deterministic non-const coupling)
  weak_kappa       : B = 0.4*locked + 0.6*independent                   (graded weak coupling)
  low_snr          : clean lock at -6 dB observation noise
  nonstationary    : clean lock with f_a drifting +-15% across the epoch
  harmonic_contam  : clean lock + strong 2nd/3rd harmonics added to BOTH signals

SANITY GATES (must hold or the harness is debugged, not the finding):
  clean   -> PLV-canonical AUC > 0.9
  bimodal -> PLV-canonical AUC < 0.6  AND  rho_entropy AUC > PLV-canonical AUC

Outputs: results/study41_technique_bakeoff.json + printed which-wins-when summary.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert

from resonance_paper import _common as C
from biotuner.resonance.coupling import (
    nm_plv_canonical, nm_pli_canonical, nm_wpli_canonical, nm_rrci_canonical,
    nm_wpli_complex_canonical,
)
from resonance_paper.nm_techniques import rho_entropy, conditional_prob, phase_mi

warnings.filterwarnings("ignore")

SF = 500.0
DUR = 24.0
N = int(SF * DUR)
T = np.arange(N) / SF

# wide coprime ratio set (a:b, a<b, coprime); u set so f_b ~ 30 Hz keeps both components in-band
RATIOS = [(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6), (5, 7)]
OFFSET = np.pi / 4

# technique registry. kind: "plv_phase" (canonical wrapper, phase, call with (a,b)),
# "plv_analytic" (canonical wrapper, analytic, call with (a,b)),
# "purpose" (call with (b,a) phases).
PLV_PHASE = {
    "plv_canonical": nm_plv_canonical,
    "pli_canonical": nm_pli_canonical,
    "wpli_canonical": nm_wpli_canonical,
    "rrci_canonical": nm_rrci_canonical,
}
PLV_ANALYTIC = {"wpli_complex_canonical": nm_wpli_complex_canonical}
PURPOSE = {"rho_entropy": rho_entropy, "conditional_prob": conditional_prob, "phase_mi": phase_mi}
TECHNIQUES = list(PLV_PHASE) + list(PLV_ANALYTIC) + list(PURPOSE)

REGIMES = ["clean", "bimodal", "multistable3", "nonsinusoidal_cf",
           "weak_kappa", "low_snr", "nonstationary", "harmonic_contam"]


# ---------------------------------------------------------------------------
# reused study39/40 helpers
# ---------------------------------------------------------------------------
def _bp(x, f, bw=3.0):
    lo = max(f - bw / 2, 0.5); hi = min(f + bw / 2, SF / 2 - 1)
    sos = butter(4, [lo / (SF / 2), hi / (SF / 2)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def _telegraph(states, switch_hz, rng):
    """Piecewise-constant telegraph cycling through a SHUFFLED order of `states` so every epoch
    spends near-equal time in each state (balanced multistability -> the first circular moment of
    the relative phase cancels). Dwell ~ 1/switch_hz s with mild gamma jitter (not heavy-tailed
    exponential, which would let one state dominate a finite epoch)."""
    out = np.empty(N)
    t = 0
    mean_dwell = SF / switch_hz
    while t < N:
        order = list(states)
        rng.shuffle(order)
        for s in order:
            dwell = max(1, int(rng.gamma(shape=8.0, scale=mean_dwell / 8.0)))
            out[t:t + dwell] = s
            t += dwell
            if t >= N:
                break
    return out[:N]


def _observe(A, B, rng, snr_db=6.0):
    """Add white observation noise at a target SNR, then z-normalize."""
    snr = 10 ** (snr_db / 10.0)
    A = np.sqrt(snr) * A + rng.standard_normal(N)
    B = np.sqrt(snr) * B + rng.standard_normal(N)
    return _norm(A), _norm(B)


def gen(a, b, regime, coupled, seed):
    """Return (A, B, fa, fb). `coupled`=False -> independent B at f_b (the negative class).

    A is always sin(phi_a) at f_a; B carries (or does not carry) a b/a phase lock per regime.
    """
    r = np.random.default_rng(seed)
    u = 30.0 / b
    fa, fb = u * a, u * b

    # base phase of A (small phase diffusion so spectra are realistic, like study40)
    drift = np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    if regime == "nonstationary":
        # f_a drifts +-15% across the epoch (slow sinusoidal sweep)
        inst_fa = fa * (1.0 + 0.15 * np.sin(2 * np.pi * (1.0 / DUR) * T))
        phi_a = 2 * np.pi * np.cumsum(inst_fa) / SF + drift
    else:
        phi_a = 2 * np.pi * fa * T + drift
    A = np.sin(phi_a)

    # independent reference at f_b (negative class, and the "unlocked" pool for mixes)
    indep_phase = 2 * np.pi * fb * T + np.cumsum(0.5 * np.sqrt(1.0 / SF) * r.standard_normal(N))
    indep = np.sin(indep_phase)

    snr_db = 6.0
    if not coupled:
        B = indep
    else:
        ratio = b / a
        if regime in ("clean", "low_snr", "nonstationary"):
            B = np.sin(ratio * phi_a + OFFSET)
            if regime == "low_snr":
                snr_db = -6.0
        elif regime == "bimodal":
            # genuinely antipodal: theta switches the relative phase psi by pi (no constant offset,
            # so the first circular moment of psi cancels -> PLV-family blind, entropy/MI not).
            theta = _telegraph([0.0, np.pi / a], switch_hz=1.0, rng=r)
            B = np.sin(ratio * phi_a + theta)
        elif regime == "multistable3":
            # three antipodal-balanced stable phases -> psi trimodal, first moment cancels.
            theta = _telegraph([0.0, 2 * np.pi / 3 / a, 4 * np.pi / 3 / a], switch_hz=1.0, rng=r)
            B = np.sin(ratio * phi_a + theta)
        elif regime == "nonsinusoidal_cf":
            B = np.sin(ratio * phi_a + OFFSET + 0.9 * np.sin(phi_a))
        elif regime == "weak_kappa":
            locked = np.sin(ratio * phi_a + OFFSET)
            B = 0.4 * locked + 0.6 * indep
        elif regime == "harmonic_contam":
            B = np.sin(ratio * phi_a + OFFSET)
        else:
            raise ValueError(regime)

    if regime == "harmonic_contam":
        # strong 2nd/3rd harmonics on BOTH signals (broadband non-sinusoidal waveforms)
        A = A + 0.6 * np.sin(2 * phi_a) + 0.4 * np.sin(3 * phi_a)
        base_b = (b / a) * phi_a + OFFSET if coupled else indep_phase
        B = B + 0.6 * np.sin(2 * base_b) + 0.4 * np.sin(3 * base_b)

    A, B = _observe(A, B, r, snr_db=snr_db)
    return A, B, fa, fb


# ---------------------------------------------------------------------------
# technique evaluation at the CORRECT multipliers
# ---------------------------------------------------------------------------
def _eval_all(A, B, fa, fb, a, b):
    """Return {technique: scalar} for one trial, all called at the correct n:m convention."""
    ana_a, ana_b = hilbert(_bp(A, fa)), hilbert(_bp(B, fb))
    pa, pb = np.angle(ana_a), np.angle(ana_b)
    out = {}
    for name, fn in PLV_PHASE.items():
        out[name] = float(fn(pa, pb, a, b))           # canonical wrapper swaps -> tests b*pa - a*pb
    for name, fn in PLV_ANALYTIC.items():
        out[name] = float(fn(ana_a, ana_b, a, b))
    for name, fn in PURPOSE.items():
        out[name] = float(fn(pa, pb, b, a))            # purpose-built use (n=b, m=a) directly
    return out


def detection_auc(regime, seeds):
    """Pooled detection AUC (coupled vs independent) across the ratio set, per technique."""
    pos = {t: [] for t in TECHNIQUES}
    neg = {t: [] for t in TECHNIQUES}
    for (a, b) in RATIOS:
        for s in seeds:
            Ac, Bc, fa, fb = gen(a, b, regime, coupled=True, seed=s)
            Au, Bu, fa2, fb2 = gen(a, b, regime, coupled=False, seed=s + 9001)
            vc = _eval_all(Ac, Bc, fa, fb, a, b)
            vu = _eval_all(Au, Bu, fa2, fb2, a, b)
            for t in TECHNIQUES:
                pos[t].append(vc[t]); neg[t].append(vu[t])
    return {t: C.bootstrap_auc_ci(pos[t], neg[t])["auc"] for t in TECHNIQUES}


def specificity(regime, seeds):
    """Does each technique still PEAK at the true ratio? Scan candidate coprime multipliers and
    report identification accuracy (argmax candidate == true)."""
    from math import gcd
    cands = sorted({(n, m) for n in range(1, 8) for m in range(1, 8) if gcd(n, m) == 1})
    ident = {t: [] for t in TECHNIQUES}
    for (a, b) in RATIOS:
        true_nm = (b, a)
        for s in seeds:
            A, B, fa, fb = gen(a, b, regime, coupled=True, seed=s)
            ana_a, ana_b = hilbert(_bp(A, fa)), hilbert(_bp(B, fb))
            pa, pb = np.angle(ana_a), np.angle(ana_b)
            for name, fn in PLV_PHASE.items():
                # candidate (cb, ca): true is (b, a); wrapper call uses (ca, cb)
                vals = {(cb, ca): fn(pa, pb, ca, cb) for (cb, ca) in cands}
                ident[name].append(1.0 if max(vals, key=vals.get) == true_nm else 0.0)
            for name, fn in PLV_ANALYTIC.items():
                vals = {(cb, ca): fn(ana_a, ana_b, ca, cb) for (cb, ca) in cands}
                ident[name].append(1.0 if max(vals, key=vals.get) == true_nm else 0.0)
            for name, fn in PURPOSE.items():
                vals = {(cb, ca): fn(pa, pb, cb, ca) for (cb, ca) in cands}
                ident[name].append(1.0 if max(vals, key=vals.get) == true_nm else 0.0)
    return {t: float(np.mean(ident[t])) for t in TECHNIQUES}


def run(quick=True):
    seeds = list(range(8 if quick else 16))
    auc = {r: detection_auc(r, seeds) for r in REGIMES}
    spec = {r: specificity(r, seeds) for r in ("clean", "bimodal")}

    # sanity gates
    gates = {
        "clean_plv_gt_0.9": auc["clean"]["plv_canonical"] > 0.9,
        "bimodal_plv_lt_0.6": auc["bimodal"]["plv_canonical"] < 0.6,
        "bimodal_rho_gt_plv": auc["bimodal"]["rho_entropy"] > auc["bimodal"]["plv_canonical"],
    }
    gates["all_passed"] = all(gates.values())

    winners = {r: max(auc[r], key=auc[r].get) for r in REGIMES}

    out = {
        "ratios": [f"{a}:{b}" for a, b in RATIOS],
        "techniques": TECHNIQUES,
        "regimes": REGIMES,
        "detection_auc": auc,
        "specificity_ident_accuracy": spec,
        "regime_winner": winners,
        "sanity_gates": gates,
        "n_seeds": len(seeds),
    }
    C.save_json(out, "study41_technique_bakeoff.json")
    _headline(out)
    return out


def _headline(out):
    auc = out["detection_auc"]
    print("\n  === Study 41 bake-off: detection AUC (coupled vs independent), regime x technique ===")
    hdr = "  " + " " * 18 + "".join(f"{t[:11]:>13s}" for t in TECHNIQUES)
    print(hdr)
    for r in REGIMES:
        row = "".join(f"{auc[r][t]:13.2f}" for t in TECHNIQUES)
        print(f"  {r:18s}{row}")
    print("\n  --- which wins when (best technique per regime + margin over 2nd) ---")
    for r in REGIMES:
        srt = sorted(auc[r].items(), key=lambda kv: -kv[1])
        best, second = srt[0], srt[1]
        print(f"  {r:18s} -> {best[0]:22s} AUC={best[1]:.2f}  (margin +{best[1]-second[1]:.2f} over {second[0]})")
    print("\n  --- specificity (ratio identification accuracy) ---")
    for r in ("clean", "bimodal"):
        print(f"  [{r}]")
        for t in TECHNIQUES:
            print(f"      {t:24s} {out['specificity_ident_accuracy'][r][t]:.2f}")
    g = out["sanity_gates"]
    print("\n  --- sanity gates ---")
    for k, v in g.items():
        print(f"      {k:24s} {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

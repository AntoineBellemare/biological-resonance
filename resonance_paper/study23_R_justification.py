"""Study 23 — Why R = H * PC?  Operator analysis + the phase-coherence gate.

Two complementary justifications for the resonance construct R = combine(H, PC),
addressing (a) why a CONJUNCTION, (b) which combine rule, (c) what R adds over its
factors. (Study 17 separately shows H is phase-blind on real signals: rho(H,kappa)=0.00.)

PART A — operator analysis (independent factors).
  Sample H and PC as INDEPENDENT latent values in [0,1] (the regime R targets but
  that a single coupled oscillator cannot produce). Define "true resonance" = both
  factors high; "confounds" = exactly one factor high. Score how well each candidate
  combine rule (and H-alone, PC-alone) separates true from the single-factor
  confounds. A CONJUNCTION (product/geomean/harmmean/min) should beat both single
  factors AND beat disjunctions (max/mean), which accept single-factor confounds.

PART B — the phase-coherence gate (real signals, artifact-free).
  Take a harmonic, phase-locked pair (high H, high PC) and PHASE-SCRAMBLE it (FFT
  phase randomization preserves |FFT| exactly, so H is unchanged BY CONSTRUCTION but
  phase coupling is destroyed). H cannot tell the two apart; R = H*PC drops. This is
  the clean demonstration that R adds a phase-coherence requirement H lacks.

Honest expectation: R (conjunction) beats BOTH single factors under independence;
product is good but min/harmmean may match or beat it — product is preferred because
log R = log H + log PC gives an additive, interpretable attribution.

Outputs: results/study23_R_justification.json, figures/study23_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from fractions import Fraction
from scipy.signal import butter, filtfilt, hilbert

from resonance_paper import _common as C
from resonance_paper import signals as S
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import ResonanceConfig

SF = 500.0
F1, F2 = 8.0, 12.0        # 2:3 harmonic pair
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, coupling_metric="nm_plv_canonical",
                      ratio_kernel="fraction", ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)
COMBINE_RULES = ["product", "geomean", "harmmean", "min", "max", "mean"]
CONJUNCTIONS = {"product", "geomean", "harmmean", "min"}


def _combine(H, PC, rule):
    H = np.asarray(H, float); PC = np.asarray(PC, float)
    if rule == "product":  return H * PC
    if rule == "geomean":  return np.sqrt(np.clip(H * PC, 0, None))
    if rule == "harmmean": return 2 * H * PC / (H + PC + 1e-9)
    if rule == "min":      return np.minimum(H, PC)
    if rule == "max":      return np.maximum(H, PC)
    if rule == "mean":     return 0.5 * (H + PC)
    raise ValueError(rule)


# ----------------------------------------------------------------- Part A
def part_a(quick=True):
    rng = np.random.default_rng(0)
    n = 4000 if quick else 20000
    H = rng.uniform(0, 1, n); PC = rng.uniform(0, 1, n)     # INDEPENDENT factors
    thr = 0.6
    true = (H > thr) & (PC > thr)
    conf = ((H > thr) & (PC <= thr)) | ((H <= thr) & (PC > thr))   # single-factor-high
    out = {}
    for name, v in [("H", H), ("PC", PC)] + [(f"R[{r}]", _combine(H, PC, r)) for r in COMBINE_RULES]:
        out[name] = C.bootstrap_auc_ci(list(v[true]), list(v[conf]))["auc"]
    return dict(n=int(n), threshold=thr, auc=out,
                conj_mean=float(np.mean([out[f"R[{r}]"] for r in CONJUNCTIONS])),
                disj_mean=float(np.mean([out[f"R[{r}]"] for r in ("max", "mean")])),
                best_rule=max(COMBINE_RULES, key=lambda r: out[f"R[{r}]"]))


# ----------------------------------------------------------------- Part B
def _nm_plv(s1, s2, f1, f2, bw=2.0):
    """Clean n:m PLV. f2/f1 = q/p, so the locked relation is q*th1 - p*th2 = const."""
    fr = Fraction(f2 / f1).limit_denominator(16)
    q, p = fr.numerator, fr.denominator           # f2/f1 = q/p
    def phase(s, f):
        b, a = butter(4, [(f - bw) / (SF / 2), (f + bw) / (SF / 2)], btype="band")
        return np.angle(hilbert(filtfilt(b, a, s)))
    return float(np.abs(np.mean(np.exp(1j * (q * phase(s1, f1) - p * phase(s2, f2))))))


def _phase_scramble(x, rng):
    """Randomize Fourier phases, preserve magnitude spectrum exactly (H unchanged)."""
    X = np.fft.rfft(x); mag = np.abs(X)
    ph = rng.uniform(0, 2 * np.pi, len(X)); ph[0] = 0.0
    return np.fft.irfft(mag * np.exp(1j * ph), n=len(x))


def _HR(s1, s2):
    r = compute_cross_resonance(s1.astype(np.float64), s2.astype(np.float64), sf=SF, config=CFG)
    fr = r.freqs; i = int(np.argmin(np.abs(fr - F1))); j = int(np.argmin(np.abs(fr - F2)))
    return float(r.harmonicity_matrix[i, j]), _nm_plv(s1, s2, F1, F2)


def part_b(quick=True):
    seeds = range(8) if quick else range(20)
    coh, scr = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        n = int(SF * 20); t = np.arange(n) / SF
        drift = np.cumsum(rng.standard_normal(n)) * 0.02            # mild shared drift
        s1 = np.sin(2 * np.pi * F1 * t + drift + rng.uniform(0, 2 * np.pi))
        s2 = np.sin(2 * np.pi * F2 * t + 1.5 * drift + rng.uniform(0, 2 * np.pi))  # locked (q/p=3/2)
        nz = 10 ** (-6 / 20.0)
        s1 = S._norm(S._norm(s1) + nz * S.pink_noise(n, SF, seed=seed + 5))
        s2 = S._norm(S._norm(s2) + nz * S.pink_noise(n, SF, seed=seed + 9))
        H1, P1 = _HR(s1, s2)
        s1s, s2s = _phase_scramble(s1, rng), _phase_scramble(s2, rng)   # same |FFT| -> same H
        H2, P2 = _HR(s1s, s2s)
        coh.append(dict(H=H1, PC=P1)); scr.append(dict(H=H2, PC=P2))
        print(f"  seed {seed}: coherent H={H1:.1f} PC={P1:.2f} | scrambled H={H2:.1f} PC={P2:.2f}", flush=True)
    from scipy.stats import wilcoxon
    Hc = np.array([r["H"] for r in coh]); Hs = np.array([r["H"] for r in scr])
    Pc = np.array([r["PC"] for r in coh]); Ps = np.array([r["PC"] for r in scr])
    Rc, Rs = Hc * Pc, Hs * Ps
    def pw(a, b):
        try: return float(wilcoxon(a, b).pvalue)
        except ValueError: return float("nan")
    return dict(coherent=coh, scrambled=scr,
                H_coherent=float(Hc.mean()), H_scrambled=float(Hs.mean()), H_p=pw(Hc, Hs),
                PC_coherent=float(Pc.mean()), PC_scrambled=float(Ps.mean()), PC_p=pw(Pc, Ps),
                R_coherent=float(Rc.mean()), R_scrambled=float(Rs.mean()), R_p=pw(Rc, Rs),
                auc_H=C.bootstrap_auc_ci(list(Hc), list(Hs))["auc"],
                auc_R=C.bootstrap_auc_ci(list(Rc), list(Rs))["auc"])


def run(quick=True):
    A = part_a(quick); B = part_b(quick)
    result = dict(quick=quick, part_a=A, part_b=B)
    C.save_json(result, "study23_R_justification.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    A = result["part_a"]; B = result["part_b"]
    print("\n  --- Study 23 headline (why R = H*PC) ---")
    print("  PART A (independent factors): specificity AUC, true=both-high vs single-factor confounds")
    print(f"    H={A['auc']['H']:.2f}  PC={A['auc']['PC']:.2f}  | "
          + "  ".join(f"{r}={A['auc'][f'R[{r}]']:.2f}" for r in COMBINE_RULES))
    print(f"    conjunctions mean AUC={A['conj_mean']:.2f}  >>  disjunctions(max/mean) mean AUC={A['disj_mean']:.2f}")
    print(f"    => a conjunction is required and beats BOTH single factors; best rule={A['best_rule']} "
          f"(product preferred for log-additive attribution)")
    print("  PART B (phase-scramble, |FFT| preserved -> H fixed by construction):")
    print(f"    H: coherent={B['H_coherent']:.1f} scrambled={B['H_scrambled']:.1f} (p={B['H_p']:.2g}, ~unchanged)")
    print(f"    PC: coherent={B['PC_coherent']:.2f} scrambled={B['PC_scrambled']:.2f} (p={B['PC_p']:.2g})")
    print(f"    R: coherent={B['R_coherent']:.1f} scrambled={B['R_scrambled']:.1f} (p={B['R_p']:.2g})")
    print(f"    AUC coherent-vs-scrambled: H={B['auc_H']:.2f} (blind) vs R={B['auc_R']:.2f} (gates on coherence)")


def _figures(result):
    plt = C.setup_mpl()
    A = result["part_a"]; B = result["part_b"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # A: combine-rule specificity AUC
    keys = ["H", "PC"] + [f"R[{r}]" for r in COMBINE_RULES]
    aucs = [A["auc"][k] for k in keys]
    cols = ["#E69F00", "#CC79A7"] + ["#D55E00" if r in CONJUNCTIONS else "#777777" for r in COMBINE_RULES]
    axes[0].bar(range(len(keys)), aucs, color=cols, alpha=0.9)
    axes[0].axhline(0.5, color="k", ls="--", lw=0.7); axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels([k.replace("R[", "").replace("]", "") for k in keys], rotation=40, fontsize=6.5)
    axes[0].set_ylabel("specificity AUC (true vs confounds)")
    axes[0].set_title("A. Independent factors:\nconjunctions beat both factors & disjunctions", fontsize=10)

    # B: phase-scramble — H unchanged, PC & R collapse
    ax = axes[1]; x = np.arange(3); w = 0.38
    coh = [B["H_coherent"] / B["H_coherent"], B["PC_coherent"], B["R_coherent"] / B["H_coherent"]]
    scr = [B["H_scrambled"] / B["H_coherent"], B["PC_scrambled"], B["R_scrambled"] / B["H_coherent"]]
    ax.bar(x - w/2, coh, w, color="#2e7d32", label="coherent")
    ax.bar(x + w/2, scr, w, color="#bdbdbd", label="phase-scrambled")
    ax.set_xticks(x); ax.set_xticklabels(["H\n(norm)", "PC", "R\n(norm)"])
    ax.set_ylabel("factor value (H-normalized)")
    ax.set_title("B. Phase-scramble (|FFT| kept):\nH unchanged, PC & R collapse", fontsize=10); ax.legend(fontsize=7)

    # C: AUC coherent-vs-scrambled — H blind, R discriminates
    ax = axes[2]
    ax.bar([0, 1], [B["auc_H"], B["auc_R"]], color=["#E69F00", "#D55E00"], width=0.6)
    ax.axhline(0.5, color="k", ls="--", lw=0.7); ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["H", "R = H·PC"])
    ax.set_ylabel("AUC (coherent vs scrambled)")
    ax.set_title("C. R adds the phase-coherence\ngate that H lacks", fontsize=10)

    fig.suptitle("Study 23 — Why R = H·PC: a conjunction (beats both factors under independence) "
                 "that gates harmonicity by phase coherence", fontweight="bold", fontsize=10)
    fig.tight_layout()
    C.save_fig(fig, "study23_R_justification")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

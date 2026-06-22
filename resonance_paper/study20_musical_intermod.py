"""Study 20 — Musical chords, intermodulation, and consonance via H / PC / R.

Musical consonance has long been linked to frequency-ratio simplicity (harmonicity)
and to nonlinear combination/difference tones generated in the auditory system.
Here we synthesize chords at just-intonation ratios, pass them through an auditory
NONLINEARITY (which creates intermodulation / combination tones), and ask whether
the framework's harmonicity H (and R) recovers the consonance ordering — and
whether the nonlinear combination tones *sharpen* that recovery.

Per chord: tones at f0·ratio summed -> (optional) static nonlinearity -> spectrum
-> H_max. Ground-truth consonance proxy = mean pairwise Tenney height log2(p·q)
of the chord's ratios (lower = more consonant).

Outputs: results/study20_musical_intermod.json, figures/study20_*.{png,pdf}
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import _norm, pink_noise
from biotuner.resonance import compute_resonance

SF = 2000.0
F0 = 100.0
# just-intonation chords (integer ratios), roughly ordered consonant -> dissonant
CHORDS = {
    "octave (1:2)": [1, 2],
    "fifth (2:3)": [2, 3],
    "major_triad (4:5:6)": [4, 5, 6],
    "minor_triad (10:12:15)": [10, 12, 15],
    "major7 (8:10:12:15)": [8, 10, 12, 15],
    "dom7 (4:5:6:7)": [4, 5, 6, 7],
    "tritone (5:7)": [5, 7],
    "minor2nd (15:16)": [15, 16],
    "cluster (16:17:18)": [16, 17, 18],
    "complex (11:13:17)": [11, 13, 17],
}


def tenney_complexity(ratios):
    base = min(ratios)
    ths = []
    for i in range(len(ratios)):
        for j in range(i + 1, len(ratios)):
            fr = Fraction(ratios[j], ratios[i])
            ths.append(np.log2(fr.numerator * fr.denominator))
    return float(np.mean(ths))


def chord_signal(ratios, g_nl, dur=8.0, noise_db=20.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SF * dur); t = np.arange(n) / SF
    base = min(ratios)
    x = sum(np.sin(2 * np.pi * (F0 * r / base) * t + rng.uniform(0, 2 * np.pi)) for r in ratios) / len(ratios)
    if g_nl > 0:                                  # auditory nonlinearity -> combination tones
        x = x + g_nl * x ** 2 + 0.6 * g_nl * x ** 3
    nz = 10 ** (-noise_db / 20.0)
    return _norm(x + nz * pink_noise(n, SF, seed=seed + 7)).astype(np.float64)


def run(quick=True):
    seeds = range(3) if quick else range(10)
    cfg = C.default_config(fmin=40, fmax=900, precision_hz=2.0)
    rows = []
    for name, ratios in CHORDS.items():
        comp = tenney_complexity(ratios)
        for g_nl, tag in [(0.0, "linear"), (0.8, "nonlinear")]:
            Hs = []
            for s in seeds:
                y = chord_signal(ratios, g_nl, seed=s)
                Hs.append(float(compute_resonance(y, sf=SF, config=cfg).summaries["H"]["max"]))
            rows.append(dict(chord=name, complexity=comp, condition=tag, H_max=float(np.mean(Hs))))
        print(f"  {name} done", flush=True)

    from scipy.stats import spearmanr
    rng = np.random.default_rng(0)

    def stats(tag):
        rr = [r for r in rows if r["condition"] == tag]
        comp = np.array([r["complexity"] for r in rr]); H = np.array([r["H_max"] for r in rr])
        m = len(comp); rho = float(spearmanr(comp, H)[0])
        bs = []
        for _ in range(5000):
            idx = rng.integers(0, m, m)
            if len(set(comp[idx])) > 2:
                bs.append(spearmanr(comp[idx], H[idx])[0])
        bs = np.array([b for b in bs if np.isfinite(b)])
        perm = np.array([spearmanr(comp, rng.permutation(H))[0] for _ in range(5000)])
        p = float((np.sum(np.abs(perm) >= abs(rho)) + 1) / (len(perm) + 1))   # two-sided
        return dict(rho=rho, lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), p=p)

    st = {tag: stats(tag) for tag in ("linear", "nonlinear")}
    # is the nonlinear "sharpening" (|rho_nl| - |rho_lin|) above chance? paired chord bootstrap
    lin = sorted([r for r in rows if r["condition"] == "linear"], key=lambda r: r["chord"])
    nl = sorted([r for r in rows if r["condition"] == "nonlinear"], key=lambda r: r["chord"])
    comp = np.array([r["complexity"] for r in lin])
    Hl = np.array([r["H_max"] for r in lin]); Hn = np.array([r["H_max"] for r in nl])
    md = len(comp); sharp = []
    for _ in range(5000):
        idx = rng.integers(0, md, md)
        if len(set(comp[idx])) > 2:
            sharp.append(abs(spearmanr(comp[idx], Hn[idx])[0]) - abs(spearmanr(comp[idx], Hl[idx])[0]))
    sharp = np.array([s for s in sharp if np.isfinite(s)])
    summary = dict(rho_H_vs_complexity_linear=st["linear"]["rho"],
                   rho_H_vs_complexity_nonlinear=st["nonlinear"]["rho"],
                   linear_ci=[st["linear"]["lo"], st["linear"]["hi"]], linear_p=st["linear"]["p"],
                   nonlinear_ci=[st["nonlinear"]["lo"], st["nonlinear"]["hi"]], nonlinear_p=st["nonlinear"]["p"],
                   sharpen_delta=float(np.mean(sharp)),
                   sharpen_ci=[float(np.percentile(sharp, 2.5)), float(np.percentile(sharp, 97.5))],
                   nonlinearity_sharpens=bool(np.percentile(sharp, 2.5) > 0))
    result = dict(quick=quick, rows=rows, summary=summary)
    C.save_json(result, "study20_musical_intermod.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 20 headline (musical chords / consonance) ---")
    print(f"  rho(H_max, chord complexity)  linear={s['rho_H_vs_complexity_linear']:+.2f}  "
          f"nonlinear={s['rho_H_vs_complexity_nonlinear']:+.2f}  (expect negative = H tracks consonance)")
    print(f"  nonlinearity (combination tones) sharpens consonance recovery: {s['nonlinearity_sharpens']}")
    nl = [r for r in result["rows"] if r["condition"] == "nonlinear"]
    nl.sort(key=lambda r: r["complexity"])
    print("  chords by complexity (consonant->dissonant), H_max (nonlinear):")
    for r in nl:
        print(f"    {r['chord']:24s} complexity={r['complexity']:.2f}  H_max={r['H_max']:.3f}")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, col, mk in [("linear", "#9e9e9e", "o"), ("nonlinear", "#b71c1c", "s")]:
        rr = [r for r in rows if r["condition"] == tag]
        ax.scatter([r["complexity"] for r in rr], [r["H_max"] for r in rr],
                   color=col, marker=mk, s=55, label=tag, edgecolor="k", linewidth=0.4, zorder=3)
    # label points (nonlinear)
    for r in [r for r in rows if r["condition"] == "nonlinear"]:
        ax.annotate(r["chord"].split(" ")[0], (r["complexity"], r["H_max"]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    s = result["summary"]
    ax.set_xlabel("chord complexity  mean log2(p·q)  (dissonant →)")
    ax.set_ylabel("harmonicity H_max")
    ax.set_title(f"Study 20 — H tracks chord consonance\n"
                 f"ρ(H,complexity): linear={s['rho_H_vs_complexity_linear']:+.2f}, "
                 f"nonlinear={s['rho_H_vs_complexity_nonlinear']:+.2f}", fontweight="bold", fontsize=10)
    ax.legend()
    fig.tight_layout()
    C.save_fig(fig, "study20_musical_intermod")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

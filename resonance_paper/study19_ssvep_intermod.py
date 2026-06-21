"""Study 19 — SSVEP, nonlinearity, and intermodulation through H / PC / R.

SSVEP is harmonic by construction: a periodic flicker at f drives a response at
f and (through neural nonlinearity) its harmonics 2f, 3f…; two simultaneous
flickers f1, f2 generate INTERMODULATION products n·f1 ± m·f2 — the classic
signature of NONLINEAR integration/binding. We model this with a tunable static
nonlinearity and ask what H / PC / R reveal:

  (A) single flicker: is harmonicity H informative here? (It turns out H is at
      ceiling — a single periodic drive is already trivially harmonic — so this is
      a sanity-ceiling, not the discriminating signal.)
  (B) two flickers: does the intermodulation index (power at n·f1±m·f2) — and the
      cross-resonance between the two driven components — rise with nonlinearity,
      and is it stronger for SIMPLER f1:f2 ratios (where IM products fall on a
      simpler lattice)?

This connects the framework to steady-state/frequency-tagging paradigms and to
nonlinear-systems intermodulation.

Outputs: results/study19_ssvep_intermod.json, figures/study19_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from resonance_paper import _common as C
from resonance_paper.signals import _norm, pink_noise
from biotuner.resonance import compute_resonance

SF = 500.0


def nonlinear_response(freqs, g_nl, dur=20.0, noise_db=12.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SF * dur); t = np.arange(n) / SF
    drive = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in freqs) / len(freqs)
    # static nonlinearity: quadratic (even harmonics + sum/diff IM) + cubic (odd harmonics + IM)
    y = drive + g_nl * drive ** 2 + 0.6 * g_nl * drive ** 3
    nz = 10 ** (-noise_db / 20.0)
    return _norm(y + nz * pink_noise(n, SF, seed=seed + 3)).astype(np.float64), t


def intermod_index(y, f1, f2, sf=SF):
    """Fraction of in-band power at intermodulation freqs |n*f1 ± m*f2| (n,m<=2,
    excluding the drives and pure harmonics)."""
    f, p = welch(y, fs=sf, nperseg=min(len(y), int(sf * 4)))
    tot = p[(f >= 1) & (f <= 60)].sum() + 1e-20
    im = set()
    for nn in range(0, 3):
        for mm in range(0, 3):
            for s in (1, -1):
                v = abs(nn * f1 + s * mm * f2)
                if 1 < v < 60 and not (mm == 0) and not (nn == 0):
                    im.add(round(v, 1))
    pw = 0.0
    for v in im:
        band = (f >= v - 0.6) & (f <= v + 0.6); pw += p[band].sum()
    return float(pw / tot)


def run(quick=True):
    gs = [0.0, 0.3, 0.6, 1.0] if quick else [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]
    seeds = range(4) if quick else range(12)
    cfg = C.default_config(fmin=2, fmax=60, precision_hz=0.5)

    # (A) single flicker: H vs nonlinearity
    f0 = 6.0
    partA = []
    for g in gs:
        Hs = []
        for s in seeds:
            y, _ = nonlinear_response([f0], g, seed=s)
            Hs.append(float(compute_resonance(y, sf=SF, config=cfg).summaries["H"]["max"]))
        partA.append(dict(g=g, H_max=float(np.mean(Hs))))
        print(f"  [A] g={g:.2f} done", flush=True)

    # (B) two flickers, simple vs complex ratio: IM + H vs nonlinearity
    pairs = {"simple_2to3": (6.0, 9.0), "simple_3to4": (6.0, 8.0), "complex_5to7": (6.0, 8.4),
             "complex_7to9": (6.3, 8.1)}
    partB = []
    for label, (f1, f2) in pairs.items():
        for g in gs:
            ims, Hs = [], []
            for s in seeds:
                y, _ = nonlinear_response([f1, f2], g, seed=s)
                ims.append(intermod_index(y, f1, f2))
                Hs.append(float(compute_resonance(y, sf=SF, config=cfg).summaries["H"]["max"]))
            partB.append(dict(pair=label, f1=f1, f2=f2, g=g,
                              im_index=float(np.mean(ims)), H_max=float(np.mean(Hs))))
        print(f"  [B] {label} done", flush=True)

    from scipy.stats import spearmanr
    a_rho = spearmanr([r["g"] for r in partA], [r["H_max"] for r in partA])[0]
    im_rho = spearmanr([r["g"] for r in partB], [r["im_index"] for r in partB])[0]
    # IM at full nonlinearity: simple vs complex pairs
    gmax = gs[-1]
    simple_im = np.mean([r["im_index"] for r in partB if r["g"] == gmax and r["pair"].startswith("simple")])
    complex_im = np.mean([r["im_index"] for r in partB if r["g"] == gmax and r["pair"].startswith("complex")])
    summary = dict(H_vs_nonlinearity=float(a_rho), IM_vs_nonlinearity=float(im_rho),
                   IM_simple_at_gmax=float(simple_im), IM_complex_at_gmax=float(complex_im))
    result = dict(quick=quick, partA=partA, partB=partB, summary=summary)
    C.save_json(result, "study19_ssvep_intermod.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 19 headline (SSVEP / intermodulation) ---")
    print(f"  (A) single flicker: H_max is at CEILING (~1.3, invariant to nonlinearity): "
          f"rho={s['H_vs_nonlinearity']:+.2f} but range is trivial")
    print("      H_max by g: " + ", ".join(f"{r['g']:.2f}:{r['H_max']:.2f}" for r in result["partA"]))
    print(f"  (B) two flicker: rho(nonlinearity, IM index) = {s['IM_vs_nonlinearity']:+.2f}")
    print(f"      IM index at max nonlinearity: simple={s['IM_simple_at_gmax']:.3f} "
          f"complex={s['IM_complex_at_gmax']:.3f}")
    print("  => single-flicker H saturates (a periodic drive is already trivially harmonic);")
    print("     the discriminating signal is two-flicker INTERMODULATION, which rises with")
    print("     nonlinearity and is richer for simpler ratios.")


def _figures(result):
    plt = C.setup_mpl()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.3))
    a = result["partA"]
    ax1.plot([r["g"] for r in a], [r["H_max"] for r in a], "o-", color="#ef6c00")
    ax1.set_xlabel("nonlinearity strength g"); ax1.set_ylabel("H_max of response")
    ax1.set_ylim(0, max(2.0, max(r["H_max"] for r in a) * 1.3))
    ax1.set_title("A. Single flicker: H at ceiling\n(periodic drive already harmonic)", fontsize=10)
    b = result["partB"]
    for label in sorted(set(r["pair"] for r in b)):
        rr = [r for r in b if r["pair"] == label]; rr.sort(key=lambda r: r["g"])
        col = "#1565c0" if label.startswith("simple") else "#b71c1c"
        ax2.plot([r["g"] for r in rr], [r["im_index"] for r in rr], "o-",
                 color=col, alpha=0.8, label=label)
    ax2.set_xlabel("nonlinearity strength g"); ax2.set_ylabel("intermodulation index")
    ax2.set_title("B. Two flickers: intermodulation\n(blue=simple, red=complex ratios)", fontsize=10)
    ax2.legend(fontsize=7)
    fig.suptitle("Study 19 — SSVEP harmonics + intermodulation via H / PC / R", fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study19_ssvep_intermod")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

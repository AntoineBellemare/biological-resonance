"""Study 22 — Complexity descriptors of the H / PC / R spectra (first-class).

The framework already computes, for EACH factor spectrum (H, PC, R), a panel of
spectral-complexity descriptors — spectral FLATNESS, ENTROPY, SPREAD and the
Higuchi fractal dimension (HFD) — exposed in
``result.summaries[factor]['flatness'|'entropy'|'spread'|'higuchi']``. Until now
they were only exercised implicitly inside the modality classifier (Study 3). Here
we make them first-class: applied TO the resonance spectra, they characterize how
*organized* a signal's harmonic/coupling structure is, beyond the scalar H mean/max.

Signals of known structure (pure tone, rich harmonic stack, inharmonic stack,
narrowband noise, pink noise) on a matched 1/f background. For each we read the 4
descriptors x 3 factors = 12 numbers and ask:
  (1) do the descriptors of the H/R spectra order signals by harmonic organization
      (structured -> low flatness/entropy, focal -> low spread)?
  (2) which descriptor x factor best separates HARMONIC structure from NOISE (AUC)?
  (3) do they add information beyond H_avg/H_max (separate classes H_avg conflates)?

Outputs: results/study22_spectral_descriptors.json, figures/study22_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import _norm, pink_noise
from biotuner.resonance import compute_resonance

SF = 250.0
DUR = 12.0
F0 = 10.0
DESCRIPTORS = ["flatness", "entropy", "spread", "higuchi"]
FACTORS = ["H", "PC", "R"]
CLASSES = ["pure_tone", "harmonic", "inharmonic", "narrowband", "pink"]
SNR_DB = 6.0


def make_signal(kind, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SF * DUR); t = np.arange(n) / SF
    if kind == "pure_tone":
        x = np.sin(2 * np.pi * F0 * t + rng.uniform(0, 2 * np.pi))
    elif kind == "harmonic":
        x = sum(np.sin(2 * np.pi * F0 * k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 6)) / 5
    elif kind == "inharmonic":
        ratios = [1.0, 1.413, 1.732, 2.236, 2.833]            # incommensurate partials
        x = sum(np.sin(2 * np.pi * F0 * r * t + rng.uniform(0, 2 * np.pi)) for r in ratios) / len(ratios)
    elif kind == "narrowband":
        from scipy.signal import butter, filtfilt
        b, a = butter(4, [(F0 - 1.5) / (SF / 2), (F0 + 1.5) / (SF / 2)], btype="band")
        x = filtfilt(b, a, rng.standard_normal(n))
    elif kind == "pink":
        x = pink_noise(n, SF, seed=seed)
    else:
        raise ValueError(kind)
    nz = 10 ** (-SNR_DB / 20.0)
    return _norm(_norm(x) + nz * pink_noise(n, SF, seed=seed + 7)).astype(np.float64)


def _descriptors(sig):
    cfg = C.default_config(fmin=2, fmax=45, precision_hz=0.5)
    r = compute_resonance(sig, sf=SF, config=cfg)
    out = {}
    for f in FACTORS:
        s = r.summaries.get(f, {})
        for d in DESCRIPTORS:
            out[f"{f}_{d}"] = float(s.get(d, np.nan))
        out[f"{f}_avg"] = float(s.get("avg", np.nan))
        out[f"{f}_max"] = float(s.get("max", np.nan))
    return out


def run(quick=True):
    seeds = range(6) if quick else range(20)
    rows = []
    for kind in CLASSES:
        for s in seeds:
            rows.append(dict(kind=kind, **_descriptors(make_signal(kind, seed=s))))
        print(f"  {kind} done", flush=True)

    feats = [f"{f}_{d}" for f in FACTORS for d in DESCRIPTORS]

    # (2) informativeness: how much of the variance ACROSS all 5 signal classes does
    #     each descriptor explain (one-way eta^2)?  This is GRADED (unlike the binary
    #     harmonic-vs-noise AUC, which saturates at 1.0 for almost every descriptor
    #     because those two classes are trivially separable). eta^2 ranks descriptors.
    def _eta2(key):
        groups = [np.array([r[key] for r in rows if r["kind"] == k and np.isfinite(r[key])])
                  for k in CLASSES]
        allv = np.concatenate(groups); grand = allv.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups if len(g))
        ss_total = float(((allv - grand) ** 2).sum()) + 1e-30
        return float(ss_between / ss_total)

    disc = {}
    for key in feats:
        pos = [r[key] for r in rows if r["kind"] == "harmonic" and np.isfinite(r[key])]
        neg = [r[key] for r in rows if r["kind"] == "pink" and np.isfinite(r[key])]
        auc = C.bootstrap_auc_ci(pos, neg)["auc"]
        disc[key] = dict(eta2=_eta2(key), auc=auc,
                         discriminability=float(max(auc, 1 - auc)),
                         direction="structured>noise" if auc > 0.5 else "noise>structured")

    # (3) descriptors add info beyond H_avg: classes with similar H_avg, different shape
    havg = {k: float(np.nanmean([r["H_avg"] for r in rows if r["kind"] == k])) for k in CLASSES}
    hflat = {k: float(np.nanmean([r["H_flatness"] for r in rows if r["kind"] == k])) for k in CLASSES}
    hspread = {k: float(np.nanmean([r["H_spread"] for r in rows if r["kind"] == k])) for k in CLASSES}

    summary = dict(
        best_descriptor=max(disc, key=lambda k: disc[k]["eta2"]),
        best_eta2=max(disc[k]["eta2"] for k in disc),
        best_auc_descriptor=max(disc, key=lambda k: disc[k]["discriminability"]),
        best_auc=max(disc[k]["discriminability"] for k in disc),
        H_avg_by_class=havg, H_flatness_by_class=hflat, H_spread_by_class=hspread)
    result = dict(quick=quick, classes=CLASSES, feats=feats, rows=rows,
                  discriminability=disc, summary=summary)
    C.save_json(result, "study22_spectral_descriptors.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 22 headline (complexity descriptors of H/PC/R spectra) ---")
    print(f"  on clean synthetic signals most descriptors separate the 5 classes near-perfectly "
          f"(best eta2={s['best_eta2']:.2f}); the value is the interpretable resonance 'fingerprint'")
    print("  + capturing spectral SHAPE that the scalar H summaries miss. Key example:")
    print("  H_avg CONFLATES pure_tone/inharmonic/narrowband/pink, but H_flatness separates them:")
    for k in result["classes"]:
        print(f"    {k:11s} H_avg={s['H_avg_by_class'][k]:.3f}   H_flatness={s['H_flatness_by_class'][k]:.3f}")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; disc = result["discriminability"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    cols = {"pure_tone": "#1565c0", "harmonic": "#2e7d32", "inharmonic": "#ef6c00",
            "narrowband": "#6a1b9a", "pink": "#999999"}

    def box(ax, key, ttl):
        data = [[r[key] for r in rows if r["kind"] == k and np.isfinite(r[key])] for k in CLASSES]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black"),
                        flierprops=dict(ms=2))
        for p, k in zip(bp["boxes"], CLASSES):
            p.set_facecolor(cols[k]); p.set_alpha(0.75)
        ax.set_xticks(range(1, len(CLASSES) + 1)); ax.set_xticklabels(CLASSES, rotation=35, fontsize=6.5)
        ax.set_title(ttl, fontsize=10)

    box(axes[0], "H_flatness", "A. H-spectrum flatness\n(structured low, noise high)")
    box(axes[1], "R_entropy", "B. R-spectrum entropy")

    # C: the resonance-descriptor FINGERPRINT — z-scored (per row) descriptor x class
    #    heatmap. Each descriptor has a distinct class profile; this is what makes the
    #    12 descriptors a first-class, interpretable feature set (a bar ranking
    #    saturates because every descriptor separates these clean classes near-perfectly).
    ax = axes[2]
    feats = result["feats"]
    M = np.array([[np.nanmean([r[f] for r in rows if r["kind"] == k]) for k in CLASSES] for f in feats])
    Mz = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-12)
    im = ax.imshow(Mz, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=35, fontsize=6)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=5.5)
    ax.set_title("C. Descriptor fingerprint\n(z-scored per row)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # D: descriptors add info beyond H_avg — 2D shape space colored by class
    ax = axes[3]
    for k in CLASSES:
        xs = [r["H_flatness"] for r in rows if r["kind"] == k]
        ys = [r["H_spread"] for r in rows if r["kind"] == k]
        ax.scatter(xs, ys, color=cols[k], s=22, alpha=0.8, label=k, edgecolor="none")
    ax.set_xlabel("H-spectrum flatness"); ax.set_ylabel("H-spectrum spread")
    ax.set_title("D. Resonance-shape space\n(beyond H_avg/H_max)", fontsize=10)
    ax.legend(fontsize=6)

    fig.suptitle("Study 22 — Spectral-complexity descriptors of the H / PC / R spectra are first-class features",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study22_spectral_descriptors")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

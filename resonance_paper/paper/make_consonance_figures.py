"""Publication figures for the CONSONANCE paper (cross-paradigm: one harmonic-resonance
descriptor recovers consonance structure across chords, SSVEP intermodulation, and the FFR).

Five title-less composite figures (paper/figures/cons_Fig1-5.{png,pdf}, 600 DPI); the
manuscript supplies captions. Build AFTER the relevant studies have run --paper.

    python -m resonance_paper.paper.make_consonance_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mfig
_mfig.Figure.suptitle = lambda self, *a, **k: None

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, SKY, YELLOW, GREY = "#CC79A7", "#56B4E9", "#F0E442", "#999999"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8,
    "axes.labelsize": 9, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
})
COL2 = 7.2


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def panel(ax, letter):
    ax.text(-0.17, 1.12, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png"); fig.savefig(FIGDIR / f"{name}.pdf"); plt.close(fig)
    print(f"  wrote paper/figures/{name}.png (+pdf)")


# --------------------------------------------------------------------------- F1
def fig1_construct():
    s = load("study17_tripartite_dissociation.json")
    if not s:
        return
    rows = s["rows"]; sp = s.get("specificity", {})
    comps = sorted(set(r["complexity"] for r in rows))
    from matplotlib.colors import Normalize
    nrm = Normalize(min(comps), max(comps))
    fig, ax = plt.subplots(1, 2, figsize=(COL2 * 0.75, 2.6))
    for c in comps:
        sub = sorted([r for r in rows if r["complexity"] == c], key=lambda r: r["kappa"])
        ax[0].plot([r["kappa"] for r in sub], [r["H"] for r in sub], "-o", ms=2.5, lw=1.1,
                   color=plt.cm.viridis(nrm(c)), alpha=0.9)
    ax[0].set_xlabel("phase-locking κ"); ax[0].set_ylabel("harmonicity H")
    ax[0].set_title("H is phase-blind"); panel(ax[0], "A")
    keys = ["H", "PC", "R"]; cols = [ORANGE, PURPLE, RED]
    aucs = [sp.get(k, {}).get("auc", np.nan) for k in keys]
    ax[1].bar(range(3), aucs, color=cols, alpha=0.9)
    ax[1].axhline(0.5, color="k", ls="--", lw=0.7); ax[1].set_ylim(0, 1.05)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(keys); ax[1].set_ylabel("specificity AUC")
    ax[1].set_title("R interpretive (≤ PC)"); panel(ax[1], "B")
    fig.tight_layout(); save(fig, "cons_Fig1_construct")


# --------------------------------------------------------------------------- F2
def fig2_chords():
    s = load("study20_musical_intermod.json")
    if not s:
        return
    rows = s["rows"]; sm = s["summary"]
    fig, ax = plt.subplots(1, 2, figsize=(COL2 * 0.82, 2.7))
    for tag, col, mk in [("linear", GREY, "o"), ("nonlinear", RED, "s")]:
        rr = [r for r in rows if r["condition"] == tag]
        ax[0].scatter([r["complexity"] for r in rr], [r["H_max"] for r in rr], color=col, marker=mk,
                      s=42, edgecolor="k", linewidth=0.4, label=tag, zorder=3)
    ax[0].set_xlabel("chord complexity  log₂(p·q)  (dissonant →)"); ax[0].set_ylabel("harmonicity H_max")
    ax[0].set_title(f"H tracks consonance\nρ_nl={sm['rho_H_vs_complexity_nonlinear']:+.2f} "
                    f"(p={sm.get('nonlinear_p', float('nan')):.2g})")
    ax[0].legend(fontsize=6.5); panel(ax[0], "A")
    # sharpening with CI
    sd = sm.get("sharpen_delta", float("nan")); sci = sm.get("sharpen_ci", [np.nan, np.nan])
    ax[1].bar([0], [sd], yerr=[[sd - sci[0]], [sci[1] - sd]], color=ORANGE, width=0.5, capsize=4)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xticks([0]); ax[1].set_xticklabels(["nonlinear − linear"])
    ax[1].set_ylabel("|ρ| sharpening (combination tones)")
    sig = "yes" if sm.get("nonlinearity_sharpens") else "n.s."
    ax[1].set_title(f"Combination tones sharpen?\n{sig} (Δ|ρ|={sd:+.2f})"); panel(ax[1], "B")
    fig.tight_layout(); save(fig, "cons_Fig2_chords")


# --------------------------------------------------------------------------- F3
def fig3_ssvep():
    s = load("study19_ssvep_intermod.json")
    if not s:
        return
    A = s["partA"]; B = s["partB"]; sm = s["summary"]
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))
    ax[0].plot([r["g"] for r in A], [r["H_max"] for r in A], "o-", color=ORANGE)
    ax[0].set_ylim(0, max(2.0, max(r["H_max"] for r in A) * 1.3))
    ax[0].set_xlabel("nonlinearity g"); ax[0].set_ylabel("single-flicker H_max")
    ax[0].set_title("Single flicker: H at ceiling\n(sanity)"); panel(ax[0], "A")
    # IM index + n:m PC vs g (pooled mean over pairs)
    gs = sorted(set(r["g"] for r in B))
    im = [np.mean([r["im_index"] for r in B if r["g"] == g]) for g in gs]
    pc = [np.mean([r["PC"] for r in B if r["g"] == g]) for g in gs]
    ax[1].plot(gs, np.array(im) / (max(im) + 1e-9), "s-", color=BLUE, label="IM index (norm)")
    ax[1].plot(gs, pc, "v-", color=PURPLE, label="n:m phase coupling")
    ax[1].set_xlabel("nonlinearity g"); ax[1].set_ylabel("intermodulation / PC")
    ax[1].set_title(f"IM = n:m coupling\nρ(g,PC)={sm.get('PC_vs_nonlinearity', float('nan')):+.2f}")
    ax[1].legend(fontsize=6.5); panel(ax[1], "B")
    # simple vs complex PC at gmax
    xs = [0, 1]
    ax[2].bar(xs, [sm.get("PC_simple_at_gmax", np.nan), sm.get("PC_complex_at_gmax", np.nan)],
              color=[BLUE, RED], alpha=0.85, width=0.6)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(["simple\nratios", "complex\nratios"])
    ax[2].set_ylabel("n:m PC at max nonlinearity")
    ax[2].set_title("Simpler ratios couple more"); panel(ax[2], "C")
    fig.tight_layout(); save(fig, "cons_Fig3_ssvep")


# --------------------------------------------------------------------------- F4
def fig4_ffr():
    s = load("study20b_ffr_consonance.json")
    if not s:
        return
    rows = s["rows"]; sm = s["summary"]; bb = s.get("brain_behavior", {})
    fig, ax = plt.subplots(1, 4, figsize=(COL2 * 1.4, 2.7))
    order = ["CC", "DC", "CI", "DI"]; cols = {"CC": BLUE, "DC": RED, "CI": SKY, "DI": "#e57373"}
    for xi, c in enumerate(order):
        vals = [r["H"] for r in rows if r["cond"] == c and r["run"] == 1 and np.isfinite(r["H"])]
        ax[0].scatter(np.full(len(vals), xi) + np.random.uniform(-0.08, 0.08, len(vals)), vals,
                      s=14, color=cols[c], alpha=0.6, edgecolor="none")
        ax[0].bar(xi, np.mean(vals), width=0.6, color=cols[c], alpha=0.25, zorder=0)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["CC", "DC", "CI", "DI"])
    ax[0].set_ylabel("neural FFR harmonicity"); ax[0].set_title("Consonant > dissonant\n(incl. missing-fundamental)")
    panel(ax[0], "A")
    # band split: CI-DI low (silent->neural) vs high
    lo, hi = sm["incomplete_low_delta"], sm["incomplete_high_delta"]
    ax[1].bar([0, 1], [lo, hi], color=[GREEN, GREY], alpha=0.85)
    ax[1].axhline(0, color="k", lw=0.6)
    for xi, (d, pv) in enumerate([(lo, sm["incomplete_low_p"]), (hi, sm["incomplete_high_p"])]):
        ax[1].annotate(f"p={pv:.1g}", (xi, d), ha="center", va="bottom" if d >= 0 else "top", fontsize=7)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["low\n(silent→neural)", "high\n(leakage zone)"], fontsize=6.5)
    ax[1].set_ylabel("CI − DI harmonicity"); ax[1].set_title("Effect is in the silent band\n(neural, not leakage)")
    panel(ax[1], "B")
    # neural vs acoustic
    bars = {"acoustic": sm["stim_consonance_effect"], "neural\ncomplete": sm["neural_consonance_complete_delta"],
            "neural\nincomplete": sm["neural_consonance_incomplete_delta"]}
    ax[2].bar(range(3), list(bars.values()), color=[GREY, BLUE, SKY], alpha=0.9)
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(list(bars), fontsize=6.5)
    ax[2].set_ylabel("consonant − dissonant H"); ax[2].set_title("Neural effect beyond\nacoustic leakage"); panel(ax[2], "C")
    # brain-behavior (honest null): neural advantage vs amma
    inc = {}
    a = {r["listener"]: r["H"] for r in rows if r["cond"] == "CI" and r["run"] == 1 and np.isfinite(r["H"])}
    b = {r["listener"]: r["H"] for r in rows if r["cond"] == "DI" and r["run"] == 1 and np.isfinite(r["H"])}
    adv = {l: a[l] - b[l] for l in set(a) & set(b)}
    rho = bb.get("incomplete_vs_amma", {}).get("rho", float("nan"))
    pp = bb.get("incomplete_vs_amma", {}).get("p", float("nan"))
    ax[3].scatter(range(len(adv)), [adv[l] for l in sorted(adv)], s=14, color=GREY, alpha=0.6)
    ax[3].axhline(np.mean(list(adv.values())), color=RED, ls="--", lw=0.8)
    ax[3].set_xlabel("listener"); ax[3].set_ylabel("CI−DI advantage")
    ax[3].set_title(f"No musicianship link\n(vs AMMA ρ={rho:+.2f}, p={pp:.2g})"); panel(ax[3], "D")
    fig.tight_layout(); save(fig, "cons_Fig4_ffr")


# --------------------------------------------------------------------------- F5
def fig5_synthesis():
    s20 = load("study20_musical_intermod.json"); s19 = load("study19_ssvep_intermod.json")
    s20b = load("study20b_ffr_consonance.json")
    fig, ax = plt.subplots(figsize=(COL2 * 0.8, 2.7))
    labels, effects, texts = [], [], []
    if s20:
        labels.append("chords\n|ρ(H,consonance)|"); effects.append(abs(s20["summary"]["rho_H_vs_complexity_nonlinear"]))
        texts.append(f"p={s20['summary'].get('nonlinear_p', float('nan')):.2g}")
    if s19:
        labels.append("SSVEP\nρ(nonlin, IM)"); effects.append(abs(s19["summary"].get("IM_vs_nonlinearity", np.nan)))
        texts.append("two-flicker")
    if s20b:
        f = s20b["contrasts"]["active"]["consonance_incomplete"]
        labels.append("FFR\nCI>DI (frac subj.)"); effects.append(f.get("frac_a_gt_b", np.nan))
        texts.append(f"p={f.get('p', float('nan')):.2g}")
    xs = np.arange(len(labels))
    ax.bar(xs, effects, color=[ORANGE, PURPLE, GREEN][:len(labels)], alpha=0.9)
    for x, e, t in zip(xs, effects, texts):
        ax.annotate(t, (x, e), ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=6.8); ax.set_ylim(0, 1.05)
    ax.set_ylabel("standardized effect (0–1)")
    ax.set_title("Cross-paradigm: one descriptor,\nsame consonance direction"); panel(ax, "A")
    fig.tight_layout(); save(fig, "cons_Fig5_synthesis")


def main():
    fig1_construct(); fig2_chords(); fig3_ssvep(); fig4_ffr(); fig5_synthesis()
    print(f"  Done -> {FIGDIR}")


if __name__ == "__main__":
    main()

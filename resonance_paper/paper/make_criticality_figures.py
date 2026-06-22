"""Publication figures for the CRITICALITY paper (discovery: spectral harmonicity H
is an observable of proximity to criticality).

Six composite figures (paper/figures/crit_Fig1-6.{png,pdf}, 600 DPI), title-less
(the manuscript supplies captions). Build AFTER:
  python -m resonance_paper.run_all --paper            # studies 10,11,12
  python -m resonance_paper.study16_criticality_indepth sleep --paper
  python -m resonance_paper.study16_criticality_indepth propofol --paper

    python -m resonance_paper.paper.make_criticality_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure as _mfig
_mfig.Figure.suptitle = lambda self, *a, **k: None   # captions in the manuscript, not the image

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


def _norm(v):
    v = np.asarray(v, float); m = np.nanmax(np.abs(v))
    return v / (m + 1e-12)


# --------------------------------------------------------------------------- F1
def fig1_schematic():
    """Conceptual: the falsifiable law. H peaks at criticality (co-located with the
    branching/susceptibility peak); R only moves where oscillations coexist."""
    fig, ax = plt.subplots(figsize=(COL2, 2.7))
    x = np.linspace(0, 2, 400)
    bell = np.exp(-((x - 1.0) ** 2) / (2 * 0.13 ** 2))
    ax.axvspan(0, 0.85, color=BLUE, alpha=0.05); ax.axvspan(1.15, 2, color=RED, alpha=0.05)
    ax.plot(x, bell, color="k", lw=1.6, label="criticality (m̂→1, susceptibility)")
    ax.plot(x, 0.92 * np.exp(-((x - 1.02) ** 2) / (2 * 0.16 ** 2)), color=ORANGE, lw=2.2,
            label="harmonicity H (prediction)")
    Rcurve = 0.9 / (1 + np.exp(-(x - 1.25) / 0.12)) * (x > 0.95)
    ax.plot(x, Rcurve, color=PURPLE, lw=1.8, ls="--", label="phase coupling / R (needs oscillation)")
    ax.axvline(1.0, color="grey", ls=":", lw=0.9)
    ax.text(0.42, 1.02, "subcritical", color=BLUE, fontsize=7.5, ha="center")
    ax.text(1.0, 1.08, "critical", color="k", fontsize=8, ha="center", fontweight="bold")
    ax.text(1.6, 1.02, "supercritical /\nsynchronized", color=RED, fontsize=7.5, ha="center")
    ax.set_xlabel("control parameter (branching ratio / coupling / gain)")
    ax.set_ylabel("normalized observable"); ax.set_ylim(0, 1.2); ax.set_xticks([])
    ax.legend(fontsize=6.4, loc="upper right")
    panel(ax, "A")
    fig.tight_layout(); save(fig, "crit_Fig1_schematic")


# --------------------------------------------------------------------------- F2
def fig2_branching():
    s = load("study10_criticality.json")
    if not s:
        return
    rows = s["rows"]; sm = s["summary"]; sig = [r["sigma"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))

    # A: criticality markers
    for k, c, mk, lab in [("susceptibility", BLUE, "o", "susceptibility"),
                          ("powerlaw_r2", GREEN, "^", "avalanche power-law R²"),
                          ("branching_est", GREY, "s", "branching m̂")]:
        ax[0].plot(sig, _norm([r[k] for r in rows]), mk + "-", color=c, ms=4, label=lab)
    ax[0].axvline(1.0, color="grey", ls="--", lw=0.8)
    ax[0].set_xlabel("branching ratio  σ"); ax[0].set_ylabel("normalized")
    ax[0].set_title("Criticality markers peak at σ≈1"); ax[0].legend(fontsize=6)
    panel(ax[0], "A")

    # B: H_max inverted-U + R floor
    H = np.array([r["H_max"] for r in rows]); He = np.array([r.get("H_max_sem", 0) for r in rows])
    ax[1].errorbar(sig, H, yerr=He, fmt="d-", color=ORANGE, capsize=2, label="harmonicity H")
    cH = sm.get("sigma_at_max_H_ci")
    if cH:
        ax[1].axvspan(cH[0], cH[1], color=ORANGE, alpha=0.13, label="H peak 95% CI")
    ax[1].axvline(1.0, color="grey", ls="--", lw=0.8)
    ax[1].text(0.03, 0.04, "R ≈ 0 throughout\n(avalanches non-oscillatory)",
               transform=ax[1].transAxes, fontsize=6.3, color=PURPLE, va="bottom")
    ax[1].set_xlabel("branching ratio  σ"); ax[1].set_ylabel("harmonicity H_max")
    ax[1].set_title("H peaks at criticality"); ax[1].legend(fontsize=6)
    panel(ax[1], "B")

    # C: co-location statistic
    ax[2].plot(sig, _norm([r["susceptibility"] for r in rows]), "o-", color=BLUE, ms=3, label="susceptibility")
    ax[2].plot(sig, _norm([r["H_max"] for r in rows]), "d-", color=ORANGE, ms=3, label="H")
    ax[2].axvline(1.0, color="grey", ls="--", lw=0.8)
    rho = sm.get("H_tracks_prox_rho", float("nan")); pp = sm.get("H_tracks_prox_p", float("nan"))
    coinc = sm.get("H_peak_coincides")
    txt = (f"peaks {'coincide' if coinc else 'differ'}\nH tracks m̂-prox: ρ={rho:+.2f}"
           + (f", p={pp:.2g}" if pp == pp else ""))
    ax[2].text(0.03, 0.05, txt, transform=ax[2].transAxes, fontsize=6.5, va="bottom")
    ax[2].set_xlabel("branching ratio  σ"); ax[2].set_ylabel("normalized")
    ax[2].set_title("H peak co-locates with criticality"); ax[2].legend(fontsize=6)
    panel(ax[2], "C")
    fig.tight_layout(); save(fig, "crit_Fig2_branching")


# --------------------------------------------------------------------------- F3
def fig3_reservoir():
    s = load("study11_reservoir_criticality.json")
    if not s:
        return
    rows = s["rows"]; sm = s["summary"]; rho = [r["rho"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))

    ax[0].errorbar(rho, [r["lyapunov"] for r in rows], yerr=[r.get("lyapunov_sem", 0) for r in rows],
                   fmt="o-", color="k", capsize=2)
    ax[0].axhline(0, color="grey", lw=0.6)
    rc, cc = sm.get("rho_critical"), sm.get("rho_critical_ci")
    if rc:
        ax[0].axvline(rc, color=RED, ls="--", lw=1.0, label=f"edge ρ_c={rc:.2f}")
        if cc:
            ax[0].axvspan(cc[0], cc[1], color=RED, alpha=0.12)
    ax[0].set_xlabel("spectral radius  ρ"); ax[0].set_ylabel("Lyapunov exponent")
    ax[0].set_title("Edge of chaos"); ax[0].legend(fontsize=6); panel(ax[0], "A")

    Hg = np.array([r["H_gain"] for r in rows]); Hge = np.array([r.get("H_gain_sem", 0) for r in rows])
    ax[1].errorbar(rho, Hg, yerr=Hge, fmt="s-", color=RED, capsize=2)
    ax[1].axhline(0, color="grey", ls=":", lw=0.8)
    onset = sm.get("generation_onset_rho")
    if onset and onset == onset:
        ax[1].axvline(onset, color=GREEN, ls="--", lw=1.0, label=f"onset ρ={onset:.2f}")
    ax[1].set_xlabel("spectral radius  ρ"); ax[1].set_ylabel("H gain over noise input")
    ax[1].set_title("Harmonicity generated from noise"); ax[1].legend(fontsize=6); panel(ax[1], "B")

    ax[2].plot(rho, _norm([r["H_gain"] for r in rows]), "s-", color=RED, ms=3, label="generation (H gain)")
    ax[2].plot(rho, _norm([r["memory_capacity"] for r in rows]), "o-", color=BLUE, ms=3, label="computation (MC)")
    d = sm.get("Hgain_minus_MC_ci"); sep = sm.get("generation_separated_from_computation")
    if d:
        ax[2].text(0.03, 0.05, f"peaks {'separated' if sep else 'overlap'}\nΔ=[{d[0]:+.1f},{d[1]:+.1f}]",
                   transform=ax[2].transAxes, fontsize=6.5, va="bottom")
    ax[2].set_xlabel("spectral radius  ρ"); ax[2].set_ylabel("normalized")
    ax[2].set_title("Generation ≠ computation"); ax[2].legend(fontsize=6); panel(ax[2], "C")
    fig.tight_layout(); save(fig, "crit_Fig3_reservoir")


# --------------------------------------------------------------------------- F4
def fig4_ei():
    s = load("study12_ei_network.json")
    if not s:
        return
    rows = s["rows"]; sm = s["summary"]; g = [r["g"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))
    gc, cgc = sm.get("g_critical"), sm.get("g_critical_ci")

    for k, c, mk, lab in [("order_param", "#455a64", "o", "order parameter"),
                          ("susceptibility_norm", BLUE, "^", "susceptibility (var/mean²)"),
                          ("autocorr_time", GREEN, "x", "autocorr time")]:
        if any(k in r for r in rows):
            ax[0].plot(g, _norm([r.get(k, np.nan) for r in rows]), mk + "-", color=c, ms=4, label=lab)
    if gc:
        ax[0].axvline(gc, color=RED, ls="--", lw=1.0, label=f"g_c={gc:.2f}")
        if cgc:
            ax[0].axvspan(cgc[0], cgc[1], color=RED, alpha=0.12)
    ax[0].set_xlabel("coupling gain  g"); ax[0].set_ylabel("normalized")
    ax[0].set_title("Edge of synchronization"); ax[0].legend(fontsize=6); panel(ax[0], "A")

    for k, c, mk, lab in [("crossEI_PC", PURPLE, "v", "E↔I phase coupling PC"),
                          ("crossEI_R", RED, "s", "E↔I resonance R")]:
        ax[1].errorbar(g, [r[k] for r in rows], yerr=[r.get(k + "_sem", 0) for r in rows],
                       fmt=mk + "-", color=c, capsize=2, label=lab)
    bp = sm.get("baseline_PC")
    if bp is not None:
        ax[1].axhline(bp, color=PURPLE, ls=":", lw=1.0, label=f"async baseline {bp:.2f}")
    if gc:
        ax[1].axvline(gc, color=RED, ls="--", lw=1.0)
    ax[1].set_xlabel("coupling gain  g"); ax[1].set_ylabel("coupling (PLV) / R")
    dpc = sm.get("delta_PC_at_g_critical", float("nan"))
    ax[1].set_title(f"PC rises at the onset (ΔPC=+{dpc:.2f})"); ax[1].legend(fontsize=6); panel(ax[1], "B")

    ax[2].errorbar(g, [r["crossEI_H"] for r in rows], yerr=[r.get("crossEI_H_sem", 0) for r in rows],
                   fmt="d-", color=ORANGE, capsize=2, label="harmonicity H")
    if gc:
        ax[2].axvline(gc, color=RED, ls="--", lw=1.0, label=f"g_c={gc:.2f}")
    ax[2].set_xlabel("coupling gain  g"); ax[2].set_ylabel("E↔I harmonicity H")
    ax[2].set_title("R is placeable here (oscillatory)"); ax[2].legend(fontsize=6); panel(ax[2], "C")
    fig.tight_layout(); save(fig, "crit_Fig4_ei_network")


# --------------------------------------------------------------------------- F5
def fig5_realdata_tension():
    s14 = load("study14_sleep.json")
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))

    # A: sleep by-state H_max (the reversal: highest in deep sleep) + m̂ per state
    if s14 and "by_state" in s14:
        bs = s14["by_state"]; order = [k for k in ["W", "Wake", "N1", "N2", "N3", "REM"] if k in bs] or list(bs)
        xs = np.arange(len(order))
        Hm = [bs[k].get("H_max", np.nan) for k in order]
        ax[0].bar(xs, Hm, color=ORANGE, alpha=0.85)
        ax[0].set_xticks(xs); ax[0].set_xticklabels(order, fontsize=6.5)
        ax[0].set_ylabel("harmonicity H_max"); ax[0].set_title("Sleep: H highest in deep N3\n(the reversal)")
        axb = ax[0].twinx(); axb.plot(xs, [bs[k].get("m_hat", np.nan) for k in order], "k.-", ms=5)
        axb.set_ylabel("branching m̂", fontsize=7); axb.spines["top"].set_visible(False)
    panel(ax[0], "A")

    # B: per-subject rho(H, m̂-proximity) across datasets — null/negative (reversal/boundary)
    labels, rhos, los, his = [], [], [], []
    for fn, lab in [("study14_sleep.json", "sleep"), ("study13_anesthesia.json", "propofol"),
                    ("study15_deep_anesthesia.json", "deep GA")]:
        d = load(fn)
        pr = (d or {}).get("primary", {}).get("H_max_vs_prox_m", {})
        if pr:
            labels.append(lab); rhos.append(pr.get("mean_rho", np.nan))
            los.append(pr.get("mean_rho", np.nan) - pr.get("lo", np.nan))
            his.append(pr.get("hi", np.nan) - pr.get("mean_rho", np.nan))
    xs = np.arange(len(labels))
    ax[1].bar(xs, rhos, yerr=[los, his], color=GREY, alpha=0.85, capsize=3)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("ρ(H_max, m̂-proximity)")
    ax[1].set_title("Raw-EEG H vs criticality:\nnull / reversed"); panel(ax[1], "B")

    # C: boundary cases — m̂ traversal range per dataset (sedation/deep-GA barely move)
    rng_lab, rng_val = [], []
    for fn, lab in [("study14_sleep.json", "sleep"), ("study13_anesthesia.json", "propofol"),
                    ("study15_deep_anesthesia.json", "deep GA")]:
        d = load(fn)
        bs = (d or {}).get("by_state", {})
        ms = [v.get("m_hat") for v in bs.values() if v.get("m_hat") is not None]
        if ms:
            rng_lab.append(lab); rng_val.append(max(ms) - min(ms))
    xs = np.arange(len(rng_lab))
    ax[2].bar(xs, rng_val, color=SKY, alpha=0.85)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(rng_lab, fontsize=7)
    ax[2].set_ylabel("m̂ traversal range (max−min)")
    ax[2].set_title("Boundary cases barely\ntraverse criticality"); panel(ax[2], "C")
    fig.tight_layout(); save(fig, "crit_Fig5_realdata_tension")


# --------------------------------------------------------------------------- F6
def fig6_resolution():
    s = load("study16_sleep.json"); sp = load("study16_propofol.json")
    if not s:
        return
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))

    # A: the sign flip — H_full (raw, reverses) vs H_aval (scale-free, recovers), across datasets
    def get(d, key):
        v = (d or {}).get("criticality", {}).get(key, {})
        return v.get("rho", np.nan), v.get("rho", np.nan) - v.get("lo", np.nan), v.get("hi", np.nan) - v.get("rho", np.nan)
    sets = [("sleep", s)] + ([("propofol", sp)] if sp else [])
    xs = np.arange(len(sets)); w = 0.38
    fr = [get(d, "H_full_vs_prox_m") for _, d in sets]
    av = [get(d, "H_aval_vs_prox_m") for _, d in sets]
    ax[0].bar(xs - w / 2, [a[0] for a in fr], w, yerr=[[a[1] for a in fr], [a[2] for a in fr]],
              color=GREY, alpha=0.85, capsize=2, label="H_full (raw EEG)")
    ax[0].bar(xs + w / 2, [a[0] for a in av], w, yerr=[[a[1] for a in av], [a[2] for a in av]],
              color=GREEN, alpha=0.85, capsize=2, label="H_aval (scale-free)")
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([n for n, _ in sets], fontsize=7)
    ax[0].set_ylabel("ρ(H, m̂-proximity)")
    pd = s.get("paired_dissociation", {}).get("across", {})
    ttl = "Observable flips the sign"
    if pd.get("p") == pd.get("p"):
        ttl += f"\n(paired p={pd['p']:.2g})"
    ax[0].set_title(ttl); ax[0].legend(fontsize=6); panel(ax[0], "A")

    # B: partial correlation (H_aval vs prox controlling slow power) survives
    pa = s.get("partial", {})
    keys = [("H_aval_vs_prox_m|slow_gfp", "H_aval\n| slow"), ("H_full_vs_prox_m|slow_raw", "H_full\n| slow")]
    xs = np.arange(len(keys)); vals, los, his = [], [], []
    for k, _ in keys:
        v = pa.get(k, {}); vals.append(v.get("rho", np.nan))
        los.append(v.get("rho", np.nan) - v.get("lo", np.nan)); his.append(v.get("hi", np.nan) - v.get("rho", np.nan))
    ax[1].bar(xs, vals, yerr=[los, his], color=[GREEN, GREY], alpha=0.85, capsize=3)
    ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([l for _, l in keys], fontsize=6.5)
    ax[1].set_ylabel("partial ρ (slow-power controlled)")
    ax[1].set_title("Recovery survives the\nslow-power confound"); panel(ax[1], "B")

    # C: sleep by-state H_full vs H_aval (raw rises into N3; scale-free does not)
    bs = s.get("by_state", {})
    order = [k for k in ["W", "Wake", "N1", "N2", "N3", "REM"] if k in bs] or list(bs)
    xs = np.arange(len(order))
    ax[2].plot(xs, [bs[k].get("H_full", np.nan) for k in order], "s-", color=GREY, label="H_full (raw)")
    ax[2].plot(xs, [bs[k].get("H_aval", np.nan) for k in order], "o-", color=GREEN, label="H_aval (scale-free)")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(order, fontsize=6.5)
    ax[2].set_ylabel("harmonicity H"); ax[2].set_title("Raw H inflates in deep N3;\nscale-free H does not")
    ax[2].legend(fontsize=6); panel(ax[2], "C")
    fig.tight_layout(); save(fig, "crit_Fig6_resolution")


def main():
    fig1_schematic(); fig2_branching(); fig3_reservoir(); fig4_ei()
    fig5_realdata_tension(); fig6_resolution()
    print(f"  Done -> {FIGDIR}")


if __name__ == "__main__":
    main()

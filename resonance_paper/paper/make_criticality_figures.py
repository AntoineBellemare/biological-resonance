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
    ax.plot(x, Rcurve, color=PURPLE, lw=1.8, ls="--", label="phase coupling / R (synchronization)")
    ax.axvline(1.0, color="grey", ls=":", lw=0.9)
    ax.text(0.42, 1.02, "subcritical", color=BLUE, fontsize=7.5, ha="center")
    ax.text(1.0, 1.08, "critical", color="k", fontsize=8, ha="center", fontweight="bold")
    ax.text(1.6, 1.02, "supercritical /\nsynchronized", color=RED, fontsize=7.5, ha="center")
    ax.set_xlabel("control parameter (branching ratio / coupling / gain)")
    ax.set_ylabel("normalized observable"); ax.set_ylim(0, 1.25); ax.set_xticks([])
    # legend in the empty lower-left (curve tails are ~0 there), clear of the region labels on top
    ax.legend(fontsize=6.4, loc="lower left", bbox_to_anchor=(0.01, 0.04), framealpha=0.9)
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
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.08, 2.7))
    edge = sm.get("g_at_max_susceptibility_norm") or sm.get("g_critical")
    edge_ci = sm.get("g_at_max_susceptibility_norm_ci")

    def mark(a):
        if edge_ci:
            a.axvspan(edge_ci[0], edge_ci[1], color=RED, alpha=0.12)
        if edge:
            a.axvline(edge, color=RED, ls="--", lw=1.0)

    # A: locate the edge — relative-fluctuation susceptibility + critical slowing
    ax[0].plot(g, _norm([r["order_param"] for r in rows]), "o-", color="#90a4ae", ms=3, alpha=0.7, label="order parameter")
    ax[0].plot(g, _norm([r.get("susceptibility_norm", np.nan) for r in rows]), "^-", color=BLUE, ms=4, label="susceptibility (var/mean²)")
    ax[0].plot(g, _norm([r["autocorr_time"] for r in rows]), "x-", color=GREEN, ms=4, label="autocorr time")
    de_g = sm.get("det_eig_g"); de_re = sm.get("det_eig_re"); ghopf = sm.get("g_hopf")
    if de_g and de_re:
        axe = ax[0].twinx()
        axe.plot(de_g, de_re, "-", color="#444444", lw=1.1, label="Re λ (eigenvalue)")
        axe.axhline(0, color="#444444", ls=":", lw=0.7); axe.tick_params(labelsize=6)
        axe.set_ylabel("Re λ (deterministic)", fontsize=6.5, color="#444444")
        if ghopf and np.isfinite(ghopf):
            ax[0].axvline(ghopf, color="#444444", ls="-.", lw=0.9)
    mark(ax[0])
    ax[0].set_xlabel("coupling gain  g"); ax[0].set_ylabel("normalized")
    ht = f"; Hopf g={ghopf:.2f}" if (ghopf and np.isfinite(ghopf)) else ""
    ax[0].set_title(f"Edge of synchronization\n(g≈{edge:.2f}{ht})")
    ax[0].legend(fontsize=5.3, loc="upper left"); panel(ax[0], "A")

    # B: the resonance factors vs the edge — H, PC, R together (the key relationship)
    for k, c, mk, lab in [("crossEI_H", ORANGE, "d", "harmonicity H"),
                          ("crossEI_PC", PURPLE, "v", "phase coupling PC"),
                          ("crossEI_R", RED, "s", "resonance R = H·PC")]:
        ax[1].plot(g, _norm([r[k] for r in rows]), mk + "-", color=c, ms=4, label=lab)
    mark(ax[1])
    ax[1].set_xlabel("coupling gain  g"); ax[1].set_ylabel("normalized  H / PC / R")
    ax[1].set_title("PC rises at the edge; R is placeable\n(R maximal in the synchronized regime)")
    ax[1].legend(fontsize=5.8, loc="lower right"); panel(ax[1], "B")

    # C: absolute PC rise (PLV units) above the asynchronous baseline
    ax[2].errorbar(g, [r["crossEI_PC"] for r in rows], yerr=[r.get("crossEI_PC_sem", 0) for r in rows],
                   fmt="v-", color=PURPLE, capsize=2, label="E↔I PC (PLV)")
    bp = sm.get("baseline_PC")
    if bp is not None:
        ax[2].axhline(bp, color=PURPLE, ls=":", lw=1.0, label=f"async baseline {bp:.2f}")
    mark(ax[2])
    dpc = sm.get("delta_PC_at_g_critical", float("nan"))
    ax[2].set_xlabel("coupling gain  g"); ax[2].set_ylabel("phase coupling PC (PLV)")
    ax[2].set_title(f"PC rises from baseline\n(ΔPC = +{dpc:.2f})"); ax[2].legend(fontsize=6); panel(ax[2], "C")
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
    s = load("study16_sleep.json"); sst = load("study16_sleep_st.json"); sp = load("study16_propofol.json")
    if not s:
        return
    fig, ax = plt.subplots(1, 4, figsize=(COL2 * 1.4, 2.7))

    def cval(d, key):
        v = (d or {}).get("criticality", {}).get(key, {})
        rho = v.get("rho", np.nan)
        return rho, rho - v.get("lo", np.nan), v.get("hi", np.nan) - rho, v.get("p", np.nan)

    def sig(p):
        return p == p and p < 0.05

    # A: the sign flip — raw H reverses, scale-free H recovers; traversing cohorts (sleep,
    #    sleep-ST) vs the non-traversing boundary case (propofol). * = p<0.05
    sets = [("sleep", s)] + ([("sleep-ST", sst)] if sst else []) + ([("propofol", sp)] if sp else [])
    xs = np.arange(len(sets)); w = 0.38
    for off, key, col, lab in [(-w / 2, "H_full_vs_prox_m", GREY, "H_full (raw)"),
                               (+w / 2, "H_aval_vs_prox_m", GREEN, "H_aval (scale-free)")]:
        vv = [cval(d, key) for _, d in sets]
        ax[0].bar(xs + off, [v[0] for v in vv], w, yerr=[[v[1] for v in vv], [v[2] for v in vv]],
                  color=col, alpha=0.85, capsize=2, label=lab)
        for x, v in zip(xs + off, vv):
            if sig(v[3]):
                ax[0].annotate("*", (x, v[0]), ha="center", va="bottom" if v[0] >= 0 else "top", fontsize=11)
    ax[0].axhline(0, color="k", lw=0.7); ax[0].set_xticks(xs); ax[0].set_xticklabels([n for n, _ in sets], fontsize=7)
    ax[0].set_ylabel("ρ(H, m̂-proximity)")
    pd = s.get("paired_dissociation", {}).get("across", {})
    pdst = (sst or {}).get("paired_dissociation", {}).get("across", {})
    ttl = f"Observable flips the sign\n(paired p: sleep {pd.get('p', float('nan')):.2g}"
    ttl += f", ST {pdst['p']:.1g})" if pdst.get("p") == pdst.get("p") else ")"
    ax[0].set_title(ttl)
    ax[0].legend(fontsize=5.6); panel(ax[0], "A")

    # B: partial correlation controlling slow power — recovery survives (H_aval | slow)
    pa = s.get("partial", {})
    keys = [("H_aval_vs_prox_m|slow_gfp", "H_aval\n| slow", GREEN), ("H_full_vs_prox_m|slow_raw", "H_full\n| slow", GREY)]
    for i, (k, lab, col) in enumerate(keys):
        v = pa.get(k, {}); rho = v.get("rho", np.nan); p = v.get("p", np.nan)
        ax[1].bar(i, rho, yerr=[[rho - v.get("lo", np.nan)], [v.get("hi", np.nan) - rho]], color=col, alpha=0.85, capsize=3)
        ax[1].annotate(("*  " if sig(p) else "") + f"p={p:.2g}", (i, rho), ha="center",
                       va="bottom" if rho >= 0 else "top", fontsize=6.3)
    ax[1].axhline(0, color="k", lw=0.7); ax[1].set_xticks(range(len(keys))); ax[1].set_xticklabels([l for _, l, _ in keys], fontsize=6.5)
    ax[1].set_ylabel("partial ρ (slow-power controlled)")
    ax[1].set_title("Recovery survives the\nslow-power confound"); panel(ax[1], "B")

    # C: slow-band-removal control — the raw-EEG reversal vanishes when the slow band is removed
    full = cval(s, "H_full_vs_prox_m"); ns = cval(s, "H_full_noslow_vs_prox_m")
    for i, (v, col) in enumerate([(full, GREY), (ns, SKY)]):
        ax[2].bar(i, v[0], yerr=[[v[1]], [v[2]]], color=col, alpha=0.85, capsize=3)
        ax[2].annotate(("*  " if sig(v[3]) else "n.s. ") + f"p={v[3]:.2g}", (i, v[0]), ha="center",
                       va="top" if v[0] < 0 else "bottom", fontsize=6.3)
    ax[2].axhline(0, color="k", lw=0.7); ax[2].set_xticks([0, 1])
    ax[2].set_xticklabels(["H_full\n(raw)", "H_full\n(no slow band)"], fontsize=6.5)
    ax[2].set_ylabel("ρ(H_full, m̂-proximity)")
    ax[2].set_title("Reversal is the slow-wave band\n(remove it → it vanishes)"); panel(ax[2], "C")

    # D: by-state mechanism — raw H inflates in deep N3; scale-free H does not
    bs = s.get("by_state", {})
    order = [k for k in ["W", "Wake", "N1", "N2", "N3", "REM"] if k in bs] or list(bs)
    xs = np.arange(len(order))
    ax[3].plot(xs, [bs[k].get("H_full", np.nan) for k in order], "s-", color=GREY, label="H_full (raw)")
    ax[3].plot(xs, [bs[k].get("H_aval", np.nan) for k in order], "o-", color=GREEN, label="H_aval (scale-free)")
    ax[3].set_xticks(xs); ax[3].set_xticklabels(order, fontsize=6.5)
    ax[3].set_ylabel("harmonicity H"); ax[3].set_title("Raw H inflates in deep N3;\nscale-free H does not")
    ax[3].legend(fontsize=6); panel(ax[3], "D")
    fig.tight_layout(); save(fig, "crit_Fig6_resolution")


# --------------------------------------------------------------------------- F7
def fig7_specificity():
    """Specificity: H's criticality peak is surrogate-specific and slope-independent. H(real)
    tracks criticality; spectrum-matched surrogates (matched-slope noise, inharmonic peak-warp,
    phase-randomized) preserve slope/power/peakiness but do NOT reproduce the criticality peak."""
    s = load("study27_h_specificity.json")
    if not s:
        return
    M = s["models"]
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))

    def plot_model(a, name, plabel):
        m = M[name]; rows = m["rows"]; x = [r["param"] for r in rows]
        a.errorbar(x, [r["H_real"] for r in rows], yerr=[r.get("H_real_sem", 0) for r in rows],
                   fmt="d-", color=ORANGE, capsize=2, ms=4, label="H (real)")
        a.plot(x, [r["H_matched_slope"] for r in rows], "s--", color=GREY, ms=3, label="matched-slope noise")
        a.plot(x, [r["H_peak_warp"] for r in rows], "^:", color=PURPLE, ms=3, label="peak-warp (inharmonic)")
        a.plot(x, [r["H_phase_random"] for r in rows], "o-", color=SKY, ms=2, lw=0.8, alpha=0.7, label="phase-random")
        a.axvline(m["crit"], color="grey", ls="--", lw=0.8)
        a.set_xlabel(plabel); a.set_ylabel("harmonicity H_max")
        tr = m["tracks"]["real"]; trs = m["tracks"]["matched_slope"]
        a.text(0.03, 0.04, f"H tracks crit ρ={tr['rho']:+.2f}\nslope-null ρ={trs['rho']:+.2f}",
               transform=a.transAxes, fontsize=6.3, va="bottom")

    plot_model(ax[0], "branching", "branching ratio  σ")
    ax[0].set_title("Branching process"); ax[0].legend(fontsize=5.5, loc="upper right"); panel(ax[0], "A")
    plot_model(ax[1], "wilson_cowan", "coupling  g")
    ax[1].set_title("Wilson–Cowan E/I"); panel(ax[1], "B")

    # C: per-seed tracking-rho across surrogates -- only real H tracks criticality
    names = ["branching", "wilson_cowan"]; kinds = ["real", "matched_slope", "peak_warp"]
    kcol = {"real": ORANGE, "matched_slope": GREY, "peak_warp": PURPLE}
    xpos = np.arange(len(names)); w = 0.26
    for i, k in enumerate(kinds):
        vals = [M[n]["tracks"][k]["rho"] for n in names]
        los = [M[n]["tracks"][k]["rho"] - M[n]["tracks"][k]["lo"] for n in names]
        his = [M[n]["tracks"][k]["hi"] - M[n]["tracks"][k]["rho"] for n in names]
        ax[2].bar(xpos + (i - 1) * w, vals, w, color=kcol[k], label=k.replace("_", "-"),
                  yerr=[los, his], capsize=2, error_kw=dict(lw=0.7))
    ax[2].axhline(0, color="k", lw=0.7); ax[2].set_xticks(xpos)
    ax[2].set_xticklabels(["branching", "Wilson–Cowan"], fontsize=6.5)
    ax[2].set_ylabel("ρ(H, criticality proximity)")
    ax[2].set_title("Only real H tracks criticality"); ax[2].legend(fontsize=5.6); panel(ax[2], "C")
    fig.tight_layout(); save(fig, "crit_Fig7_specificity")


# --------------------------------------------------------------------------- F8
def fig8_oscillatory():
    """Spiking E/I network where avalanches and oscillations coexist: R peaks with synchronization,
    above the avalanche-critical point -> H and R are separable axes."""
    s = load("study26_critical_oscillations.json")
    if not s:
        return
    rows = s["rows"]; sm = s["summary"]; sig = [r["sigma"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))

    # A: coexistence of avalanches + oscillation; supercritical runaway
    a0 = ax[0]
    a0.plot(sig, [r["powerlaw_r2"] for r in rows], "^-", color=GREEN, ms=4, label="avalanche power-law R²")
    a0b = a0.twinx()
    a0b.plot(sig, _norm([r["peak_prom"] for r in rows]), "o-", color=PURPLE, ms=3, label="oscillation prominence")
    a0b.plot(sig, _norm([r["max_aval"] for r in rows]), "s:", color=RED, ms=3, label="max avalanche (runaway)")
    a0.set_xlabel("branching gain  σ"); a0.set_ylabel("avalanche power-law R²", color=GREEN, fontsize=7)
    a0b.set_ylabel("normalized", fontsize=7)
    a0.set_title("Avalanches + oscillation coexist")
    h0, l0 = a0.get_legend_handles_labels(); h1, l1 = a0b.get_legend_handles_labels()
    a0.legend(h0 + h1, l0 + l1, fontsize=5.3, loc="center right"); panel(a0, "A")

    # B: H, PC, R vs sigma; avalanche-critical point vs R peak
    sc = sm["sigma_at_max_crit"]; sR = sm["sigma_at_max_R"]
    for k, c, mk in [("H", ORANGE, "d"), ("PC", SKY, "o"), ("R", PURPLE, "^")]:
        ax[1].plot(sig, _norm([r[k] for r in rows]), mk + "-", color=c, ms=3, label=k)
    ax[1].axvline(sc, color=GREEN, ls="--", lw=0.9, label="avalanche-critical σ")
    ax[1].axvline(sR, color=PURPLE, ls=":", lw=0.9, label="R peak")
    ax[1].set_xlabel("branching gain  σ"); ax[1].set_ylabel("normalized")
    ax[1].set_title("R peaks above the critical point"); ax[1].legend(fontsize=5.5); panel(ax[1], "B")

    # C: cross-observable contrast — same metric (H), opposite sign, set by the observable
    s10 = load("study10_criticality.json")
    rho_branch = (s10["summary"].get("H_tracks_prox_rho", float("nan")) if s10 else float("nan"))
    rho_spik = sm["H_tracks_crit"]["rho"]
    labels = ["H on scale-free\navalanche signal\n(branching)", "H on oscillation-laden\npop. signal\n(spiking net)"]
    vals = [rho_branch, rho_spik]; cols = [GREEN, PURPLE]
    ax[2].bar(np.arange(2), vals, color=cols, width=0.6)
    ax[2].axhline(0, color="k", lw=0.7); ax[2].set_xticks(np.arange(2))
    ax[2].set_xticklabels(labels, fontsize=5.6)
    ax[2].set_ylabel("ρ(H, avalanche-criticality)")
    ax[2].set_title("The observable sets the sign"); panel(ax[2], "C")
    fig.tight_layout(); save(fig, "crit_Fig8_oscillatory")


def main():
    fig1_schematic(); fig2_branching(); fig3_reservoir(); fig4_ei()
    fig5_realdata_tension(); fig6_resolution(); fig7_specificity(); fig8_oscillatory()
    print(f"  Done -> {FIGDIR}")


if __name__ == "__main__":
    main()

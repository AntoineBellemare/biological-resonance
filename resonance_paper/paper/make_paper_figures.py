"""Assemble publication-ready composite figures from the paper-grade result JSONs.

Produces four main figures (paper/figures/Fig1-4.{png,pdf}) at 600 DPI with a
consistent publication style (Wong colorblind-safe palette, panel letters,
journal two-column width). Run AFTER `python -m resonance_paper.run_all --paper`.

    python -m resonance_paper.paper.make_paper_figures
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGDIR = HERE / "figures"
FIGDIR.mkdir(exist_ok=True, parents=True)

# Wong (2011) colorblind-safe palette
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
PURPLE, SKY, YELLOW, GREY = "#CC79A7", "#56B4E9", "#F0E442", "#999999"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "legend.frameon": False, "lines.linewidth": 1.6, "lines.markersize": 5,
})
COL2 = 7.2   # two-column journal width (inches)


def load(name):
    return json.loads((RESULTS / name).read_text())


def panel(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="right")


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png")
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.png (+pdf)")


# ---------------------------------------------------------------------------
# Figure 1 — Framework validation on synthetic ground truth
# ---------------------------------------------------------------------------
def figure1():
    s1 = load("study1_ground_truth.json"); s5 = load("study5_cross_signal.json")
    s6 = load("study6_resonance_conjunction.json")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))

    # A: harmonic-structure recovery — H_avg by signal kind
    rows = s1["part_a"]["rows"]
    order = ["pink", "inharmonic", "pure_tone", "dyad", "triad", "rich_stack"]
    data = [[r["H_avg"] for r in rows if r["kind"] == k] for k in order]
    bp = axes[0].boxplot(data, labels=["noise", "inharm", "tone", "dyad", "triad", "rich"],
                         patch_artist=True, widths=0.6,
                         medianprops=dict(color="black"), flierprops=dict(ms=2))
    for p in bp["boxes"]:
        p.set_facecolor(BLUE); p.set_alpha(0.75)
    axes[0].set_ylabel("mean harmonicity H")
    axes[0].set_title("Harmonic-structure recovery")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].text(0.04, 0.92, f"ρ(richness,H)={s1['part_a']['spearman_richness_Havg']:.2f}",
                 transform=axes[0].transAxes, fontsize=7)
    panel(axes[0], "A")

    # B: coupling detection AUC (single-signal + cross-signal)
    b = s1["part_b"]
    s5auc = {r["regime"]: r["auc"] for r in s5["auc"]}
    names = ["PC_z\n(single)", "matrix\n(single)", "1:1\n(cross)", "1:2\n(cross)", "2:3\n(cross)"]
    vals = [b["auc_PC_z"], b["auc_matrix_z"], s5auc.get("lock_1to1", np.nan),
            s5auc.get("lock_1to2", np.nan), s5auc.get("lock_2to3", np.nan)]
    axes[1].bar(range(len(vals)), vals, color=[BLUE, BLUE, RED, RED, RED], alpha=0.85)
    axes[1].axhline(0.5, color="k", ls="--", lw=0.7)
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("detection AUC")
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, fontsize=6.5)
    axes[1].set_title("Phase-coupling recovery")
    panel(axes[1], "B")

    # C: polyrhythm recovery (locked vs scrambled R peak/median)
    pb = s6["part_b"]
    axes[2].bar([0, 1], [pb["within_locked_ratio"], pb["within_scrambled_ratio"]],
                color=[RED, GREY], alpha=0.85, width=0.6)
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["locked", "scrambled"])
    axes[2].set_ylabel("R peak/median, 2:3:4 bands")
    axes[2].set_title(f"Polyrhythm recovery\n(AUC={pb['auc_within']['auc']:.2f})")
    panel(axes[2], "C")

    fig.suptitle("Figure 1 — Resonance recovers harmonic structure and phase coupling (synthetic ground truth)",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "Fig1_ground_truth")


# ---------------------------------------------------------------------------
# Figure 2 — Real biosignals
# ---------------------------------------------------------------------------
def figure2():
    s2 = load("study2_eeg_states.json"); s3 = load("study3_cross_modality.json")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    # A: EEG decoding — resonance vs band power, both contrasts
    contrasts = list(s2["contrasts"].items())
    labels, res, pwr = [], [], []
    for cname, c in contrasts:
        clf = c["classification"]
        labels.append("EO/EC" if "open" in cname else "rest/motor")
        res.append(clf["auc_resonance"]); pwr.append(clf["auc_band_power"])
    x = np.arange(len(labels)); w = 0.38
    axes[0].bar(x - w/2, res, w, color=GREEN, label="resonance", alpha=0.9)
    axes[0].bar(x + w/2, pwr, w, color=GREY, label="band power", alpha=0.9)
    axes[0].axhline(0.5, color="k", ls="--", lw=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("decoding AUC (LOSO)")
    axes[0].set_title("EEG state decoding"); axes[0].legend(loc="lower right")
    panel(axes[0], "A")

    # B: the 1/f confound — H_max EO/EC with vs without aperiodic removal
    ap = s2["contrasts"].get("eyes_open_vs_closed", {}).get("aperiodic_check", {})
    axes[1].bar([0, 1], [ap.get("H_max_auc_with_removal", np.nan),
                         ap.get("H_max_auc_without_removal", np.nan)],
                color=[GREEN, RED], alpha=0.85, width=0.6)
    axes[1].axhline(0.5, color="k", ls="--", lw=0.7)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["aperiodic\nremoved", "kept"])
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("H_max AUC (EO vs EC)")
    axes[1].set_title("1/f confound control")
    panel(axes[1], "B")

    # C: cross-modality example resonance spectra
    spec = s3.get("example_spectra", {})
    colors = {"ECG_real": RED, "ECG_synth": ORANGE, "EEG_alpha": BLUE,
              "PPG": PURPLE, "RSP": SKY, "harmonic_stack": GREEN, "pink_noise": GREY}
    for lab in ["harmonic_stack", "EEG_alpha", "ECG_real", "pink_noise"]:
        if lab in spec:
            sd = spec[lab]
            axes[2].plot(sd["freqs"], np.array(sd["R"]) / (max(sd["R"]) + 1e-12),
                         label=lab.replace("_", " "), color=colors.get(lab, None), lw=1.2)
    axes[2].set_xlabel("Frequency (Hz)"); axes[2].set_ylabel("resonance R (norm.)")
    clf = s3["classification"]
    axes[2].set_title(f"Cross-modality\n(acc={clf['cv_accuracy_mean']:.2f}, chance {clf['chance']:.2f})")
    axes[2].legend(fontsize=6)
    panel(axes[2], "C")

    fig.suptitle("Figure 2 — Resonance discriminates brain states and fingerprints biosignal modality",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "Fig2_real_biosignals")


# ---------------------------------------------------------------------------
# Figure 3 — Harmonic complexity governs lockability (Arnold tongues)
# ---------------------------------------------------------------------------
def figure3():
    s8 = load("study8_arnold_tongues.json")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.7))

    # A: devil's staircase
    st = s8["staircase"]
    axes[0].plot([d["Omega"] for d in st], [d["rho"] for d in st], ".",
                 color=BLUE, ms=3)
    for t in s8["tongues"]:
        axes[0].axhline(t["rho"], color=GREEN, alpha=0.2, lw=0.6)
    axes[0].set_xlabel("drive ratio  Ω = f_drive / f0")
    axes[0].set_ylabel("rotation number  ρ")
    axes[0].set_title("Devil's staircase (forced Van der Pol)")
    panel(axes[0], "A")

    # B: tongue width vs complexity, colored by framework harmonicity
    t = s8["tongues"]
    comps = np.array([r["complexity"] for r in t])
    widths = np.array([r["width"] for r in t])
    hs = np.array([r["harmsim_pair"] for r in t])
    sc = axes[1].scatter(comps, widths, c=hs, cmap="viridis", s=70,
                         edgecolor="k", linewidth=0.5, zorder=3)
    for r in t:
        axes[1].annotate(r["ratio"], (r["complexity"], r["width"]),
                         fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    cb = plt.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.03)
    cb.set_label("framework harmonicity", fontsize=7)
    axes[1].set_xlabel("ratio complexity  p·q")
    axes[1].set_ylabel("Arnold tongue width (Ω)")
    c = s8["corr"]
    axes[1].set_title(f"Lockability vs complexity\n"
                      f"ρ(width,cplx)={c['width_vs_complexity']:+.2f}, "
                      f"ρ(width,harm)={c['width_vs_harmsim']:+.2f}")
    panel(axes[1], "B")

    fig.suptitle("Figure 3 — Harmonic simplicity governs phase-locking width; the framework's harmonicity tracks it",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "Fig3_arnold_tongues")


# ---------------------------------------------------------------------------
# Figure 4 — Resonance, computation and criticality
# ---------------------------------------------------------------------------
def figure4():
    s10 = load("study10_criticality.json")
    s11 = load("study11_reservoir_criticality.json")
    s12 = load("study12_ei_network.json")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.7))

    def nrm(v):
        v = np.asarray(v, float); return v / (np.nanmax(np.abs(v)) + 1e-12)

    # A: branching network — H peaks at criticality
    r = s10["rows"]; sig = [x["sigma"] for x in r]
    axes[0].plot(sig, nrm([x["susceptibility"] for x in r]), "o-", color=BLUE, label="susceptibility")
    axes[0].plot(sig, nrm([x["powerlaw_r2"] for x in r]), "^-", color=GREEN, label="avalanche power-law")
    axes[0].plot(sig, nrm([x["H_max"] for x in r]), "d-", color=ORANGE, label="harmonicity H")
    axes[0].axvline(1.0, color="k", ls="--", lw=0.7)
    axes[0].set_xlabel("branching ratio σ"); axes[0].set_ylabel("normalized")
    axes[0].set_title("Branching network\n(H peaks at criticality)")
    axes[0].legend(loc="lower center", fontsize=6)
    panel(axes[0], "A")

    # B: reservoir — resonance vs computation vs edge of chaos
    r = s11["rows"]; rho = [x["rho"] for x in r]; rc = s11["summary"]["rho_critical"]
    axes[1].plot(rho, nrm([x["memory_capacity"] for x in r]), "o-", color=BLUE, label="memory capacity")
    axes[1].plot(rho, nrm([x["R_internal"] for x in r]), "s-", color=RED, label="resonance R")
    if np.isfinite(rc):
        axes[1].axvline(rc, color="k", ls="--", lw=0.7, label=f"edge of chaos")
    axes[1].set_xlabel("spectral radius ρ"); axes[1].set_ylabel("normalized")
    axes[1].set_title("Reservoir\n(R highest when ordered; dies in chaos)")
    axes[1].legend(loc="lower center", fontsize=6)
    panel(axes[1], "B")

    # C: E/I network — phase coupling RISES (from a non-zero baseline) at sync onset
    r = s12["rows"]; g = [x["g"] for x in r]; gc = s12["summary"]["g_critical"]
    base_pc = s12["summary"].get("baseline_PC", float(r[0]["crossEI_PC"]))
    axes[2].plot(g, nrm([x["order_param"] for x in r]), "o-", color=GREY, label="order param")
    axes[2].plot(g, nrm([x["susceptibility"] for x in r]), "^-", color=BLUE, label="susceptibility")
    axes[2].plot(g, [x["crossEI_PC"] for x in r], "v-", color=PURPLE, label="E↔I PC (raw PLV)")
    axes[2].axhline(base_pc, color=PURPLE, ls=":", lw=0.8, label=f"async baseline {base_pc:.2f}")
    axes[2].axvline(gc, color="k", ls="--", lw=0.7)
    axes[2].set_xlabel("coupling gain g"); axes[2].set_ylabel("normalized  /  PLV")
    axes[2].set_title("E/I network\n(PC rises at sync onset, not from 0)")
    axes[2].legend(loc="lower right", fontsize=6)
    panel(axes[2], "C")

    fig.suptitle("Figure 4 — Harmonic structure tracks criticality; phase-coupling resonance rises at the synchronization transition",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "Fig4_criticality")


def main():
    print("Building paper-ready figures from paper-grade results ...")
    figure1(); figure2(); figure3(); figure4()
    print(f"Done -> {FIGDIR}")


if __name__ == "__main__":
    main()

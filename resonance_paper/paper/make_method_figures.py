"""Publication figures for the METHODS paper (M-A spine: harmonicity vs phase
coupling, validated framework).

Four composite figures (paper/figures/method_Fig1-4.{png,pdf}, 600 DPI):
  Fig 1  Ground-truth recovery (H recovery + n:m coupling detection + polyrhythm)   [Studies 1,5,6]
  Fig 2  The construct: H is phase-blind, PC tracks locking, R is interpretive       [Study 17]
  Fig 3  Real biosignals: state discrimination, 1/f confound, modality fingerprint   [Studies 2,3]
  Fig 4  Method choices + mechanism: strategy registry + harmonic complexity governs lockability [Studies 4,8]

Criticality (Studies 9-16) and consonance (20/20b) belong to the other papers and
are NOT included here. Run AFTER `python -m resonance_paper.run_all --paper`.

    python -m resonance_paper.paper.make_method_figures
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
COL2 = 7.2


def load(name):
    return json.loads((RESULTS / name).read_text())


def panel(ax, letter):
    # place above the axes (clears y-labels, tick labels, and titles)
    ax.text(-0.16, 1.12, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(FIGDIR / f"{name}.png")
    fig.savefig(FIGDIR / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.png (+pdf)")


# --------------------------------------------------------------------------- F1
def figure1():
    s1 = load("study1_ground_truth.json"); s5 = load("study5_cross_signal.json")
    s6 = load("study6_resonance_conjunction.json")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))

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

    pb = s6["part_b"]
    axes[2].bar([0, 1], [pb["within_locked_ratio"], pb["within_scrambled_ratio"]],
                color=[RED, GREY], alpha=0.85, width=0.6)
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["locked", "scrambled"])
    axes[2].set_ylabel("R peak/median, 2:3:4")
    axes[2].set_title(f"Polyrhythm recovery\n(AUC={pb['auc_within']['auc']:.2f})")
    panel(axes[2], "C")

    fig.suptitle("Figure 1 — The framework recovers harmonic structure and n:m phase coupling (synthetic ground truth)",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "method_Fig1_ground_truth")


# --------------------------------------------------------------------------- F2
def figure2():
    """The construct: H is phase-blind; PC tracks locking; R is interpretive, not a better detector."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    s = load("study17_tripartite_dissociation.json")
    rows = s["rows"]; ind = s["independence"]; sp = s["specificity"]
    comps = sorted(set(r["complexity"] for r in rows))
    nrm = Normalize(vmin=min(comps), vmax=max(comps))

    fig, axes = plt.subplots(1, 4, figsize=(COL2 * 1.32, 2.7))

    # A/B: H and PC vs locking κ, one line per ratio-complexity.
    #   H lines are FLAT across κ (phase-blind) and separated by complexity;
    #   PC lines RISE with κ (track locking). This is the dissociation.
    for ax, key, ttl in [(axes[0], "H", "H is flat across κ\n(phase-blind)"),
                         (axes[1], "PC", "PC rises with κ\n(tracks locking)")]:
        for c in comps:
            sub = sorted([r for r in rows if r["complexity"] == c], key=lambda r: r["kappa"])
            ax.plot([r["kappa"] for r in sub], [r[key] for r in sub], "-o",
                    color=plt.cm.viridis(nrm(c)), ms=2.5, lw=1.2, alpha=0.9)
        ax.set_xlabel("phase-locking κ"); ax.set_title(ttl)
    axes[0].set_ylabel("harmonicity H"); axes[1].set_ylabel("phase coupling PC")
    sm = ScalarMappable(norm=nrm, cmap="viridis"); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cb.set_label("ratio complexity log2(pq)", fontsize=6.5)
    panel(axes[0], "A"); panel(axes[1], "B")

    # C: independence — each factor vs each axis (the dissociation: H⟂κ ≈ 0)
    ax = axes[2]
    x = np.arange(2); w = 0.38
    h_vals = [ind["H_vs_complexity"], ind["H_vs_kappa"]]
    pc_vals = [ind["PC_vs_complexity"], ind["PC_vs_kappa"]]
    ax.bar(x - w/2, h_vals, w, color=ORANGE, label="H", alpha=0.9)
    ax.bar(x + w/2, pc_vals, w, color=PURPLE, label="PC", alpha=0.9)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(["vs ratio\ncomplexity", "vs phase\nlocking κ"], fontsize=7)
    ax.set_ylabel("Spearman ρ"); ax.set_ylim(-0.75, 0.75)
    ax.annotate("H vs κ ≈ 0\n(phase-blind)", (1 - w/2, h_vals[1]), xytext=(0.45, 0.55),
                textcoords="axes fraction", fontsize=6.5, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_title("Independence"); ax.legend(loc="lower left")
    panel(ax, "C")

    # D: specificity — R does NOT beat PC (honest)
    ax = axes[3]
    keys = ["H", "PC", "R"]; cols = [ORANGE, PURPLE, RED]
    aucs = [sp[k]["auc"] for k in keys]
    los = [sp[k]["auc"] - sp[k]["lo"] for k in keys]
    his = [sp[k]["hi"] - sp[k]["auc"] for k in keys]
    ax.bar(range(3), aucs, color=cols, alpha=0.9, yerr=[los, his], capsize=3,
           error_kw=dict(lw=0.8))
    ax.axhline(0.5, color="k", ls="--", lw=0.7)
    ax.set_xticks(range(3)); ax.set_xticklabels(keys); ax.set_ylim(0, 1.05)
    ax.set_ylabel("specificity AUC")
    ax.set_title("R = H·PC is interpretive\n(does not beat PC)")
    panel(ax, "D")

    fig.suptitle("Figure 2 — The construct: harmonicity is phase-blind, phase coupling tracks locking, "
                 "and R is an interpretive decomposition",
                 fontsize=9.5, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "method_Fig2_dissociation")


# --------------------------------------------------------------------------- F3
def figure3():
    s2 = load("study2_eeg_states.json"); s3 = load("study3_cross_modality.json")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

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
    axes[0].set_title("EEG state decoding\n(= band power: honest scope)")
    axes[0].legend(loc="lower right")
    panel(axes[0], "A")

    ap = s2["contrasts"].get("eyes_open_vs_closed", {}).get("aperiodic_check", {})
    axes[1].bar([0, 1], [ap.get("H_max_auc_with_removal", np.nan),
                         ap.get("H_max_auc_without_removal", np.nan)],
                color=[GREEN, RED], alpha=0.85, width=0.6)
    axes[1].axhline(0.5, color="k", ls="--", lw=0.7)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["aperiodic\nremoved", "kept"])
    axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("H_max AUC (EO vs EC)")
    axes[1].set_title("1/f confound control")
    panel(axes[1], "B")

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

    fig.suptitle("Figure 3 — Real biosignals: resonance discriminates states (= band power) and fingerprints modality",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "method_Fig3_real_biosignals")


# --------------------------------------------------------------------------- F4
def figure4():
    s4 = load("study4_strategy_comparison.json"); s8 = load("study8_arnold_tongues.json")
    table = s4["table"]; rt = s4["runtime_per_call_s"]
    fig, axes = plt.subplots(1, 4, figsize=(COL2 * 1.28, 2.6))

    # A: ALL 20 configs — accuracy saturates (choice barely matters on this task);
    #    the honest message is "every strategy works; differentiate on speed (B)".
    aucs = sorted(r["auc_mean"] for r in table)
    best_label = "harmsim|fraction|nm_plv_canonical"
    rec = next((r["auc_mean"] for r in table if r["label"] == best_label), max(aucs))
    worst = min(table, key=lambda r: r["auc_mean"])
    x = np.arange(len(aucs))
    axes[0].scatter(x, aucs, s=18, color=GREY, alpha=0.8, zorder=2)
    axes[0].scatter([len(aucs) - 1], [rec], s=40, color=GREEN, zorder=3, label="recommended")
    axes[0].scatter([0], [worst["auc_mean"]], s=40, color=RED, zorder=3, label="weakest")
    axes[0].set_ylim(0.975, 1.004); axes[0].set_xticks([])
    axes[0].set_xlabel("20 kernel × metric configs (sorted)")
    axes[0].set_ylabel("mean AUC")
    axes[0].set_title(f"Accuracy saturates\n(all configs {min(aucs):.3f}–{max(aucs):.3f})")
    axes[0].legend(loc="lower right", fontsize=6)
    axes[0].annotate(worst["label"].replace("nm_", "").replace("|", "\n"),
                     (0, worst["auc_mean"]), xytext=(8, -2), textcoords="offset points", fontsize=5.2)
    panel(axes[0], "A")

    # B: runtime — the real differentiator (harmsim vs subharm_tension per call)
    hs, sub = rt.get("harmsim", np.nan), rt.get("subharm_tension", np.nan)
    axes[1].bar([0, 1], [hs, sub], color=[GREEN, GREY], alpha=0.85, width=0.6)
    for i, v in enumerate([hs, sub]):
        axes[1].text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=7)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["harmsim", "subharm_\ntension"], fontsize=7)
    axes[1].set_ylabel("runtime / call (s)")
    axes[1].set_title(f"Speed is the differentiator\n(harmsim ~{sub / max(hs, 1e-9):.0f}× faster)")
    panel(axes[1], "B")

    # C: devil's staircase
    st = s8["staircase"]
    axes[2].plot([d["Omega"] for d in st], [d["rho"] for d in st], ".", color=BLUE, ms=2.5)
    axes[2].set_xlabel("drive ratio  Ω"); axes[2].set_ylabel("rotation number  ρ")
    axes[2].set_title("Devil's staircase\n(forced Van der Pol)")
    panel(axes[2], "C")

    # D: tongue width vs complexity, colored by harmonicity
    t = s8["tongues"]
    comps = np.array([r["complexity"] for r in t]); widths = np.array([r["width"] for r in t])
    hs = np.array([r["harmsim_pair"] for r in t])
    sc = axes[3].scatter(comps, widths, c=hs, cmap="viridis", s=55, edgecolor="k", linewidth=0.5, zorder=3)
    cb = plt.colorbar(sc, ax=axes[3], fraction=0.046, pad=0.03); cb.set_label("harmonicity", fontsize=7)
    axes[3].set_xlabel("ratio complexity  p·q"); axes[3].set_ylabel("Arnold tongue width")
    c = s8["corr"]
    axes[3].set_title(f"Lockability vs complexity\nρ={c['width_vs_complexity']:+.2f}")
    panel(axes[3], "D")

    fig.suptitle("Figure 4 — Method choices (strategy registry) and the mechanism: harmonic simplicity governs lockability",
                 fontsize=9.5, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "method_Fig4_strategy_mechanism")


def main():
    print("Building METHOD-paper figures (M-A spine) from paper-grade results ...")
    figure1(); figure2(); figure3(); figure4()
    print(f"Done -> {FIGDIR}")


if __name__ == "__main__":
    main()

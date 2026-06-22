"""Publication figures for the METHODS paper (M-A spine: harmonicity vs phase
coupling, validated framework).

Nine composite figures (paper/figures/method_Fig1-9.{png,pdf}, 600 DPI), in the
CONSTRUCT-FIRST narrative order (define -> justify -> validate -> characterize -> apply):
  Fig 1  The construct: H is phase-blind, PC tracks locking, R is interpretive       [Study 17]
  Fig 2  Why R = H*PC: conjunction (beats both factors) + combine-rule justification  [Study 23]
  Fig 3  Ground-truth recovery (H recovery + n:m coupling detection + polyrhythm)     [Studies 1,5,6]
  Fig 4  Operating characteristics: SNR robustness, null calibration, scaling         [Study 24]
  Fig 5  Baselines: framework factors vs established measures (n:m PLV, HNR)          [Study 25]
  Fig 6  Method choices + mechanism: strategy registry + harmonic complexity -> lockability [Studies 4,8]
  Fig 7  Real biosignals: state discrimination, 1/f confound, modality fingerprint   [Studies 2,3]
  Fig 8  The multichannel layer: cross-resonance connectivity (honest capability demo) [Study 21]
  Fig 9  Complexity descriptors (flatness/entropy/spread/HFD) of the H/PC/R spectra   [Study 22]

(Figure builder functions are named figureN by content, not output order — e.g.
figure2() = the dissociation = output Fig 1; the save() name is authoritative.)
Criticality (Studies 9-16) and consonance (19/20/20b) belong to the other papers
and are NOT included here. Run AFTER `python -m resonance_paper.run_all --paper`.

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

    fig.suptitle("Figure 3 — The framework recovers harmonic structure and n:m phase coupling (synthetic ground truth)",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "method_Fig3_ground_truth")


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

    # D: GENERATIVE — harmonically-coupled phase oscillators. As coupling K rises each
    #    pair locks at its n:m ratio, but the threshold K* RISES with ratio complexity
    #    (n:m via higher/weaker harmonics) — the Arnold tongue in the coupling dimension.
    ax = axes[3]
    s7 = load("study7_coupled_oscillators.json"); tr = s7["transition"]; cx = s7["complexity"]
    kcols = {"2:3": GREEN, "3:4": BLUE, "4:5": RED}
    for label, rows in tr.items():
        ax.plot([max(d["K"], 1.0) for d in rows], [d["PC"] for d in rows], "o-",
                color=kcols.get(label), ms=3, lw=1.2, label=f"{label} (n·m={cx[label]})")
    ax.axhline(0.5, color="grey", ls=":", lw=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("coupling strength K (log)"); ax.set_ylabel("phase coupling PC")
    rho = s7["corr"]["kstar_vs_complexity"]
    ax.set_title(f"Generative: complex ratios\nlock later (K*↑ with n·m, ρ={rho:+.1f})")
    ax.legend(fontsize=5.5)
    panel(ax, "D")

    fig.suptitle("Figure 1 — The construct: harmonicity is phase-blind, phase coupling tracks locking "
                 "(synthetic grid + generative n:m oscillators)",
                 fontsize=9, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "method_Fig1_dissociation")


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

    fig.suptitle("Figure 7 — Real biosignals: resonance discriminates states (= band power) and fingerprints modality",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "method_Fig7_real_biosignals")


# --------------------------------------------------------------------------- F4
def figure4():
    s4 = load("study4_strategy_comparison.json"); s8 = load("study8_arnold_tongues.json")
    table = s4["table"]; rt = s4["runtime_per_call_s"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.6))

    # A: ALL 20 configs — accuracy saturates; the practical decision is runtime, noted inline.
    aucs = sorted(r["auc_mean"] for r in table)
    best_label = "harmsim|fraction|nm_plv_canonical"
    rec = next((r["auc_mean"] for r in table if r["label"] == best_label), max(aucs))
    worst = min(table, key=lambda r: r["auc_mean"])
    speed = rt.get("subharm_tension", np.nan) / max(rt.get("harmsim", np.nan), 1e-9)
    x = np.arange(len(aucs))
    axes[0].scatter(x, aucs, s=18, color=GREY, alpha=0.8, zorder=2)
    axes[0].scatter([len(aucs) - 1], [rec], s=40, color=GREEN, zorder=3, label="recommended")
    axes[0].scatter([0], [worst["auc_mean"]], s=40, color=RED, zorder=3, label="weakest")
    axes[0].set_ylim(0.975, 1.004); axes[0].set_xticks([])
    axes[0].set_xlabel("20 kernel × metric configs (sorted)")
    axes[0].set_ylabel("mean AUC")
    axes[0].set_title(f"Accuracy saturates ({min(aucs):.3f}–{max(aucs):.3f})\n→ choose on speed (harmsim ~{speed:.0f}× faster)")
    axes[0].legend(loc="lower right", fontsize=6)
    panel(axes[0], "A")

    # B: devil's staircase
    st = s8["staircase"]
    axes[1].plot([d["Omega"] for d in st], [d["rho"] for d in st], ".", color=BLUE, ms=2.5)
    axes[1].set_xlabel("drive ratio  Ω"); axes[1].set_ylabel("rotation number  ρ")
    axes[1].set_title("Devil's staircase\n(forced Van der Pol)")
    panel(axes[1], "B")

    # C: tongue width vs complexity, colored by harmonicity
    t = s8["tongues"]
    comps = np.array([r["complexity"] for r in t]); widths = np.array([r["width"] for r in t])
    hs = np.array([r["harmsim_pair"] for r in t])
    sc = axes[2].scatter(comps, widths, c=hs, cmap="viridis", s=55, edgecolor="k", linewidth=0.5, zorder=3)
    cb = plt.colorbar(sc, ax=axes[2], fraction=0.046, pad=0.03); cb.set_label("harmonicity", fontsize=7)
    axes[2].set_xlabel("ratio complexity  p·q"); axes[2].set_ylabel("Arnold tongue width")
    c = s8["corr"]
    axes[2].set_title(f"Lockability vs complexity\nρ={c['width_vs_complexity']:+.2f}")
    panel(axes[2], "C")

    fig.suptitle("Figure 6 — Method choices (strategy registry; accuracy saturates, choose on speed) "
                 "and the mechanism: harmonic simplicity governs lockability",
                 fontsize=9, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "method_Fig6_strategy_mechanism")


# --------------------------------------------------------------------------- F5
def figure5():
    """Network layer: cross-resonance connectivity recovers a planted coupled cluster
    (H diffuse, PC sharper, R ~ PC; recovery is overlap-driven — honest caveat)."""
    s = load("study21_connectivity.json")
    ex = s["example"]; summ = s["summary"]; cl = s["cluster"]; c0, c1 = min(cl), max(cl)
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.7))

    for ax, key, ttl in [(axes[0], "H", "H connectivity\n(diffuse)"),
                         (axes[1], "PC", "PC connectivity\n(sharper; cyan = cluster)")]:
        im = ax.imshow(np.array(ex[key], float), cmap="magma")
        ax.add_patch(plt.Rectangle((c0 - 0.5, c0 - 0.5), c1 - c0 + 1, c1 - c0 + 1,
                                   fill=False, edgecolor=SKY, lw=1.6))
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("channel"); ax.set_ylabel("channel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    keys = ["H", "PC", "R"]; cols = [ORANGE, PURPLE, RED]
    aucs = [summ[k]["auc_mean"] for k in keys]
    lo = [summ[k]["auc_mean"] - summ[k]["ci"]["lo"] for k in keys]
    hi = [summ[k]["ci"]["hi"] - summ[k]["auc_mean"] for k in keys]
    axes[2].bar(range(3), aucs, color=cols, alpha=0.9, yerr=[lo, hi], capsize=3, error_kw=dict(lw=0.8))
    axes[2].axhline(0.5, color="k", ls="--", lw=0.7); axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(keys)
    axes[2].set_ylabel("AUC (within-cluster vs rest)"); axes[2].set_title("Cluster recovery\n(R ≈ PC)")
    for a, l in zip(axes, "ABC"):
        panel(a, l)

    fig.suptitle("Figure 8 — Cross-resonance connectivity recovers a planted coupled cluster "
                 "(overlap-driven; surrogate-z isolates pure phase)", fontsize=9, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "method_Fig8_connectivity")


# --------------------------------------------------------------------------- F6
def figure6():
    """Spectral-complexity descriptors of the H/PC/R spectra are first-class features."""
    s = load("study22_spectral_descriptors.json")
    rows = s["rows"]; disc = s["discriminability"]; classes = s["classes"]; feats = s["feats"]
    cols = {"pure_tone": BLUE, "harmonic": GREEN, "inharmonic": ORANGE,
            "narrowband": PURPLE, "pink": GREY}
    fig, axes = plt.subplots(1, 3, figsize=(COL2 * 1.05, 2.7))

    # A: H-spectrum flatness by class
    data = [[r["H_flatness"] for r in rows if r["kind"] == k and np.isfinite(r["H_flatness"])] for k in classes]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black"),
                         flierprops=dict(ms=2))
    for p, k in zip(bp["boxes"], classes):
        p.set_facecolor(cols[k]); p.set_alpha(0.8)
    axes[0].set_xticks(range(1, len(classes) + 1)); axes[0].set_xticklabels(classes, rotation=35, fontsize=6.5)
    axes[0].set_ylabel("H-spectrum flatness"); axes[0].set_title("Structured → low flatness")
    panel(axes[0], "A")

    # B: the resonance-descriptor FINGERPRINT — z-scored (per row) descriptor × class
    #    heatmap (a bar ranking saturates: every descriptor separates these clean
    #    classes near-perfectly, so the informative view is the per-class PATTERN)
    M = np.array([[np.nanmean([r[f] for r in rows if r["kind"] == k]) for k in classes] for f in feats])
    Mz = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-12)
    im = axes[1].imshow(Mz, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6)
    axes[1].set_xticks(range(len(classes))); axes[1].set_xticklabels(classes, rotation=35, fontsize=6)
    axes[1].set_yticks(range(len(feats))); axes[1].set_yticklabels(feats, fontsize=5.5)
    axes[1].set_title("Descriptor fingerprint (z per row)")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    panel(axes[1], "B")

    # C: resonance-shape space — descriptors add info beyond H_avg
    for k in classes:
        xs = [r["H_flatness"] for r in rows if r["kind"] == k]
        ys = [r["H_spread"] for r in rows if r["kind"] == k]
        axes[2].scatter(xs, ys, color=cols[k], s=20, alpha=0.8, label=k, edgecolor="none")
    axes[2].set_xlabel("H-spectrum flatness"); axes[2].set_ylabel("H-spectrum spread")
    axes[2].set_title("Shape space (beyond H_avg)"); axes[2].legend(fontsize=5.5)
    panel(axes[2], "C")

    fig.suptitle("Figure 9 — Complexity descriptors (flatness/entropy/spread/HFD) of the H/PC/R spectra are first-class features",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "method_Fig9_descriptors")


# --------------------------------------------------------------------------- F7
def figure7():
    """Why R = H*PC: conjunction beats both factors (independent axes) + phase gate."""
    s = load("study23_R_justification.json"); A = s["part_a"]; B = s["part_b"]
    rules = ["product", "geomean", "harmmean", "min", "max", "mean"]
    conj = {"product", "geomean", "harmmean", "min"}
    fig, axes = plt.subplots(1, 4, figsize=(COL2 * 1.32, 2.7))

    keys = ["H", "PC"] + [f"R[{r}]" for r in rules]
    aucs = [A["auc"][k] for k in keys]
    cols = [ORANGE, PURPLE] + [RED if r in conj else GREY for r in rules]
    axes[0].bar(range(len(keys)), aucs, color=cols, alpha=0.9)
    axes[0].axhline(0.5, color="k", ls="--", lw=0.7); axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(range(len(keys)))
    axes[0].set_xticklabels([k.replace("R[", "").replace("]", "") for k in keys], rotation=40, fontsize=6.5)
    axes[0].set_ylabel("specificity AUC"); axes[0].set_title("Conjunctions beat both\nfactors & disjunctions")
    panel(axes[0], "A")

    x = np.arange(3); w = 0.38
    hc = B["H_coherent"]
    coh = [1.0, B["PC_coherent"], B["R_coherent"] / hc]
    scr = [B["H_scrambled"] / hc, B["PC_scrambled"], B["R_scrambled"] / hc]
    axes[1].bar(x - w/2, coh, w, color=GREEN, label="coherent")
    axes[1].bar(x + w/2, scr, w, color=GREY, label="phase-scrambled")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["H\n(norm)", "PC", "R\n(norm)"])
    axes[1].set_ylabel("factor (H-normalized)"); axes[1].set_title("Phase-scramble: H fixed,\nPC & R collapse")
    axes[1].legend(fontsize=7); panel(axes[1], "B")

    axes[2].bar([0, 1], [B["auc_H"], B["auc_R"]], color=[ORANGE, RED], width=0.6)
    axes[2].axhline(0.5, color="k", ls="--", lw=0.7); axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["H", "R = H·PC"])
    axes[2].set_ylabel("AUC (coherent vs scrambled)"); axes[2].set_title("R adds the phase gate\nH lacks")
    panel(axes[2], "C")

    # D: GENERATIVE — harmonically-coupled 2:3 oscillators: as coupling K rises, PC and
    #    R = H·PC rise (R gates on the emergent coupling) while H stays high throughout —
    #    R is low when phase coherence is absent despite present harmonicity. Analog of B.
    s7 = load("study7_coupled_oscillators.json"); p = s7["transition"]["2:3"]
    for key, color, lbl in [("H", ORANGE, "H"), ("PC", PURPLE, "PC"), ("R", RED, "R = H·PC")]:
        v = np.array([d[key] for d in p], float); v = v / (v.max() + 1e-12)
        axes[3].plot([d["K"] for d in p], v, "o-", color=color, ms=3, label=lbl)
    axes[3].set_xlabel("coupling strength K"); axes[3].set_ylabel("normalized")
    axes[3].set_title("Generative (2:3 oscillators):\nR gates H by emergent coupling")
    axes[3].legend(fontsize=6); panel(axes[3], "D")

    fig.suptitle("Figure 2 — Why R = H·PC: a conjunction (beats both factors; gates harmonicity by phase "
                 "coherence) — shown abstractly and generatively", fontsize=9, fontweight="bold", y=1.04)
    fig.tight_layout(); save(fig, "method_Fig2_R_justification")


# --------------------------------------------------------------------------- F8
def figure8():
    """Operating characteristics: SNR robustness, null calibration, scaling."""
    s = load("study24_operating_characteristics.json")
    A = s["snr"]; B = s["null"]; sc = s["scaling"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.7))

    axes[0].plot(A["snrs"], A["H_auc"], "o-", color=BLUE, label="harmonicity H")
    axes[0].plot(A["cpl_snrs"], A["PC_auc"], "s-", color=PURPLE, label="coupling cross PC_z")
    axes[0].axhline(0.5, color="k", ls="--", lw=0.7); axes[0].set_ylim(0.4, 1.05)
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("detection AUC")
    axes[0].set_title("Operating range vs SNR"); axes[0].legend(fontsize=7); panel(axes[0], "A")

    axes[1].bar([0, 1], [B["fpr_005"], B["fpr_001"]], color=GREEN, width=0.6, alpha=0.85)
    axes[1].axhline(0.05, color="k", ls="--", lw=0.8); axes[1].axhline(0.01, color=GREY, ls=":", lw=0.8)
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["α=0.05", "α=0.01"])
    axes[1].set_ylabel("per-frequency false-positive rate")
    axes[1].set_title("PC_z null calibration\n(targets dashed)"); panel(axes[1], "B")

    nf = [r["n_freqs"] for r in sc["by_resolution"]]; ms = [r["sec_per_call"] * 1000 for r in sc["by_resolution"]]
    axes[2].plot(nf, ms, "o-", color=RED)
    axes[2].set_xlabel("n frequency bins"); axes[2].set_ylabel("runtime (ms/call)")
    axes[2].set_title("Scaling: set by n_freqs\n(flat in signal length)"); panel(axes[2], "C")

    fig.suptitle("Figure 4 — Operating characteristics: SNR robustness, null calibration, scaling",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout(); save(fig, "method_Fig4_operating")


# --------------------------------------------------------------------------- F9
def figure9():
    """Baselines: framework factors vs established measures (n:m PLV, HNR)."""
    s = load("study25_baselines.json"); cpl = s["coupling"]; harm = s["harmonicity"]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.7))

    axes[0].plot(cpl["snrs"], cpl["PCz_auc"], "o-", color=PURPLE, label="framework PC_z")
    axes[0].plot(cpl["snrs"], cpl["rawPLV_auc"], "s--", color=GREY, label="raw n:m PLV")
    axes[0].axhline(0.5, color="k", ls=":", lw=0.7); axes[0].set_ylim(0.4, 1.05)
    axes[0].set_xlabel("SNR (dB)"); axes[0].set_ylabel("coupling AUC")
    axes[0].set_title("Coupling: PC_z vs raw PLV"); axes[0].legend(fontsize=7); panel(axes[0], "A")

    axes[1].plot(harm["snrs"], harm["H_auc_inh"], "o-", color=BLUE, label="framework H")
    axes[1].plot(harm["snrs"], harm["HNR_auc_inh"], "s--", color=GREY, label="HNR")
    axes[1].axhline(0.5, color="k", ls=":", lw=0.7); axes[1].set_ylim(0.4, 1.05)
    axes[1].set_xlabel("SNR (dB)"); axes[1].set_ylabel("AUC")
    axes[1].set_title("H vs HNR\n(harmonic vs inharmonic)"); axes[1].legend(fontsize=7); panel(axes[1], "B")

    axes[2].plot(harm["snrs"], harm["H_auc_noise"], "o-", color=BLUE, label="framework H")
    axes[2].plot(harm["snrs"], harm["HNR_auc_noise"], "s--", color=GREY, label="HNR")
    axes[2].axhline(0.5, color="k", ls=":", lw=0.7); axes[2].set_ylim(0.4, 1.05)
    axes[2].set_xlabel("SNR (dB)"); axes[2].set_ylabel("AUC")
    axes[2].set_title("H vs HNR\n(harmonic vs noise)"); axes[2].legend(fontsize=7); panel(axes[2], "C")

    fig.suptitle("Figure 5 — Baselines: framework factors vs established measures (n:m PLV, HNR)",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout(); save(fig, "method_Fig5_baselines")


def main():
    print("Building METHOD-paper figures (M-A spine) from paper-grade results ...")
    figure1(); figure2(); figure3(); figure4(); figure5(); figure6()
    figure7(); figure8(); figure9()
    print(f"Done -> {FIGDIR}")


if __name__ == "__main__":
    main()

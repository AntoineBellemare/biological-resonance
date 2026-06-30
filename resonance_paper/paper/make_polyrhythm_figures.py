"""Generate the polyrhythm-soundness paper figures from the study JSONs.

Fig 1  cardiorespiratory polyrhythm on real data (study44)          -> biosignal motivation
Fig 2  defaults anti-detect ground truth + the convention bug       (study38, study39)
Fig 3  the corrected estimator is sound across ratios               (study40)
Fig 4  no universal technique: regime x technique heatmap           (study41)
Fig 5  the harmonics confound + which null separates it             (study42, study43)
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIGDIR = os.path.join(HERE, "polyrhythm_figs")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120,
    "savefig.dpi": 300, "legend.frameon": False,
})
C_PLV = "#1f3a93"; C_RHO = "#b7410e"; C_MI = "#117733"; C_GREY = "#888888"; C_BAD = "#c0392b"
METRIC_COL = {"plv": C_PLV, "nm_plv": C_PLV, "plv_canonical": C_PLV,
              "rho_entropy": C_RHO, "phase_mi": C_MI, "conditional_prob": "#8856a7",
              "pli": "#5b8fd6", "wpli": "#7aa6c2", "rrci": "#a0b9c9", "wpli_complex": "#6699aa"}


def _load(name):
    return json.load(open(os.path.join(RES, name), encoding="utf-8"))


def _save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".png"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png/.pdf")


# ---------------------------------------------------------------- Fig 1 — cardiorespiratory
def fig1_cardio():
    d = _load("study44_cardiorespiratory.json")
    sync = d["headline_sync"]; rows = d["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    t = np.array(sync["times_s"]) / 60.0; R = np.array(sync["Rw"]); thr = sync["thr95"]
    ax[0].plot(t, R, color=C_PLV, lw=1.5)
    ax[0].axhline(thr, color=C_BAD, ls="--", lw=1, label="surrogate 95th pct")
    ax[0].fill_between(t, thr, R, where=R > thr, color=C_PLV, alpha=0.25, step="mid")
    ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("windowed sync. index $R$")
    ax[0].set_ylim(0, 1.02); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"A  {d['headline_subject']}: intermittent {d['headline_ratio']} cardiorespiratory lock")
    # panel B: per-subject best-ratio z
    subs = [r["subject"] for r in rows]; z = [r["z"] for r in rows]
    cols = [C_PLV if r["rank_p"] <= 0.05 else C_GREY for r in rows]
    ax[1].bar(range(len(subs)), z, color=cols)
    ax[1].axhline(1.96, color=C_BAD, ls="--", lw=1)
    ax[1].set_xticks(range(len(subs)))
    ax[1].set_xticklabels([s.replace("f1", "") for s in subs], rotation=45, ha="right", fontsize=7)
    ax[1].set_ylabel("surrogate $z$ (predicted ratio)")
    ax[1].set_title(f"B  {d['n_significant']}/{d['n_subjects']} subjects locked (p$\\leq$0.05)")
    for i, r in enumerate(rows):
        ax[1].text(i, z[i] + 0.1, r["ratio"], ha="center", fontsize=6, rotation=90)
    fig.suptitle("Polyrhythms are present in biosignals: n:m cardiorespiratory phase coupling (Fantasia)",
                 fontweight="bold", y=1.04)
    _save(fig, "fig1_cardiorespiratory")


# ---------------------------------------------------------------- Fig 2 — defaults fail
def fig2_defaults():
    d38 = _load("study38_polyrhythm_probe.json")["rows"]
    d39 = _load("study39_metric_soundness.json")
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.3), gridspec_kw={"width_ratios": [1.4, 1]})
    labels = ["default\n(stft+plv+binary)", "corrected\n(hilbert+canonical+fraction)", "oracle\n(handed ratio)"]
    deltas = [r["delta"] for r in d38]; cols = [C_BAD if x < 0.1 else C_PLV for x in deltas]
    ax[0].bar(range(3), deltas, color=cols)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(labels, fontsize=7.5)
    ax[0].set_ylabel("detection $\\Delta$ (coupled $-$ uncoupled)")
    ax[0].set_title("A  Default config anti-detects a 2:3 lock")
    for i, v in enumerate(deltas):
        ax[0].text(i, v + (0.03 if v >= 0 else -0.06), f"{v:+.2f}", ha="center", fontsize=8)
    pa = d39["out"]["detection"]["pi/2"]["pooled_auc"]
    names = ["plv", "plv_SWAPPED"]; vals = [pa["plv"], pa["plv_SWAPPED"]]
    ax[1].bar([0, 1], vals, color=[C_PLV, C_BAD])
    ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["correct\nconvention", "kernel\nconvention"], fontsize=7.5)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("detection AUC (pooled)")
    ax[1].set_title("B  The swapped-multiplier bug")
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    _save(fig, "fig2_defaults_fail")


# ---------------------------------------------------------------- Fig 3 — corrected is sound
def fig3_sound():
    d = _load("study40_polyrhythm_battery.json")["out"]
    ident = d["part_a_specificity"]["identification_accuracy"]
    dose = d["part_b_robustness"]["dose_response_kappa"]
    snr = d["part_b_robustness"]["snr_auc"]
    metrics = [m for m in ("plv", "wpli", "rrci") if m in dose]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    ax[0].bar(range(len(ident)), [ident[m] for m in ident],
              color=[METRIC_COL.get(m, C_GREY) for m in ident])
    ax[0].set_xticks(range(len(ident))); ax[0].set_xticklabels(list(ident), rotation=45, ha="right", fontsize=7)
    ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("ratio-ID accuracy")
    ax[0].set_title("A  Specificity: identifies the true ratio")
    for m in metrics:
        ks = sorted(dose[m], key=lambda s: float(s.split("=")[1]))
        ax[1].plot([float(k.split("=")[1]) for k in ks], [dose[m][k] for k in ks],
                   "-o", ms=3, color=METRIC_COL[m], label=m)
    ax[1].set_xlabel("coupling strength $\\kappa$"); ax[1].set_ylabel("mean coupling")
    ax[1].set_title("B  Monotone dose-response"); ax[1].legend(fontsize=7)
    for m in metrics:
        ks = sorted(snr[m], key=lambda s: float(s.replace("dB", "")))
        ax[2].plot([float(k.replace("dB", "")) for k in ks], [snr[m][k] for k in ks],
                   "-o", ms=3, color=METRIC_COL[m], label=m)
    ax[2].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[2].set_xlabel("SNR (dB)"); ax[2].set_ylabel("detection AUC")
    ax[2].set_title("C  Sensitivity vs noise"); ax[2].set_ylim(0.45, 1.03)
    fig.suptitle("The corrected estimator is sound across the ratio range (1:2 … 5:7)",
                 fontweight="bold", y=1.03)
    _save(fig, "fig3_corrected_sound")


# ---------------------------------------------------------------- Fig 4 — regime x technique
def fig4_regime():
    d = _load("study41_technique_bakeoff.json")
    A = d["detection_auc"]; regimes = list(A)
    techs = ["plv_canonical", "pli_canonical", "wpli_canonical", "rrci_canonical",
             "wpli_complex_canonical", "conditional_prob", "rho_entropy", "phase_mi"]
    M = np.array([[A[r][t] for t in techs] for r in regimes])
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(techs)))
    ax.set_xticklabels([t.replace("_canonical", "").replace("_", " ") for t in techs], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(regimes))); ax.set_yticklabels(regimes, fontsize=8)
    for i in range(len(regimes)):
        for j in range(len(techs)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 0.55 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="detection AUC")
    ax.set_title("No universal technique: the best n:m detector is regime-dependent\n"
                 "(first-moment PLV family anti-detects multimodal locks; entropy / MI recover them)")
    _save(fig, "fig4_regime_technique")


# ---------------------------------------------------------------- Fig 5 — the confound
def fig5_confound():
    d42 = _load("study42_false_positive_control.json")["out"]
    d43 = _load("study43_confound_scope.json")["out"]
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.4))
    metrics = ["plv", "rho_entropy", "phase_mi"]
    x = np.arange(len(metrics)); w = 0.38
    neg = [d42["negative"][m]["mean_z"] for m in metrics]
    pos = [d42["positive"][m]["mean_z"] for m in metrics]
    ax[0].bar(x - w / 2, neg, w, color=C_BAD, label="harmonic artifact (1 oscillator)")
    ax[0].bar(x + w / 2, pos, w, color=C_PLV, label="genuine coupling")
    ax[0].axhline(1.96, color="k", ls=":", lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(metrics, fontsize=8)
    ax[0].set_ylabel("surrogate $z$ (IAAFT null)")
    ax[0].set_title("A  Within-signal harmonics false-positive"); ax[0].legend(fontsize=7)
    nulls = ["iaaft", "phase_randomize", "time_shuffle"]
    negm = {n: np.mean([d43["within_harmonic_NEG"][f"{n}/{m}"] for m in metrics]) for n in nulls}
    posm = {n: np.mean([d43["cross_genuine_POS"][f"{n}/{m}"] for m in metrics]) for n in nulls}
    xn = np.arange(len(nulls))
    ax[1].bar(xn - w / 2, [negm[n] for n in nulls], w, color=C_BAD, label="artifact (want $\\approx$0)")
    ax[1].bar(xn + w / 2, [posm[n] for n in nulls], w, color=C_PLV, label="genuine")
    ax[1].axhline(1.96, color="k", ls=":", lw=0.8)
    ax[1].set_xticks(xn); ax[1].set_xticklabels(["IAAFT", "phase-rand", "time-shuffle"], fontsize=8)
    ax[1].set_ylabel("mean surrogate $z$")
    ax[1].set_title("B  No null fully removes the artifact"); ax[1].legend(fontsize=7)
    fig.suptitle("n:m at harmonically related frequencies is confounded within one signal "
                 "→ sound only cross-signal", fontweight="bold", y=1.03)
    _save(fig, "fig5_harmonic_confound")


def fig6_sleep():
    d = _load("study47_sleep_spectrum.json")
    freqs = np.array(d["freqs"]); stages = d["stages"]; metrics = list(d["pc_mean"].keys())
    scol = {"Wake": "#e69f00", "N2": "#56b4e9", "N3": "#0072b2", "REM": "#cc79a7"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.1), sharex=True)
    for ax, m in zip(axes, metrics):
        for s in stages:
            sp = d["pc_mean"][m].get(s)
            if sp:
                ax.plot(freqs, np.array(sp), color=scol.get(s, "#888"), label=s, lw=1.6)
        ax.set_title(m.replace("nm_", "").replace("_canonical", ""))
        ax.set_xlabel("frequency (Hz)")
        sd = d["stage_dependence"][m]
        ax.text(0.96, 0.96, f"{sd['n_freqs_sig']}/{sd['n_freqs']} bins\nstage-dep.\n(FDR$\\leq$0.05)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    axes[0].set_ylabel("n:m phase coupling\n(power-independent)")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("The n:m phase-coupling spectrum reorganizes across sleep stages "
                 "(Fpz-Cz, Sleep-EDF; whole-spectrum, all pairs)", fontweight="bold", y=1.02)
    _save(fig, "fig6_sleep_spectrum")


def main():
    made = []
    for fn, name in [(fig1_cardio, "Fig1"), (fig2_defaults, "Fig2"), (fig3_sound, "Fig3"),
                     (fig4_regime, "Fig4"), (fig5_confound, "Fig5")]:
        try:
            fn(); made.append(name)
        except Exception as exc:
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
    print(f"  done: {made} -> {FIGDIR}")


if __name__ == "__main__":
    main()

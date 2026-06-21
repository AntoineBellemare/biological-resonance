"""Compose the real-data criticality/state composite figure (Fig 5) + a summary.

Reads results/study13_anesthesia.json, study14_sleep.json,
study15_deep_anesthesia.json and assembles:
  A. state decoding accuracy (resonance vs band power vs criticality vs all) per
     dataset  -- the SOLID result: H/R discriminate brain states.
  B. H_max by state per dataset                -- H drops in the deepest states.
  C. per-subject rho(H, criticality proximity) -- the (honest, mixed) criticality test.

Run after the paper-grade studies 13-15. Prints a plain-text synthesis too.
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

BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999", "#CC79A7"
DATASETS = [("study13_anesthesia.json", "Propofol\nsedation"),
            ("study14_sleep.json", "Sleep\n(wake→N3)"),
            ("study15_deep_anesthesia.json", "Deep GA\n(wake→LOC)")]
# canonical "depth" ordering per dataset for panel B
ORDER = {"study13_anesthesia.json": ["baseline", "mild sedation", "moderate sedation", "recovery"],
         "study14_sleep.json": ["Wake", "N1", "N2", "N3", "REM"],
         "study15_deep_anesthesia.json": ["wake", "unconscious"]}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    data = [(load(f), lab, f) for f, lab in DATASETS]
    data = [(d, lab, f) for d, lab, f in data if d is not None]
    if not data:
        print("no results yet"); return
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
                         "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    # A. state decoding accuracy
    ax = axes[0]; n = len(data); w = 0.2
    for i, key in enumerate(["band_power", "resonance", "criticality", "all"]):
        col = {"band_power": GREY, "resonance": ORANGE, "criticality": GREEN, "all": PURPLE}[key]
        vals = [d["state_decoding"].get(key, np.nan) for d, _, _ in data]
        ax.bar(np.arange(n) + (i - 1.5) * w, vals, w, color=col, label=key.replace("_", " "))
    ch = [d["state_decoding"]["chance"] for d, _, _ in data]
    for j, c in enumerate(ch):
        ax.plot([j - 0.4, j + 0.4], [c, c], "k--", lw=0.8)
    ax.set_xticks(range(n)); ax.set_xticklabels([lab for _, lab, _ in data], fontsize=7)
    ax.set_ylabel("multiclass accuracy (LOSO)"); ax.set_ylim(0, 1.05)
    ax.set_title("A. Resonance/H discriminate brain state", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    # B. H_max by state (normalized within dataset to show the shape)
    ax = axes[1]
    for (d, lab, f), mk, col in zip(data, "os^", [RED, BLUE, GREEN]):
        states = [s for s in ORDER.get(f, d["states"]) if s in d["by_state"]]
        H = np.array([d["by_state"][s]["H_max"] for s in states], float)
        Hn = (H - np.nanmin(H)) / (np.nanmax(H) - np.nanmin(H) + 1e-12)
        ax.plot(range(len(states)), Hn, mk + "-", color=col, label=lab.replace("\n", " "))
    ax.set_xlabel("state (light → deep →)"); ax.set_ylabel("H_max (norm. within dataset)")
    ax.set_title("B. H drops in the deepest states", fontsize=9); ax.legend(fontsize=6)

    # C. rho(H, criticality proximity) with CIs
    ax = axes[2]; labels, rhos, los, his = [], [], [], []
    for d, lab, f in data:
        for mk in ["H_max_vs_prox_m", "H_max_vs_prox_dfa"]:
            v = d["primary"].get(mk, {})
            if np.isfinite(v.get("mean_rho", np.nan)):
                labels.append(f"{lab.splitlines()[0][:6]}:{'m̂' if 'prox_m' in mk else 'DFA'}")
                rhos.append(v["mean_rho"]); los.append(v["mean_rho"] - v["lo"]); his.append(v["hi"] - v["mean_rho"])
    y = range(len(labels))
    ax.barh(list(y), rhos, xerr=[los, his], color=[BLUE if r > 0 else RED for r in rhos], capsize=2)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("ρ(H, criticality proximity)")
    ax.set_title("C. H vs criticality (mixed sign)", fontsize=9); ax.invert_yaxis()

    fig.suptitle("Figure 5 — Resonance tracks brain state across modalities; criticality link is state-mediated",
                 fontsize=9.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"Fig5_realdata.{ext}")
    plt.close(fig)
    print(f"wrote {FIGDIR}/Fig5_realdata.png (+pdf)")

    # text synthesis
    print("\n=== REAL-DATA SYNTHESIS ===")
    for d, lab, f in data:
        sd = d["state_decoding"]
        pm = d["primary"].get("H_max_vs_prox_m", {})
        print(f"\n{lab.replace(chr(10),' ')} (n={d['n_subjects']}, {d['n_windows']} windows):")
        print(f"  state decoding (chance {sd['chance']:.2f}): resonance={sd['resonance']:.2f} "
              f"band_power={sd['band_power']:.2f} criticality={sd['criticality']:.2f} all={sd['all']:.2f}")
        if np.isfinite(pm.get("mean_rho", np.nan)):
            print(f"  rho(H, prox_m)={pm['mean_rho']:+.2f} [{pm['lo']:+.2f},{pm['hi']:+.2f}] (p={pm['wilcoxon_p']:.2g})")


if __name__ == "__main__":
    main()

"""Study 17 — The tripartite dissociation: H ⟂ PC, and R's specificity.

The non-circular capstone for the H / PC / R construct. A 2-D grid independently
varies:
  * harmonicity   — frequency-ratio complexity p:q (simple 1:2 -> complex 5:11),
                    via the Tenney height log2(p*q). Sets how *harmonic* the pair is.
  * phase-locking — coupling strength kappa (0 = drifting phases, 1 = rigid n:m
                    lock). Sets how *phase-coupled* the pair is, independent of the
                    ratio (any rational p:q can mode-lock).

Predictions:
  (1) H tracks the harmonicity axis and is FLAT along locking (H ⟂ phase);
  (2) PC tracks the locking axis and is FLAT along harmonicity (PC ⟂ ratio);
  (3) R = H·PC peaks ONLY in the simple-AND-locked corner;
  (4) the payoff — R is MORE SPECIFIC than H or PC alone: it rejects the two
      single-factor confounds (harmonic-but-unlocked; complex-but-locked) that
      H-alone / PC-alone falsely accept. AUC(true resonance) for R > H, PC.

H = framework harmonicity of the pair (harmonicity_matrix); PC = clean n:m PLV;
R = H_norm * PC.

Outputs: results/study17_tripartite_dissociation.json, figures/study17_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from resonance_paper import _common as C
from resonance_paper import signals as S
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import ResonanceConfig

# ratios spanning complexity (Tenney height p*q), all with resolvable f1,f2
RATIOS = [(1, 2), (2, 3), (3, 4), (2, 5), (3, 7), (4, 7), (3, 8), (4, 9), (5, 9), (5, 11)]
BASE = 4.0
SF = 500.0
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, coupling_metric="nm_plv_canonical",
                      ratio_kernel="fraction", ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)


def make_pair(p, q, kappa, dur=20.0, snr_db=6.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(SF * dur); t = np.arange(n) / SF
    f1 = BASE; f2 = BASE * q / p
    ph1 = 2 * np.pi * f1 * t + rng.uniform(0, 2 * np.pi)
    drift = np.cumsum(rng.standard_normal(n)) * (1.0 - kappa) * 0.6   # unlocking = phase random walk
    ph2 = 2 * np.pi * f2 * t + rng.uniform(0, 2 * np.pi) + drift
    s1 = S._norm(np.sin(ph1)); s2 = S._norm(np.sin(ph2))
    nz = 10 ** (-snr_db / 20.0)
    s1 = S._norm(s1 + nz * S.pink_noise(n, SF, seed=seed + 5))
    s2 = S._norm(s2 + nz * S.pink_noise(n, SF, seed=seed + 9))
    return s1.astype(np.float64), s2.astype(np.float64), f1, f2


def nm_plv(s1, s2, f1, f2, p, q, bw=2.0):
    def phase(s, f):
        b, a = butter(4, [(f - bw) / (SF / 2), (f + bw) / (SF / 2)], btype="band")
        return np.angle(hilbert(filtfilt(b, a, s)))
    th1 = phase(s1, f1); th2 = phase(s2, f2)
    return float(np.abs(np.mean(np.exp(1j * (q * th1 - p * th2)))))


def run(quick=True):
    kappas = [0.0, 0.25, 0.5, 0.75, 1.0] if quick else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    seeds = range(4) if quick else range(12)
    rows = []
    for (p, q) in RATIOS:
        comp = float(np.log2(p * q))            # Tenney height = inverse harmonicity
        for kappa in kappas:
            Hs, PCs = [], []
            for seed in seeds:
                s1, s2, f1, f2 = make_pair(p, q, kappa, seed=seed)
                r = compute_cross_resonance(s1, s2, sf=SF, config=CFG)
                fr = r.freqs; i = int(np.argmin(np.abs(fr - f1))); j = int(np.argmin(np.abs(fr - f2)))
                Hs.append(float(r.harmonicity_matrix[i, j]))
                PCs.append(nm_plv(s1, s2, f1, f2, p, q))
            rows.append(dict(p=p, q=q, complexity=comp, kappa=kappa,
                             H=float(np.mean(Hs)), PC=float(np.mean(PCs))))
        print(f"  ratio {p}:{q} (complexity {comp:.1f}) done", flush=True)

    H = np.array([r["H"] for r in rows]); PC = np.array([r["PC"] for r in rows])
    Hn = (H - H.min()) / (H.ptp() + 1e-12)
    for r, hn in zip(rows, Hn):
        r["H_norm"] = float(hn); r["R"] = float(hn * r["PC"])

    # (1)(2) independence: corr of each factor with each axis
    from scipy.stats import spearmanr
    comp = np.array([r["complexity"] for r in rows]); kap = np.array([r["kappa"] for r in rows])
    Hv = np.array([r["H"] for r in rows]); PCv = np.array([r["PC"] for r in rows]); Rv = np.array([r["R"] for r in rows])
    independence = dict(
        H_vs_complexity=float(spearmanr(comp, Hv)[0]), H_vs_kappa=float(spearmanr(kap, Hv)[0]),
        PC_vs_complexity=float(spearmanr(comp, PCv)[0]), PC_vs_kappa=float(spearmanr(kap, PCv)[0]))

    # (4) specificity: true resonance = simple (complexity<=median) AND locked (kappa>=0.75)
    cthr = np.median(comp); true = (comp <= cthr) & (kap >= 0.75)
    confound = ((comp <= cthr) & (kap <= 0.25)) | ((comp > cthr) & (kap >= 0.75))
    spec = {}
    for name, v in [("H", Hv), ("PC", PCv), ("R", Rv)]:
        ci = C.bootstrap_auc_ci(list(v[true]), list(v[confound]))
        spec[name] = dict(auc=ci["auc"], lo=ci["lo"], hi=ci["hi"])

    result = dict(quick=quick, ratios=[list(r) for r in RATIOS], rows=rows,
                  independence=independence, specificity=spec)
    C.save_json(result, "study17_tripartite_dissociation.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    ind = result["independence"]; sp = result["specificity"]
    print("\n  --- Study 17 headline (tripartite dissociation) ---")
    print("  Independence (each factor should track ITS axis, be flat on the other):")
    print(f"    H : vs complexity rho={ind['H_vs_complexity']:+.2f} | vs kappa rho={ind['H_vs_kappa']:+.2f}")
    print(f"    PC: vs complexity rho={ind['PC_vs_complexity']:+.2f} | vs kappa rho={ind['PC_vs_kappa']:+.2f}")
    print("  Specificity AUC (true=simple&locked vs single-factor confounds; R should win):")
    for k in ("H", "PC", "R"):
        print(f"    {k}: AUC={sp[k]['auc']:.2f} [{sp[k]['lo']:.2f},{sp[k]['hi']:.2f}]")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]
    comps = sorted(set(r["complexity"] for r in rows)); kaps = sorted(set(r["kappa"] for r in rows))
    def grid(key):
        M = np.full((len(comps), len(kaps)), np.nan)
        for r in rows:
            M[comps.index(r["complexity"]), kaps.index(r["kappa"])] = r[key]
        return M
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key, title in zip(axes[:3], ["H", "PC", "R"],
                              ["H (harmonicity)", "PC (phase locking)", "R = H·PC"]):
        im = ax.imshow(grid(key), aspect="auto", origin="lower", cmap="viridis",
                       extent=[min(kaps), max(kaps), min(comps), max(comps)])
        ax.set_xlabel("phase-locking κ"); ax.set_ylabel("ratio complexity log2(pq)")
        ax.set_title(title, fontsize=10); plt.colorbar(im, ax=ax, fraction=0.046)
    sp = result["specificity"]
    axes[3].bar(["H", "PC", "R"], [sp[k]["auc"] for k in ("H", "PC", "R")],
                color=["#ef6c00", "#6a1b9a", "#b71c1c"])
    axes[3].axhline(0.5, color="k", ls="--", lw=0.6); axes[3].set_ylim(0, 1.05)
    axes[3].set_ylabel("AUC (true resonance vs confounds)")
    axes[3].set_title("R rejects single-factor\nconfounds best", fontsize=10)
    fig.suptitle("Study 17 — H ⟂ PC dissociation; R = H·PC is the specific conjunction",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study17_tripartite_dissociation")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 4 — Strategy comparison (method guidance).

The resonance framework is a registry of swappable strategies. Which
combination of harmonic kernel x ratio kernel x coupling metric best serves a
given goal? We score every combination on a clean discrimination task with
known ground truth: separating harmonic signals from (a) pink noise and
(b) an inharmonic pair, using the resonance peak R_max.

Output: a ranked table + heatmap of discrimination AUC by strategy, giving
practical guidance on configuration.

Outputs: results/study4_strategy_comparison.json, figures/study4_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import signals as S
from resonance_paper.study1_ground_truth import _ladder_signal
from biotuner.resonance import compute_resonance


def run(quick=True):
    import time as _time
    sf, duration, snr_db = 500.0, 30.0, 6.0
    seeds = range(4) if quick else range(15)
    # subharm_tension is ~37x slower than harmsim per call (a reportable
    # performance finding); use a coarser grid for the sweep so it is tractable.
    precision_hz = 1.0 if quick else 0.5
    timing = {}
    # build the labeled signal bank once (independent of strategy)
    bank = {"harmonic": [], "pink": [], "inharmonic": []}
    for seed in seeds:
        bank["harmonic"].append(_ladder_signal("triad", sf, duration, snr_db, seed)[0])
        bank["pink"].append(_ladder_signal("pink", sf, duration, snr_db, seed)[0])
        bank["inharmonic"].append(_ladder_signal("inharmonic", sf, duration, snr_db, seed)[0])

    harmonic_kernels = C.HARMONIC_KERNELS          # harmsim, subharm_tension
    ratio_kernels = C.RATIO_KERNELS                # binary, fraction
    coupling_metrics = C.COUPLING_METRICS          # 5 metrics

    results = []
    for hk in harmonic_kernels:
        for rk in ratio_kernels:
            for cm in coupling_metrics:
                from biotuner.resonance import ResonanceConfig
                cfg = ResonanceConfig(
                    precision_hz=precision_hz, fmin=2, fmax=45,
                    harmonic_kernel=hk, ratio_kernel=rk,
                    ratio_kernel_params=C.ratio_params_for(rk),
                    coupling_metric=cm, combine="product",
                )
                feats = {k: [] for k in bank}
                _t0 = _time.time()
                for label, sigs in bank.items():
                    for s in sigs:
                        res = compute_resonance(s, sf=sf, config=cfg)
                        f = C.resonance_features(res)
                        # use R_max (combines H and PC) and H_max as descriptors
                        feats[label].append((f["R_max"], f["H_max"]))
                rmax = {k: [v[0] for v in feats[k]] for k in feats}
                hmax = {k: [v[1] for v in feats[k]] for k in feats}
                timing.setdefault(hk, []).append(_time.time() - _t0)
                auc_h_pink = C.roc_auc(rmax["harmonic"], rmax["pink"])
                auc_h_inh = C.roc_auc(rmax["harmonic"], rmax["inharmonic"])
                results.append(dict(
                    harmonic_kernel=hk, ratio_kernel=rk, coupling_metric=cm,
                    label=f"{hk}|{rk}|{cm}",
                    auc_harmonic_vs_pink=auc_h_pink,
                    auc_harmonic_vs_inharmonic=auc_h_inh,
                    auc_mean=np.nanmean([auc_h_pink, auc_h_inh]),
                ))
        print(f"  harmonic_kernel={hk} swept")

    results.sort(key=lambda r: -(r["auc_mean"] if np.isfinite(r["auc_mean"]) else 0))
    n_sig = 3 * len(list(seeds))
    runtime_per_call = {hk: float(np.mean(ts) / n_sig) for hk, ts in timing.items()}
    out = dict(quick=quick, n_seeds=len(list(seeds)), precision_hz=precision_hz,
               runtime_per_call_s=runtime_per_call, table=results)
    C.save_json(out, "study4_strategy_comparison.json")
    _figures(out)
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 4 headline ---")
    print("  Best strategies (R_max discrimination, mean of harmonic-vs-pink & vs-inharmonic):")
    for r in out["table"][:5]:
        print(f"    {r['label']:45s} AUC_mean={r['auc_mean']:.3f}  "
              f"(pink {r['auc_harmonic_vs_pink']:.2f}, inh {r['auc_harmonic_vs_inharmonic']:.2f})")
    print("  Worst:")
    for r in out["table"][-2:]:
        print(f"    {r['label']:45s} AUC_mean={r['auc_mean']:.3f}")
    print("  Runtime per compute_resonance by harmonic kernel:")
    for hk, t in out.get("runtime_per_call_s", {}).items():
        print(f"    {hk:16s} {t:.2f}s")


def _figures(out):
    plt = C.setup_mpl()
    tbl = out["table"]
    # heatmap: rows = harmonic_kernel|ratio_kernel, cols = coupling_metric
    hks = C.HARMONIC_KERNELS; rks = C.RATIO_KERNELS; cms = C.COUPLING_METRICS
    rowlabels = [f"{h}|{r}" for h in hks for r in rks]
    M = np.full((len(rowlabels), len(cms)), np.nan)
    for r in tbl:
        ri = rowlabels.index(f"{r['harmonic_kernel']}|{r['ratio_kernel']}")
        ci = cms.index(r["coupling_metric"])
        M[ri, ci] = r["auc_mean"]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(cms))); ax.set_xticklabels(cms, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(rowlabels))); ax.set_yticklabels(rowlabels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.8 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, label="discrimination AUC (mean)")
    ax.set_title("Study 4 — Strategy comparison: harmonic-signal discrimination AUC\n"
                 "(rows: harmonic kernel | ratio kernel; cols: coupling metric)",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study4_strategy_comparison")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 21 — Network layer: cross-resonance CONNECTIVITY matrices.

Validates the multichannel layer (n_elec x n_elec): given a planted coupled cluster,
the cross-resonance connectivity matrix recovers it. We plant a harmonically
mode-locked family (channels 0-3 share one drifting phase at 6/12/18 Hz, simple
ratios) among independent channels (4-7, incommensurate frequencies, independent
phase), and ask how well each factor's connectivity matrix separates the within-
cluster pairs from the rest (AUC).

HONEST SCOPE (important): the per-pair scalar reducer (`peak_to_median` of the
cross spectrum) is **broadband/overlap-biased** — it rewards channels that share
spectral structure, so the recovery here is driven by the cluster's shared
harmonics as much as by phase coupling. The H matrix is diffuse, the PC matrix is
sharper, and R = H*PC ~ PC (H is near-uniform within the harmonic cluster). To
isolate PURE cross-channel phase coupling above a power-preserving null, use
`compute_cross_resonance_connectivity_zscore(surrogate_kind='iaaft')` (which
isolates phase: H does not survive a PSD-preserving surrogate).

Outputs: results/study21_connectivity.json, figures/study21_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.signals import _norm, pink_noise
from biotuner.harmonic_connectivity import harmonic_connectivity
from biotuner.resonance import ResonanceConfig

SF = 500.0
DUR = 20.0
N_ELEC = 8
CLUSTER = [0, 1, 2, 3]
NOISE = 0.6
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, noverlap=400,
                      coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                      ratio_kernel_params={"max_denom": 16, "beta": 1.0}, return_intermediates=True)


def make_network(seed=0):
    rng = np.random.default_rng(seed)
    n = int(SF * DUR); t = np.arange(n) / SF
    drift = np.cumsum(rng.standard_normal(n)) * 0.0025                  # SHARED slow drift
    theta = 2 * np.pi * 6.0 * t + drift
    chans = [np.sin(theta),                                            # 6 Hz   (cluster)
             np.sin(2 * theta + rng.uniform(0, 2 * np.pi)),            # 12 Hz  1:2
             np.sin(3 * theta + rng.uniform(0, 2 * np.pi)),            # 18 Hz  1:3
             np.sin(theta + np.pi / 3)]                                # 6 Hz   1:1 phase-shift
    for f in [7.3, 9.7, 11.1, 13.9]:                                  # independent channels
        d = np.cumsum(rng.standard_normal(n)) * 0.02
        chans.append(np.sin(2 * np.pi * f * t + d + rng.uniform(0, 2 * np.pi)))
    data = np.array([_norm(_norm(c) + NOISE * pink_noise(n, SF, seed=int(rng.integers(1_000_000))))
                     for c in chans])
    return data.astype(np.float64)


def _labels():
    cl = set(CLUSTER); lab = {}
    for i in range(N_ELEC):
        for j in range(N_ELEC):
            if i != j:
                lab[(i, j)] = 1 if (i in cl and j in cl) else 0
    return lab


def _auc(M, labels):
    pos = [M[i, j] for (i, j), y in labels.items() if y == 1 and np.isfinite(M[i, j])]
    neg = [M[i, j] for (i, j), y in labels.items() if y == 0 and np.isfinite(M[i, j])]
    return C.bootstrap_auc_ci(pos, neg)


def run(quick=True):
    seeds = range(4) if quick else range(10)
    labels = _labels()
    per_factor = {f: [] for f in ("H", "PC", "R")}
    example = {}
    for si, seed in enumerate(seeds):
        data = make_network(seed)
        hc = harmonic_connectivity(sf=SF, data=data, precision=0.5, min_freq=2, max_freq=45)
        for f in ("H", "PC", "R"):
            M = hc.compute_cross_resonance_connectivity(config=CFG, factor=f,
                                                        aggregate="peak_to_median", graph=False)
            per_factor[f].append(_auc(M, labels)["auc"])
            if si == 0:
                example[f] = M.tolist()
        print(f"  seed {seed}: " + " ".join(f"AUC_{f}={per_factor[f][-1]:.2f}" for f in ("H", "PC", "R")), flush=True)

    summary = {f: dict(auc_mean=float(np.mean(per_factor[f])), ci=C.mean_ci(per_factor[f]))
               for f in ("H", "PC", "R")}
    result = dict(quick=quick, n_elec=N_ELEC, cluster=CLUSTER, per_factor=per_factor,
                  example=example, summary=summary)
    C.save_json(result, "study21_connectivity.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    s = result["summary"]
    print("\n  --- Study 21 headline (cross-resonance connectivity) ---")
    for f in ("H", "PC", "R"):
        print(f"    {f} matrix: AUC(within-cluster vs rest) = {s[f]['auc_mean']:.2f} "
              f"[{s[f]['ci']['lo']:.2f},{s[f]['ci']['hi']:.2f}]")
    print("  H is diffuse, PC sharper, R ~ PC (H near-uniform in the harmonic cluster).")
    print("  Recovery is partly spectral-overlap-driven; surrogate-z isolates pure phase.")


def _figures(result):
    plt = C.setup_mpl()
    ex = result["example"]; summ = result["summary"]
    c0, c1 = min(CLUSTER), max(CLUSTER)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    for ax, f, ttl in [(axes[0], "H", "H connectivity (diffuse)"),
                       (axes[1], "PC", "PC connectivity (sharper)")]:
        im = ax.imshow(np.array(ex[f], float), cmap="magma")
        ax.add_patch(plt.Rectangle((c0 - 0.5, c0 - 0.5), c1 - c0 + 1, c1 - c0 + 1,
                                   fill=False, edgecolor="#00e5ff", lw=1.8))
        ax.set_title(ttl, fontsize=10); ax.set_xlabel("channel"); ax.set_ylabel("channel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    keys = ["H", "PC", "R"]; cols = ["#E69F00", "#CC79A7", "#D55E00"]
    aucs = [summ[k]["auc_mean"] for k in keys]
    lo = [summ[k]["auc_mean"] - summ[k]["ci"]["lo"] for k in keys]
    hi = [summ[k]["ci"]["hi"] - summ[k]["auc_mean"] for k in keys]
    axes[2].bar(range(3), aucs, color=cols, alpha=0.9, yerr=[lo, hi], capsize=3, error_kw=dict(lw=0.8))
    axes[2].axhline(0.5, color="k", ls="--", lw=0.7); axes[2].set_ylim(0, 1.05)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(keys)
    axes[2].set_ylabel("AUC (within-cluster vs rest)"); axes[2].set_title("Cluster recovery\n(R ~ PC)", fontsize=10)

    fig.suptitle("Study 21 — Cross-resonance connectivity recovers a planted coupled cluster "
                 "(overlap-driven; surrogate-z for pure phase)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    C.save_fig(fig, "study21_connectivity")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

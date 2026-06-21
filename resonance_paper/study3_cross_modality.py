"""Study 3 — Cross-modality generality.

Does the same single-signal resonance framework yield meaningful,
modality-characteristic structure across signal types? We profile four
modalities at matched sf/duration:

  ECG (neurokit2, known HR)  — QRS gives a dense integer-harmonic series
  EEG-alpha (eyes-closed-like) — 1/f + bursty alpha + weak beta harmonic
  harmonic stack (synthetic)   — clean phase-locked partials (max harmonicity)
  pink noise                   — aperiodic control (min harmonicity)

Predictions
-----------
P1  Harmonicity ranks ECG, harmonic-stack > EEG-alpha > pink noise.
P2  The resonance/complexity feature vector separates modalities (high
    multiclass classification accuracy; clear clustering in feature space).

Outputs: results/study3_cross_modality.json, figures/study3_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import datasets as D
from biotuner.resonance import compute_resonance


FEATURE_KEYS = [
    "H_avg", "H_max", "H_flatness", "H_entropy", "H_spread",
    "PC_avg", "R_avg", "R_max", "R_flatness", "R_entropy",
    "R_peak_harmsim_avg", "R_peak_harmsim_max",
]


def run(quick=True):
    sf, duration = 500.0, 30.0
    n_per = 4 if quick else 12
    bundle = D.modality_bundle(sf=sf, duration=duration, n_per_class=n_per, seed=0)
    if not bundle:
        print("  modality bundle empty; aborting study 3.")
        return None
    print(f"  built {len(bundle)} signals across modalities")

    # Shared wide band starting at 0.5 Hz so the ECG harmonic series (fundamental
    # ~1 Hz at 60 bpm and its integer multiples) is captured alongside EEG bands.
    cfg = C.default_config(fmin=0.5, fmax=45, precision_hz=0.5)
    rows = []
    spectra = {}  # one example R spectrum per label
    for k, item in enumerate(bundle):
        res = compute_resonance(item["signal"], sf=item["sf"], config=cfg)
        feats = C.resonance_features(res)
        row = dict(modality=item["modality"], label=item["label"])
        for fk in FEATURE_KEYS:
            row[fk] = feats.get(fk, np.nan)
        rows.append(row)
        if item["label"] not in spectra:
            spectra[item["label"]] = dict(freqs=res.freqs.tolist(),
                                          R=res.resonance_spectrum.tolist(),
                                          H=res.factors["H"].tolist())
        if (k + 1) % 8 == 0:
            print(f"    {k+1}/{len(bundle)}")

    # harmonicity ranking by label, across several summaries. H_avg (mean over
    # the spectrum) is confounded by broadband energy (pink noise accrues
    # spurious harmonicity from accidental simple ratios), whereas peak-based
    # summaries (H_max, R_max, peak_harmsim) cleanly track harmonic richness.
    labels = sorted(set(r["label"] for r in rows))
    ranking = {}
    for metric in ("H_avg", "H_max", "R_max", "R_peak_harmsim_avg"):
        ranking[metric] = {lab: float(np.nanmean([r[metric] for r in rows if r["label"] == lab]))
                           for lab in labels}

    # multiclass separability
    clf = _classify(rows)

    result = dict(quick=quick, n_signals=len(rows), rows=rows,
                  harmonicity_ranking=ranking, example_spectra=spectra,
                  classification=clf, feature_keys=FEATURE_KEYS)
    C.save_json(result, "study3_cross_modality.json")
    _figures(result)
    _headline(result)
    return result


def _classify(rows):
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
    labels = sorted(set(r["label"] for r in rows))
    lab_idx = {l: i for i, l in enumerate(labels)}
    X = np.nan_to_num(np.array([[r[fk] for fk in FEATURE_KEYS] for r in rows], dtype=float))
    y = np.array([lab_idx[r["label"]] for r in rows])
    n_splits = min(4, np.bincount(y).min())
    if n_splits < 2:
        return {"error": "too few samples per class"}
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=0))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    obs_acc = float(scores.mean())

    # permutation test: shuffle labels, recompute CV accuracy
    n_perm = 200
    rng = np.random.default_rng(0)

    def _null(_rng):
        yp = _rng.permutation(y)
        sk = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        return float(cross_val_score(model, X, yp, cv=sk, scoring="accuracy").mean())
    perm = C.permutation_test_value(obs_acc, _null, n_perm=n_perm, seed=0)
    return dict(labels=labels, cv_accuracy_mean=obs_acc,
                cv_accuracy_std=float(scores.std()), chance=1.0 / len(labels),
                n_splits=n_splits, perm_p=perm["p"], perm_null_mean=perm["null_mean"])


def _headline(result):
    print("\n  --- Study 3 headline ---")
    for metric, ranks in result["harmonicity_ranking"].items():
        order = sorted(ranks.items(), key=lambda kv: -kv[1])
        print(f"  {metric} ranking: " + "  ".join(f"{lab}={v:.3f}" for lab, v in order))
    clf = result["classification"]
    if "error" not in clf:
        pp = clf.get("perm_p", float("nan"))
        print(f"  Multiclass modality classification accuracy = "
              f"{clf['cv_accuracy_mean']:.3f} +/- {clf['cv_accuracy_std']:.3f} "
              f"(chance {clf['chance']:.2f}, perm p={pp:.3g})")


def _figures(result):
    plt = C.setup_mpl()
    rows = result["rows"]; FEATURE_KEYS = result["feature_keys"]
    labels = sorted(set(r["label"] for r in rows))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    # (1) example R spectra
    spec = result["example_spectra"]
    colors = {"ECG": "#c62828", "EEG_alpha": "#1565c0",
              "harmonic_stack": "#2e7d32", "pink_noise": "#757575"}
    for lab, sd in spec.items():
        axes[0].plot(sd["freqs"], sd["R"], label=lab, color=colors.get(lab, None))
    axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("Resonance R(f)")
    axes[0].set_title("Example resonance spectra by modality"); axes[0].legend(fontsize=8)

    # (2) feature fingerprint heatmap (z-scored across signals, mean per label)
    X = np.array([[r[fk] for fk in FEATURE_KEYS] for r in rows], dtype=float)
    Xz = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-12)
    M = np.array([[np.nanmean(Xz[[i for i, r in enumerate(rows) if r["label"] == lab], j])
                   for lab in labels] for j in range(len(FEATURE_KEYS))])
    im = axes[1].imshow(M, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
    axes[1].set_xticks(range(len(labels))); axes[1].set_xticklabels(labels, rotation=30, fontsize=8)
    axes[1].set_yticks(range(len(FEATURE_KEYS))); axes[1].set_yticklabels(FEATURE_KEYS, fontsize=7)
    axes[1].set_title("Resonance-feature fingerprints (z-scored)")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # (3) PCA scatter
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Z = StandardScaler().fit_transform(np.nan_to_num(X))
        pc = PCA(n_components=2).fit_transform(Z)
        for lab in labels:
            idx = [i for i, r in enumerate(rows) if r["label"] == lab]
            axes[2].scatter(pc[idx, 0], pc[idx, 1], label=lab, s=40,
                            color=colors.get(lab, None), alpha=0.8)
        axes[2].set_xlabel("PC1"); axes[2].set_ylabel("PC2")
        clf = result["classification"]
        acc = clf.get("cv_accuracy_mean", float("nan"))
        axes[2].set_title(f"Feature-space PCA (modality acc={acc:.2f})")
        axes[2].legend(fontsize=8)
    except Exception as e:  # pragma: no cover
        axes[2].text(0.5, 0.5, f"PCA unavailable\n{e}", ha="center")

    fig.suptitle("Study 3 — Cross-modality: resonance features fingerprint biosignal type",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study3_cross_modality")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

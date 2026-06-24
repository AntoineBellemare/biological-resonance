"""Study 2 — EEG state discrimination (rigorous).

Two real-EEG state contrasts from PhysioNet eegbci:
  * eyes-open vs eyes-closed (occipital)  — alpha-enhancement manipulation
  * rest vs motor-execution (motor cortex) — mu/beta task modulation

Tier-1 rigor: per-subject paired tests (Wilcoxon on subject means) + rank-biserial
effect sizes, bootstrap CIs on per-feature AUC, and a permutation test on the
leave-one-subject-out classification AUC. Tier-2: harmonicity computed with
aperiodic removal by default, plus an explicit with-vs-without comparison to show
the 1/f confound is controlled on real data.

Outputs: results/study2_eeg_states.json, figures/study2_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import datasets as D
from biotuner.resonance import compute_resonance


def _alpha_power(signal, sf, lo=8, hi=12):
    from scipy.signal import welch
    f, p = welch(signal, fs=sf, nperseg=min(len(signal), int(sf * 2)))
    band = (f >= lo) & (f <= hi)
    total = (f >= 2) & (f <= 45)
    return float(p[band].sum() / (p[total].sum() + 1e-20))


FEATURE_KEYS = [
    "H_avg", "H_max", "H_flatness", "H_entropy",
    "R_avg", "R_max", "R_flatness", "R_peak_harmsim_avg", "R_peak_top",
]


def _epoch_rows(epochs, cfg, band):
    rows = []
    for ep in epochs:
        res = compute_resonance(ep["signal"], sf=ep["sf"], config=cfg)
        feats = C.resonance_features(res)
        row = dict(subject=ep["subject"], condition=ep["condition"],
                   channel=ep["channel"],
                   band_power=_alpha_power(ep["signal"], ep["sf"], *band))
        for fk in FEATURE_KEYS:
            row[fk] = feats.get(fk, np.nan)
        rows.append(row)
    return rows


def _specparam_Hmax(signal, sf, cfg_keep, fmin=2.0, fmax=45.0):
    """H_max with a specparam (FOOOF) aperiodic removal instead of the lightweight 2-parameter NLS:
    FOOOF separates peaks from the aperiodic component, so a strong alpha peak cannot bias the slope.
    We fit FOOOF to the PSD, whiten the signal by the fitted aperiodic spectrum, then compute H with
    the framework's own removal disabled -- a like-for-like aperiodic cross-check on H_max."""
    from fooof import FOOOF
    from scipy.signal import welch
    n = len(signal)
    f, P = welch(signal, fs=sf, nperseg=min(n, int(2 * sf)))
    fm = FOOOF(max_n_peaks=6, aperiodic_mode="fixed", verbose=False)
    fm.fit(f, P, [fmin, fmax])
    apf = fm.freqs; ap = 10.0 ** fm._ap_fit                       # aperiodic power over [fmin,fmax]
    fr = np.fft.rfftfreq(n, 1.0 / sf)
    apg = np.exp(np.interp(np.log(np.clip(fr, apf[0], apf[-1])), np.log(apf), np.log(ap)))
    sig_w = np.fft.irfft(np.fft.rfft(signal) / np.sqrt(apg + 1e-30), n=n)
    return float(compute_resonance(sig_w, sf=sf, config=cfg_keep).summaries["H"]["max"])


def _analyze_contrast(name, epochs, pos_cond, neg_cond, band, quick, do_specparam=False):
    cfg_rm = C.default_config(fmin=2, fmax=45, remove_aperiodic=True)
    cfg_keep = C.default_config(fmin=2, fmax=45, remove_aperiodic=False)
    rows = _epoch_rows(epochs, cfg_rm, band)

    pos = [r for r in rows if r["condition"] == pos_cond]
    neg = [r for r in rows if r["condition"] == neg_cond]

    # per-feature: bootstrap AUC CI (pos vs neg) + per-subject paired stats
    per_feat = {}
    for fk in FEATURE_KEYS + ["band_power"]:
        ci = C.bootstrap_auc_ci([r[fk] for r in pos], [r[fk] for r in neg],
                                n_boot=1000 if quick else 3000)
        paired = C.paired_by_group(rows, fk, "subject", "condition", neg_cond, pos_cond)
        per_feat[fk] = dict(**ci, paired_p=paired["p"],
                            rank_biserial=paired["rank_biserial"],
                            n_subjects=paired["n_groups"])

    clf = _classify(rows, pos_cond, quick)

    # aperiodic confound check: does H_max discrimination survive removal?
    rows_keep = _epoch_rows(epochs, cfg_keep, band)
    pk = [r["H_max"] for r in rows_keep if r["condition"] == pos_cond]
    nk = [r["H_max"] for r in rows_keep if r["condition"] == neg_cond]
    aperiodic_check = dict(
        H_max_auc_with_removal=per_feat["H_max"]["auc"],
        H_max_auc_without_removal=C.roc_auc(pk, nk),
    )
    if do_specparam:   # specparam (FOOOF) cross-check of the lightweight aperiodic fit
        try:
            ps = [_specparam_Hmax(ep["signal"], ep["sf"], cfg_keep)
                  for ep in epochs if ep["condition"] == pos_cond]
            ns = [_specparam_Hmax(ep["signal"], ep["sf"], cfg_keep)
                  for ep in epochs if ep["condition"] == neg_cond]
            aperiodic_check["H_max_auc_specparam"] = C.roc_auc(ps, ns)
        except Exception as e:
            aperiodic_check["H_max_auc_specparam"] = None
            aperiodic_check["specparam_error"] = str(e)
    return dict(name=name, n_epochs=len(rows), pos=pos_cond, neg=neg_cond,
                per_feature=per_feat, classification=clf,
                aperiodic_check=aperiodic_check, rows=rows)


def _classify(rows, pos_cond, quick):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_auc_score
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}
    subj = np.array([r["subject"] for r in rows])
    y = np.array([1 if r["condition"] == pos_cond else 0 for r in rows])
    Xr = np.nan_to_num(np.array([[r[fk] for fk in FEATURE_KEYS] for r in rows], float))
    Xp = np.nan_to_num(np.array([[r["band_power"]] for r in rows], float))

    def loso(X, y_):
        preds = np.full(len(y_), np.nan)
        for s in sorted(set(subj)):
            tr, te = subj != s, subj == s
            if len(set(y_[tr])) < 2:
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            m.fit(X[tr], y_[tr]); preds[te] = m.predict_proba(X[te])[:, 1]
        ok = ~np.isnan(preds)
        return roc_auc_score(y_[ok], preds[ok]) if len(set(y_[ok])) > 1 else np.nan

    auc_res = loso(Xr, y)
    # permutation test: shuffle labels within subject, recompute LOSO AUC
    n_perm = 200 if quick else 1000
    rng = np.random.default_rng(0)

    def _null(_rng):
        yp = y.copy()
        for s in set(subj):
            idx = np.where(subj == s)[0]
            yp[idx] = _rng.permutation(yp[idx])
        return loso(Xr, yp)
    perm = C.permutation_test_value(auc_res, _null, n_perm=n_perm, seed=0)
    return dict(auc_resonance=float(auc_res), auc_band_power=float(loso(Xp, y)),
                perm_p=perm["p"], perm_null_mean=perm["null_mean"],
                n_features=len(FEATURE_KEYS))


def run(quick=True):
    subjects = (1, 2, 3) if quick else tuple(range(1, 11))
    contrasts = {}

    eo_ec = D.load_eeg_states(subjects=subjects, channels=D.OCCIPITAL,
                              epoch_len=8.0, max_epochs_per_run=7)
    if eo_ec:
        print(f"  EO/EC: {len(eo_ec)} epochs")
        contrasts["eyes_open_vs_closed"] = _analyze_contrast(
            "eyes_open_vs_closed", eo_ec, "EC", "EO", (8, 12), quick, do_specparam=True)

    motor = D.load_eeg_motor_contrast(subjects=subjects, channels=D.MOTOR_CHANNELS,
                                      epoch_len=8.0, max_epochs_per_run=7)
    if motor:
        print(f"  REST/MOTOR: {len(motor)} epochs")
        contrasts["rest_vs_motor"] = _analyze_contrast(
            "rest_vs_motor", motor, "MOTOR", "REST", (8, 30), quick)

    if not contrasts:
        print("  no EEG loaded; aborting study 2.")
        return None
    result = dict(quick=quick, subjects=list(subjects), contrasts=contrasts)
    C.save_json(result, "study2_eeg_states.json")
    _figures(result)
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 2 headline ---")
    for cname, c in result["contrasts"].items():
        print(f"  [{cname}] n={c['n_epochs']} epochs")
        pf = c["per_feature"]
        ranked = sorted(pf.items(), key=lambda kv: abs((kv[1]["auc"] or 0.5) - 0.5), reverse=True)
        for k, v in ranked[:4]:
            sig = "*" if (v["paired_p"] is not None and np.isfinite(v["paired_p"]) and v["paired_p"] < 0.05) else " "
            print(f"    {k:20s} AUC={v['auc']:.3f} [{v['lo']:.2f},{v['hi']:.2f}]  "
                  f"paired p={v['paired_p']:.3g}{sig} (n_subj={v['n_subjects']})")
        clf = c["classification"]
        if "error" not in clf:
            print(f"    LOSO AUC resonance={clf['auc_resonance']:.3f} "
                  f"(perm p={clf['perm_p']:.3g}), band_power={clf['auc_band_power']:.3f}")
        ap = c["aperiodic_check"]
        print(f"    H_max AUC with removal={ap['H_max_auc_with_removal']:.3f}  "
              f"without={ap['H_max_auc_without_removal']:.3f}")


def _figures(result):
    plt = C.setup_mpl()
    contrasts = result["contrasts"]
    n = len(contrasts)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
    for row, (cname, c) in enumerate(contrasts.items()):
        pf = c["per_feature"]
        # left: AUC with bootstrap CI for top features
        ranked = sorted(pf.items(), key=lambda kv: abs((kv[1]["auc"] or 0.5) - 0.5), reverse=True)[:6]
        names = [k for k, _ in ranked]
        aucs = [v["auc"] for _, v in ranked]
        los = [v["auc"] - v["lo"] for _, v in ranked]
        his = [v["hi"] - v["auc"] for _, v in ranked]
        ax = axes[row][0]
        ax.barh(range(len(names)), aucs, xerr=[los, his], color="#42a5f5",
                error_kw=dict(lw=1))
        ax.axvline(0.5, color="k", ls="--", lw=0.6)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1); ax.set_xlabel("AUC (95% CI)")
        ax.set_title(f"{cname}: feature discrimination", fontsize=10)
        # right: classification + aperiodic check
        ax2 = axes[row][1]
        clf = c["classification"]; ap = c["aperiodic_check"]
        bars = {"band\npower": clf.get("auc_band_power", np.nan),
                "resonance\nLOSO": clf.get("auc_resonance", np.nan),
                "H_max\n(aperiodic\nremoved)": ap["H_max_auc_with_removal"],
                "H_max\n(kept)": ap["H_max_auc_without_removal"]}
        ax2.bar(range(len(bars)), list(bars.values()),
                color=["#9e9e9e", "#66bb6a", "#26a69a", "#ef9a9a"])
        ax2.set_xticks(range(len(bars))); ax2.set_xticklabels(list(bars), fontsize=8)
        ax2.axhline(0.5, color="k", ls="--", lw=0.6); ax2.set_ylim(0, 1)
        ax2.set_ylabel("AUC"); ax2.set_title(f"{cname}: decoding + 1/f control", fontsize=10)
    fig.suptitle("Study 2 — EEG state discrimination (per-subject stats, bootstrap CIs)",
                 fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, "study2_eeg_states")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

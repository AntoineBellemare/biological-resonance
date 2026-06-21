"""Study 15 — H vs criticality across LOSS OF CONSCIOUSNESS (deep GA, ds004541)

The strongest real-data criticality traversal: human scalp EEG (58 ch) sliced into
WAKE (baseline->anesthesia start) vs UNCONSCIOUS (LOC->ROC). Tests the same
prediction as Studies 10/13/14 — is harmonicity H maximized when the brain is
closest to criticality? — where consciousness is genuinely lost, and with enough
channels to power the avalanche branching ratio m_hat.

Anesthetic agent is not stated in the dataset metadata (general-anesthesia LOC,
agent unconfirmed).

Outputs: results/study15_deep_anesthesia.json, figures/study15_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from resonance_paper import _common as C
from resonance_paper import deep_anesthesia as D
from resonance_paper import crit_resonance as CR

BP_FEATS = ["rel_alpha", "rel_slow", "alpha_slow", "rel_beta"]
RES_FEATS = ["H_max", "H_avg", "R_max"]
CRIT_FEATS = ["m_hat", "dfa"]


def _contrast(rows, feature, deep="unconscious", base="wake"):
    diffs = []
    for sub in sorted(set(r["subject"] for r in rows)):
        d = [r[feature] for r in rows if r["subject"] == sub and r["state"] == deep and np.isfinite(r[feature])]
        b = [r[feature] for r in rows if r["subject"] == sub and r["state"] == base and np.isfinite(r[feature])]
        if d and b:
            diffs.append(float(np.nanmean(d) - np.nanmean(b)))
    if len(diffs) < 3:
        return dict(mean_diff=float("nan"), wilcoxon_p=float("nan"), n=len(diffs))
    ci = C.mean_ci(diffs)
    try:
        wp = float(wilcoxon(diffs).pvalue)
    except ValueError:
        wp = float("nan")
    return dict(mean_diff=ci["mean"], lo=ci["lo"], hi=ci["hi"], wilcoxon_p=wp,
                frac_neg=float(np.mean(np.array(diffs) < 0)), n=len(diffs))


def _decode(rows, feats, pos="unconscious", neg="wake"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    rr = [r for r in rows if r["state"] in (pos, neg)]
    subj = np.array([r["subject"] for r in rr])
    y = np.array([1 if r["state"] == pos else 0 for r in rr])
    X = np.nan_to_num(np.array([[r[f] for f in feats] for r in rr], float))
    preds = np.full(len(y), np.nan)
    for s in sorted(set(subj)):
        tr, te = subj != s, subj == s
        if len(set(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr]); preds[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(preds)
    return float(roc_auc_score(y[ok], preds[ok])) if len(set(y[ok])) > 1 else float("nan")


def run(quick=True):
    n_sub = 2 if quick else 8
    max_win = 4 if quick else 8
    cfg = CR.state_config()
    print(f"Study 15 — loading {n_sub} deep-GA subjects (wake vs unconscious, 30s windows) ...")
    items = D.load_deepGA_multichannel(D.SUBJECTS[:n_sub], win_len=30.0,
                                       max_win=max_win, download=True)
    print(f"  {len(items)} windows; computing features (H frontal, m_hat 58-ch) ...")
    rows = []
    for i, it in enumerate(items):
        rows.append(CR.window_features(it["X"], it["sf"], it["subject"], it["state"],
                                       cfg, h_idx=it["h_idx"]))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(items)}", flush=True)
    result = CR.analyze(rows, "deep GA / LOC (ds004541)")
    result["quick"] = quick
    result["contrast"] = {k: _contrast(rows, k) for k in RES_FEATS + CRIT_FEATS + BP_FEATS}
    result["decoding"] = dict(
        auc_band_power=_decode(rows, BP_FEATS), auc_resonance=_decode(rows, RES_FEATS),
        auc_criticality=_decode(rows, CRIT_FEATS), auc_all=_decode(rows, BP_FEATS + RES_FEATS + CRIT_FEATS))
    C.save_json(result, "study15_deep_anesthesia.json")
    CR.figure(result, "study15_deep_anesthesia")
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 15 headline (deep GA/LOC: H vs criticality) ---")
    pr = result["primary"]
    print(f"  H maximal near criticality? per-subject ρ(H, proximity), n={result['n_subjects']}:")
    for k in ["H_max_vs_prox_m", "H_max_vs_prox_dfa", "R_max_vs_prox_m"]:
        v = pr.get(k, {})
        if np.isfinite(v.get("mean_rho", np.nan)):
            print(f"    {k:20s} ρ={v['mean_rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] "
                  f"p={v['wilcoxon_p']:.3g} ({int(v['frac_pos']*100)}% +)")
    for st in ["wake", "unconscious"]:
        if st in result["by_state"]:
            b = result["by_state"][st]
            print(f"    {st:12s} m_hat={b['m_hat']:.3f} dfa={b['dfa']:.3f} H_max={b['H_max']:.3f} "
                  f"lzc={b['lzc']:.3f} (n={b['n_windows']})")
    d = result["decoding"]
    print(f"  decode unconscious vs wake (LOSO): band={d['auc_band_power']:.2f} "
          f"resonance={d['auc_resonance']:.2f} criticality={d['auc_criticality']:.2f} all={d['auc_all']:.2f}")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

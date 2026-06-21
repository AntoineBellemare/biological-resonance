"""Study 13 — Does harmonicity track criticality in a real brain? (propofol)

Real-data leg of the criticality thread. Uses PROPER neural-criticality
estimators (DFA/LRTC + avalanche branching ratio m_hat, validated in
criticality.py) rather than stand-ins, and tests the in-silico prediction
(Study 10: H peaks at sigma=1): is harmonicity H maximized when the brain is
closest to criticality?

Data: Chennu graded-propofol EEG (baseline/mild/moderate/recovery), frontal H +
whole-head branching ratio, 30 s windows. Also reports the deepest-vs-baseline H
contrast and a moderate-vs-baseline decoding (resonance vs band power vs
criticality markers).

CAVEAT: Chennu is titrated *sedation*, not deep GA/LOC, so the brain may not move
far across criticality — see Studies 14 (sleep) and 15 (deep anesthesia) for
stronger traversals.

Outputs: results/study13_anesthesia.json, figures/study13_*.{png,pdf}
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from resonance_paper import _common as C
from resonance_paper import anesthesia as A
from resonance_paper import crit_resonance as CR

BP_FEATS = ["rel_alpha", "rel_slow", "alpha_slow", "rel_beta"]
RES_FEATS = ["H_max", "H_avg", "R_max"]
CRIT_FEATS = ["m_hat", "dfa"]


def _paired_contrast(rows, feature, deep="moderate", base="baseline"):
    """Per-subject (deepest - baseline) change, tested across subjects."""
    diffs = []
    for sub in sorted(set(r["subject"] for r in rows)):
        d = [r[feature] for r in rows if r["subject"] == sub and deep in r["state"].lower()
             and np.isfinite(r[feature])]
        b = [r[feature] for r in rows if r["subject"] == sub and base in r["state"].lower()
             and np.isfinite(r[feature])]
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


def _decode(rows, feats, pos="moderate", neg="baseline"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    rr = [r for r in rows if pos in r["state"].lower() or neg in r["state"].lower()]
    subj = np.array([r["subject"] for r in rr])
    y = np.array([1 if pos in r["state"].lower() else 0 for r in rr])
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
    n_sub = 3 if quick else 10
    h_channels = ["Fz", "F3", "F4"] if quick else ["Fz", "F3", "F4", "Fp1", "Fp2"]
    max_win = 2 if quick else 4
    cfg = CR.state_config()

    print(f"Study 13 — loading {n_sub} propofol subjects (multichannel, 30s windows) ...")
    items = A.load_chennu_multichannel(A.SUBJECTS[:n_sub], h_channels=h_channels,
                                       win_len=30.0, max_win=max_win, download=True)
    print(f"  {len(items)} windows; computing features (H per frontal ch, m_hat whole-head) ...")
    rows = []
    for i, it in enumerate(items):
        rows.append(CR.window_features(it["X"], it["sf"], it["subject"], it["state"],
                                       cfg, h_idx=it["h_idx"]))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(items)}", flush=True)

    result = CR.analyze(rows, "propofol (Chennu sedation)")
    result["quick"] = quick
    # secondary: deepest-vs-baseline contrast + decoding
    result["contrast"] = {k: _paired_contrast(rows, k) for k in RES_FEATS + CRIT_FEATS + BP_FEATS}
    result["decoding"] = dict(
        auc_band_power=_decode(rows, BP_FEATS), auc_resonance=_decode(rows, RES_FEATS),
        auc_criticality=_decode(rows, CRIT_FEATS), auc_all=_decode(rows, BP_FEATS + RES_FEATS + CRIT_FEATS))
    C.save_json(result, "study13_anesthesia.json")
    CR.figure(result, "study13_anesthesia")
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 13 headline (propofol: H vs criticality) ---")
    pr = result["primary"]
    print(f"  H maximal near criticality? per-subject ρ(H, proximity-to-critical), n={result['n_subjects']}:")
    for k in ["H_max_vs_prox_m", "H_max_vs_prox_dfa", "R_max_vs_prox_m"]:
        v = pr.get(k, {})
        if np.isfinite(v.get("mean_rho", np.nan)):
            print(f"    {k:20s} ρ={v['mean_rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] "
                  f"p={v['wilcoxon_p']:.3g} ({int(v['frac_pos']*100)}% +)")
    print("  marker traversal across states (mean):")
    for st in result["states"]:
        b = result["by_state"][st]
        print(f"    {st[:18]:18s} m_hat={b['m_hat']:.3f} dfa={b['dfa']:.3f} H_max={b['H_max']:.3f}")
    d = result["decoding"]
    print(f"  decode moderate vs baseline (LOSO): band={d['auc_band_power']:.2f} "
          f"resonance={d['auc_resonance']:.2f} criticality={d['auc_criticality']:.2f} all={d['auc_all']:.2f}")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

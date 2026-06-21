"""Study 16 — In-depth in-vivo resonance↔criticality (investigations #1/#2/#3/#6/#4)

Addresses the in-vivo "reversal" by measuring resonance on the RIGHT observable.
Per 30 s multichannel window we compute, on the SAME data:
  * raw-EEG resonance      H_full, R_full   (oscillatory; frontal channels)        [#2 oscillatory]
  * avalanche-resonance    H_aval, R_aval   (resonance of the population activity /
                            global field power -- the scale-free signal Study 10's
                            H was actually computed on)                             [#1, #2 scale-free]
  * validated criticality axis m_hat (peaks at sigma=1) + DFA                       [#3]
Then:
  #1/#2: per-subject rho(H_full, m̂-prox) vs rho(H_aval, m̂-prox) -- does the
         scale-free (avalanche) resonance track criticality like the model, while
         the oscillatory raw-EEG H reverses?
  #6:    same for R (cross-frequency coupling) vs H.
  #4:    within-state, per-window rho (controls between-state oscillation confounds).

Usage: python -m resonance_paper.study16_criticality_indepth sleep|deepga [--paper]
Outputs: results/study16_<dataset>.json, figures/study16_<dataset>_*.{png,pdf}
"""
from __future__ import annotations

import sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from resonance_paper import _common as C
from resonance_paper import crit_resonance as CR
from resonance_paper import criticality as Cr
from biotuner.resonance import compute_resonance


def _load(dataset, quick):
    if dataset == "sleep":
        from resonance_paper.study14_sleep import load_sleep
        return load_sleep(n_subjects=3 if quick else 8, max_epochs_per_stage=6 if quick else 15)
    if dataset == "deepga":
        from resonance_paper import deep_anesthesia as D
        n = 2 if quick else 8
        return D.load_deepGA_multichannel(D.SUBJECTS[:n], win_len=30.0, max_win=4 if quick else 8, download=True)
    if dataset == "propofol":
        from resonance_paper import anesthesia as A
        n = 3 if quick else 10
        return A.load_chennu_multichannel(A.SUBJECTS[:n], win_len=30.0, max_win=2 if quick else 4, download=True)
    raise ValueError(dataset)


def _window(it, cfg):
    X = np.atleast_2d(it["X"]); sf = it["sf"]; hidx = it.get("h_idx") or list(range(min(5, len(X))))
    # raw-EEG (oscillatory) resonance: mean over frontal channels
    Hf, Rf = [], []
    for ci in hidx:
        sn = (X[ci] - X[ci].mean()) / (X[ci].std() + 1e-12)
        r = compute_resonance(sn.astype(np.float64), sf=sf, config=cfg)
        Hf.append(float(r.summaries["H"]["max"])); Rf.append(float(r.summaries["R"]["max"]))
    # avalanche / population-activity (scale-free) resonance: GFP of all channels
    gfp = Cr.population_activity(X, sf, mode="gfp")
    gz = (gfp - gfp.mean()) / (gfp.std() + 1e-12)
    rg = compute_resonance(gz.astype(np.float64), sf=sf, config=cfg)
    m = Cr.eeg_branching_ratio(X, sf)
    return dict(subject=it["subject"], state=it["state"],
                H_full=float(np.nanmean(Hf)), R_full=float(np.nanmean(Rf)),
                H_aval=float(rg.summaries["H"]["max"]), R_aval=float(rg.summaries["R"]["max"]),
                m_hat=m, prox_m=(-abs(m - 1) if np.isfinite(m) else np.nan))


def _agg(rows, feat, pred="prox_m"):
    """per-subject Spearman aggregated (across-window, within each subject)."""
    rhos = []
    for s in sorted(set(r["subject"] for r in rows)):
        sr = [r for r in rows if r["subject"] == s and np.isfinite(r[feat]) and np.isfinite(r[pred])]
        if len(sr) < 5:
            continue
        rho, _ = spearmanr([r[pred] for r in sr], [r[feat] for r in sr])
        if np.isfinite(rho):
            rhos.append(rho)
    if len(rhos) < 3:
        return dict(rho=float("nan"), n=len(rhos))
    ci = C.mean_ci(rhos)
    try:
        p = float(wilcoxon(rhos).pvalue)
    except ValueError:
        p = float("nan")
    return dict(rho=ci["mean"], lo=ci["lo"], hi=ci["hi"], p=p,
                frac_pos=float(np.mean(np.array(rhos) > 0)), n=len(rhos))


def _within_state_agg(rows, feat, pred="prox_m"):
    """#4 — Spearman(feat, prox) computed WITHIN each (subject, state), then
    aggregated. Removes between-state confounds (different stages = different
    dominant oscillations); tests the moment-to-moment relationship."""
    rhos = []
    for s in sorted(set(r["subject"] for r in rows)):
        for st in sorted(set(r["state"] for r in rows)):
            sr = [r for r in rows if r["subject"] == s and r["state"] == st
                  and np.isfinite(r[feat]) and np.isfinite(r[pred])]
            if len(sr) < 6 or len(set(round(r[pred], 6) for r in sr)) < 4:
                continue
            rho, _ = spearmanr([r[pred] for r in sr], [r[feat] for r in sr])
            if np.isfinite(rho):
                rhos.append(rho)
    if len(rhos) < 4:
        return dict(rho=float("nan"), n=len(rhos))
    ci = C.mean_ci(rhos)
    try:
        p = float(wilcoxon(rhos).pvalue)
    except ValueError:
        p = float("nan")
    return dict(rho=ci["mean"], lo=ci["lo"], hi=ci["hi"], p=p,
                frac_pos=float(np.mean(np.array(rhos) > 0)), n=len(rhos))


def run(dataset="sleep", quick=True):
    cfg = CR.state_config()
    print(f"Study 16 [{dataset}] — loading ...", flush=True)
    items = _load(dataset, quick)
    print(f"  {len(items)} windows; computing raw + avalanche resonance ...", flush=True)
    rows = []
    for i, it in enumerate(items):
        rows.append(_window(it, cfg))
        if (i + 1) % 40 == 0:
            print(f"    {i+1}/{len(items)}", flush=True)

    states = sorted(set(r["state"] for r in rows))
    by_state = {s: {k: float(np.nanmean([r[k] for r in rows if r["state"] == s]))
                    for k in ["H_full", "H_aval", "R_full", "R_aval", "m_hat"]} for s in states}
    # #1/#2/#6 — oscillatory vs scale-free resonance vs criticality (across-state)
    crit = {f"{feat}_vs_prox_m": _agg(rows, feat) for feat in ["H_full", "H_aval", "R_full", "R_aval"]}
    # #4 — WITHIN-state per-window correlations (controls between-state confounds)
    within = {feat: _within_state_agg(rows, feat) for feat in ["H_full", "H_aval", "R_full", "R_aval"]}
    result = dict(dataset=dataset, quick=quick, n_subjects=len(set(r["subject"] for r in rows)),
                  n_windows=len(rows), states=states, by_state=by_state,
                  criticality=crit, within_state=within)
    C.save_json(result, f"study16_{dataset}.json")
    _headline(result)
    return result


def _headline(result):
    print(f"\n  --- Study 16 [{result['dataset']}] headline ---")
    print("  Oscillatory (raw-EEG) vs scale-free (avalanche/GFP) resonance vs criticality (m̂-proximity):")
    for k in ["H_full_vs_prox_m", "H_aval_vs_prox_m", "R_full_vs_prox_m", "R_aval_vs_prox_m"]:
        v = result["criticality"][k]
        if np.isfinite(v.get("rho", np.nan)):
            print(f"    {k:20s} rho={v['rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']:.2g} ({int(v['frac_pos']*100)}%+)")
    print("  by-state H_full vs H_aval (does the avalanche signal behave differently?):")
    for s, b in result["by_state"].items():
        print(f"    {s:12s} H_full={b['H_full']:.3f} H_aval={b['H_aval']:.3f} m_hat={b['m_hat']:.3f}")
    print("  WITHIN-state (per-window, controls between-state confounds):")
    for k in ["H_full", "H_aval"]:
        v = result["within_state"].get(k, {})
        if np.isfinite(v.get("rho", np.nan)):
            print(f"    {k:8s} rho={v['rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']:.2g} (n={v['n']})")
    print("  KEY: if H_aval tracks m̂-proximity POSITIVELY while H_full is negative,")
    print("       the reversal is the observable (oscillatory vs scale-free), per the model.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ds = args[0] if args else "sleep"
    run(dataset=ds, quick="--paper" not in sys.argv)

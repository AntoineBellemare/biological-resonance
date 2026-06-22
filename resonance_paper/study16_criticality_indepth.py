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
from scipy.stats import spearmanr, wilcoxon, rankdata
from scipy.signal import welch

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


def _rel_slow_power(x, sf, lo=0.5, hi=4.0):
    """Relative power in the slow (delta) band — the confound that inflates raw-EEG H
    via the slow-wave harmonic series in synchronized (subcritical) states."""
    x = np.asarray(x, float)
    f, p = welch(x, fs=sf, nperseg=int(min(len(x), sf * 4)))
    tot = np.trapz(p, f) + 1e-20
    band = np.trapz(p[(f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)])
    return float(band / tot)


def _window(it, cfg):
    X = np.atleast_2d(it["X"]); sf = it["sf"]; hidx = it.get("h_idx") or list(range(min(5, len(X))))
    # raw-EEG (oscillatory) resonance: mean over frontal channels
    Hf, Rf = [], []
    for ci in hidx:
        sn = (X[ci] - X[ci].mean()) / (X[ci].std() + 1e-12)
        r = compute_resonance(sn.astype(np.float64), sf=sf, config=cfg)
        Hf.append(float(r.summaries["H"]["max"])); Rf.append(float(r.summaries["R"]["max"]))
    raw_mean = X[hidx].mean(0) if len(hidx) else X.mean(0)
    # avalanche / population-activity (scale-free) resonance: GFP of all channels
    gfp = Cr.population_activity(X, sf, mode="gfp")
    gz = (gfp - gfp.mean()) / (gfp.std() + 1e-12)
    rg = compute_resonance(gz.astype(np.float64), sf=sf, config=cfg)
    m = Cr.eeg_branching_ratio(X, sf)
    return dict(subject=it["subject"], state=it["state"],
                H_full=float(np.nanmean(Hf)), R_full=float(np.nanmean(Rf)),
                H_aval=float(rg.summaries["H"]["max"]), R_aval=float(rg.summaries["R"]["max"]),
                slow_pow_raw=_rel_slow_power(raw_mean, sf),   # delta power of the raw (H_full) signal
                slow_pow_gfp=_rel_slow_power(gz, sf),         # delta power of the GFP (H_aval) signal
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


def _partial_agg(rows, feat, ctrl, pred="prox_m"):
    """Per-subject Spearman partial correlation of feat vs pred CONTROLLING ctrl
    (rank-based: partial of the three pairwise Spearman r), aggregated across subjects.
    Tests whether the feat↔criticality link survives removing the slow-power confound."""
    prs = []
    for s in sorted(set(r["subject"] for r in rows)):
        sr = [r for r in rows if r["subject"] == s
              and np.isfinite(r[feat]) and np.isfinite(r[pred]) and np.isfinite(r[ctrl])]
        if len(sr) < 6:
            continue
        x = rankdata([r[feat] for r in sr]); y = rankdata([r[pred] for r in sr]); z = rankdata([r[ctrl] for r in sr])
        rxy = np.corrcoef(x, y)[0, 1]; rxz = np.corrcoef(x, z)[0, 1]; ryz = np.corrcoef(y, z)[0, 1]
        denom = np.sqrt(max((1 - rxz**2) * (1 - ryz**2), 1e-12))
        pr = (rxy - rxz * ryz) / denom
        if np.isfinite(pr):
            prs.append(pr)
    if len(prs) < 3:
        return dict(rho=float("nan"), n=len(prs))
    ci = C.mean_ci(prs)
    try:
        p = float(wilcoxon(prs).pvalue)
    except ValueError:
        p = float("nan")
    return dict(rho=ci["mean"], lo=ci["lo"], hi=ci["hi"], p=p,
                frac_pos=float(np.mean(np.array(prs) > 0)), n=len(prs))


def _paired_diff(rows, fa, fb, pred="prox_m"):
    """Paired per-subject test of the SIGN DISSOCIATION: for each subject compute
    ρ(fa, pred) and ρ(fb, pred), then Wilcoxon on the per-subject differences (fb − fa).
    This formally tests H_aval (scale-free) vs H_full (raw) rather than two separate tests."""
    ra, rb, diffs = [], [], []
    for s in sorted(set(r["subject"] for r in rows)):
        sr = [r for r in rows if r["subject"] == s
              and np.isfinite(r[fa]) and np.isfinite(r[fb]) and np.isfinite(r[pred])]
        if len(sr) < 5:
            continue
        a = spearmanr([r[pred] for r in sr], [r[fa] for r in sr])[0]
        b = spearmanr([r[pred] for r in sr], [r[fb] for r in sr])[0]
        if np.isfinite(a) and np.isfinite(b):
            ra.append(a); rb.append(b); diffs.append(b - a)
    if len(diffs) < 3:
        return dict(n=len(diffs))
    try:
        p = float(wilcoxon(diffs).pvalue)
    except ValueError:
        p = float("nan")
    return dict(rho_a=float(np.mean(ra)), rho_b=float(np.mean(rb)), diff=float(np.mean(diffs)),
                p=p, frac_b_gt_a=float(np.mean(np.array(diffs) > 0)), n=len(diffs))


def _bh_adjust(pdict):
    """Benjamini–Hochberg FDR adjustment over a name→p-value dict (skips NaN)."""
    items = [(k, v) for k, v in pdict.items() if np.isfinite(v)]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1]); m = len(items); adj = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        prev = min(prev, p * m / (i + 1))
        adj[k] = float(min(prev, 1.0))
    return adj


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
    # PAIRED sign-dissociation test (scale-free H_aval vs raw H_full), across- and within-state
    paired = dict(across=_paired_diff(rows, "H_full", "H_aval"))
    # PARTIAL correlation controlling slow-band power (does the link survive the slow-wave confound?)
    partial = {"H_aval_vs_prox_m|slow_gfp": _partial_agg(rows, "H_aval", "slow_pow_gfp"),
               "H_full_vs_prox_m|slow_raw": _partial_agg(rows, "H_full", "slow_pow_raw")}
    # multiple-comparison (BH-FDR) over the four primary across-state correlations
    crit_p_adj = _bh_adjust({k: v.get("p", float("nan")) for k, v in crit.items()})
    result = dict(dataset=dataset, quick=quick, n_subjects=len(set(r["subject"] for r in rows)),
                  n_windows=len(rows), states=states, by_state=by_state,
                  criticality=crit, criticality_p_bh=crit_p_adj,
                  paired_dissociation=paired, partial=partial, within_state=within)
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
    pd = result.get("paired_dissociation", {}).get("across", {})
    if np.isfinite(pd.get("p", np.nan)):
        print(f"  PAIRED dissociation (H_aval − H_full ρ): Δ={pd['diff']:+.2f} "
              f"(H_full {pd['rho_a']:+.2f} vs H_aval {pd['rho_b']:+.2f}), p={pd['p']:.2g}, "
              f"{int(pd['frac_b_gt_a']*100)}% aval>full (n={pd['n']})")
    for k, v in result.get("partial", {}).items():
        if np.isfinite(v.get("rho", np.nan)):
            print(f"  PARTIAL {k}: ρ={v['rho']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}] p={v['p']:.2g} (n={v['n']})")
    if result.get("criticality_p_bh"):
        print("  BH-FDR adjusted p (primary across-state): " +
              ", ".join(f"{k.replace('_vs_prox_m','')}={p:.2g}" for k, p in result["criticality_p_bh"].items()))
    print("  KEY: if H_aval tracks m̂-proximity POSITIVELY while H_full is negative (paired Δ>0),")
    print("       AND the partial (slow-power-controlled) H_aval link survives, the reversal is")
    print("       the observable (oscillatory vs scale-free), per the model — robustly.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ds = args[0] if args else "sleep"
    run(dataset=ds, quick="--paper" not in sys.argv)

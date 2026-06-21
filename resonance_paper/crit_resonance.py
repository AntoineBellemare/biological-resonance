"""Shared real-data criticality<->resonance analysis (Studies 13-15).

Tests the in-silico prediction (Study 10: H peaks at the critical branching ratio
sigma=1) in real brains: across epochs spanning brain states, is harmonicity H
MAXIMIZED when the brain is closest to criticality?

Per multichannel window we compute:
  * H_max / H_avg / R_max         -- the resonance framework (aperiodic removed)
  * dfa (alpha-envelope LRTC)      -- criticality marker (alpha->1 near critical)
  * m_hat (avalanche branching)    -- criticality marker (m->1 critical); the
                                      real-brain analog of Study 10's sigma
  * rel_alpha / rel_slow, lzc, exp_1f -- band-power baseline + descriptors

Core test: per-subject Spearman of H against criticality PROXIMITY
(prox = -|marker - 1|; larger = closer to critical), aggregated across subjects.
Positive => H is higher the closer the brain sits to criticality.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr, wilcoxon

from resonance_paper import _common as C
from resonance_paper import criticality as Cr
from biotuner.resonance import compute_resonance, ResonanceConfig

BANDS = {"slow": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def state_config():
    """Resonance config for the brain-state studies (13-15), chosen by parameter
    sweep: fmin=0.5 so the <2 Hz slow-wave band (which defines N3 / deep
    anesthesia) is included; precision 0.25 Hz to resolve low-frequency peaks;
    aperiodic removal on (discrimination reflects harmonic structure, not the
    stage-varying 1/f). Requires data high-passed at <=0.5 Hz.

    Kernel = harmsim. (subharm_tension was *more* sensitive in the sweeps but is
    ~40x slower at fmin=0.5 -- 7 s vs 0.2 s/call -- so infeasible at paper scale;
    harmsim still discriminates states strongly and gives the same criticality
    sign.)"""
    return ResonanceConfig(fmin=0.5, fmax=45, precision_hz=0.25,
                           harmonic_kernel="harmsim", remove_aperiodic=True)
RES_FEATS = ["H_max", "H_avg", "R_max"]
MARKERS = ["m_hat", "dfa"]


def _band_power(sig, sf):
    f, p = welch(sig, fs=sf, nperseg=min(len(sig), int(sf * 2)))
    tot = p[(f >= 0.5) & (f <= 45)].sum() + 1e-20
    out = {f"rel_{b}": float(p[(f >= lo) & (f <= hi)].sum() / tot) for b, (lo, hi) in BANDS.items()}
    out["alpha_slow"] = out["rel_alpha"] / (out["rel_slow"] + 1e-9)
    return out


def _exp_1f(sig, sf):
    f, p = welch(sig, fs=sf, nperseg=min(len(sig), int(sf * 2)))
    m = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    if m.sum() < 5:
        return float("nan")
    return float(-np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)[0])


def _lzc(sig):
    b = (sig > np.median(sig)).astype(int)
    s = "".join(map(str, b)); n = len(s)
    if n < 2:
        return 0.0
    i, c, l, k, k_max = 0, 1, 1, 1, 1
    while l + k <= n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            k_max = max(k, k_max); i += 1
            if i == l:
                c += 1; l += k_max; i = 0; k = 1; k_max = 1
            else:
                k = 1
    if k != 1:
        c += 1
    return float(c * np.log2(n) / n)


def window_features(X, sf, subject, state, cfg, h_idx=None, alpha_band=(8, 13)):
    """X: (n_ch, n_times) one multichannel window. H/DFA/band-power are computed
    on channels ``h_idx`` (default all); the branching ratio m_hat uses ALL
    channels (more channels = better avalanche statistics)."""
    X = np.atleast_2d(np.asarray(X, float))
    idx = list(range(len(X))) if h_idx is None else list(h_idx)
    Hs, Havg, Rs, dfas, bps, lzcs, exps = [], [], [], [], [], [], []
    for ci in idx:
        ch = X[ci]
        sn = (ch - ch.mean()) / (ch.std() + 1e-12)
        res = compute_resonance(sn.astype(np.float64), sf=sf, config=cfg)
        f = C.resonance_features(res)
        Hs.append(f.get("H_max", np.nan)); Havg.append(f.get("H_avg", np.nan))
        Rs.append(f.get("R_max", np.nan))
        dfas.append(Cr.lrtc_envelope(sn, sf, band=alpha_band))
        lzcs.append(_lzc(sn)); exps.append(_exp_1f(sn, sf))
        bps.append(_band_power(sn, sf))
    row = dict(subject=subject, state=state,
               H_max=float(np.nanmean(Hs)), H_avg=float(np.nanmean(Havg)),
               R_max=float(np.nanmean(Rs)), dfa=float(np.nanmean(dfas)),
               m_hat=float(Cr.eeg_branching_ratio(X, sf)),
               lzc=float(np.nanmean(lzcs)), exp_1f=float(np.nanmean(exps)))
    for k in bps[0]:
        row[k] = float(np.nanmean([b[k] for b in bps]))
    return row


def _per_subject_corr(rows, feature, predictor):
    rhos = []
    for sub in sorted(set(r["subject"] for r in rows)):
        sr = [r for r in rows if r["subject"] == sub
              and np.isfinite(r.get(feature, np.nan)) and np.isfinite(r.get(predictor, np.nan))]
        if len(sr) < 5:
            continue
        rho, _ = spearmanr([r[predictor] for r in sr], [r[feature] for r in sr])
        if np.isfinite(rho):
            rhos.append(float(rho))
    if len(rhos) < 3:
        return dict(mean_rho=float("nan"), lo=float("nan"), hi=float("nan"),
                    wilcoxon_p=float("nan"), frac_pos=float("nan"), n=len(rhos))
    ci = C.mean_ci(rhos)
    try:
        wp = float(wilcoxon(rhos).pvalue)
    except ValueError:
        wp = float("nan")
    return dict(mean_rho=ci["mean"], lo=ci["lo"], hi=ci["hi"], wilcoxon_p=wp,
                frac_pos=float(np.mean(np.array(rhos) > 0)), n=len(rhos))


BP_GROUP = ["rel_slow", "rel_theta", "rel_alpha", "rel_beta", "alpha_slow"]
RES_GROUP = ["H_max", "H_avg", "R_max"]
CRIT_GROUP = ["m_hat", "dfa", "lzc", "exp_1f"]


def _decode_states(rows, feats):
    """Leave-one-subject-out multiclass accuracy decoding brain state from feats."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
    except Exception:
        return float("nan")
    subj = np.array([r["subject"] for r in rows])
    y = np.array([r["state"] for r in rows])
    X = np.nan_to_num(np.array([[r.get(f, np.nan) for f in feats] for r in rows], float))
    if len(set(y)) < 2 or len(set(subj)) < 2:
        return float("nan")
    correct = tot = 0
    for s in sorted(set(subj)):
        tr, te = subj != s, subj == s
        if len(set(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        correct += int((m.predict(X[te]) == y[te]).sum()); tot += int(te.sum())
    return float(correct / tot) if tot else float("nan")


def analyze(rows, dataset):
    """Run the H-maximal-near-criticality test + supporting analyses."""
    # criticality proximity (closer to critical = larger)
    for r in rows:
        r["prox_m"] = -abs(r["m_hat"] - 1.0) if np.isfinite(r["m_hat"]) else np.nan
        r["prox_dfa"] = -abs(r["dfa"] - 1.0) if np.isfinite(r["dfa"]) else np.nan

    # PRIMARY: per-subject Spearman(H, proximity-to-criticality), aggregated
    primary = {}
    for feat in ["H_max", "H_avg", "R_max"]:
        primary[f"{feat}_vs_prox_m"] = _per_subject_corr(rows, feat, "prox_m")
        primary[f"{feat}_vs_prox_dfa"] = _per_subject_corr(rows, feat, "prox_dfa")

    # do the criticality markers actually traverse states? (validation)
    states = sorted(set(r["state"] for r in rows))
    by_state = {}
    for st in states:
        sr = [r for r in rows if r["state"] == st]
        by_state[st] = {k: float(np.nanmean([r[k] for r in sr]))
                        for k in ["m_hat", "dfa", "H_max", "H_avg", "R_max",
                                  "lzc", "exp_1f", "rel_alpha", "rel_slow"]}
        by_state[st]["n_windows"] = len(sr)

    # pooled direction (raw marker, not proximity)
    def pooled(a, b):
        xs = [r[a] for r in rows if np.isfinite(r[a]) and np.isfinite(r[b])]
        ys = [r[b] for r in rows if np.isfinite(r[a]) and np.isfinite(r[b])]
        rho, p = spearmanr(xs, ys)
        return dict(rho=float(rho), p=float(p), n=len(xs))
    pooled_dir = dict(H_max_vs_m_hat=pooled("H_max", "m_hat"),
                      H_max_vs_dfa=pooled("H_max", "dfa"))

    # SOLID deliverable: do H/R discriminate brain state? (multiclass LOSO accuracy)
    n_states = len(states)
    state_decoding = dict(
        chance=1.0 / n_states if n_states else float("nan"),
        resonance=_decode_states(rows, RES_GROUP),
        band_power=_decode_states(rows, BP_GROUP),
        criticality=_decode_states(rows, CRIT_GROUP),
        resonance_plus_power=_decode_states(rows, RES_GROUP + BP_GROUP),
        all=_decode_states(rows, RES_GROUP + BP_GROUP + CRIT_GROUP))

    return dict(dataset=dataset, n_subjects=len(set(r["subject"] for r in rows)),
                n_windows=len(rows), primary=primary, by_state=by_state,
                pooled_direction=pooled_dir, states=states,
                state_decoding=state_decoding)


def figure(result, name):
    plt = C.setup_mpl()
    p = result["primary"]; bs = result["by_state"]; states = result["states"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # A: per-subject H-vs-criticality-proximity correlations (the core test)
    keys = ["H_max_vs_prox_m", "H_max_vs_prox_dfa", "H_avg_vs_prox_m",
            "R_max_vs_prox_m", "R_max_vs_prox_dfa"]
    keys = [k for k in keys if np.isfinite(p.get(k, {}).get("mean_rho", np.nan))]
    rhos = [p[k]["mean_rho"] for k in keys]
    err = [[p[k]["mean_rho"] - p[k]["lo"] for k in keys], [p[k]["hi"] - p[k]["mean_rho"] for k in keys]]
    axes[0].barh(range(len(keys)), rhos, xerr=err, color="#ef6c00", capsize=2)
    axes[0].axvline(0, color="k", lw=0.6)
    axes[0].set_yticks(range(len(keys))); axes[0].set_yticklabels(keys, fontsize=7)
    axes[0].set_xlabel("per-subject ρ (H vs criticality proximity)")
    axes[0].set_title("A. Is H higher near criticality?\n(ρ>0 = yes)", fontsize=9)
    axes[0].invert_yaxis()

    # B: marker traversal across states (m_hat, dfa)
    x = range(len(states))
    axes[1].plot(x, [bs[s]["m_hat"] for s in states], "o-", color="#1565c0", label="branching m̂")
    axes[1].plot(x, [bs[s]["dfa"] for s in states], "^-", color="#2e7d32", label="DFA α")
    axes[1].axhline(1.0, color="grey", ls="--", lw=0.8, label="critical (=1)")
    axes[1].set_xticks(list(x)); axes[1].set_xticklabels(states, rotation=30, fontsize=7)
    axes[1].set_ylabel("criticality marker"); axes[1].legend(fontsize=7)
    axes[1].set_title("B. Do states traverse criticality?", fontsize=9)

    # C: H vs criticality marker across states (the prediction: H peaks near 1)
    mhat = [bs[s]["m_hat"] for s in states]
    H = [bs[s]["H_max"] for s in states]
    axes[2].scatter(mhat, H, c=range(len(states)), cmap="viridis", s=70, zorder=3)
    for s, mx, hy in zip(states, mhat, H):
        axes[2].annotate(s[:4], (mx, hy), fontsize=6, xytext=(3, 3), textcoords="offset points")
    axes[2].axvline(1.0, color="grey", ls="--", lw=0.8)
    axes[2].set_xlabel("branching ratio m̂ (state mean)"); axes[2].set_ylabel("H_max")
    axes[2].set_title("C. H vs criticality (state means)", fontsize=9)

    fig.suptitle(f"{name} — H maximal near criticality? ({result['dataset']}, "
                 f"n={result['n_subjects']} subj)", fontweight="bold")
    fig.tight_layout()
    C.save_fig(fig, name)

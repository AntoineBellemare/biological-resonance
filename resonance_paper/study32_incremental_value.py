"""Study 32 — Incremental cross-subject value of the resonance axes.

Reviewer challenge: does H or PC surface something a SIMPLER band-power analysis cannot,
on a real (unplanted) effect? Even if neither axis beats band power alone, a legitimate
"surfaces something band power misses" result is: adding H/PC features to band-power
features IMPROVES cross-subject decoding, with a permutation-tested lift.

Two real-EEG state contrasts (PhysioNet eegbci, subjects 1..10, epoch_len=8.0):
  * eyes-open vs eyes-closed (occipital)   — D.load_eeg_states
  * rest vs motor-execution  (motor)       — D.load_eeg_motor_contrast

Feature groups per epoch:
  BAND_POWER = relative delta/theta/alpha/beta power, alpha/theta ratio,
               spectral centroid, spectral edge (95%)  -- non-harmonic spectral features
  RESONANCE  = H_max, H_avg(=H_auc proxy), H_flatness, PC_max, PC_avg, R_max
               (aperiodic removal ON for H, per the paper's finding)

For each contrast we compute subject-grouped LOSO AUC for (a) band-power only,
(b) resonance only, (c) combined; then permutation-test the DELTA AUC
(combined - bandpower) by shuffling labels WITHIN subject (n_perm>=500), one-sided
p that combined > bandpower.

Outputs: results/study32_incremental_value.json
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import datasets as D
from biotuner.resonance import compute_resonance


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def _psd(signal, sf):
    from scipy.signal import welch
    f, p = welch(signal, fs=sf, nperseg=min(len(signal), int(sf * 2)))
    return f, p


def _band_power_features(signal, sf):
    """Non-harmonic spectral features band power gives you."""
    f, p = _psd(signal, sf)
    total = p[(f >= 2) & (f <= 45)].sum() + 1e-20
    bands = {"delta": (2, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (12, 30)}
    rel = {}
    for name, (lo, hi) in bands.items():
        rel[f"rel_{name}"] = float(p[(f >= lo) & (f <= hi)].sum() / total)
    rel["alpha_theta_ratio"] = float(rel["rel_alpha"] / (rel["rel_theta"] + 1e-12))
    # spectral centroid + 95% spectral edge (within 2-45 Hz)
    band = (f >= 2) & (f <= 45)
    fb, pb = f[band], p[band]
    cumsum = np.cumsum(pb)
    rel["spectral_centroid"] = float((fb * pb).sum() / (pb.sum() + 1e-20))
    edge_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    rel["spectral_edge"] = float(fb[min(edge_idx, len(fb) - 1)])
    return rel


RESONANCE_KEYS = ["H_max", "H_avg", "H_flatness", "PC_max", "PC_avg", "R_max"]
BAND_KEYS = ["rel_delta", "rel_theta", "rel_alpha", "rel_beta",
             "alpha_theta_ratio", "spectral_centroid", "spectral_edge"]


def _resonance_features(signal, sf, cfg):
    r = compute_resonance(signal, sf=sf, config=cfg)
    s = r.summaries
    return {
        "H_max": float(s["H"]["max"]),
        "H_avg": float(s["H"]["avg"]),       # mean over spectrum ~ "H_auc" proxy
        "H_flatness": float(s["H"]["flatness"]),
        "PC_max": float(s["PC"]["max"]),
        "PC_avg": float(s["PC"]["avg"]),
        "R_max": float(s["R"]["max"]),
    }


def _rows(epochs, pos_cond, cfg):
    rows = []
    for ep in epochs:
        feat = _band_power_features(ep["signal"], ep["sf"])
        feat.update(_resonance_features(ep["signal"], ep["sf"], cfg))
        feat["subject"] = ep["subject"]
        feat["y"] = 1 if ep["condition"] == pos_cond else 0
        rows.append(feat)
    return rows


# --------------------------------------------------------------------------
# LOSO + permutation lift
# --------------------------------------------------------------------------
def _loso_auc(X, y, subj):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    preds = np.full(len(y), np.nan)
    for s in sorted(set(subj)):
        tr, te = subj != s, subj == s
        if len(set(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        preds[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(preds)
    if len(set(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], preds[ok]))


def _analyze(name, epochs, pos_cond, cfg, n_perm):
    rows = _rows(epochs, pos_cond, cfg)
    subj = np.array([r["subject"] for r in rows])
    y = np.array([r["y"] for r in rows])

    Xb = np.nan_to_num(np.array([[r[k] for k in BAND_KEYS] for r in rows], float))
    Xr = np.nan_to_num(np.array([[r[k] for k in RESONANCE_KEYS] for r in rows], float))
    Xc = np.concatenate([Xb, Xr], axis=1)

    auc_band = _loso_auc(Xb, y, subj)
    auc_res = _loso_auc(Xr, y, subj)
    auc_comb = _loso_auc(Xc, y, subj)
    delta = auc_comb - auc_band

    # permutation: shuffle labels WITHIN subject, recompute the SAME delta
    def _null(rng):
        yp = y.copy()
        for s in set(subj):
            idx = np.where(subj == s)[0]
            yp[idx] = rng.permutation(yp[idx])
        return _loso_auc(Xc, yp, subj) - _loso_auc(Xb, yp, subj)

    perm = C.permutation_test_value(delta, _null, n_perm=n_perm, seed=0, tail="greater")

    return dict(name=name, n_epochs=len(rows), n_subjects=int(len(set(subj))),
                pos=pos_cond,
                auc_band_power=auc_band, auc_resonance=auc_res, auc_combined=auc_comb,
                delta_auc=float(delta), lift_p=perm["p"],
                lift_null_mean=perm["null_mean"])


def run(quick=True):
    subjects = tuple(range(1, 6)) if quick else tuple(range(1, 11))
    n_perm = 500 if quick else 1000
    cfg = C.default_config(fmin=2, fmax=45, remove_aperiodic=True)
    out = {}

    eo_ec = D.load_eeg_states(subjects=subjects, channels=D.OCCIPITAL,
                              epoch_len=8.0, max_epochs_per_run=7)
    if eo_ec:
        print(f"  EO/EC: {len(eo_ec)} epochs", flush=True)
        out["eyes_open_vs_closed"] = _analyze("eyes_open_vs_closed", eo_ec, "EC", cfg, n_perm)

    motor = D.load_eeg_motor_contrast(subjects=subjects, channels=D.MOTOR_CHANNELS,
                                      epoch_len=8.0, max_epochs_per_run=7)
    if motor:
        print(f"  REST/MOTOR: {len(motor)} epochs", flush=True)
        out["rest_vs_motor"] = _analyze("rest_vs_motor", motor, "MOTOR", cfg, n_perm)

    if not out:
        print("  no EEG loaded; aborting study 32.")
        return None
    C.save_json(dict(quick=quick, subjects=list(subjects), contrasts=out),
                "study32_incremental_value.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 32 headline (incremental cross-subject value) ---")
    for cname, c in out.items():
        print(f"  [{cname}] n={c['n_epochs']} epochs, {c['n_subjects']} subjects")
        print(f"      band-power LOSO AUC = {c['auc_band_power']:.3f}")
        print(f"      resonance  LOSO AUC = {c['auc_resonance']:.3f}")
        print(f"      combined   LOSO AUC = {c['auc_combined']:.3f}")
        print(f"      delta (comb-band)   = {c['delta_auc']:+.3f}  "
              f"(perm null mean {c['lift_null_mean']:+.3f}, one-sided p={c['lift_p']:.3g})")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

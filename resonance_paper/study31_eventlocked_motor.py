"""Study 31 — Confound-free EVENT-LOCKED motor lateralization: does H or cross-channel
PC add to (or beat) the lateralized mu/beta ERD that band power already gives, cross-subject?

The prior motor contrast (study2 / study30) was RUN-LEVEL (rest run 1 vs motor run 3), which
confounds arousal / time-on-task / electrode drift. Here we remove that confound by working
*within* the eegbci motor-execution runs (3, 7, 11) and contrasting the two movement classes
against each other using the trial event annotations:

    T1 = open/close LEFT  fist  -> contralateral mu/beta ERD at C4 (right hemisphere)
    T2 = open/close RIGHT fist  -> contralateral ERD at C3 (left hemisphere)
    T0 = rest

Because T1 and T2 trials are interleaved within the same run, arousal/time/drift are matched;
the only systematic difference is which hemisphere desynchronizes. This is the textbook clean
lateralization effect and a very strong band-power baseline.

Per epoch (tmin=0.5, tmax=4.0 s, picks C3/C4) we compute:
  * mu (8-12) and beta (18-26) band power at C3 and C4 (FFT, Hann)
  * H_max at C3 and C4 (compute_resonance, aperiodic removal ON)
  * cross-channel PC_z(C3,C4) at 1:1 (10,10) and 1:2 (10,20) (cross_target_z, IAAFT-of-B null)

LATERALIZED features (left T1 vs right T2):
  * band-power lateralization: (C4-C3) mu power, (C4-C3) beta power
  * H lateralization: (H_C4 - H_C3)
  * PC: cross-channel PC_z at 1:1 and 1:2 (symmetric; tests coupling, not lateralization)

We also run the easier movement-vs-rest contrast (T1|T2 vs T0) as a fallback.

Evaluation: single-feature bootstrap AUC + cross-subject LOSO logistic regression (mirrors
study2), for band-power, H, PC, and band+H combined.

QUESTION: does H or cross-channel PC BEAT or ADD TO the lateralized band-power ERD, cross-subject?

Outputs: results/study31_eventlocked_motor.json
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import datasets as D
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import compute_resonance, ResonanceConfig
from biotuner.resonance.nulls import iaaft_surrogate

MOTOR_RUNS = (3, 7, 11)
MU = (8.0, 12.0)
BETA = (18.0, 26.0)
TMIN, TMAX = 0.5, 4.0

H_CFG = C.default_config(fmin=2, fmax=45, remove_aperiodic=True)
PC_CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, noverlap=1,
                         coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                         ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                         return_intermediates=True)
PC_PAIRS = {"1:1": [(10.0, 10.0)], "1:2": [(10.0, 20.0)]}


def _band_power(x, sf, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sf)
    band = (f >= lo) & (f <= hi)
    total = (f >= 2) & (f <= 45)
    return float(X[band].sum() / (X[total].sum() + 1e-20))


def _h_max(x, sf):
    return float(compute_resonance(x, sf=sf, config=H_CFG).summaries["H"]["max"])


def _pc_z_pairs(A, B, sf, pair_dict, n, seed):
    """Cross-channel PC surrogate z for SEVERAL n:m ratio pairs at once, serial (no joblib).

    Mirrors study5.cross_target_z (channel-B IAAFT null on the phase-coupling matrix entry),
    but (a) computes all ratio pairs from a SINGLE observed + surrogate ensemble and (b) runs
    serially -- on Windows the per-call joblib process-pool spawn in cross_target_z dominates
    runtime when invoked ~90x/subject, so we share the ensemble and skip the pool entirely.
    """
    obs = compute_cross_resonance(A, B, sf=sf, config=PC_CFG)
    fr = obs.freqs
    idx = lambda f: int(np.argmin(np.abs(fr - f)))

    def measure(result):
        M = result.phase_coupling_matrix
        return {pk: max(M[idx(fa), idx(fb)] for fa, fb in pairs)
                for pk, pairs in pair_dict.items()}

    obs_v = measure(obs)
    rng = np.random.default_rng(seed)
    surr = {pk: [] for pk in pair_dict}
    for _ in range(n):
        Bs = iaaft_surrogate(B, rng)
        mv = measure(compute_cross_resonance(A, Bs, sf=sf, config=PC_CFG))
        for pk in pair_dict:
            surr[pk].append(mv[pk])
    out = {}
    for pk in pair_dict:
        s = np.asarray(surr[pk], float)
        out[pk] = float((obs_v[pk] - s.mean()) / (s.std() + 1e-12))
    return out


def _epochs_for_subject(subj, n_pc_surr, skip_pc=False, include_rest=False):
    """Return list of per-trial dicts with raw C3/C4 signals + derived features.

    Labels: LEFT (T1), RIGHT (T2), and (if include_rest) REST (T0).
    skip_pc=True omits the (expensive) cross-channel PC_z features -- used for the
    fast cross-subject ERD-vs-H comparison; PC is profiled separately on a subject
    subset because each PC_z needs n surrogate cross-resonance fits.
    """
    import mne
    segs = []
    keep = ("T0", "T1", "T2") if include_rest else ("T1", "T2")
    lab = {"T0": "REST", "T1": "LEFT", "T2": "RIGHT"}
    for run in MOTOR_RUNS:
        try:
            raw = D._load_eegbci_raw(subj, run)
        except Exception as exc:
            print(f"  [skip] subj {subj} run {run}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if "C3" not in raw.ch_names or "C4" not in raw.ch_names:
            continue
        raw.filter(1.0, 45.0, verbose="ERROR")
        events, ev_id = mne.events_from_annotations(raw)
        want = {k: v for k, v in ev_id.items() if k in keep}
        if not want:
            continue
        epochs = mne.Epochs(raw, events, event_id=want, tmin=TMIN, tmax=TMAX,
                            baseline=None, picks=["C3", "C4"], preload=True,
                            verbose="ERROR")
        sf = float(epochs.info["sfreq"])
        data = epochs.get_data()  # (n_ep, 2, n_times); ch order from picks=["C3","C4"]
        ch = list(epochs.ch_names)
        i3, i4 = ch.index("C3"), ch.index("C4")
        codes = epochs.events[:, 2]
        inv = {v: k for k, v in want.items()}
        for k in range(data.shape[0]):
            segs.append(dict(subj=subj, label=lab[inv[codes[k]]], sf=sf,
                             C3=data[k, i3].astype(np.float64),
                             C4=data[k, i4].astype(np.float64)))
    # derive features per trial
    rows = []
    for s in segs:
        c3, c4, sf = s["C3"], s["C4"], s["sf"]
        row = dict(subj=s["subj"], label=s["label"])
        for band, (lo, hi) in (("mu", MU), ("beta", BETA)):
            p3 = _band_power(c3, sf, lo, hi)
            p4 = _band_power(c4, sf, lo, hi)
            row[f"bp_{band}_C3"] = p3
            row[f"bp_{band}_C4"] = p4
            row[f"bp_{band}_lat"] = p4 - p3  # (C4 - C3): higher -> LEFT-fist ERD at C4 makes this LOWER
            row[f"bp_{band}_mean"] = 0.5 * (p3 + p4)  # bilateral power (for movement vs rest)
        h3, h4 = _h_max(c3, sf), _h_max(c4, sf)
        row["H_C3"], row["H_C4"], row["H_lat"] = h3, h4, h4 - h3
        row["H_mean"] = 0.5 * (h3 + h4)
        if not skip_pc:
            pcz = _pc_z_pairs(c3, c4, sf, PC_PAIRS, n=n_pc_surr,
                              seed=hash((s["subj"], s["label"], len(rows))) % (2**31))
            for pk in PC_PAIRS:
                row[f"PCz_{pk}"] = pcz[pk]
        rows.append(row)
    return rows


# ---- evaluation ---------------------------------------------------------
def _loso(rows, feat_keys, pos_label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import roc_auc_score
    subj = np.array([r["subj"] for r in rows])
    y = np.array([1 if r["label"] == pos_label else 0 for r in rows])
    X = np.nan_to_num(np.array([[r[k] for k in feat_keys] for r in rows], float))
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
        return float("nan")
    return float(roc_auc_score(y[ok], preds[ok]))


def _single_auc(rows, key, pos_label):
    pos = [r[key] for r in rows if r["label"] == pos_label]
    neg = [r[key] for r in rows if r["label"] != pos_label]
    ci = C.bootstrap_auc_ci(pos, neg, n_boot=2000)
    # orient so AUC >= 0.5 (sign of feature is not known a priori)
    if ci["auc"] < 0.5:
        ci = dict(auc=1 - ci["auc"], lo=1 - ci["hi"], hi=1 - ci["lo"])
    return ci


def _contrast(rows, pos_label, neg_label, bp_keys, h_keys, has_pc):
    """rows already filtered to the two labels of interest. pos_label is class 1.
    bp_keys / h_keys = the band-power and harmonicity feature sets for this contrast."""
    pc_keys = ["PCz_1:1", "PCz_1:2"] if has_pc else []
    feats_single = bp_keys + h_keys + pc_keys
    single = {k: _single_auc(rows, k, pos_label) for k in feats_single}

    loso = dict(
        band_power=_loso(rows, bp_keys, pos_label),
        H=_loso(rows, h_keys, pos_label),
        band_plus_H=_loso(rows, bp_keys + h_keys, pos_label),
    )
    if has_pc:
        loso["PC"] = _loso(rows, pc_keys, pos_label)
        loso["band_plus_PC"] = _loso(rows, bp_keys + pc_keys, pos_label)
        loso["band_plus_H_plus_PC"] = _loso(rows, bp_keys + h_keys + pc_keys, pos_label)
    n_subj = len(set(r["subj"] for r in rows))
    n_pos = sum(1 for r in rows if r["label"] == pos_label)
    return dict(pos=pos_label, neg=neg_label, n_epochs=len(rows), n_subjects=n_subj,
                n_pos=n_pos, single_feature_auc=single, loso_auc=loso)


BP_LAT = ["bp_mu_lat", "bp_beta_lat"]          # lateralized ERD (LEFT vs RIGHT)
H_LAT = ["H_lat"]
BP_MEAN = ["bp_mu_mean", "bp_beta_mean"]        # bilateral power (movement vs rest)
H_MEAN = ["H_mean"]


def run(quick=True):
    # Core question (does H/PC beat ERD lateralization, cross-subject) is answered with
    # the cheap band-power + H features over more subjects; PC -- whose per-epoch
    # surrogate cost dominates -- is profiled on a 2-subject subset (matches study30's
    # finding that scalp n:m PC sits at the null).
    subjects = (1, 2, 3, 4, 5, 6) if quick else tuple(range(1, 11))
    # compute_cross_resonance is ~30x compute_resonance; the PC subset dominates runtime.
    # quick: skip PC here (profiled separately at QUICK; see _pc_z_pairs note). paper: 2 subjects.
    pc_subjects = () if quick else (1, 2)
    n_pc_surr = 25 if quick else 50

    all_rows = []
    for subj in subjects:
        r = _epochs_for_subject(subj, n_pc_surr, skip_pc=True, include_rest=True)
        n = lambda L: sum(1 for x in r if x["label"] == L)
        print(f"  subj {subj}: {len(r)} trials (L={n('LEFT')}, R={n('RIGHT')}, REST={n('REST')})",
              flush=True)
        all_rows.extend(r)
    if not all_rows:
        print("  no epochs loaded; aborting study 31.")
        return None

    # PC profile on a small subset (LEFT/RIGHT only)
    pc_rows = []
    for subj in pc_subjects:
        pc_rows.extend(_epochs_for_subject(subj, n_pc_surr, skip_pc=False, include_rest=False))
        print(f"  [PC subset] subj {subj} done", flush=True)

    fist_rows = [r for r in all_rows if r["label"] in ("LEFT", "RIGHT")]
    move_rows = [dict(r, label=("MOVE" if r["label"] in ("LEFT", "RIGHT") else "REST"))
                 for r in all_rows]

    result = dict(
        quick=quick, subjects=list(subjects), pc_subjects=list(pc_subjects), n_pc_surr=n_pc_surr,
        # confound-free lateralization (band-power + H, more subjects, no PC)
        contrast_left_vs_right=_contrast(fist_rows, "LEFT", "RIGHT", BP_LAT, H_LAT, has_pc=False),
        # movement vs rest (bilateral power vs H), event-locked within motor runs
        contrast_move_vs_rest=_contrast(move_rows, "MOVE", "REST", BP_MEAN, H_MEAN, has_pc=False),
    )
    if pc_rows:
        # PC profile (cross-channel n:m), subset only
        result["contrast_left_vs_right_pc"] = _contrast(
            pc_rows, "LEFT", "RIGHT", BP_LAT, H_LAT, has_pc=True)
    C.save_json(result, "study31_eventlocked_motor.json")
    _headline(result)
    return result


def _print_contrast(name, c):
    print(f"\n  [{name}]  pos={c['pos']} vs {c['neg']}  n_epochs={c['n_epochs']}  "
          f"n_subj={c['n_subjects']}  n_pos={c['n_pos']}")
    sf = c["single_feature_auc"]
    print("   single-feature bootstrap AUC (oriented >=0.5):")
    for k, v in sf.items():
        print(f"     {k:12s} AUC={v['auc']:.3f} [{v['lo']:.2f},{v['hi']:.2f}]")
    print("   cross-subject LOSO logistic-regression AUC:")
    for k, v in c["loso_auc"].items():
        print(f"     {k:22s} AUC={v:.3f}")


def _headline(result):
    print("\n  --- Study 31 headline (confound-free event-locked motor) ---")
    _print_contrast("LEFT vs RIGHT (lateralized ERD; band-power + H)", result["contrast_left_vs_right"])
    _print_contrast("MOVE vs REST (bilateral power vs H)", result["contrast_move_vs_rest"])
    if "contrast_left_vs_right_pc" in result:
        _print_contrast("LEFT vs RIGHT + PC subset", result["contrast_left_vs_right_pc"])
    print("\n  => value-add iff H or PC LOSO beats band_power, or band_plus_* > band_power alone.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

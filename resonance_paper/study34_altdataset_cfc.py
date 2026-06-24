"""Study 34 — n:m phase-coupling on a dataset where CFC is DOCUMENTED (Sleep-EDF NREM).

Scalp resting EEG (Study 30) showed genuine n!=m coupling sits at the surrogate null. The
standard objection is that scalp rest is simply a poor place to look: cross-frequency coupling
(CFC) is strongest in NREM sleep, where the slow oscillation organises spindle/faster activity.
This study moves the PC probe to a dataset where CFC is textbook-documented — PhysioNet Sleep-EDF
(already cached locally; no download) — and asks the same two questions as Study 30:

  (1) IS THERE REAL n:m COUPLING band power is blind to?  For each (channel-config, n:m pair) we
      take NREM (N2/N3) epochs and compute the fraction with surrogate-significant PC_z (IAAFT-of-B
      null; z>1.645 ~= rank p<=0.05 one-sided). Band power is phase-blind, so any genuine, replicable
      n!=m coupling here is a value-add by kind. We probe within-channel (Pz-Oz, the posterior
      channel carrying spindles) and cross-channel (Fpz-Cz <-> Pz-Oz) at low-integer pairs that fall
      inside our 2-45 Hz window: SO/delta-theta (3,6), delta-alpha (2,8)->1:4 out of family so use
      theta-alpha (5,10)=1:2, (4,8)=1:2, alpha-beta (6,9)=2:3, (8,12)=2:3, (10,20)=1:2, (10,30)=1:3.
      1:1 (10,10) is included as the coherence positive-control.
  (2) DOES PC BEAT BAND POWER on a real state contrast?  Wake vs N2 (deep-vs-light is the classic
      CFC-bearing contrast). Per-subject AUC for PC_z at the best n:m pair vs delta/theta/alpha/sigma
      band power, aggregated cross-subject with bootstrap CI.

Honest framing: if n!=m PC_z fractions track the FPR (~5%) and PC does not beat band power, that is a
NULL even in the CFC-rich regime, and we report it as such with the real numbers.

Outputs: results/study34_altdataset_cfc.json
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper.study5_cross_signal import cross_target_z
from resonance_paper.study14_sleep import load_sleep
from biotuner.resonance import ResonanceConfig

# sf is 100 Hz in Sleep-EDF; keep fmax well under Nyquist.
CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=40, noverlap=1,
                      coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                      ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)

# n:m pairs (fA, fB) inside 2-40 Hz. Labels note the integer ratio.
PAIRS = {
    "1:1_alpha":        [(10.0, 10.0)],   # coherence positive control
    "1:2_theta-alpha":  [(5.0, 10.0)],
    "1:2_alpha-beta":   [(10.0, 20.0)],
    "1:3_alpha-beta":   [(10.0, 30.0)],
    "2:3_alpha-beta":   [(8.0, 12.0)],
    "1:2_delta-theta":  [(3.0, 6.0)],
    "1:4_theta-beta":   [(5.0, 20.0)],
}
NM_KEYS = [k for k in PAIRS if not k.startswith("1:1")]  # genuine n!=m


def _band_power(x, sf, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sf)
    m = (f >= lo) & (f <= hi)
    return float(X[m].mean()) if m.any() else float("nan")


def _nm_pc_fraction(epochs_AB, pair_keys, n_surr, label):
    """For each n:m pair, fraction of NREM epochs with surrogate-significant cross-PC_z."""
    res = {}
    for pk in pair_keys:
        pairs = PAIRS[pk]
        zs = []
        for k, (A, B, sf) in enumerate(epochs_AB):
            z = cross_target_z(A, B, sf, CFG, pairs, "PC", n=n_surr,
                               seed=(hash((label, pk, k)) % (2 ** 31)))
            if np.isfinite(z):
                zs.append(z)
        zs = np.asarray(zs)
        res[pk] = dict(n=int(zs.size),
                       frac_sig=float(np.mean(zs > 1.645)) if zs.size else float("nan"),
                       median_z=float(np.median(zs)) if zs.size else float("nan"),
                       mean_z=float(np.mean(zs)) if zs.size else float("nan"))
    return res


def run(quick=True):
    n_sub = 4 if quick else 8
    mxep = 8 if quick else 18
    n_surr = 30 if quick else 60

    print(f"Study 34 — loading {n_sub} Sleep-EDF nights (cached) ...", flush=True)
    items = load_sleep(n_subjects=n_sub, max_epochs_per_stage=mxep)
    # items: dict(X=(n_ch,n_times), h_idx, sf, subject, state); ch0=Fpz-Cz, ch1=Pz-Oz
    print(f"  {len(items)} epochs; states present: "
          + ", ".join(sorted(set(it['state'] for it in items))), flush=True)

    nrem = [it for it in items if it["state"] in ("N2", "N3")]
    print(f"  NREM (N2/N3) epochs: {len(nrem)}", flush=True)

    # ---- Q1: real n:m coupling in NREM (within Pz-Oz, and cross Fpz-Cz<->Pz-Oz) ----
    within = [(it["X"][1], it["X"][1], it["sf"]) for it in nrem if it["X"].shape[0] > 1]
    cross  = [(it["X"][0], it["X"][1], it["sf"]) for it in nrem if it["X"].shape[0] > 1]
    # cap epoch count in quick mode to keep runtime sane
    cap = 24 if quick else 1000
    within, cross = within[:cap], cross[:cap]
    print(f"  probing within Pz-Oz n={len(within)}, cross Fpz-Cz<->Pz-Oz n={len(cross)} ...", flush=True)

    out = {}
    out["within_pzoz"] = _nm_pc_fraction(within, list(PAIRS), n_surr, "within")
    print("  within done", flush=True)
    out["cross_fpzcz_pzoz"] = _nm_pc_fraction(cross, list(PAIRS), n_surr, "cross")
    print("  cross done", flush=True)

    # ---- Q2: state contrast Wake vs N2, PC_z (best n!=m pair, cross) vs band power ----
    # build per-subject epoch lists for Wake and N2
    state_contrast = {}
    subjects = sorted(set(it["subject"] for it in items))
    # pick the best n!=m pair by cross frac_sig for the PC feature
    cross_nm = {k: out["cross_fpzcz_pzoz"][k]["frac_sig"] for k in NM_KEYS}
    best_pair = max(cross_nm, key=lambda k: (cross_nm[k] if np.isfinite(cross_nm[k]) else -1))
    print(f"  state contrast uses best n!=m pair = {best_pair}", flush=True)

    bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16)}
    pc_pos, pc_neg = [], []          # PC_z at best pair: N2(pos) vs Wake(neg)
    bp_pos = {b: [] for b in bands}
    bp_neg = {b: [] for b in bands}
    per_subj = {}
    for subj in subjects:
        wake = [it for it in items if it["subject"] == subj and it["state"] == "Wake"]
        n2 = [it for it in items if it["subject"] == subj and it["state"] == "N2"]
        if len(wake) < 2 or len(n2) < 2:
            continue
        ws = wake[:6]; ns = n2[:6]
        sp = {"pc_n2": [], "pc_wake": []}
        for tag, eps in (("pc_n2", ns), ("pc_wake", ws)):
            for k, it in enumerate(eps):
                A, B = it["X"][0], it["X"][1]
                z = cross_target_z(A, B, it["sf"], CFG, PAIRS[best_pair], "PC",
                                   n=n_surr, seed=(hash((subj, tag, k)) % (2 ** 31)))
                if np.isfinite(z):
                    sp[tag].append(z)
        for it in ns:
            for b, (lo, hi) in bands.items():
                bp_pos[b].append(_band_power(it["X"][1], it["sf"], lo, hi))
        for it in ws:
            for b, (lo, hi) in bands.items():
                bp_neg[b].append(_band_power(it["X"][1], it["sf"], lo, hi))
        pc_pos.extend(sp["pc_n2"]); pc_neg.extend(sp["pc_wake"])
        per_subj[subj] = dict(n_wake=len(ws), n_n2=len(ns),
                              pc_n2_mean=float(np.mean(sp["pc_n2"])) if sp["pc_n2"] else float("nan"),
                              pc_wake_mean=float(np.mean(sp["pc_wake"])) if sp["pc_wake"] else float("nan"))

    pc_auc = C.bootstrap_auc_ci(pc_pos, pc_neg) if pc_pos and pc_neg else dict(auc=float("nan"), lo=float("nan"), hi=float("nan"))
    bp_auc = {}
    for b in bands:
        bp_auc[b] = C.bootstrap_auc_ci(bp_pos[b], bp_neg[b]) if bp_pos[b] and bp_neg[b] else dict(auc=float("nan"))
    state_contrast = dict(best_pair=best_pair, n_pos=len(pc_pos), n_neg=len(pc_neg),
                          pc_z_auc=pc_auc, band_power_auc=bp_auc, per_subject=per_subj)
    out["state_contrast_wake_vs_n2"] = state_contrast

    result = dict(quick=quick, n_subjects=n_sub, n_surrogates=n_surr,
                  dataset="Sleep-EDF (Sleep Cassette, cached)", out=out)
    C.save_json(result, "study34_altdataset_cfc.json")
    _headline(out)
    return result


def _headline(out):
    print("\n  --- Study 34 headline (NREM n:m PC value-add probe) ---")
    for cfgname in ("within_pzoz", "cross_fpzcz_pzoz"):
        print(f"  [{cfgname}]")
        for pk, r in out[cfgname].items():
            star = " <-- coherence" if pk.startswith("1:1") else ""
            print(f"      {pk:18s} frac_sig={r['frac_sig']:.2f}  median_z={r['median_z']:+.2f}  "
                  f"mean_z={r['mean_z']:+.2f}  (n={r['n']}){star}")
    sc = out["state_contrast_wake_vs_n2"]
    print(f"  [state contrast Wake vs N2]  PC pair={sc['best_pair']}")
    print(f"      PC_z AUC = {sc['pc_z_auc']['auc']:.2f} "
          f"[{sc['pc_z_auc'].get('lo', float('nan')):.2f},{sc['pc_z_auc'].get('hi', float('nan')):.2f}]"
          f"  (n_pos={sc['n_pos']}, n_neg={sc['n_neg']})")
    bp = sc["band_power_auc"]
    print("      band-power AUC: " + "  ".join(f"{b}={bp[b]['auc']:.2f}" for b in bp))
    print("  => value-add only if an n!=m pair has frac_sig clearly above ~0.05 (FPR) AND/OR")
    print("     PC_z AUC clearly exceeds the best band-power AUC.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

"""Study 33 — Exhaustive search for surrogate-significant, cross-subject-REPLICABLE
n!=m cross-channel phase coupling in real EEG.

Study 30 settled the easy cases: cross-channel 1:1 (same-frequency) PC_z is robustly
surrogate-significant (it is coherence, not a novel n:m claim), while the two n!=m ratios
it tried (mu-beta 1:2, alpha-theta 3:2) sat at the null. This study widens the net to
authoritatively answer: is there ANY genuine n!=m (channel-pair, ratio) cross-channel
phase coupling in scalp EEG that (a) clearly exceeds the ~0.05 false-positive rate AND
(b) replicates across a majority of subjects (not driven by one)?

Method (per epoch, per (pair, ratio)): targeted rank-p test on the cross-channel
phase-coupling matrix entry Phi_AB[fa, fb], with an IAAFT-of-B surrogate distribution
(n>=49) that preserves both PSDs while destroying only the A-B phase relation. This
mirrors study29's within_pc_z rank_p logic but for the CROSS-channel targeted entry,
reusing the compute_cross_resonance machinery behind cross_target_z.

  rank_p = (1 + #{surr >= obs}) / (n_surr + 1)

Decision per (dataset, pair, ratio):
  frac_sig          = fraction of epochs with rank_p <= 0.05            (vs ~0.05 FPR)
  n_subj_replicate  = #distinct subjects whose MEDIAN epoch rank_p <= 0.05
A real effect should show frac_sig clearly above chance AND replicate across a majority
of subjects. A 1:1 reference pair is included as a positive control (it should light up).

QUICK-grade: subjects 1..8, ~49 surrogates. Outputs: results/study33_broad_nm_search.json
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import datasets as D
from biotuner.harmonic_connectivity import compute_cross_resonance
from biotuner.resonance import ResonanceConfig
from biotuner.resonance.nulls import iaaft_surrogate

CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, noverlap=1,
                      coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                      ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)

# n!=m ratios to test (fa, fb), label.  Plus a 1:1 positive-control reference.
RATIOS = [
    ((10.0, 20.0), "1:2@10-20"),
    ((10.0, 30.0), "1:3@10-30"),
    ((6.0, 12.0), "1:2@6-12"),
    ((6.0, 9.0), "2:3@6-9"),
    ((8.0, 12.0), "2:3@8-12"),
    ((10.0, 15.0), "2:3@10-15"),
    ((6.0, 10.0), "3:5@6-10"),
    ((4.0, 8.0), "1:2@4-8"),
    ((12.0, 18.0), "2:3@12-18"),
    ((5.0, 15.0), "1:3@5-15"),
    ((10.0, 10.0), "1:1@10-10[ctrl]"),  # positive control — should be significant
]

# channel pairs (use whichever are present in the 64-ch montage)
PAIRS = [("C3", "C4"), ("C3", "Cz"), ("O1", "O2"), ("Oz", "O1"),
         ("C3", "O1"), ("F3", "F4")]


def _epochs(subjects, runs, chans, epoch_s=8.0, max_ep=7):
    """Load multi-channel epochs: each entry holds a {ch: signal} dict for the present channels."""
    out = []
    want = set(chans)
    for subj in subjects:
        for cond, run in runs:
            try:
                raw = D._load_eegbci_raw(subj, run)
            except Exception as exc:
                print(f"  [skip] subj {subj} run {run}: {type(exc).__name__}", flush=True)
                continue
            raw.filter(1.0, 45.0, verbose="ERROR")
            sf = float(raw.info["sfreq"])
            present = [c for c in chans if c in raw.ch_names]
            if len(present) < 2:
                continue
            data = raw.get_data(picks=present)
            seg = int(epoch_s * sf)
            n_ep = min(max_ep, data.shape[1] // seg)
            for i in range(n_ep):
                chans_sig = {ch: data[ci, i * seg:(i + 1) * seg].astype(np.float64)
                             for ci, ch in enumerate(present)}
                out.append(dict(subj=subj, cond=cond, sf=sf, sig=chans_sig))
    return out


def cross_target_rank_p(A, B, sf, fa, fb, n_surr, seed):
    """Targeted rank-p for the cross-channel phase-coupling matrix entry Phi_AB[fa, fb].

    Observed = Phi_AB[fa, fb]; null = IAAFT-of-B distribution (preserves both PSDs,
    destroys the A-B phase relation). rank_p = (1 + #{surr >= obs}) / (n_surr + 1).
    Mirrors study29.within_pc_z but on the CROSS targeted entry (reuses
    compute_cross_resonance internals). SERIAL over surrogates — parallelism is at the
    epoch level (one joblib pool per pair/ratio) to avoid Windows pool-spawn overhead.
    """
    obs = compute_cross_resonance(A, B, sf=sf, config=CFG)
    fr = obs.freqs
    ia = int(np.argmin(np.abs(fr - fa)))
    ib = int(np.argmin(np.abs(fr - fb)))
    obs_v = float(obs.phase_coupling_matrix[ia, ib])
    rng = np.random.default_rng(seed)
    sv = np.empty(n_surr)
    for k, s in enumerate(rng.integers(0, 2 ** 31 - 1, size=n_surr)):
        Bs = iaaft_surrogate(B, np.random.default_rng(int(s)))
        r = compute_cross_resonance(A, Bs, sf=sf, config=CFG)
        sv[k] = float(r.phase_coupling_matrix[ia, ib])
    return float((1 + np.sum(sv >= obs_v)) / (n_surr + 1))


def _probe_dataset(name, epochs, n_surr):
    """For each (pair, ratio): per-epoch rank_p (parallel over epochs), then frac_sig +
    per-subject replicability."""
    subjects = sorted(set(e["subj"] for e in epochs))
    try:
        from joblib import Parallel, delayed
        _par = Parallel(n_jobs=-1)
    except Exception:
        _par = None
    out = {}
    for (ca, cb) in PAIRS:
        for (fa, fb), rlab in RATIOS:
            key = f"{ca}-{cb} {rlab}"
            jobs = [ep for ep in epochs if ca in ep["sig"] and cb in ep["sig"]]
            if not jobs:
                continue
            seeds = [abs(hash((ep["subj"], ep["cond"], ca, cb, rlab))) % (2 ** 31) for ep in jobs]
            if _par is not None:
                rps = _par(delayed(cross_target_rank_p)(
                    ep["sig"][ca], ep["sig"][cb], ep["sf"], fa, fb, n_surr, sd)
                    for ep, sd in zip(jobs, seeds))
            else:
                rps = [cross_target_rank_p(ep["sig"][ca], ep["sig"][cb], ep["sf"],
                                           fa, fb, n_surr, sd) for ep, sd in zip(jobs, seeds)]
            per_epoch = [(ep["subj"], rp) for ep, rp in zip(jobs, rps)]
            if not per_epoch:
                continue
            rps = np.array([p for _, p in per_epoch])
            frac_sig = float(np.mean(rps <= 0.05))
            # replicability: per-subject median rank_p <= 0.05
            subj_sig = 0
            subj_present = sorted(set(s for s, _ in per_epoch))
            for s in subj_present:
                med = np.median([p for ss, p in per_epoch if ss == s])
                if med <= 0.05:
                    subj_sig += 1
            out[key] = dict(n_epochs=len(per_epoch), n_subj=len(subj_present),
                            frac_sig=frac_sig, n_subj_replicate=subj_sig,
                            median_rank_p=float(np.median(rps)))
            print(f"    {key:26s} frac_sig={frac_sig:.2f}  "
                  f"replicate={subj_sig}/{len(subj_present)} subj  "
                  f"med_p={np.median(rps):.3f}  (n_ep={len(per_epoch)})", flush=True)
    return dict(subjects=subjects, results=out)


def run(quick=True):
    subjects = tuple(range(1, 9)) if quick else tuple(range(1, 11))
    n_surr = 49 if quick else 99
    all_chans = sorted({c for p in PAIRS for c in p})
    out = {}

    print(f"  loading motor (REST/MOTOR) epochs, subjects {subjects[0]}..{subjects[-1]}", flush=True)
    motor = _epochs(subjects, (("REST", D.REST_RUN), ("MOTOR", D.MOTOR_RUN)), all_chans)
    if motor:
        print(f"  motor epochs: {len(motor)} — probing {len(PAIRS)} pairs x {len(RATIOS)} ratios", flush=True)
        out["motor"] = _probe_dataset("motor", motor, n_surr)

    print(f"  loading occipital (EO/EC) epochs", flush=True)
    occ = _epochs(subjects, (("EO", D.EO_RUN), ("EC", D.EC_RUN)), all_chans)
    if occ:
        print(f"  occipital epochs: {len(occ)} — probing {len(PAIRS)} pairs x {len(RATIOS)} ratios", flush=True)
        out["occipital"] = _probe_dataset("occipital", occ, n_surr)

    result = dict(quick=quick, n_surr=n_surr, subjects=list(subjects), out=out)
    C.save_json(result, "study33_broad_nm_search.json")
    _headline(result)
    return result


def _headline(result):
    print("\n  --- Study 33 headline (exhaustive n!=m cross-channel coupling search) ---")
    # Collect all n!=m entries (exclude the 1:1 control) and the control separately.
    nm_entries = []  # (dataset, key, rec)
    ctrl_entries = []
    for ds, blk in result["out"].items():
        n_subj_total = len(blk["subjects"])
        for key, rec in blk["results"].items():
            if "[ctrl]" in key:
                ctrl_entries.append((ds, key, rec, n_subj_total))
            else:
                nm_entries.append((ds, key, rec, n_subj_total))

    if ctrl_entries:
        print("  positive control (1:1, should be significant):")
        for ds, key, rec, nt in ctrl_entries:
            maj = rec["n_subj_replicate"] > rec["n_subj"] / 2
            print(f"    [{ds}] {key}: frac_sig={rec['frac_sig']:.2f}  "
                  f"replicate={rec['n_subj_replicate']}/{rec['n_subj']} "
                  f"{'(majority)' if maj else ''}")

    # Best n!=m by frac_sig and the replication picture.
    nm_sorted = sorted(nm_entries, key=lambda x: x[2]["frac_sig"], reverse=True)
    hits = [(ds, key, rec) for ds, key, rec, nt in nm_entries
            if rec["frac_sig"] > 0.15 and rec["n_subj_replicate"] > rec["n_subj"] / 2]
    print(f"\n  n!=m entries tested: {len(nm_entries)}")
    if nm_sorted:
        ds, key, rec, _ = nm_sorted[0]
        print(f"  max n!=m frac_sig: {rec['frac_sig']:.2f}  ({key}, {ds}; "
              f"replicate={rec['n_subj_replicate']}/{rec['n_subj']})")
        print("  top 5 n!=m by frac_sig:")
        for ds, key, rec, _ in nm_sorted[:5]:
            print(f"    [{ds}] {key}: frac_sig={rec['frac_sig']:.2f}  "
                  f"replicate={rec['n_subj_replicate']}/{rec['n_subj']}  med_p={rec['median_rank_p']:.3f}")
    if hits:
        print(f"\n  ** POTENTIAL HIT(S): {len(hits)} n!=m pair(s) exceed 0.15 frac_sig AND replicate across a majority:")
        for ds, key, rec in hits:
            print(f"     [{ds}] {key}: frac_sig={rec['frac_sig']:.2f}  "
                  f"replicate={rec['n_subj_replicate']}/{rec['n_subj']}")
    else:
        print("\n  => NO n!=m pair exceeds chance AND replicates across a majority of subjects.")
        print("     Genuine n:m cross-channel phase coupling is at the null in scalp EEG;")
        print("     only 1:1 (coherence) survives. Settles the n!=m question.")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

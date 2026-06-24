"""Study 30 — Real-data value-add probe: does the PC (n:m phase-coupling) axis surface structure
that band power is blind to, in real EEG?

Band power is phase-blind, so the framework's distinctive claim must be demonstrated on the PHASE
axis in a real recording. We probe CROSS-CHANNEL n:m phase coupling (PC_z, IAAFT-of-B null) between
motor channels (C3<->C4) and occipital channels (O1<->O2) at candidate low-integer ratios, across
the eegbci rest/motor and eyes-open/closed contrasts. For each (contrast, channel-pair, n:m) we ask:

  (1) IS THERE REAL COUPLING?  fraction of epochs with surrogate-significant PC_z (rank p<=0.05) --
      band power cannot measure a cross-channel phase relation at all, so any genuine, replicable
      n:m coupling here is a value-add by kind (substantiates the paper's motivating premise).
  (2) DOES IT BEAT BAND POWER?  AUC(PC_z) vs AUC(band power) for the state contrast.

Exploratory: surveys which n:m pairs carry real coupling before committing to a headline.
Run-level epochs (fast); promote to event-locked if a pair shows clean signal.

Outputs: results/study30_realdata_coupling.json
"""
from __future__ import annotations

import numpy as np

from resonance_paper import _common as C
from resonance_paper import datasets as D
from resonance_paper.study5_cross_signal import cross_target_z
from biotuner.resonance import ResonanceConfig

CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=45, noverlap=1,
                      coupling_metric="nm_plv_canonical", ratio_kernel="fraction",
                      ratio_kernel_params={"max_denom": 16, "beta": 1.0},
                      return_intermediates=True)
PAIRS = {"mu1:1": [(10.0, 10.0)], "mu-beta1:2": [(10.0, 20.0)], "beta1:1": [(20.0, 20.0)],
         "alpha-theta3:2": [(6.0, 9.0)], "alpha1:1": [(10.0, 10.0)]}


def _band_power(x, sf, lo, hi):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sf)
    return float(X[(f >= lo) & (f <= hi)].mean())


def _paired_epochs(subjects, runs, chans, epoch_s=8.0, max_ep=7):
    out = []
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
            data = raw.get_data(picks=present[:2])
            seg = int(epoch_s * sf); n_ep = min(max_ep, data.shape[1] // seg)
            for i in range(n_ep):
                a = data[0, i * seg:(i + 1) * seg]; b = data[1, i * seg:(i + 1) * seg]
                out.append(dict(subj=subj, cond=cond, sf=sf, A=a.astype(np.float64),
                                B=b.astype(np.float64)))
    return out


def _probe(name, epochs, pos, neg, pair_keys, n_surr):
    res = {}
    for pk in pair_keys:
        pairs = PAIRS[pk]
        recs = []
        for ep in epochs:
            z = cross_target_z(ep["A"], ep["B"], ep["sf"], CFG, pairs, "PC", n=n_surr, seed=hash((ep["subj"], ep["cond"], pk)) % (2**31))
            # crude rank-p via the same surrogate count isn't returned by cross_target_z; use z>1.645 as the flag
            recs.append(dict(cond=ep["cond"], z=z))
        pz = [r["z"] for r in recs if r["cond"] == pos and np.isfinite(r["z"])]
        nz = [r["z"] for r in recs if r["cond"] == neg and np.isfinite(r["z"])]
        allz = np.array([r["z"] for r in recs if np.isfinite(r["z"])])
        res[pk] = dict(n=len(allz), frac_sig=float(np.mean(allz > 1.645)) if len(allz) else float("nan"),
                       median_z=float(np.median(allz)) if len(allz) else float("nan"),
                       auc_pcz=C.bootstrap_auc_ci(pz, nz)["auc"] if pz and nz else float("nan"))
    return res


def run(quick=True):
    subjects = (1, 2, 3, 4, 5) if quick else tuple(range(1, 11))
    n_surr = 30 if quick else 60
    out = {}

    motor = _paired_epochs(subjects, (("REST", D.REST_RUN), ("MOTOR", D.MOTOR_RUN)), ["C3", "C4"])
    if motor:
        print(f"  motor C3-C4 epochs: {len(motor)}", flush=True)
        bp = {}
        for band, (lo, hi) in [("mu", (8, 12)), ("beta", (18, 22))]:
            p = [_band_power(e["A"], e["sf"], lo, hi) for e in motor if e["cond"] == "MOTOR"]
            n = [_band_power(e["A"], e["sf"], lo, hi) for e in motor if e["cond"] == "REST"]
            bp[band] = C.bootstrap_auc_ci(p, n)["auc"]
        out["motor"] = dict(pc=_probe("motor", motor, "MOTOR", "REST",
                                      ["mu1:1", "mu-beta1:2", "beta1:1"], n_surr),
                            band_power_auc=bp)

    occ = _paired_epochs(subjects, (("EO", D.EO_RUN), ("EC", D.EC_RUN)), ["O1", "O2"])
    if occ:
        print(f"  occipital O1-O2 epochs: {len(occ)}", flush=True)
        p = [_band_power(e["A"], e["sf"], 8, 12) for e in occ if e["cond"] == "EC"]
        n = [_band_power(e["A"], e["sf"], 8, 12) for e in occ if e["cond"] == "EO"]
        out["occipital"] = dict(pc=_probe("occ", occ, "EC", "EO",
                                          ["alpha1:1", "mu-beta1:2", "alpha-theta3:2"], n_surr),
                                band_power_auc={"alpha": C.bootstrap_auc_ci(p, n)["auc"]})

    C.save_json(dict(quick=quick, out=out), "study30_realdata_coupling.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 30 headline (real-data PC value-add probe) ---")
    for ds, blk in out.items():
        bp = blk.get("band_power_auc", {})
        print(f"  [{ds}] band-power AUC: " + "  ".join(f"{k}={v:.2f}" for k, v in bp.items()))
        for pk, r in blk["pc"].items():
            print(f"      PC_z {pk:14s}: frac_sig={r['frac_sig']:.2f}  median_z={r['median_z']:+.2f}  "
                  f"state-AUC={r['auc_pcz']:.2f}  (n={r['n']})")
    print("  => look for: a pair with high frac_sig (real coupling band power can't see) AND/OR")
    print("     state-AUC clearly above the band-power AUC (PC surfaces what power misses).")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

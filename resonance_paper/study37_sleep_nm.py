"""Study 37 — Proper-instrument n:m coupling in NREM sleep: positive-control / last value-add shot.

Scalp resting + event-locked motor showed genuine 1:1 cross-channel coupling but NO n!=m (studies
35/36/31). Sleep is the dataset where cross-frequency coupling is documented -- so it is the decisive
remaining test of whether the framework's n:m axis finds n!=m phase-phase coupling WHERE IT PLAUSIBLY
EXISTS, using the PROPER instrument (peak-based + Hilbert + volume-conduction-robust wpli).

Caveat stated up front: sleep's famous CFC is SO(~0.75 Hz)-spindle(~13 Hz) PHASE-AMPLITUDE coupling at
~1:17 -- a different phenomenon at an out-of-range ratio from the low-integer (n,m<=3) phase-PHASE
coupling this framework measures. So a null here would COMPLETE a clean exhaustive negative (low-integer
n!=m phase-phase coupling is genuinely rare in macroscopic EEG); a positive would be a real value-add.

We pool N2+N3 epochs, take the long-range frontal (Fpz-Cz) <-> posterior (Pz-Oz) bipolar pair, and run
the study36 n:m-composition test (genuine 1:1 vs genuine n!=m peak pairs, wpli, IAAFT-of-B rank-p).

Outputs: results/study37_sleep_nm.json
"""
from __future__ import annotations

import warnings

import numpy as np

from resonance_paper import _common as C
from resonance_paper.study14_sleep import load_sleep
from resonance_paper.study36_nm_composition import _probe

warnings.filterwarnings("ignore")


def run(quick=True):
    n_sub = 3 if quick else 6
    n_surr = 24 if quick else 49
    items = load_sleep(n_subjects=n_sub, max_epochs_per_stage=8)
    states = {}
    for it in items:
        states[it["state"]] = states.get(it["state"], 0) + 1
    print(f"  sleep epochs by stage: {states}", flush=True)

    nrem = [it for it in items if it.get("state") in ("N2", "N3") and np.asarray(it["X"]).shape[0] >= 2]
    epochs = [dict(A=np.asarray(it["X"])[0], B=np.asarray(it["X"])[1], sf=it["sf"],
                   subj=it["subject"], cond=it["state"]) for it in nrem]
    if not epochs:
        print("  no NREM 2-channel epochs loaded; aborting study 37.")
        return None
    print(f"  NREM (N2+N3) frontal<->posterior epochs: {len(epochs)} (proper instrument, wpli)", flush=True)
    out = {"sleep_nrem": _probe(epochs, n_surr)}
    C.save_json(dict(quick=quick, n_subjects=n_sub, n_epochs=len(epochs), out=out),
                "study37_sleep_nm.json")
    _headline(out)
    return out


def _headline(out):
    print("\n  --- Study 37 headline (NREM sleep, proper peak-based n:m instrument) ---")
    blk = out["sleep_nrem"]; nm = blk["nm"]; one = blk["one"]
    print(f"  1:1 wpli frac_sig={one['frac_sig']:.2f} | n!=m present in {nm['frac_epochs_with_pair']:.2f} "
          f"of epochs, wpli frac_sig={nm['frac_sig']:.2f}  top n!=m ratios={nm.get('top_ratios')}")
    print("  => n!=m frac_sig >> 0.05 even in sleep => the framework finds genuine n:m where it exists (value-add).")
    print("     n!=m at ~0.05 => low-integer n:m phase-phase coupling is rare even in sleep (exhaustive negative).")


if __name__ == "__main__":
    import sys
    run(quick="--paper" not in sys.argv)

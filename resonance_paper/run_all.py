"""Run the full resonance validation suite.

Usage
-----
    python -m resonance_paper.run_all            # quick pass (minutes)
    python -m resonance_paper.run_all --paper    # paper-grade (long; more
                                                 # seeds, more surrogates)

Each study writes results/<study>.json and figures/<study>_*.{png,pdf}, and
prints a headline summary. ``--only N`` runs a single study (1-4).
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="paper-grade pass (slow)")
    ap.add_argument("--only", type=str, default="", help="run only study N (e.g. 5, 16, 20, 20b)")
    args = ap.parse_args()
    quick = not args.paper

    from resonance_paper import (
        study1_ground_truth, study2_eeg_states,
        study3_cross_modality, study4_strategy_comparison,
        study5_cross_signal, study6_resonance_conjunction,
        study8_arnold_tongues, study9_reservoir,
        study10_criticality, study11_reservoir_criticality,
        study12_ei_network, study13_anesthesia, study14_sleep,
        study15_deep_anesthesia, study16_criticality_indepth,
        study17_tripartite_dissociation, study18_stochastic_resonance,
        study19_ssvep_intermod, study20_musical_intermod,
        study20b_ffr_consonance,
    )
    studies = {
        1: ("Ground-truth recovery", study1_ground_truth.run),
        2: ("EEG state discrimination", study2_eeg_states.run),
        3: ("Cross-modality generality", study3_cross_modality.run),
        4: ("Strategy comparison", study4_strategy_comparison.run),
        5: ("Cross-signal coupling recovery", study5_cross_signal.run),
        6: ("Polyrhythm recovery", study6_resonance_conjunction.run),
        8: ("Arnold tongues / harmonic complexity", study8_arnold_tongues.run),
        9: ("Reservoir: resonance vs computation", study9_reservoir.run),
        10: ("Criticality (branching network)", study10_criticality.run),
        11: ("Reservoir: edge of chaos", study11_reservoir_criticality.run),
        12: ("E/I network: edge of synchronization", study12_ei_network.run),
        13: ("Real data: propofol sedation (H vs depth/criticality)", study13_anesthesia.run),
        14: ("Real data: sleep wake->N3 (H vs depth/criticality)", study14_sleep.run),
        15: ("Real data: deep anesthesia / LOC (H vs depth/criticality)", study15_deep_anesthesia.run),
        16: ("Real data: criticality in-depth (H/R observables, controls)",
             lambda quick=True: study16_criticality_indepth.run(dataset="sleep", quick=quick)),
        17: ("Tripartite dissociation (H ⟂ PC; R specificity)", study17_tripartite_dissociation.run),
        18: ("Stochastic resonance across ratios", study18_stochastic_resonance.run),
        19: ("SSVEP harmonics + intermodulation", study19_ssvep_intermod.run),
        20: ("Musical chords / consonance via H", study20_musical_intermod.run),
        # 20b real-data FFR companion (downloads OSF 5puhb on first run; --only 20b)
        "20b": ("Real EEG: FFR consonance harmonicity (Study 20b)", study20b_ffr_consonance.run),
    }
    # Study 7 omitted from the default sweep (Direction B confounded; see file).
    # 13-16 + 20b download real EEG (Chennu/Sleep-EDF/ds004541/OSF-5puhb) on first run;
    # 16 and 20b are investigations/companions (kept out of the default sweep, run via --only).
    def _key(s):
        return int(s) if s.isdigit() else s
    todo = [_key(args.only)] if args.only else [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,
                                                17, 18, 19, 20]
    print(f"=== Resonance validation suite ({'PAPER' if args.paper else 'QUICK'} mode) ===\n")
    for n in todo:
        title, fn = studies[n]
        print(f"\n########## Study {n}: {title} ##########")
        t0 = time.time()
        fn(quick=quick)
        print(f"  [study {n} finished in {time.time()-t0:.0f}s]")
    print("\n=== suite complete; see resonance_paper/results and resonance_paper/figures ===")


if __name__ == "__main__":
    main()

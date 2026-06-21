# resonance_paper — validation suite for the biotuner resonance module

A self-contained analysis suite that validates the
`biotuner.resonance` framework for a methods paper. It exercises the toolbox
capacities — **single-signal and cross-signal H × PC × R spectra + complexity
metrics** and **strategy-registry comparison** — across twelve studies spanning
synthetic ground truth, real biosignals, and generative dynamical systems.

> The top-level repository [`README.md`](../README.md) is the authoritative
> overview; this file documents the package internals.

## Studies

| # | File | Question | Data |
|---|------|----------|------|
| 1 | `study1_ground_truth.py` | Does resonance recover known structure? | synthetic |
| 2 | `study2_eeg_states.py` | Do resonance features separate brain states? | PhysioNet eegbci (eyes open/closed) |
| 3 | `study3_cross_modality.py` | Do features fingerprint signal modality? | ECG (neurokit2) + EEG + synthetic |
| 4 | `study4_strategy_comparison.py` | Which strategy for which goal? | synthetic ground truth |
| 5 | `study5_cross_signal.py` | Cross-signal coupling recovery (1:1/1:2/2:3) | synthetic pairs |
| 6 | `study6_resonance_conjunction.py` | Polyrhythm recovery (Part A = decomposition illustration) | synthetic |
| 7 | `study7_coupled_oscillators.py` | Coupled Van der Pol (Dir A sound; Dir B confounded — see caveat) | generative |
| 8 | `study8_arnold_tongues.py` | Harmonic complexity governs lockability (devil's staircase) | forced Van der Pol |
| 9 | `study9_reservoir.py` | Resonance vs reservoir computation (memory) | echo-state network |
| 10 | `study10_criticality.py` | Resonance vs criticality (avalanches; H peaks at σ=1) | branching network |
| 11 | `study11_reservoir_criticality.py` | Resonance vs the edge of chaos | echo-state network |
| 12 | `study12_ei_network.py` | Resonance vs the edge of synchronization | Wilson-Cowan E/I |

See `paper/resonance_and_criticality.md` for the Studies 9–11 synthesis
(reservoir computing + criticality).

Each writes `results/<study>.json` and `figures/<study>_*.{png,pdf}` and prints
a headline summary.

## Running

```bash
# quick pass (minutes) — for iteration
python -m resonance_paper.run_all

# paper-grade pass (long: more seeds, 150-200 surrogates)
python -m resonance_paper.run_all --paper

# a single study
python -m resonance_paper.run_all --only 1
python -m resonance_paper.study2_eeg_states            # quick
python -m resonance_paper.study2_eeg_states --paper    # full
```

Surrogate loops are parallelized with joblib. EEG data auto-downloads via
`mne.datasets.eegbci` (PhysioNet) and is cached under `~/mne_data`.

## Core engine (`_common.py`)

- `default_config()` / `legacy_config()` — recommended vs paper-reproduction presets
- `factor_surrogate_z()` — **per-frequency surrogate z for each factor (H, PC, R)**.
  The shipped `with_surrogate_null` only z-scores R; the validation shows R is
  H-dominated (PSD-driven) and blind to phase coupling under a PSD-preserving
  null, so the suite z-scores all three factors against the same surrogate
  ensemble.
- `resonance_features()` — flatten a `ResonanceResult` into scalar features
- `roc_auc()` — rank-based detection AUC
- `strategy_grid()` — cartesian product over kernels × ratio kernels × metrics

## Key findings (see `paper/resonance_paper_draft.md`)

1. **Harmonic-structure recovery is strong.** Harmonicity tracks known harmonic
   richness (Spearman ≈ 0.98); harmonic-vs-inharmonic and tone-vs-noise
   separation is near-perfect.
2. **n:m phase-coupling detection succeeds (post phase-alignment fix).** Read
   against a PSD-preserving (AAFT) null, single-signal `PC_z` detects coupling at
   AUC ≈ 0.99 and the matrix entry Φ[f1,f2] at AUC = 1.0; cross-signal 1:1/1:2/2:3
   locks all detect at AUC = 1.0. Raw reduced PC is PSD-weighted and small, so
   coupling must be read as a surrogate z-score, not in absolute terms. (Before
   the fix these sat near chance — a bug artifact, not a property of the method.)
3. **EEG states separate cleanly.** Eyes-open vs eyes-closed decodes at high AUC
   from resonance features; eyes-closed shows higher harmonicity and lower
   spectral entropy.
4. **Modalities are fingerprinted multivariately.** The resonance/complexity
   feature vector classifies ECG / EEG / harmonic / pink with high accuracy,
   though no single harmonicity scalar gives a clean ranking (mean harmonicity
   is noise-confounded — peak-based summaries are preferred).

## Toolbox issues surfaced (issues 1 & 2 now fixed in core)

- **[FIXED]** `with_surrogate_null(surr_type='IAAFT')` default was non-functional
  (the single-signal generator implemented only
  `AAFT/TFT/phase/shuffle/white/pink/brown/blue`). IAAFT, `phase_randomize`, and
  `time_shuffle` are now wired into the single-signal path, so the documented
  default works.
- **[FIXED]** `with_surrogate_null` now returns **factor-level** z-scores —
  `result.factor_z["H" | "PC" | "R"]` (plus `factor_surrogate_mean/std`) — so
  phase-coupling inference can use `factor_z["PC"]` directly. (`_common.py` still
  provides `factor_surrogate_z` for arbitrary surrogate types in the sweeps.)
- **Harmonicity is confounded by the aperiodic (1/f) component.** On *pure*
  colored noise with no oscillations, both `H_avg` and `H_max` rise
  monotonically with the 1/f slope (e.g. H_avg 0.28→0.43, H_max 1.8→7.7 as
  β goes 0.5→2.0). The fix is `ResonanceConfig(remove_aperiodic=True)` (FOOOF-
  style aperiodic removal), which flattens the spurious harmonicity of noise
  (H_avg ≈ 0.19 across all slopes). Comparisons across conditions that differ
  in 1/f activity (age, arousal, anesthesia, task/rest, eyes open/closed) MUST
  remove the aperiodic component or match it across conditions — otherwise a
  harmonicity difference may simply reflect a 1/f difference. Peak summaries
  help only once backgrounds are matched/removed; they are not a substitute for
  aperiodic removal.

## Layout

```
resonance_paper/
  _common.py        signals.py        datasets.py
  study1_..4_*.py   run_all.py
  results/  figures/  paper/resonance_paper_draft.md
```

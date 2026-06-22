# biological-resonance

**Quantifying resonance in biological systems.**

This repository is the validation suite and methods-paper material for a new way
of characterizing biosignals: a *tripartite resonance representation* built on
top of the [`biotuner`](https://github.com/AntoineBellemare/biotuner) library.
Every signal is described by three frequency-resolved spectra,

| spectrum | symbol | what it measures |
|----------|--------|------------------|
| **harmonicity**    | `H(f)`  | how harmonically organized the spectrum is around `f` |
| **phase coupling** | `PC(f)` | n:m phase locking involving `f` |
| **resonance**      | `R(f) = H(f) × PC(f)` | where harmonic structure *and* phase coupling coincide |

plus complexity summaries, surrogate-normalized inference, and a
strategy-registry so kernels / coupling metrics can be swapped and compared. The
same machinery extends to **cross-signal** resonance (signal ↔ signal, channel ↔
channel) via `biotuner.harmonic_connectivity`.

The suite tests this representation against synthetic ground truth, real
biosignals (EEG, ECG, PPG, respiration), and several generative dynamical systems
(forced/coupled oscillators, echo-state reservoirs, branching networks,
Wilson–Cowan E/I networks).

---

## Headline results

All numbers below are produced by the paper-grade pass and stored in
[`resonance_paper/results/`](resonance_paper/results); the figures are
regenerated from those JSONs by
[`resonance_paper/paper/make_paper_figures.py`](resonance_paper/paper/make_paper_figures.py).

1. **Harmonic-structure recovery is essentially perfect.** Mean harmonicity `H`
   rank-orders a ladder of synthetic signals by harmonic richness (Spearman
   ρ = 0.98); harmonic-vs-inharmonic and tone-vs-noise separation are at
   AUC = 1.0.
2. **Phase-coupling detection succeeds** once the phase rows are aligned to the
   analysis frequencies and read against a PSD-preserving (AAFT) null:
   single-signal `PC_z` AUC ≈ 0.99 and the matrix entry Φ[f₁,f₂] AUC = 1.0;
   cross-signal n:m locks (1:1 / 1:2 / 2:3) detect at AUC = 1.0 each (vs a
   single-signal baseline of 0.59). Raw reduced `PC` is small and PSD-weighted —
   coupling must be read as a surrogate z-score, not in absolute terms.
3. **Polyrhythmic structure is recovered** — the 2:3:4 resonance peak/median
   ratio separates locked from scrambled stacks at AUC = 1.0.
4. **Resonance discriminates the eyes-open/closed state at AUC ≈ 0.81**
   (leave-one-subject-out, p = 0.001), matching a relative-alpha-power baseline,
   with eyes-closed more harmonically organized (after aperiodic removal).
   *Scope note:* on motor ERD — a single-band *amplitude* phenomenon — a fair
   event-locked test (movement vs rest within-run) finds band power wins (≈0.59)
   and resonance is at chance (≈0.49). This correctly bounds the method: it adds
   value for harmonic / cross-frequency-coupling structure, **not** band-amplitude
   changes. (The motor contrast in the committed `study2` is confounded —
   baseline-run vs task-run — and is being replaced by the event-locked design.)
5. **Harmonicity is 1/f-confounded — and the suite controls for it.** Peak
   harmonicity discriminates eyes-open/closed at AUC = 0.83 *with* FOOOF-style
   aperiodic removal but collapses to 0.29 *without* it. Cross-condition
   comparisons **must** use `remove_aperiodic=True`.
6. **The feature vector fingerprints modality.** A 7-way classifier over
   resonance + complexity features separates ECG / EEG / PPG / RSP / harmonic /
   noise at accuracy 0.99 (chance 0.14, permutation p = 0.005).
7. **Harmonic simplicity governs lockability.** For a forced Van der Pol
   oscillator, Arnold-tongue width vs ratio complexity correlates ρ = −0.50, and
   the framework's harmonicity tracks tongue width at ρ = +0.41.
8. **Resonance is a signature of criticality, across three independent systems**
   (paper-grade, 12–20 seed replicates + bootstrap CIs). (a) Harmonic structure
   `H` peaks at the critical branching ratio (σ = 1.00 [0.97, 1.08], co-located
   with the susceptibility peak). (b) A reservoir driven by **white noise**
   *generates* harmonic structure specifically at the **edge of chaos** — H rises
   above the noise baseline only from ρ ≈ 1.3, peaking at ρ_c = 1.57 [1.44, 1.68]
   (and distinct from generic spectral peakedness, which keeps rising into chaos,
   and from its memory-capacity optimum at ρ ≈ 0.70). (c) Cross E↔I phase coupling
   **rises** from a non-zero asynchronous baseline (PLV ≈ 0.43 → 0.99, ΔPC ≈ +0.55)
   at the synchronization onset (g_c = 1.0) — it does not switch on from zero.

## Figures

![Figure 1 — ground truth](resonance_paper/paper/figures/Fig1_ground_truth.png)

![Figure 2 — real biosignals](resonance_paper/paper/figures/Fig2_real_biosignals.png)

![Figure 3 — Arnold tongues](resonance_paper/paper/figures/Fig3_arnold_tongues.png)

![Figure 4 — criticality](resonance_paper/paper/figures/Fig4_criticality.png)

---

## Installation

```bash
git clone https://github.com/AntoineBellemare/biological-resonance.git
cd biological-resonance
pip install -r requirements.txt
```

> **Note on the biotuner dependency.** The resonance fixes this suite relies on
> (phase-row alignment to the analysis grid; factor-level surrogate z-scores)
> landed on biotuner `main` *after* the v0.4.1 PyPI release, and the version was
> not bumped. `pip install biotuner` therefore installs a build that **does not
> reproduce these results**. `requirements.txt` pins biotuner to `main`
> (`git+https://github.com/AntoineBellemare/biotuner.git@main`). Once a biotuner
> release (≥ 0.4.2) ships these fixes, this can be relaxed to a PyPI pin.

EEG data auto-downloads via `mne.datasets.eegbci` (PhysioNet) on first run and is
cached under `~/mne_data`. ECG/PPG/RSP are simulated with `neurokit2`.

## Reproducing the results

```bash
# quick pass (minutes) — for iteration
python -m resonance_paper.run_all

# paper-grade pass (long: more seeds, 100-200 surrogates) — regenerates results/
python -m resonance_paper.run_all --paper

# a single study
python -m resonance_paper.run_all --only 1
python -m resonance_paper.study2_eeg_states --paper

# rebuild the composite paper figures from the stored result JSONs (fast)
python -m resonance_paper.paper.make_paper_figures
```

Surrogate loops are parallelized with `joblib`. Run all commands from the
repository root (the studies form the importable `resonance_paper` package).

---

## Study index

| # | File | Question | Data | Result |
|---|------|----------|------|--------|
| 1 | [study1_ground_truth.py](resonance_paper/study1_ground_truth.py) | Recover known harmonic structure & n:m coupling? | synthetic | H ranks richness ρ=0.98; coupling PC_z AUC 0.99, matrix 1.0 |
| 2 | [study2_eeg_states.py](resonance_paper/study2_eeg_states.py) | Separate brain states? | PhysioNet eegbci | EO/EC AUC 0.81 (= alpha-power baseline, p=0.001); motor ERD is band-power's domain (resonance ≈ chance — see scope note) |
| 3 | [study3_cross_modality.py](resonance_paper/study3_cross_modality.py) | Fingerprint signal modality? | EEG+ECG+PPG+RSP+synthetic | 7-way acc 0.99 (chance 0.14) |
| 4 | [study4_strategy_comparison.py](resonance_paper/study4_strategy_comparison.py) | Which strategy for which goal? | synthetic | 20 strategies scored; `harmsim·binary·nm_plv` AUC 1.0 |
| 5 | [study5_cross_signal.py](resonance_paper/study5_cross_signal.py) | Cross-signal coupling recovery (1:1/1:2/2:3)? | synthetic pairs | AUC 1.0 each vs 0.59 single-signal baseline |
| 6 | [study6_resonance_conjunction.py](resonance_paper/study6_resonance_conjunction.py) | Polyrhythm (2:3:4) recovery? | synthetic | AUC 1.0 |
| 7 | [study7_coupled_oscillators.py](resonance_paper/study7_coupled_oscillators.py) | Coupled Van der Pol H/PC/R co-variation | generative | Direction A sound; **Direction B confounded** — excluded from the sweep (see file caveat) |
| 8 | [study8_arnold_tongues.py](resonance_paper/study8_arnold_tongues.py) | Does harmonic complexity govern lockability? | forced Van der Pol | width~complexity ρ=−0.50, width~harmonicity ρ=+0.41 |
| 9 | [study9_reservoir.py](resonance_paper/study9_reservoir.py) | Resonance vs reservoir memory? | echo-state network | R vs memory ρ=−0.32 (honest null) |
| 10 | [study10_criticality.py](resonance_paper/study10_criticality.py) | Resonance vs criticality (avalanches)? | branching network | H peaks at σ=1.00 [0.97,1.08]; R≈0 (avalanches non-oscillatory) |
| 11 | [study11_reservoir_criticality.py](resonance_paper/study11_reservoir_criticality.py) | Does criticality GENERATE resonance (noise-driven)? | echo-state network | noise→harmonic structure generated at the edge of chaos (onset ρ≈1.3, peak ρ_c=1.57) |
| 12 | [study12_ei_network.py](resonance_paper/study12_ei_network.py) | Resonance vs the edge of synchronization? | Wilson–Cowan E/I | E↔I phase coupling rises (PLV 0.43→0.99) at g_c=1.0, not from 0 |
| 13 | [study13_anesthesia.py](resonance_paper/study13_anesthesia.py) | Real data: does H track depth/criticality? | Chennu propofol (10 subj) | H tracks state but ≈ band power; mild sedation = weak contrast |
| 14 | [study14_sleep.py](resonance_paper/study14_sleep.py) | Real data: H across sleep stages? | Sleep-EDF (8 subj) | H highest in **N3** (slow-wave harmonics); decode < band power; ρ(H,m̂-prox)=−0.24 (reversal) |
| 15 | [study15_deep_anesthesia.py](resonance_paper/study15_deep_anesthesia.py) | Real data: H across loss of consciousness? | ds004541 deep GA (7 subj) | wake vs LOC decodable (all-features AUC 0.82); resonance ≈ band power; markers disagree on criticality sign |
| 16 | [study16_criticality_indepth.py](resonance_paper/study16_criticality_indepth.py) | Why does the in-silico criticality prediction reverse in vivo? | Sleep-EDF + ds004541 | raw-EEG H reversal is a slow-wave artifact; on scale-free population activity + within-state, H tracks criticality (sleep ρ=+0.11, p=0.03) — recovers Study 10 (run via `--only 16`) |
| 17 | [study17_tripartite_dissociation.py](resonance_paper/study17_tripartite_dissociation.py) | Are H/PC/R separable; does R beat its factors? | synthetic grid | **H ⟂ phase (ρ=0.00)**; PC not ratio-blind (−0.47, mode-locking); R AUC 0.66 < PC 0.78 — R is *interpretive decomposition*, not a better detector |
| 18 | [study18_stochastic_resonance.py](resonance_paper/study18_stochastic_resonance.py) | Can noise induce resonance at complex ratios? | forced Van der Pol | noise-induced H gain ~7× larger for complex vs simple ratios (+0.024 vs +0.004) |
| 19 | [study19_ssvep_intermod.py](resonance_paper/study19_ssvep_intermod.py) | SSVEP harmonics + intermodulation under nonlinearity? | synthetic | two-flicker intermodulation rises with nonlinearity (ρ=+0.61), richer for simple ratios (0.35 vs 0.13); single-flicker H at ceiling |
| 20 | [study20_musical_intermod.py](resonance_paper/study20_musical_intermod.py) | Does H track musical consonance? | synthetic chords | ρ(H, chord complexity) = −0.73; auditory nonlinearity (combination tones) *sharpens* the recovery |
| 20b | [study20b_ffr_consonance.py](resonance_paper/study20b_ffr_consonance.py) | Real EEG: is the brainstem FFR more harmonic for consonant dyads? | FFR to dyads (OSF 5puhb, 36 subj) | **consonant>dissonant H** (CC>DC p=1.3e-8; **CI>DI missing-fundamental p=4e-5**); effect lives in the stimulus-**silent** band (leakage-null p=0.27), FFR reconstructs each dyad's own fundamental (240/225 Hz) — **neural, adversarially verified** (run via `--only 20b`) |
| 21 | [study21_connectivity.py](resonance_paper/study21_connectivity.py) | Multichannel: does cross-resonance *connectivity* recover a planted coupled cluster? | synthetic 8-ch network | R/PC/H connectivity matrices recover the cluster (R AUC≈0.76; IAAFT z-score sharpens it) — validates the n_elec×n_elec layer |
| 22 | [study22_spectral_descriptors.py](resonance_paper/study22_spectral_descriptors.py) | Are complexity descriptors of the H/PC/R spectra informative? | synthetic (known structure) | flatness/entropy/spread/HFD of the resonance spectra separate harmonic from noise (H_flatness AUC 1.0) **where the scalar H_avg conflates them** — descriptors are first-class |

**The construct & its dynamical regimes (Studies 17–20)** (see
[`paper/construct_and_dynamics.md`](resonance_paper/paper/construct_and_dynamics.md)):
Study 17 is the non-circular capstone — H is *exactly* phase-blind (ρ=0.00), so
harmonicity and phase-coupling are distinct measurements; but mode-locking ties PC
to ratio simplicity, so R = H·PC is best read as an **interpretive decomposition**
(why is/isn't this a resonance?) rather than a detector that beats PC. Studies 18–20
show resonance is *generated* across three regimes: by noise at complex ratios
(stochastic resonance), by nonlinear intermodulation (SSVEP/frequency-tagging), and
tracking **auditory consonance** (sharpened by combination tones). Study **20b**
then confirms it in **real EEG**: the human brainstem FFR is more harmonically
organized for consonant than dissonant dyads — even for *missing-fundamental* dyads,
where the brain reconstructs the harmonic relations (the effect lives in the
stimulus-silent band, is frequency-specific to each dyad's own fundamental, and
survives SNR/leakage/mains/permutation controls — adversarially verified).

**Real-data criticality (Studies 13–16) — honest summary** (see
[`paper/realdata_criticality.md`](resonance_paper/paper/realdata_criticality.md)):
H/R discriminate brain state above chance but **do not beat band power**. The
in-silico "H peaks at criticality" (Study 10) at first appeared to **reverse in
vivo** — but Study 16 shows that was a **measurement artifact**: raw-EEG H is
dominated by the slow-oscillation harmonic series (test B1: the effect lives
entirely in the <2 Hz band, survives controlling for slow-wave power). Recomputing
resonance on the **scale-free population-activity signal** (the in-vivo analog of
what Study 10 actually used) and controlling between-state oscillation confounds
(within-state), in-vivo **H positively tracks criticality-proximity** (sleep
ρ=+0.11, p=0.03), **recovering the model prediction** — and it is H, not R, exactly
as in silico. Criticality axis = validated branching ratio m̂ (DCC was tried but
failed ground-truth validation and is not used).

Study 7 is retained for the record but **excluded from the default sweep**: its
Direction B is confounded (natural frequencies set at exact rational ratios, so
the n:m phase combination cancels deterministically regardless of coupling). A
correct Arnold-tongue test with detuned frequencies and real coupling lives in
Study 8.

## Methods notes

- **Factor-level surrogate inference.** The suite z-scores each factor (`H`,
  `PC`, `R`) per frequency against the *same* PSD-preserving surrogate ensemble
  (`resonance_paper/_common.py::factor_surrogate_z`). This is what makes
  phase-coupling detection work: raw reduced `PC` is PSD-weighted and small, but
  `PC_z` cleanly separates locked from unlocked signals.
- **Aperiodic removal is mandatory across conditions.** On pure colored noise,
  harmonicity rises monotonically with the 1/f slope. Use
  `ResonanceConfig(remove_aperiodic=True)` whenever conditions differ in
  background 1/f (arousal, task/rest, eyes open/closed, age, anesthesia).

## Repository layout

```
biological-resonance/
├── README.md  requirements.txt  .gitignore
└── resonance_paper/                  # the importable analysis package
    ├── _common.py                    # config presets, surrogate-z, AUC/stats, plotting
    ├── signals.py  datasets.py       # synthetic generators / real-data loaders
    ├── study1..22_*.py  run_all.py   # the studies + driver
    ├── criticality.py                # validated DFA/LRTC/branching-ratio estimators
    ├── crit_resonance.py             # shared real-data criticality analysis
    ├── results/   *.json             # paper-grade headline metrics (committed)
    ├── figures/   study*_*.{png,pdf} # per-study diagnostic figures
    └── paper/
        ├── resonance_paper_draft.md          # main manuscript draft
        ├── resonance_and_criticality.md      # studies 9–12 synthesis (in-silico)
        ├── realdata_criticality.md           # studies 13–16 synthesis (real data)
        ├── construct_and_dynamics.md         # studies 17–20 synthesis
        ├── make_method_figures.py            # METHOD paper: method_Fig1–6 (M-A spine)
        ├── make_paper_figures.py             # legacy composite Fig 1–4 (criticality paper seed)
        └── figures/   method_Fig1..6, Fig1..4_*.{png,pdf}  # publication composites (600 DPI)
```

The package is named `resonance_paper` because it *is* the paper's reproducible
validation suite; the manuscript and stored result JSONs reference that path.

## Relationship to biotuner

The resonance engine itself (kernels, phase estimators, coupling metrics,
orchestration, surrogate nulls, cross-signal connectivity) lives in the
[`biotuner`](https://github.com/AntoineBellemare/biotuner) library under
`biotuner.resonance` and `biotuner.harmonic_connectivity`. This repository holds
only the validation/paper work that depends on it.

## License

See the upstream [`biotuner`](https://github.com/AntoineBellemare/biotuner)
project; a license file will be added here to match.

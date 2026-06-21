# Harmonic Resonance Spectra of Biosignals: A Validated Single-Signal Framework

*Draft — biotuner resonance module validation. Headline numbers are from the
paper-grade (`--paper`) pass (10 subjects; 100–200 surrogates); a few secondary
descriptives are flagged where they still come from the lighter moderate pass.
Regenerate with `python -m resonance_paper.run_all --paper`.*

---

## Abstract

We introduce and validate a single-signal **harmonic resonance** framework that
decomposes a time series into a per-frequency harmonicity spectrum **H(f)**, a
phase-coupling spectrum **PC(f)**, and their product, a resonance spectrum
**R(f) = H(f) · PC(f)**, together with a panel of spectral-complexity
descriptors. The framework is organized as a strategy registry (swappable
harmonic kernels, ratio kernels, phase estimators, coupling metrics, and combine
rules), enabling systematic method comparison. Using synthetic signals with
known structure and real biosignals (PhysioNet scalp EEG; synthetic ECG with a
known heart rate), we establish four results. **(1)** Harmonicity recovers known
harmonic structure with high fidelity: it rank-orders signals by harmonic
richness (Spearman ρ = 0.98) and separates harmonic from inharmonic content and
oscillation from noise at AUC = 1.0. **(2)** With a frequency-alignment fix to the
phase pipeline (the phase rows had been offset by ``fmin``), surrogate-normalized
phase-coupling detection recovers known n:m coupling at near-perfect AUC — single
signal PC_z AUC = 0.99, the phase-coupling matrix entry AUC = 1.0, and
cross-signal coupling AUC = 1.0 for 1:1/1:2/2:3 locks. (Before the fix these sat
near chance, ≈0.5–0.6 — an artifact, not a limitation of the construct.) The raw
reduced PC value is small and PSD-weighted, so it must be read against a surrogate
null, not in absolute terms.
**(3)** Resonance and complexity features discriminate brain states: eyes-open
vs eyes-closed EEG decodes at AUC = 0.81 (leave-one-subject-out, 10 subjects),
matching a relative-alpha-power baseline, with eyes-closed showing higher
harmonicity and lower spectral entropy — though the harmonicity difference holds
only after aperiodic (1/f) removal (peak-harmonicity AUC 0.83 with removal vs
0.29 without). **(4)** The feature vector fingerprints signal modality across
seven classes (real and synthetic ECG, EEG, PPG, respiration, a harmonic stack,
and pink noise) at 99% accuracy (chance 14%). We give practical guidance on
strategy selection and surface two implementation issues in the reference
toolbox.

---

## 1. Introduction

Biological time series are rich in *harmonic* structure: the QRS complex of the
ECG generates a dense integer-harmonic series; cortical rhythms exhibit
cross-frequency relationships and harmonic overtones; and many physiological
oscillators are weakly coupled across frequency. A unifying way to quantify "how
harmonically organized" a signal is — and at which frequencies that organization
concentrates — would support comparison across states and across modalities with
a single, interpretable descriptor.

We formalize this as a **resonance spectrum**. For a signal with power spectrum
`p(f)`, a harmonic kernel `S[i,j]` scores the musical/harmonic simplicity of each
frequency pair, and a phase-coupling kernel `Φ[i,j]` scores the n:m phase
consistency of each pair. PSD-weighted reductions give two per-frequency factors,

```
H(f_i)  = p(f_i) · Σ_j S[i,j] · p(f_j)        (harmonicity)
PC(f_i) = p(f_i) · Σ_j W[i,j] Φ[i,j] · p(f_j)  (phase coupling)
```

and their product `R(f) = H(f)·PC(f)` is the resonance spectrum. Each spectrum is
summarized by complexity metrics (mean, max, spectral flatness, spectral entropy,
spectral spread, Higuchi fractal dimension, and peak-harmonic-similarity).

The implementation is a **strategy registry**: harmonic kernels
(`harmsim`, `subharm_tension`), ratio kernels (`binary`, `fraction`), phase
estimators (`stft`), coupling metrics (`nm_plv`, `nm_plv_canonical`, `nm_pli`,
`nm_wpli`, `nm_rrci`, `nm_wpli_complex`), and combine rules (`product`,
`geomean`, `harmmean`, `min`, `weighted_log`). This makes the *choice of method*
an explicit, testable variable.

This paper validates the framework along four axes — ground-truth recovery, EEG
state discrimination, cross-modality generality, and strategy comparison — being
explicit about both what it does well and where it does not.

## 2. Methods

### 2.1 Resonance computation
All analyses use the single-signal `compute_resonance` pipeline (STFT phase
estimator; per-frequency factor reduction; product combine) over a 2–45 Hz band
at 0.5 Hz resolution unless noted (0.5–45 Hz for ECG, whose ~1 Hz fundamental
lies below 2 Hz). The recommended configuration pairs the `harmsim` harmonic
kernel and the `fraction` ratio kernel with the convention-correct
`nm_plv_canonical` coupling metric.

### 2.2 Surrogate inference (factor-level)
For statistical inference we compare an observed spectrum to an ensemble of
surrogates and form a per-frequency z-score. Crucially, we z-score **each factor
(H, PC, R) separately** against the same surrogate ensemble, rather than only the
resonance spectrum, because (Results 3.1) R is H-dominated and therefore blind to
phase coupling under a power-preserving null. Unless noted we use **AAFT**
(amplitude-adjusted Fourier transform) surrogates, which preserve the power
spectrum and amplitude distribution while randomizing phase — the appropriate
null for "is there phase structure beyond the spectral shape?". A `shuffle`
(time-permutation) null, which destroys the power spectrum, is used only to test
for the presence of *any* temporal/oscillatory structure.

### 2.3 Synthetic signals (ground truth)
- **Harmonic ladder**: pure tone → dyad → triad → 5-partial stack (increasing,
  known harmonic richness), an **inharmonic** pair (incommensurate partials), and
  **pink noise**, all on a 1/f background at matched SNR.
- **Non-stationary coupling**: a fundamental whose phase performs a random walk,
  with a harmonic partner that either *follows* it (locked = genuine n:m mode
  lock) or drifts independently (unlocked = identical power spectrum, no
  coupling). Low per-sample diffusion keeps the oscillation coherent within an
  STFT window while the phase wanders across the recording — the regime in which
  a power-preserving null can, in principle, expose coupling.

### 2.4 Real biosignals
- **EEG**: PhysioNet EEG Motor Movement/Imagery database (`eegbci`), baseline
  run 1 (eyes open) vs run 2 (eyes closed), occipital channels O1/O2/Oz, 8 s
  epochs, 1–45 Hz band-pass.
- **ECG**: neurokit2 synthesis at a known heart rate (60 ± 6 bpm), giving a
  ground-truth harmonic series.

### 2.5 Evaluation
Detection/discrimination is quantified by the rank-based area under the ROC curve
(AUC; 0.5 = chance). EEG decoding uses leave-one-subject-out logistic regression;
modality classification uses stratified-CV random forests. Code:
`resonance_paper/` in the biotuner repository; one command reproduces every
number and figure.

## 3. Results

### 3.1 Harmonic-structure recovery and phase-coupling detection both succeed
*(Study 1; Fig. 1)*

**Harmonic structure (Fig. 1A).** Mean harmonicity increased monotonically with
known harmonic richness across the ladder (Spearman ρ(richness, H_avg) = **0.98**;
ρ for R_avg = 0.92). Harmonic signals were perfectly separated from an
inharmonic pair of matched partial count (triad vs inharmonic, AUC = **1.00**)
and a single tone was perfectly separated from pink noise (AUC = **1.00**). The
harmonicity factor thus recovers the property it is designed to measure.

**Targeted n:m coupling (Fig. 1B).** On non-stationary locked-vs-unlocked signals
with identical power spectra, surrogate-normalized detection against an AAFT null
recovers the coupling: single-signal **PC_z AUC = 0.99** [0.98, 1.0], and the
phase-coupling **matrix entry AUC = 1.0** (25 vs 25 seeds, 150 surrogates;
locked PC_z 3.1 vs unlocked −1.0). Cross-signal coupling reaches AUC = **1.0**
for 1:1, 1:2 and 2:3 locks (Study 5). *Note:* these numbers required a
frequency-alignment fix to the phase pipeline — the STFT phase rows had been
indexed against the fmin-clipped frequency grid, offsetting every phase by
``fmin``; before the fix detection sat near chance (≈0.5–0.6), which was a bug
artifact, not a property of the construct. The raw reduced PC value is small and
PSD-weighted, so coupling must be read against a **surrogate null** (PC_z), not in
absolute terms.

### 3.2 Resonance features discriminate EEG states (eyes open vs eyes closed)
*(Study 2; Fig. 2)*

Across 420 occipital epochs from 10 subjects (paper-grade pass), eyes-closed
epochs showed higher harmonicity and lower spectral entropy/flatness than
eyes-open, indicating a more harmonically organized spectrum when posterior alpha
is engaged. A leave-one-subject-out classifier on ten resonance/complexity
features decoded state at **AUC = 0.81**, essentially matching a
relative-alpha-power baseline (**0.81**; permutation p = 0.001) — expected for
this alpha-dominated contrast — while contributing an interpretable
harmonic-organization axis beyond raw band power. Critically, the
peak-harmonicity difference is **1/f-dependent**: H_max separates the states at
AUC = 0.83 with FOOOF-style aperiodic removal but at only 0.29 without it, so the
eyes-closed "more harmonic" effect must be read after the aperiodic background is
removed. The harder rest-vs-motor contrast decodes at AUC = 0.62 (vs 0.54 for
band power), where resonance features add the most over raw band power.

### 3.3 The feature vector fingerprints signal modality
*(Study 3; Fig. 3)*

Profiling seven modalities (real and synthetic ECG, eyes-closed-like EEG, PPG,
respiration, a clean harmonic stack, and pink noise) at a shared 0.5–45 Hz band,
a random-forest classifier on the resonance/complexity feature vector identified
modality at **99% accuracy** (chance 14%, permutation p = 0.005). Notably, *no
single* harmonicity scalar gave a clean cross-modality
ranking, because harmonicity is confounded by the **aperiodic (1/f) component**:
on pure colored noise with no oscillations, both H_avg and H_max rise
monotonically with the 1/f exponent (H_avg 0.28->0.43, H_max 1.8->7.7 for
beta = 0.5->2.0), and these modalities differ markedly in their aperiodic
backgrounds. Removing the aperiodic component (`remove_aperiodic=True`) flattens
the spurious harmonicity of noise (H_avg ~ 0.19 across slopes), restoring
specificity to genuine harmonic peaks. Modality identity is best read as a
**multivariate** signature in resonance-feature space rather than a single
scalar.

### 3.4 Strategy comparison
*(Study 4; Fig. 4)*

We swept all 20 harmonic-kernel × ratio-kernel × coupling-metric combinations on
the harmonic-vs-(pink, inharmonic) discrimination task scored by R_max. The
best configurations reached **AUC = 1.00** (e.g. `harmsim | fraction |
nm_plv_canonical`); the weakest fell to **0.84** (`subharm_tension | binary |
nm_pli`). Marginal means isolate each axis's contribution:

| axis | best → worst (mean AUC) |
|------|--------------------------|
| ratio kernel | `fraction` 0.97 > `binary` 0.92 |
| harmonic kernel | `harmsim` 0.96 > `subharm_tension` 0.93 |
| coupling metric | `nm_plv_canonical` 0.97 ≥ `nm_wpli` 0.96 ≥ `nm_rrci` 0.95 ≥ `nm_plv` 0.95 > `nm_pli` 0.89 |

Three practical recommendations follow. **(i)** Prefer the `fraction` ratio
kernel over `binary` — it tests the exact closest rational for every pair rather
than a small preset table, and scored higher everywhere. **(ii)** Among coupling
metrics, the convention-correct `nm_plv_canonical` was best and `nm_pli` clearly
weakest; `nm_plv` is retained only for bit-exact reproduction of pre-refactor
results. **(iii)** `harmsim` is the practical harmonic kernel: it matched or beat
`subharm_tension` on accuracy while running **~32× faster** (0.05 s vs 1.59 s per
`compute_resonance` call), making `subharm_tension` impractical for large sweeps.

## 4. Discussion

The framework validates on both fronts. As a **descriptor of harmonic
organization** it is accurate and useful: it recovers known harmonic structure
almost perfectly, tracks a physiologically meaningful brain-state change, and
fingerprints signal modality. As a **phase-coupling detector** it also succeeds —
once the phase rows are correctly aligned to the analysis frequencies and the
coupling factor is read against a surrogate null (PC_z), it recovers known n:m
coupling at near-perfect AUC both within a signal (0.99) and across signals
(1.0). The one firm caveat: the *raw* reduced PC is small and PSD-weighted, so
coupling claims must rest on surrogate normalization (PC_z) — never on the
absolute PC value. The earlier impression that coupling detection was "limited"
traced to a frequency-misalignment bug (now fixed), not to the construct.

Two methodological cautions generalize beyond this toolbox: (i) spectral
harmonicity (both mean and peak summaries) is confounded by the **aperiodic
(1/f) component** and must be computed after aperiodic removal
(`remove_aperiodic=True`) — or across conditions matched on 1/f — whenever a
harmonicity *difference* is to be interpreted; and (ii) surrogate inference must
target the **factor that carries the effect** (PC for coupling), not a composite
(R) dominated by another factor (H).

## 5. Limitations and future work

Numbers here are from a moderate pass (3 EEG subjects, 6–8 synthetic seeds,
30–40 surrogates); the `--paper` configuration (10 subjects, 20–30 seeds,
150–200 surrogates) is provided for publication-grade estimates. ECG is
synthetic; real multi-lead ECG and additional modalities (EMG, respiration)
would strengthen the cross-modality claim. The phase estimator is STFT-only;
Hilbert/wavelet estimators (registry slots already present) may improve
phase-coupling sensitivity and are a natural next test.

## 6. Reproducibility and toolbox issues surfaced

The suite (`resonance_paper/`) regenerates all results and figures with a single
command. During validation we identified two issues in the reference
implementation:

1. **`with_surrogate_null(surr_type='IAAFT')` default is non-functional** for
   single signals: the surrogate generator implements only
   `AAFT/TFT/phase/shuffle/white/pink/brown/blue`, so the documented default
   raises `ValueError`. We used `AAFT`; the default should be fixed.
2. **`with_surrogate_null` returns only R_z.** Factor-level z-scoring (H_z, PC_z)
   is required for phase-coupling inference and is implemented in the suite;
   exposing it in the library is recommended.

## Figures

Main figures are the composite, publication-styled panels in
`paper/figures/Fig1-4.{png,pdf}` (600 DPI, regenerated by
`python -m resonance_paper.paper.make_paper_figures`). Per-study diagnostic
figures remain in `figures/study*.{png,pdf}`.

- **Fig. 1 — Ground truth (synthetic).** `paper/figures/Fig1_ground_truth`.
  (A) Mean harmonicity H rank-orders a ladder of signals by harmonic richness
  (pink noise < inharmonic < pure tone < dyad < triad < rich stack; Spearman
  ρ = 0.98). (B) Surrogate-normalized phase-coupling recovery: single-signal
  PC_z (AUC 0.99) and matrix entry Φ[f₁,f₂] (AUC 1.0), and cross-signal n:m
  locks 1:1 / 1:2 / 2:3 (AUC 1.0 each) against PSD-matched nulls. (C) Polyrhythm
  recovery: the 2:3:4 resonance peak/median ratio separates locked from
  scrambled stacks (AUC 1.0).

- **Fig. 2 — Real biosignals.** `paper/figures/Fig2_real_biosignals`.
  (A) EEG state decoding (leave-one-subject-out): resonance features vs band
  power for eyes-open/closed and rest/motor. (B) The 1/f confound — peak
  harmonicity H_max discriminates EO/EC at AUC 0.83 with aperiodic removal but
  collapses (0.29) without it, motivating `remove_aperiodic=True`.
  (C) Cross-modality fingerprinting: example resonance spectra and a 7-way
  modality classifier (accuracy 0.99, chance 0.14, permutation p < 0.01).

- **Fig. 3 — Harmonic complexity governs lockability.**
  `paper/figures/Fig3_arnold_tongues`. (A) Devil's staircase for a forced
  Van der Pol oscillator. (B) Arnold-tongue width vs ratio complexity (p·q),
  colored by the framework's harmonicity for the ratio pair: simpler ratios lock
  over wider drive ranges (ρ_width,complexity = −0.50) and score higher
  harmonicity (ρ_width,harmonicity = +0.41).

- **Fig. 4 — Resonance, computation and criticality.**
  `paper/figures/Fig4_criticality`. (A) Branching network: harmonic structure H
  peaks at the critical branching ratio σ = 1, tracking susceptibility and
  avalanche power-law fit. (B) Echo-state reservoir: resonance R and memory
  capacity both peak in the ordered regime, below the edge of chaos.
  (C) Wilson–Cowan E/I network: cross E↔I phase coupling switches on at the
  synchronization onset, coincident with the susceptibility peak.

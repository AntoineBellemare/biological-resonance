# Sound measurement of polyrhythmic phase coupling: diagnosing and correcting an n:m coupling estimator

*Draft v1 — companion to the resonance methods paper. Numbers are quick-grade unless noted; promote to
paper-grade (≥15 seeds, ≥99 surrogates) before submission.*

---

## Abstract

n:m phase coupling — the locking of two rhythms at an integer frequency ratio (a "polyrhythm", e.g. 2:3)
— is widely measured in neuroscience and physiology, yet the estimators are rarely validated against
controllable ground truth, and a single signal's own harmonics can masquerade as coupling. We treat n:m
coupling measurement as an object of study in its own right. Using phase-locked oscillator pairs with
known ratio, coupling strength, and waveform, we show that the *default* settings of a representative
resonance estimator do not merely lose power — they **anti-detect** a clean 2:3 lock, scoring genuinely
coupled signals *below* their uncoupled controls. We trace this to four independent defects (a swapped
phase-multiplier convention, a leakage-limited phase estimator, a ratio table that cannot reach beyond
3:3, and a complexity weight folded into the coupling statistic) and correct each. With the corrected
estimator we find no universally best technique: the n:m phase-locking value (PLV) is the most sensitive
detector on clean, unimodal locks and at low signal-to-noise, but it is **blind by construction to
multimodal locks** (multistable or antipodal relative phase), where entropy- and mutual-information-based
indices recover the coupling the PLV misses. Finally, we confirm and quantify the harmonics-versus-coupling
confound: within a single non-sinusoidal oscillator, *every* technique reports significant n:m coupling,
and no standard surrogate fully removes it — so n:m coupling is soundly interpretable only between
independent sources. We package these results as a validated cross-signal detector that reports a panel of
complementary techniques with surrogate inference and an explicit scope guard, and we give concrete
recommendations for measuring polyrhythmic coupling.

---

## Introduction

When two oscillations lock at an integer ratio n:m — three cycles of one rhythm to two of another — their
generalized phase difference `nφ_a − mφ_b` becomes stationary. Such polyrhythmic coupling is a basic
prediction of weakly coupled oscillator theory and is invoked across cross-frequency coupling, cortical
travelling waves, motor coordination, and auditory rhythm. The workhorse estimator is the n:m
phase-locking value (PLV) introduced by Tass and colleagues, together with its phase-lag variants and,
less commonly, entropy- and information-theoretic indices.

Two problems recur. First, the estimators are seldom validated against ground truth: a pipeline is
assembled from a phase estimator, a ratio rule, and a coupling statistic, each with defaults, and applied
to data without checking that it recovers a *known* polyrhythm. Second, a single non-sinusoidal rhythm
contains harmonics that are phase-locked by construction, so any n:m statistic computed within one signal
can report "coupling" that is merely waveform shape — the confound emphasized by Aru and colleagues and by
Scheffer-Teixeira and Tort.

Here we make the measurement itself the experiment. We build polyrhythmic ground truth with controllable
ratio, coupling strength, phase offset, amplitude asymmetry, drift, noise, and waveform, and we ask, of a
representative resonance estimator and of the broader technique family: does it retrieve the lock, is it
specific to the true ratio, which technique wins in which regime, and does it survive the harmonics
confound? The answers are, in order: not with its defaults; yes once corrected; it depends on the regime;
and only between independent sources. We turn each answer into a fix and ship a validated detector.

---

## Methods

**Construction.** For two signals we band-pass each at its own component frequency, take the Hilbert
analytic phase, and form the generalized phase difference `nφ_a − mφ_b`. The ratio enters *only* as the
integer multipliers `(n, m)`; for a frequency ratio `f_b/f_a = p/q` the stationary combination requires
`(n, m) = (p, q)` (the Tass convention). The coupling techniques summarize the distribution of the
relative phase and differ only in *which* feature they read:

- **PLV** = `|⟨exp(i·Δφ)⟩|` — the first circular moment (resultant length); maximal for a single
  concentrated phase, and its phase-lag siblings PLI / wPLI / RRCi add zero-lag robustness but remain
  first-moment statistics.
- **Entropy index** ρ = `(H_max − H)/H_max` from the Shannon entropy of the relative-phase histogram —
  sensitive to *any* departure from uniformity (all moments).
- **Conditional-probability index** — the concentration of one phase given binned values of the other.
- **Phase mutual information** — normalized model-free dependence between the two phases.

**Ground truth.** We generate n:m phase-locked oscillator pairs (one component per channel) with a
controllable coupling strength κ (independent → perfect lock via amplitude mixing of a locked and an
independent component), phase offset, amplitude ratio, frequency drift, observation SNR, and waveform.
Two emission modes are used: cross-signal (two independent sources) and within-signal (one signal vs its
own harmonics). Each coupled condition is contrasted with a PSD-matched uncoupled control.

**Inference.** Detection is summarized by the area under the ROC curve (AUC) for coupled vs uncoupled
across seeds, and by a surrogate z-score / rank-p against an iterated amplitude-adjusted Fourier transform
(IAAFT) null of one channel (which preserves that channel's spectrum and amplitude distribution while
destroying its relation to the other). Specificity is the fraction of trials whose strongest coupling, over
a grid of candidate coprime multipliers, lands on the true ratio.

An *oracle* — the n:m PLV computed on band-pass-Hilbert phase at the known frequencies and ratio — provides
the best-case target a discovery pipeline should approach.

---

## Results

### Default settings anti-detect a clean polyrhythm

On a genuine 2:3 lock (10 and 15 Hz), the oracle separates coupled from uncoupled cleanly (mean targeted
phase-coherence 0.99 vs 0.23; Δ = 0.76). The estimator's default configuration does the opposite: the
coupled signals score *below* their uncoupled controls (0.22 vs 0.30; Δ = **−0.09**). The corrected
configuration recovers the right sign but only a sliver of the effect (Δ = 0.04) **(Figure 1)**.

Four independent defects explain the failure. (i) *Convention.* The ratio rule returns `(n, m)` under a
swapped convention, and the raw statistics apply the multipliers literally, so the relative phase is
non-stationary even for a perfect lock; across the full ratio range every raw technique anti-detects
ground truth (AUC ≈ 0.13). (ii) *Phase estimator.* The default short-time-Fourier-transform bin phase is
leakage-limited and is not true oscillation phase; it misses n:m locking that the Hilbert estimator
recovers. (iii) *Ratio coverage.* The default ratio table reaches only 3:3 and silently relabels
unmatched pairs as 1:1, so polyrhythms such as 3:4 or 4:5 are never tested at their true ratio. (iv)
*Weighting.* A musical complexity weight is multiplied into the coupling value, capping a perfect 2:3 lock
at ≈ 0.17. Each defect is corrected (canonical multipliers, Hilbert phase, exact-fraction ratios, and an
unweighted coupling readout); the remainder of the paper uses the corrected estimator.

### The corrected estimator is sound across the ratio range

With the canonical convention, the n:m PLV detects every tested lock from 1:2 through 6:7 at ceiling
(AUC = 1.00), identifies the true ratio against a grid of candidate multipliers with perfect accuracy,
rises monotonically with coupling strength, and remains reliable down to low SNR (AUC → 1.00 by 0 dB)
**(Figure 2)**. The zero-lag-robust variants (PLI, wPLI, RRCi) recover the lock at most ratios but carry
ratio-dependent blind spots and plateau below ceiling (AUC ≈ 0.90); their robustness to instantaneous
mixing buys little here because, at n ≠ m, the two channels are read at different frequencies and a
same-frequency leak does not contaminate the cross-frequency statistic.

### No single technique is best — the winner is regime-dependent

Across eight coupling regimes, the techniques separate into two families with complementary failure modes
**(Figure 3)**. On clean unimodal locks — and under non-sinusoidal coupling, non-stationary drift, and
added harmonics — the PLV is supreme (AUC = 1.00) and is the most sensitive at low SNR. But when the lock
is **multimodal** — a multistable or antipodal relative phase — the PLV and the conditional-probability
index *anti-detect* (AUC ≈ 0.34), because a balanced multimodal distribution is *less* unimodally
concentrated than the independent null. Here the all-moment indices dominate: the entropy index and phase
mutual information recover the coupling (AUC 0.90–0.99) where every first-moment statistic fails. Weak
graded coupling is genuinely hard for all techniques (AUC ≈ 0.6–0.7). The practical reading is a panel:
the PLV for sensitivity on clean locks and at low SNR, the entropy/information indices for generality
against multimodal locks. This pattern was reproduced independently, with separate generators and AUC
implementations, by adversarial verification.

### Within a single signal, every technique false-positives — and no surrogate fully fixes it

A single non-sinusoidal oscillator has harmonics that are phase-locked by construction. Tested for "1:2
coupling" between its fundamental and second harmonic, *every* technique reports strong, significant
coupling against an IAAFT null — surrogate z of 7 (PLV), 42 (entropy), 37 (mutual information), all 100%
significant **(Figure 4)** — although there is only one rhythm. The IAAFT null is the wrong control here:
it randomizes the cross-frequency phase relation, so a signal's genuine harmonic lock appears "coupled"
relative to the surrogate. Critically, the more sensitive the instrument, the worse the false positive.

Testing which null separates the artifact from genuine coupling, no standard surrogate fully removes it.
IAAFT and phase randomization fail badly (artifact z = 6–43); a waveform-preserving block-shuffle reduces
the artifact roughly eight-fold (z = 3.5–5.7) while keeping genuine coupling strongly detectable (z =
10–49), but a residual remains. Within-signal n:m at harmonically related frequencies is therefore
fundamentally confounded with waveform shape. The sound regime is cross-signal: between independent
sources, the IAAFT-of-one-channel null destroys a real inter-signal relationship with no within-signal
harmonic confound — the regime in which the ground-truth recovery above holds.

### A validated detector

We package these results as a cross-signal detector. It band-passes each component in its own band, applies
the canonical multipliers, reports a panel of complementary techniques (PLV plus the entropy and mutual-
information indices) each with an IAAFT surrogate z and rank-p, scans candidate ratios for specificity, and
emits an explicit scope-guard warning when the two inputs are too correlated to separate coupling from
shared-source harmonics. On held-out ground truth it recovers genuine locks across the ratio range
(panel z ≫ 0), stays null for independent signals and off-target ratios, detects the multimodal locks the
PLV alone misses, and warns on within-signal input **(Figure 5)**. The detector and its regression tests
are released in the framework.

---

## Discussion

Treated as an experiment, n:m coupling *measurement* yields three lessons. First, estimator defaults can be
not merely weak but **directionally wrong**: a plausible pipeline anti-detected a textbook polyrhythm, and
the failure was invisible without ground truth. Validating an estimator against a known lock should be a
precondition for interpreting it on data. Second, there is **no universal n:m statistic**: the first-moment
PLV family and the all-moment entropy/information family have complementary blind spots, so a panel — not a
single default — is the honest instrument; reporting the PLV alone will silently miss multistable locks,
and reporting an entropy index alone sacrifices sensitivity on clean ones. Third, the **harmonics confound
is a property of the design, not of the metric**: every technique false-positives within one non-sinusoidal
signal and no surrogate fully removes it, so the only sound n:m inference is between independent sources —
a constraint our detector enforces by construction.

These results address, on the measurement side, the long-standing critique that cross-frequency coupling
claims are confounded by waveform shape and inflated by mismatched surrogates: the corrected estimator does
not manufacture coupling where there is none (it stays null on independent signals and off-target ratios),
it flags the within-signal regime where it cannot be trusted, and it reports the surrogate-calibrated
inference rather than a raw statistic.

Limitations. The within-signal confound is not solved, only scoped; a definitive within-signal test would
require waveform-aware controls beyond phase surrogates. Weak graded coupling sits near the detection floor
for all techniques. We did not implement a debiased estimator (pairwise phase consistency) for comparing
coupling across epochs of differing length, which would be a useful addition where trial counts vary. All
ground truth here is synthetic; the companion real-data work shows that, with this corrected instrument,
low-integer n ≠ m phase-phase coupling is at the noise floor in resting and sleep scalp EEG, consistent
with the view that genuine polyrhythmic coupling is rare in macroscopic recordings and must be claimed only
with the controls established here.

---

## Figures (to generate)

- **Figure 1** — Default settings anti-detect a 2:3 lock; corrected config recovers the sign; oracle
  separates cleanly. (bar chart of coupled/uncoupled/Δ for default, corrected, oracle)
- **Figure 2** — The corrected PLV is sound across 1:2…6:7: detection AUC per ratio (line at 1.00),
  ratio-identification accuracy, κ dose-response, and SNR sweep.
- **Figure 3** — Regime × technique detection-AUC heatmap; the PLV/entropy complementarity, with the
  multimodal columns where the PLV anti-detects and entropy/MI dominate.
- **Figure 4** — The harmonics confound: within-signal false-positive surrogate z by technique, and the
  null-comparison (IAAFT vs phase-randomize vs time-shuffle) of artifact vs genuine.
- **Figure 5** — The validated detector: panel z across the ratio range; null on independent/off-target;
  multimodal recovery; scope-guard demonstration.

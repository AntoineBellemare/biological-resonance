# The resonance construct: structure and dynamical regimes

*Synthesis of Studies 17–20. Study 17 is the non-circular capstone for the
H / PC / R construct itself; Studies 18–20 probe the three dynamical regimes in
which resonance is generated — noise-induced (stochastic) resonance, driven
intermodulation (SSVEP / frequency-tagging), and auditory consonance — with a
real-EEG capstone (Study 20b: brainstem FFR to consonant/dissonant dyads). All
numbers below are paper-grade. Regenerate with `python -m resonance_paper.run_all
--paper` (or `--only 17`, etc.; Study 20b: `python -m
resonance_paper.study20b_ffr_consonance --paper`, downloads the FFR data on first
run).*

---

## 1. What the three factors are

- **H — harmonicity.** A property of the *spectrum*: how close the detected peaks
  sit to integer (harmonic) relations. Computed from the (single- or cross-)
  harmonicity matrix. Phase-agnostic by construction.
- **PC — phase coupling.** A property of the *temporal dynamics*: how stably two
  components hold an n:m phase relation (clean n:m PLV, surrogate-normalised).
- **R = H × PC.** The conjunction: structure that is *both* harmonic *and*
  phase-locked.

The honest question Study 17 settles is whether these are genuinely separable, and
whether the product R buys anything over its factors.

## 2. Study 17 — the tripartite dissociation (the capstone)

A 2-D grid independently varies **ratio complexity** (Tenney height of p:q, simple
1:2 → complex 5:11) and **phase-locking strength** κ (0 = drifting phases, 1 =
rigid n:m lock). Predictions: (1) H tracks complexity, flat along κ; (2) PC tracks
κ, flat along complexity; (3) R peaks only in the simple-AND-locked corner; (4) the
payoff — R rejects the two single-factor confounds (harmonic-but-unlocked,
complex-but-locked) better than H or PC alone.

**Result (paper-grade, n = 12):**

| factor | vs complexity ρ | vs locking κ ρ |
|---|---|---|
| **H** | −0.46 | **+0.00** |
| **PC** | −0.47 | +0.49 |

| separability of "true resonance" (simple & locked) vs single-factor confounds | AUC |
|---|---|
| H | 0.59 [0.35, 0.81] |
| **PC** | **0.78 [0.57, 0.94]** |
| R = H·PC | 0.66 [0.42, 0.85] |

**What holds, and what doesn't:**

- **Prediction (1) holds cleanly: H is phase-blind.** ρ(H, κ) = **+0.00** —
  harmonicity is *exactly* invariant to whether the components are phase-locked.
  This is the clean dissociation the construct claims: H measures spectral
  harmonic structure independent of temporal coupling.
- **Prediction (2) half-holds: PC tracks locking (+0.49) but is *not*
  ratio-blind** (−0.47 vs complexity). This is not a bug — it is physics. Stable
  mode-locking is *easier* at simple ratios (the Arnold-tongue / devil's-staircase
  structure of Studies 6 & 8): a 1:2 lock is wide and robust, a 5:11 lock is a
  razor-thin tongue. So in any real oscillator H and PC are **physically coupled
  through the ratio axis** — they are conceptually distinct but not statistically
  independent.
- **Prediction (4) does NOT hold: R does not out-detect PC** (AUC R = 0.66 < PC =
  0.78, overlapping CIs). Because PC already declines with complexity, PC alone
  does most of the work of rejecting both confounds; multiplying by H_norm adds
  variance, not specificity.

**The honest conclusion.** The construct's clean claim survives — **H ⟂ phase**
(ρ = 0.00), so harmonicity and phase-coupling are genuinely different measurements.
But the stronger claim — that the *product* R is a more specific resonance detector
than its factors — is **not** supported, because mode-locking ties PC to ratio
simplicity, so H and PC are not independent and their conjunction does not beat the
better single factor. **R's value is interpretive, not detective:** it decomposes a
resonance into *why* it is (or isn't) one — is the structure failing because the
spectrum isn't harmonic (low H) or because the components aren't locked (low PC)? —
rather than detecting resonance better than PC alone. We report it this way.

## 3. Study 18 — stochastic resonance across many ratios

A forced van der Pol oscillator is driven across a dense ratio axis Ω = f_drive/f0
(simple → complex) at several noise levels; per cell we take the framework
harmonicity H of the response. The single-signal reduced R is PSD-diluted to ≈ 0
(see §6), so H is the resonance readout here.

**Result (paper-grade, 60 ratios × 6 noise levels × 5 seeds):** mean noise-induced
H gain (H_best_noise − H_noiseless) is **+0.0067 for complex ratios vs +0.0032 for
simple ratios** (~2× larger where deterministic locking is weak), and the fraction
of ratios whose harmonicity peaks at *non-zero* noise is **0.37 for complex vs 0.23
for simple**. (The quick pass over-estimated the gap at ~7×; the honest paper-grade
effect is real but modest.)

**Interpretation.** Simple ratios lock deterministically and gain little from
noise (they are already at/near their harmonic best at zero noise). Complex ratios,
which sit between wide Arnold tongues and do *not* lock deterministically, are the
ones where **added noise can induce harmonic structure** — the signature of
stochastic resonance / noise-induced mode-locking. Resonance is not only a property
of clean deterministic forcing; in the complex-ratio regime it is *generated by
noise*. (Magnitudes are small; the paper-grade pass tightens them.)

## 4. Study 19 — SSVEP, nonlinearity, and intermodulation

A tunable static nonlinearity models neural frequency-tagging. (A) single flicker
at f0; (B) two flickers f1, f2 at simple vs complex ratios.

**Result (paper-grade):**
- **(B) two-flicker intermodulation is the substantive signal.** The
  intermodulation index (power at n·f1 ± m·f2) **rises with nonlinearity**
  (ρ = +0.62) and is **richer for simple ratios** (0.346 vs 0.130 at full
  nonlinearity) — simpler f1:f2 lattices place IM products on fewer, stronger
  lines.
- **(A) single-flicker H is at ceiling.** H_max ≈ 1.3 essentially regardless of
  nonlinearity (1.33 → 1.31 across g = 0 → 1): a single periodic flicker is
  *already* trivially harmonic (its comb falls on exact integer multiples), so
  adding harmonics via nonlinearity does not raise the peak harmonicity. The
  discriminating case is the two-flicker lattice, not the single comb.

**Interpretation.** The framework's harmonic machinery is informative exactly where
the neuroscience is interesting — multi-frequency integration / binding — and the
two-flicker intermodulation result connects H/PC/R directly to frequency-tagging
paradigms. The single-flicker case is a sanity-ceiling, reported as such.

## 5. Study 20 — musical chords, consonance, and combination tones

Chords are synthesised at just-intonation ratios, optionally passed through an
auditory nonlinearity (generating combination/difference tones), and we ask whether
H recovers the consonance ordering. Ground-truth consonance = mean pairwise Tenney
height (lower = more consonant).

**Result (quick):** ρ(H, chord complexity) = **−0.71 (linear), −0.73 (nonlinear)** —
H decreases monotonically from consonant to dissonant chords (octave H ≈ 0.44 →
clusters ≈ 0.15), and the **nonlinear combination tones *sharpen* the recovery**
(|ρ| larger). 

**Interpretation.** H reconstructs the centuries-old consonance ↔ ratio-simplicity
relationship directly from the acoustic spectrum, and the auditory nonlinearity —
which physically generates the combination tones long argued to underlie consonance
perception — makes the relationship *cleaner*, not noisier. This ties the framework
back to its musical roots and to a concrete perceptual prediction — which the
real-EEG study below then confirms in the human brain.

## 5b. Study 20b — REAL EEG: the FFR is more harmonic for consonant dyads (neural)

The synthetic prediction was tested on real brainstem frequency-following responses
(FFR) to musical dyads (Andermann, Reineke, Riedel & Rupp, *Eur. J. Neurosci.*
2026; OSF `5puhb`; phase-locked, 20 kHz; n = 36 listeners with usable data). Four
conditions cross consonance × completeness: **CC** consonant complete (3:2 fifth,
160 + 240 Hz), **DC** dissonant complete (45:32 tritone, 160 + 225 Hz), and **CI /
DI** the same dyads with the **fundamentals physically removed** (stimulus energy
only ≥ 640 Hz). Per epoch we decimate to 4 kHz, extract the spectral partials, and
take the peak-based harmonicity (Gill–Purves harmsim — the framework's harmonic
metric) — per-listener paired Wilcoxon.

**Result — neural harmonicity tracks consonance, and the load-bearing contrast is
purely neural:**

- **Consonant > dissonant in the FFR**, in both runs: complete CC>DC (passive
  Δ=+7.66, p=1.3×10⁻⁸, 92% of listeners; active Δ=+3.58, p=4.6×10⁻³); and —
  crucially — **incomplete CI>DI** (passive Δ=+4.83, p=1.3×10⁻⁵; active Δ=+5.28,
  p=4.0×10⁻⁵).
- **The CI>DI effect is neurally generated, not stimulus leakage.** The incomplete
  stimuli carry **0.000%** energy below 600 Hz, so harmonic structure the FFR shows
  there cannot be acoustic. A **band-split** confirms it: the effect lives entirely
  in the stimulus-silent low band ([70,639) Hz: Δ=+4.96, p=3.3×10⁻⁵) and is **null
  in the high band where stimulus energy actually exists** ([640,1100] Hz: Δ=+0.46,
  p=0.27) — the opposite of what leakage would produce.
- **The FFR reconstructs each dyad's own missing fundamental.** Grand-average FFR
  amplitude is higher at 240 Hz (consonant fundamental) for CI than DI
  (p=5.8×10⁻¹¹) and higher at 225 Hz (dissonant fundamental) for DI than CI
  (p=7.3×10⁻¹⁰) — frequency-specific neural reconstruction (missing-fundamental +
  combination tones), not a generic artifact.
- **Controls rule out the obvious confounds.** Total FFR power is flat across
  consonance (not a loudness effect); the effect survives **SNR-matching by noise
  injection** (raising the cleaner consonant epochs' noise floor to the dissonant
  level leaves CI>DI intact, Δ=+4.67, p=3.3×10⁻⁵); and the dissonant condition
  carries *more* 60 Hz line noise (p=8.9×10⁻⁹), so a mains-artifact account would
  predict the wrong direction. The result is robust across a peak-count/maxdenom/
  band/window grid (verified by independent reimplementation; the neural CI>DI is
  significant across the entire grid). The one disclosed fragility is the
  *complete* active CC>DC contrast, which attenuates (never flips) at large peak
  counts; we treat the missing-fundamental CI>DI as the primary, load-bearing
  result.

**Interpretation.** The human brainstem represents consonant dyads more
harmonically than dissonant ones — and does so even when the fundamentals are
absent from the stimulus, by *reconstructing* the harmonic relationships through
its own nonlinear dynamics (missing-fundamental + combination/difference tones).
This is direct neural confirmation of the framework's consonance claim (Study 20)
and realizes the intermodulation / nonlinear-dynamics angle in real EEG. The claim
was adversarially verified: four independent skeptics (stimulus-leakage, metric
robustness, power/SNR confound, permutation null) each tried and failed to refute
it, several strengthening it.

## 6. A recurring methodological point: the single-signal R dilution

In Studies 18–20 (and the in-silico criticality studies) the **single-signal**
reduced R reads ≈ 0, while H is informative. This is expected: the reduced PC factor
is a PSD-weighted average that dilutes localized phase structure across the whole
spectrum. The construct's phase-coupling factor is meaningful in its **cross-signal
/ matrix** form (Studies 5, 12, 17), where a specific n:m pair is targeted, or via
surrogate-normalised PC_z. So for *single-channel* analyses, **H is the resonance
observable**; R/PC require a paired or targeted readout. This is consistent across
the whole suite and is now stated wherever single-signal R appears.

## 7. Conclusion

- **The construct is genuinely tripartite where it counts:** H is exactly
  phase-blind (Study 17, ρ = 0.00), so harmonicity and phase-coupling are distinct
  measurements.
- **But H and PC are not statistically independent in real oscillators** —
  mode-locking ties phase coupling to ratio simplicity — so the product R is best
  read as an *interpretive decomposition* (why is/ isn't this a resonance?), not as
  a detector that beats its factors.
- **Resonance is generated across three dynamical regimes:** by noise at complex
  ratios (stochastic resonance, Study 18), by nonlinear multi-frequency integration
  (intermodulation, Study 19), and it tracks **auditory consonance** sharpened by
  combination tones (Study 20).
- **It holds in the real human brain:** the brainstem FFR is more harmonically
  organized for consonant than dissonant dyads — even for missing-fundamental
  dyads, where the brain *reconstructs* the harmonic relationships (Study 20b).
  This is the framework's first load-bearing real-data result that beats the
  obvious confounds, adversarially verified.

## Figures
- `figures/study17_tripartite_dissociation.{png,pdf}` — H/PC/R over the
  complexity × locking grid + specificity AUC bars.
- `figures/study18_stochastic_resonance.{png,pdf}` — H across ratio × noise; SR
  gain vs ratio complexity.
- `figures/study19_ssvep_intermod.{png,pdf}` — single-flicker H; two-flicker
  intermodulation (simple vs complex).
- `figures/study20_musical_intermod.{png,pdf}` — H vs chord consonance, linear vs
  nonlinear.
- `figures/study20b_ffr_consonance.{png,pdf}` — real FFR: consonance harmonicity by
  condition; per-listener advantage; neural-vs-acoustic; band-split leakage control.

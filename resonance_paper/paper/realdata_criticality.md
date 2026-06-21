# Resonance, brain state, and criticality in real EEG (Studies 13–15)

*The real-data counterpart to the in-silico criticality studies (10–12). Tests, on
three independent human EEG datasets that traverse arousal/consciousness, whether
the framework's harmonicity H (and resonance R) (a) discriminate brain state and
(b) reproduce the in-silico prediction that **H is maximized at criticality**
(Study 10). Paper-grade; harmonicity computed with fmin = 0.5 Hz so the slow-wave
band is included (see "A measurement lesson" below). Markers: avalanche branching
ratio m̂ (Wilting–Priesemann) and alpha-envelope DFA/LRTC, both validated against
ground truth in `criticality.py`.*

## Datasets
| # | Dataset | Traversal | n | windows |
|---|---------|-----------|---|---------|
| 13 | Chennu propofol (FieldTrip BIDS) | baseline→mild→moderate *sedation*→recovery | 10 | 160 |
| 14 | Sleep-EDF (MNE) | Wake→N1→N2→**N3**→REM | 8 | 600 |
| 15 | ds004541 (OpenNeuro) | **Wake → LOC → ROC** (deep general anesthesia) | 7 | 80 |

## Result 1 — Resonance tracks brain state, but does not beat band power
Multiclass leave-one-subject-out decoding of state (accuracy; chance in parens):

| dataset | resonance (H/R) | band power | criticality markers | all |
|---|---|---|---|---|
| Propofol (0.25) | 0.37 | 0.38 | 0.39 | 0.36 |
| Sleep (0.20) | 0.37 | **0.57** | 0.56 | **0.65** |
| Deep GA (0.50) | 0.64 | 0.66 | **0.74** | **0.82** |

Resonance features are above chance everywhere, but **do not exceed band power**;
in sleep they are clearly below it. The criticality markers and the combined
feature vector decode best. Honest read: **H/R are a real, interpretable spectral
descriptor of state, not a superior state classifier.** (Chennu's *titrated
sedation* is a weak contrast for everything — all feature sets ≈ 0.37–0.39.)

## Result 2 — The in-silico "H peaks at criticality" does NOT transfer; it reverses
In-silico (Study 10), on the subcritical side H **rises toward** σ = 1
(H_max 0.31→0.53 as σ 0.70→1.00). In vivo it does the opposite: H is highest in
the **most subcritical** state.
- Sleep: H_max is highest in **N3** (0.531 vs ~0.41 elsewhere), the lowest-m̂ stage
  (m̂ = 0.775); ρ(H, m̂-proximity) = **−0.24 [−0.30,−0.18], p = 0.008** (all 8
  subjects negative).
- Deep GA: ρ(H, m̂-proximity) = −0.17; H slightly higher unconscious.
- **The markers disagree:** ρ(H, DFA-proximity) flips sign vs ρ(H, m̂-proximity)
  in deep GA (+0.17 vs −0.17). So "distance from criticality" is **not cleanly
  operationalized** in these data — different estimators place the critical state
  differently.

## Result 3 — Why it reverses (the mechanism, test B1)
H conflates two generators of harmonic structure:
1. **scale-free / critical** multi-scale structure — the only source in Study 10's
   non-oscillatory branching model, which makes H peak at σ = 1;
2. a strong **periodic oscillation's harmonic series** — N3's non-sinusoidal slow
   wave produces peaks at f, 2f, 3f… → high H.

Test B1 (sleep, per-epoch): the H↔criticality relationship lives **entirely in the
slow-wave band** —
- ρ(H_with-slow, m̂-prox) = −0.12; **survives controlling for slow-wave power**
  (partial −0.11) → it is the slow-oscillation *harmonic structure*, not amplitude;
- ρ(H_without-slow [fmin = 2], m̂-prox) = **+0.01** → exclude the slow band and the
  relationship vanishes.

So **in vivo, H is dominated by the slow-oscillation harmonic series, which is
strongest in deep/synchronized states and therefore anti-tracks
avalanche-criticality.** The two regimes measure *different generators of harmonic
structure*; the in-silico result (Studies 10–12) holds for the scale-free
generator, not the oscillatory one. This is a precise dissociation, not a vague null.

## Resolution — the reversal is the observable (Study 16)

Study 10's H was computed on the **avalanche activity** A(t) (a scale-free,
non-oscillatory signal); in vivo we had computed H on the **raw oscillatory EEG**.
Study 16 recomputes resonance on the **population-activity / global-field-power**
signal (the in-vivo analog of A(t)) alongside raw-EEG H, against the validated m̂
axis (DCC was attempted but dropped — it did not minimize at σ=1 in ground-truth
simulation; the avalanche size exponent τ→~1.5 at criticality validates and is kept
descriptively).

Sleep (n=8, 600 windows):
| | across-state ρ(·, m̂-prox) | within-state ρ (controls oscillation confound) |
|---|---|---|
| **H_full** (oscillatory, raw EEG) | **−0.24** (p=0.008) — the reversal | −0.08 (p=0.03) |
| **H_aval** (scale-free, population activity) | +0.07 (neutral) | **+0.11 (p=0.031, positive)** |
| R_aval | −0.07 | +0.01 (null) |

**The reversal is a measurement artifact of oscillatory H.** On the scale-free
observable, and controlling for between-state oscillation differences
(within-state, moment-to-moment), in-vivo H **positively** tracks
criticality-proximity (ρ = +0.11, p = 0.03) — **recovering the Study-10
prediction**. It is **H, not R** that does so (R_aval within-state ≈ 0), exactly
as the model says (H tracks criticality; R requires oscillations). Deep GA
(ds004541) is underpowered for this: m̂ ≈ 0.98 in *both* wake and LOC, so that
dataset does not actually traverse criticality and cannot test the prediction.

In one line: **once resonance is measured on the right (scale-free) observable and
between-state oscillation confounds are removed, the in-silico "H is maximized near
criticality" holds in vivo too; the earlier "reversal" was oscillatory H (slow-wave
harmonics), not a real contradiction.**

## A measurement lesson (parameter sweeps)
- **Frequency-bin precision matters.** At fmin = 0.5 with 0.5 Hz bins the grid
  (0.5, 1.0, 1.5…) cannot pin a ~0.75 Hz slow-wave peak and H is artifactually
  flat; **0.25 Hz bins** restore discrimination (Chennu moderate-vs-baseline AUC
  0.53→0.65). 0.125 Hz adds nothing. The state config uses precision = 0.25 Hz.
- **Include the slow band.** fmin = 2 (the original choice) discards the <2 Hz band
  that defines N3 / deep anesthesia and flattens H. The loaders now high-pass at
  0.3 Hz and resonance uses fmin = 0.5.
- **Kernel.** `subharm_tension` is the most *sensitive* kernel (KW p ~1e-18 across
  sleep stages) but ~40× slower (7 s vs 0.2 s/call) — infeasible at paper scale;
  `harmsim` is used, with the same qualitative conclusions.
- **R vs H.** In Chennu, R (coupling-weighted) decodes the 4 levels better than H
  (acc 0.42 vs ≤0.27); the peak-based Tenney harmonicity adds nothing.

## Honest conclusion
- The **in-silico criticality result is solid** (Studies 10–12): H peaks at
  branching criticality, the reservoir generates harmonic structure at the edge of
  chaos, E↔I coupling rises at the synchronization onset.
- **In real EEG**, H/R are valid state descriptors but **do not beat band power**.
  The H-at-criticality prediction *appeared* to reverse — but Study 16 shows that
  was an **observable artifact**: raw-EEG H is dominated by the slow-oscillation
  harmonic series (B1). On the **scale-free** observable (population activity), with
  between-state oscillation confounds removed (within-state), **in-vivo H positively
  tracks criticality-proximity (sleep ρ=+0.11, p=0.03), recovering the model
  prediction — and it is H, not R, as in silico.**
- Net, mechanistic and falsifiable: *resonance indexes harmonic organization; on the
  scale-free generator it tracks criticality both in silico and in vivo; measured on
  the raw oscillatory signal it instead indexes slow-oscillation synchronization,
  which anti-correlates with avalanche-criticality.* The two are now reconciled, not
  contradictory.

See `figures/Fig5_realdata.{png,pdf}`, `study16_*` (results), and `figures/study1[345]_*`.

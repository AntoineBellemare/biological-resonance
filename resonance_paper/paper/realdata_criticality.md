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
- The **in-silico criticality result is the strong one** (Studies 10–12): H peaks
  at branching criticality, the reservoir generates harmonic structure at the edge
  of chaos, E↔I coupling rises at the synchronization onset.
- **In real EEG**, H/R are valid state descriptors but **do not beat band power**,
  and the **H-at-criticality prediction reverses** because in-vivo H is governed by
  slow-oscillation harmonics (a synchronization/subcritical phenomenon), while
  "criticality" itself is estimator-dependent here. The forward-looking statement
  is mechanistic and falsifiable, not a failure: *resonance indexes harmonic
  organization; whether that aligns with criticality depends on whether the
  harmonic structure is scale-free (critical) or oscillatory (synchronized).*

See `figures/Fig5_realdata.{png,pdf}` and `figures/study1[345]_*` for the per-dataset panels.

# Resonance, reservoir computing, and criticality

*Synthesis of the dynamical-systems studies (Studies 9–12) addressing two
questions: (1) can reservoir computing model harmonic resonance, and (2) how does
resonance relate to criticality in biological/complex systems? Numbers are from
the paper-grade (`--paper`) pass with bootstrap-over-seeds 95% CIs (study 10:
n=20; study 11: n=16; study 12: n=12). Regenerate with `python -m
resonance_paper.run_all --paper`.*

---

## The connecting idea

A reservoir's **edge of chaos** and a neural network's **criticality** are the
same object: the critical point of a dynamical phase transition between an ordered
(contracting) and a disordered (expanding/chaotic) regime. Studying resonance
across that transition — in two complementary models — answers both questions at
once.

- **Reservoir / echo-state network** — tunable via spectral radius ρ; driven by
  white noise to test whether near-critical dynamics *generate* harmonic resonance.
- **Branching neural network** — the canonical model of neuronal criticality;
  tunable via branching ratio σ, with power-law avalanches at σ = 1. Scale-free
  but *not* oscillatory.

## What we found

### Reservoir (Studies 9, 11) — does criticality GENERATE resonance?
Study 11 was redesigned. Echoing a *fixed harmonic input* is flat-until-chaos and
uninformative (the ordered reservoir just passes the input through). Instead we
drive the reservoir with **white noise** (no harmonic content, flat spectrum) and
ask whether its dynamics *manufacture* harmonic structure in the dominant emergent
mode (PC1), as a function of ρ. Edge of chaos (Lyapunov λ = 0) at
ρ_c ≈ 1.57 [1.44, 1.68] (n = 16 reservoirs).

- **The reservoir generates harmonic resonance from noise — specifically at the
  edge of chaos.** Harmonicity of the emergent mode sits at/below the white-noise
  baseline (H_input ≈ 0.45) throughout the ordered regime (H_gain ≤ 0 for
  ρ ≲ 1.1), then **switches on at the approach to criticality** — H_gain's 95% CI
  clears 0 at ρ ≈ 1.30, just below ρ_c — and **peaks at the edge of chaos**
  (ρ ≈ 1.50, H_gain = +0.15 [+0.06, +0.25]), staying elevated in the chaotic
  regime. The tanh nonlinearity folds the emergent collective oscillation into
  integer harmonics.
- **Harmonicity and generic peakedness dissociate.** Generic spectral peakedness
  rises *monotonically* with ρ (0.13 → 0.69, deep into chaos), but *harmonic*
  (integer-related) structure specifically onsets and peaks at the edge of chaos.
  So criticality generates **resonance**, not just any spectral structure.
- Memory capacity (computation) peaks deeper in the ordered regime (ρ ≈ 0.70),
  **distinct** from where harmonic generation switches on — resonance generation
  and linear computation are different regimes.

### Branching network (Study 10)
Sweeping σ through the critical point (criticality markers — susceptibility,
power-law avalanches — peak near σ = 1):

- **Harmonicity H peaks AT criticality** (σ = 1.00 [0.97, 1.08], co-located with
  the susceptibility peak at σ = 1.08): the harmonic structure of population
  activity is a criticality signature.
- **Resonance R ≈ 0 throughout**: bare avalanche dynamics are scale-free but not
  oscillatory/phase-locked, so the phase-coupling factor — and hence R — stays
  near zero.

### E/I network (Study 12) — the oscillation-capable test
A stochastic Wilson–Cowan E/I network swept across its **synchronization onset**
(the "edge of synchronization", where criticality and oscillations coincide):

- Susceptibility (order-parameter fluctuations) peaks at g_c ≈ 1.0; the
  oscillation amplitude rises through the transition.
- The single-signal reduced PC again reads ≈ 0 (within-signal harmonic coupling
  is invisible to it). Measuring **cross-resonance between the E and I
  populations** (1:1 PING phase locking, targeted matrix entry — the Study-5
  readout) gives a non-trivial signal. Crucially, E↔I phase coupling is **already
  non-zero in the asynchronous regime** (baseline PLV ≈ 0.43, because E drives I
  even without macroscopic synchrony), so the honest statement is that **PC RISES
  at the onset**, not that it switches on from zero: PLV climbs from ≈ 0.43 to
  ≈ 0.99 (ΔPC ≈ +0.55) as g crosses g_c ≈ 1.0 [1.0, 1.0]. So phase-coupling
  resonance **rises sharply at the edge of synchronization** and marks the
  synchronized regime.

This is the first model where phase-coupling **R** (not just H) is both non-trivial
and placeable relative to criticality — and it confirms R marks the
structured-oscillatory regime that *emerges at* the synchronization transition.

## Unified conclusion

The two factors of the resonance construct behave **differently** with respect to
criticality, and the difference is interpretable:

- **Harmonicity (H)** — a property of the *spectrum* — is a criticality marker:
  it is maximized at the critical point (branching network), where multi-scale
  structure is richest.
- **Phase-coupling resonance (R = H × PC)** — which additionally requires
  sustained, phase-locked *oscillations* — is **not** a property of the critical
  point per se. It appears only when oscillatory drive is present, and then peaks
  in the **structured, near-critical-but-ordered** regime, collapsing in disorder
  (chaos) and absent in non-oscillatory avalanche criticality.

In one line: **harmonic/resonant structure is a signature of criticality across
three independent systems — it peaks at branching criticality, is generated from
noise at the reservoir's edge of chaos, and rises at the E/I synchronization onset.**

Across all three models the story is consistent:
- branching network: H peaks AT criticality (σ = 1.00 [0.97, 1.08]); R ≈ 0 (no oscillations to phase-lock).
- reservoir: driven by NOISE, harmonic structure is GENERATED at the edge of chaos (onset ρ ≈ 1.3, peak at ρ_c ≈ 1.57), and is distinct from generic peakedness (which keeps rising into chaos).
- E/I network: E↔I phase coupling rises from a non-zero asynchronous baseline (≈0.43 → ≈0.99, ΔPC ≈ +0.55) at the synchronization onset (g_c ≈ 1.0) and saturates.

## Can reservoir computing "model" resonance?
Yes, in a specific sense. Driven by structureless **noise**, a reservoir
**generates** harmonic structure once it reaches the edge of chaos (H_gain clears
0 at ρ ≈ 1.3 and peaks at ρ_c ≈ 1.57) — the tanh nonlinearity folds the emergent
collective oscillation into integer harmonics. It does *not* do this in the ordered
regime (where it merely filters the input), and the generation is **distinct from**
its computational optimum (memory capacity peaks at ρ ≈ 0.70). So RC is a genuine
*generative* model of resonance, but only near/beyond criticality — and resonance
is not a proxy for reservoir computation.

## Limitations and next steps
- **Quick-mode, few seeds** — the H-peak-at-criticality (Study 10) and the
  R-peak-below-ρ_c (Study 11) are suggestive but should be tightened with the
  `--paper` pass (more seeds, finer grids, confidence intervals).
- **The natural unifying model is an E/I balanced network** that exhibits *both*
  criticality (avalanches) *and* oscillations — there, phase-coupling resonance R
  (not just H) could be tested against criticality directly. The bare branching
  model cannot (no oscillations); the reservoir can but is not a spiking neural
  system.
- **Computation beyond linear memory** — information-processing capacity /
  nonlinear tasks may relate to resonance differently than linear memory does.
- **Real data** — long-range temporal correlations / 1-f and avalanche analyses
  on the EEG already in the suite could test whether resonance covaries with
  empirical criticality markers across brain states.

## Figures
- `figures/study9_reservoir.{png,pdf}` — memory capacity vs internal resonance vs ρ.
- `figures/study11_reservoir_criticality.{png,pdf}` — Lyapunov edge-of-chaos;
  resonance vs computation vs criticality.
- `figures/study10_criticality.{png,pdf}` — branching network: criticality
  markers + harmonicity vs σ.

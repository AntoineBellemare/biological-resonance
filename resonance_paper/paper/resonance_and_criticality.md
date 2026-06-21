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

- **Reservoir / echo-state network** — tunable via spectral radius ρ; can be
  driven by oscillatory input, so it supports genuine phase-coupling resonance.
- **Branching neural network** — the canonical model of neuronal criticality;
  tunable via branching ratio σ, with power-law avalanches at σ = 1. Scale-free
  but *not* oscillatory.

## What we found

### Reservoir (Studies 9, 11)
Locating the edge of chaos by a Lyapunov divergence estimate (ρ_c ≈ 1.57
[1.44, 1.68], n = 16 reservoirs):

| quantity | peaks at | regime |
|---|---|---|
| memory capacity (computation) | ρ ≈ 0.70 | ordered, below ρ_c |
| harmonic resonance R (driven) | no sharp peak; ~flat for ρ ≲ 1.5 | ordered |
| — R collapses for | ρ > ρ_c (chaos) | chaotic |

- Harmonic resonance R is **small and roughly flat across the ordered regime**
  (its argmax CI is wide — ρ ≈ 0.3 [0.2, 1.5] — i.e. there is *no* well-defined
  peak), and then **declines monotonically once the network passes the edge of
  chaos**: chaos destroys harmonic resonance. (The earlier quick pass that put a
  peak at ρ ≈ 1.3 was a few-seed artifact.)
- Memory capacity (computation) peaks in the ordered regime (ρ ≈ 0.70), below
  ρ_c — so **both** resonance and computation live below the edge of chaos, not
  at it.
- The reservoir does **not** amplify input harmonic resonance: internal/input R
  ≈ 0.8–0.9 (< 1) across the ordered regime and falls further in chaos. RC is a
  useful testbed for *how dynamics shape resonance*, but it neither generates nor
  amplifies it.

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

In one line: **harmonic structure tracks criticality; harmonic resonance lives in
the ordered regime and rises at the synchronization transition, collapsing in chaos.**

Across all three models the story is consistent:
- branching network: H peaks AT criticality (σ = 1.00 [0.97, 1.08]); R ≈ 0 (no oscillations).
- reservoir: R is highest in the ordered regime (flat, no sharp peak) and collapses past the edge of chaos (ρ_c ≈ 1.57 [1.44, 1.68]).
- E/I network: E↔I phase coupling rises from a non-zero asynchronous baseline (≈0.43 → ≈0.99, ΔPC ≈ +0.55) at the synchronization onset (g_c ≈ 1.0) and saturates.

## Can reservoir computing "model" resonance?
Mostly not. A driven reservoir does **not** amplify the harmonic resonance of its
input (internal/input ≈ 0.8–0.9, < 1, across the ordered regime; it falls further
in chaos), and its internal resonance is a readable but weak function of the
dynamical regime. Resonance is **not** a proxy for computational capacity either
(memory capacity peaks at ρ ≈ 0.70, where R is flat). So RC is a useful testbed
for *how dynamics shape resonance* — chiefly, that chaos destroys it — but it
neither generates, amplifies, nor is gated by reservoir computation.

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

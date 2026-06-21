# Resonance, reservoir computing, and criticality

*Synthesis of the dynamical-systems studies (Studies 9–11) addressing two
questions: (1) can reservoir computing model harmonic resonance, and (2) how does
resonance relate to criticality in biological/complex systems? All numbers are
from the moderate ("quick") pass; regenerate with `python -m
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
Locating the edge of chaos by a Lyapunov divergence estimate (ρ_c ≈ 1.94 here):

| quantity | peaks at | regime |
|---|---|---|
| memory capacity (computation) | ρ ≈ 0.9 | ordered |
| harmonic resonance R (driven) | ρ ≈ 1.3 | ordered, **approaching** criticality |
| — collapses for | ρ > ρ_c | chaotic |

- Harmonic resonance R **peaks in the ordered regime just below the edge of
  chaos**, then **collapses in chaos**. (Study 9's narrower sweep made R look
  monotonically rising; the extended sweep + located ρ_c corrects this.)
- R's peak is **distinct from** the computational (memory) optimum — they are not
  the same regime.
- The reservoir **amplifies** input harmonic resonance only mildly (~1.1×) and
  only near-critically — RC reflects/sharpens resonance rather than strongly
  generating it.

### Branching network (Study 10)
Sweeping σ through the critical point (criticality markers — susceptibility,
power-law avalanches — peak near σ = 1):

- **Harmonicity H peaks AT criticality** (σ = 1): the harmonic structure of
  population activity is a criticality signature.
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
  readout) gives a non-trivial signal: **E↔I phase coupling rises sharply at the
  onset and saturates at 1.0** once the network oscillates (0.61 → 1.0 across
  g_c). So phase-coupling resonance **switches on at the edge of synchronization**
  and marks the synchronized regime.

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

In one line: **harmonic structure tracks criticality; harmonic resonance tracks
the ordered approach to it and switches on at the synchronization transition.**

Across all three models the story is consistent:
- branching network: H peaks AT criticality; R ≈ 0 (no oscillations).
- reservoir: R peaks in the ordered regime just below the edge of chaos, collapses in chaos.
- E/I network: phase-coupling R switches on at the synchronization onset and saturates.

## Can reservoir computing "model" resonance?
Partially. A driven reservoir sharpens/amplifies the harmonic resonance of its
input (mildly, ~1.1×) in the near-critical ordered regime, and its internal
resonance is a readable function of the dynamical regime. It does **not** strongly
generate resonance, and resonance is **not** a proxy for its computational
capacity (memory peaks elsewhere). So RC is a useful testbed for *how dynamics
shape resonance*, but resonance is not a marker of reservoir computation.

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

# Plan — oscillatory-criticality network: test R (not just H) against m̂

**Goal.** Build one model that exhibits BOTH (a) avalanche/branching criticality (power-law
avalanches, branching ratio m̂→1) AND (b) sustained oscillations, so that the framework's
flagship R = H·PC is non-trivial *and* m̂ is well-defined at the same critical point. Then test
whether **R tracks proximity to criticality (m̂→1) directly** — making the criticality claim about
R, not only H.

## Why this is the right tier-up lever
Current models split the two requirements:
- branching network — has m̂, but non-oscillatory → R sits at the floor;
- reservoir — emergent oscillation, but not neural / no avalanches;
- Wilson–Cowan E/I (rate) — oscillations + a Hopf edge where R is placeable, but no avalanches,
  so R is tested against susceptibility, **not against m̂**.

So no current model lets R be validated against the *branching* criticality axis. A model where
avalanches and oscillations coexist closes that gap and reframes the paper from "H is the marker; R
needs oscillations (shown only at a Hopf edge)" to "**at a critical point that is simultaneously an
avalanche-criticality point and an oscillatory regime, R tracks m̂**."

**Reviewer points this addresses:** #2 (is H/R specific to criticality vs spectral shape — a model
where R co-locates with *independent* avalanche markers is direct evidence), #8 (negative controls /
specificity), #9 (align R with independent criticality indicators), and the standing "why is the
flagship R sidelined?" concern. (It does NOT address the EEG-side points #1/#3/#5/#6/#7 — those are
separate revision items.)

## The model (primary route): critical-oscillations E/I network
Lineage: **Poil et al. 2012** (J Neurosci — "critical-state dynamics of avalanches *and*
oscillations jointly emerge from balanced E/I") and **di Santo et al. 2018** (edge of synchronization:
scale-free avalanches emerge at the synchronization onset). This is the canonical class where
power-law avalanches and an emergent rhythm (alpha/gamma-like, with long-range temporal correlations)
coexist at one critical point.

Concrete reduced implementation (numpy, self-contained like the other studies):
- N units, ~75% excitatory / 25% inhibitory, sparse random connectivity.
- Probabilistic threshold units: integrate synaptic input, fire with a sigmoid/threshold prob,
  then refractory; E synapses excite, I synapses inhibit.
- **Control parameter** = a global synaptic gain (or E/I weight ratio) swept sub → critical → super.
- Outputs per run: population activity A(t) = spikes/bin (for avalanches/m̂) and an LFP-proxy =
  summed (or low-pass) synaptic current (for the oscillation and H/PC/R); also keep E and I
  population signals for an E↔I cross-resonance read (as in the Wilson–Cowan study).

## What to measure (per control value, n seeds → CIs)
**Independent criticality axis (none derived from H/PC/R):**
- branching ratio m̂ via the multistep-regression (Wilting–Priesemann) estimator on A(t);
- avalanche size/duration power-law fit (R², exponents τ, α) + the crackling-noise scaling relation;
- susceptibility (variance of A) and its normalized form (var/mean²);
- LRTC/DFA exponent of the oscillation-amplitude envelope (Poil's criticality hallmark);
- critical slowing (autocorrelation time).

**Resonance:** H, PC, R on the LFP-proxy, and the E↔I cross-resonance (H, PC, R) as in the
Wilson–Cowan study (the matrix-entry readout that works).

## Key tests
1. **R vs m̂ (the headline):** does R (and PC) peak at the critical control value, **co-located with
   m̂→1** and the avalanche/LRTC peaks? Use the formal co-location bootstrap already built for the
   branching study (peak-location difference CI) + per-seed Spearman(R, −|m̂−1|) with Wilcoxon.
2. **H, PC, R together vs the critical point** (one panel) — show all three peak there, with R now
   genuinely non-trivial.
3. **Cross-check independent markers co-locate:** m̂, power-law R², LRTC, susceptibility(norm),
   critical slowing all optimize at the same control value (eigenvalue / order-parameter scaling
   where tractable, per reviewer #9).

## Specificity / negative controls (reviewer #2, #8)
- **Matched-spectrum non-critical oscillator:** a signal with the *same* PSD (peak frequency, slope,
  power) but generated off-criticality (e.g., a driven linear oscillator + colored noise) → R should
  NOT show the m̂ co-location. Establishes R isn't just tracking "there's an oscillation."
- **Power-matched sub- vs super-critical points:** compare control values on either side of the
  critical point with matched oscillation power but different m̂ → R should differ with m̂, not power.
- **Surrogates:** phase-randomized and amplitude-adjusted (AAFT) surrogates of the population signal
  → R collapses; confirms R needs genuine phase structure, not the spectrum.
- **AR / colored-noise controls** with matched autocorrelation (reviewer #8) → no R peak.

## Paper integration
- New figure (insert after the current E/I figure): (A) avalanches + oscillations coexist (power-law
  + PSD peak at the critical control value); (B) m̂ / LRTC / susceptibility locate the critical point;
  (C) **H, PC, R all peak at m̂→1** with the co-location stat; (D) specificity — R co-locates with m̂
  but not in the matched-spectrum control.
- Reframe the model narrative: H is the general (even non-oscillatory) criticality marker; **in the
  oscillatory-criticality regime, R itself tracks m̂** — the framework's flagship earns a direct
  criticality claim. Update abstract/discussion accordingly.
- This is a model result (construct validity), kept separate from the EEG evidence — which also
  answers the reviewer's "separate discovery from confirmation."

## Risks & fallbacks
- **Main risk:** tuning a reduced model so avalanches + a clean oscillation + a tunable critical
  point all coexist (this coexistence is the scientific content of Poil 2012; it takes iteration).
- **Fallback 1:** use **Brian2** (spiking) for a faithful Poil/di-Santo network if the numpy reduced
  model won't produce clean avalanches+oscillations (adds a dependency, but well-trodden).
- **Fallback 2 (lightest):** extend the existing **Wilson–Cowan** model to a spatially-extended /
  many-column version (di Santo style), estimate m̂ from the summed field, and show the Hopf edge
  where R/PC peak **coincides with m̂→1** — reuses Study 12 machinery; weaker "avalanches" but a fast
  route to "R co-locates with m̂."
- Decision: attempt the reduced critical-oscillations network first; if avalanche power-laws are not
  clean after a bounded tuning budget, drop to Fallback 2 (Wilson–Cowan + m̂ co-location), which still
  delivers the headline (R tracks m̂) with less novelty.

## Execution steps
1. Implement the reduced E/I critical-oscillations network (`study26_critical_oscillations.py`);
   verify it produces avalanches (power-law) AND a spectral peak at some control value.
2. Add the criticality estimators (m̂ MR-estimator, avalanche fits, LRTC, susceptibility) — reuse
   `criticality.py` / `crit_resonance.py` where possible.
3. Sweep the control parameter × seeds; compute the independent markers + H/PC/R; store per-seed.
4. Co-location stats (R vs m̂, PC vs m̂, H vs m̂) using the branching-study bootstrap + per-seed Spearman.
5. Specificity controls (matched-spectrum oscillator, surrogates, power-matched points).
6. Build the figure; write the prose; integrate into `criticality_paper` (new model section + reframe).
7. Recompile; commit.

## Effort / gate
Medium–large (model build + tuning is the uncertain part; analysis/figure reuse existing machinery).
**Go/no-go gate after step 1:** only proceed if the network shows a clean coexistence of power-law
avalanches and a spectral peak with a tunable critical point; otherwise switch to Fallback 2.

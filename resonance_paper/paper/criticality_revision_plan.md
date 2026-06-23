# Criticality paper — major-revision plan (response to reviewer)

Maps every major reviewer point to a concrete action, with files/sections and effort (S/M/L).
The oscillatory-criticality network (separate detailed doc: `oscillatory_criticality_plan.md`) is
Workstream 6 here — the one new experiment that both lifts the paper a tier and answers #2/#8/#9.

## Triage
- **Essential for resubmission (do regardless of venue):** W1 definitions (#1), W2 EEG methods (#5),
  W3 aperiodic + spectral controls (#6,#7), W7 reframing/honesty (#2,#4,#10, minors), W8 prospective
  EEG-observable logic (#3), W9 housekeeping (#11).
- **High-value new science:** W4 model negative controls (#8), W5 Wilson–Cowan independent indicators
  (#9), W6 oscillatory-criticality network (R vs m̂; the tier-up).

---

## W1 — Formal definitions of H, PC, R (#1, essential, writing)
Write explicit equations in Methods so the paper is self-contained (no delegation to biotuner). Cover:
PSD preprocessing; the pairwise harmonic kernel (harmsim $=(p+q-1)/(pq)$ on the reduced fraction
$p/q$ of $f_j/f_i$ via `limit_denominator(max_denom)`); the ratio set / max-denominator + tolerance;
PSD power-weighting; reduction pairwise→$H(f)$; $H_{\max}$; PC (n:m PLV and surrogate-z); $R=H\cdot$PC;
normalization across simulations and EEG. **Source:** transcribe the biotuner kernels to equations.
Effort **M**.

## W2 — EEG methods + criticality-estimation detail (#5, essential, writing)
Add a full subsection: datasets (Sleep-EDF Sleep Cassette n=8 + Sleep Telemetry n=12; PhysioNet
eegmmidb; Chennu propofol; ds004541), 30 s window counts, sampling rates, montage (Fpz-Cz/Pz-Oz;
frontal h_idx), filtering, GFP computation, m̂ via the multistep-regression (Wilting–Priesemann)
estimator with the **lag range** + stationarity handling, DFA, per-(subject,state) aggregation, stage
balancing, REM handling, and that stage labels come from the dataset hypnograms. **Source:** surface
what is already in `study14/16`, `datasets.py`, `criticality.py`. Effort **M**.

## W3 — Aperiodic procedure + spectral/confound controls (#6, #7, essential, compute)
- State the aperiodic removal exactly (2-parameter power-law least-squares subtraction, **not** FOOOF)
  and show robustness to its band/settings.
- **Spectral specificity battery** — recompute H under: raw spectrum; aperiodic-removed;
  phase-randomized; amplitude-matched (AAFT); slope+power-matched but peak-locations randomized;
  peaks-preserved but frequency-labels shuffled. Show H tracks harmonic organization, not slope/power.
- **Confound battery for the in-vivo H_aval recovery:** partial correlations controlling delta power,
  spectral slope, total power, peak frequency, #detected peaks, SNR, and waveform sharpness /
  non-sinusoidality (we already have the slow-power partial; add the rest). Explicitly separate
  harmonics-from-waveform-shape vs harmonics-from-multi-scale-dynamics. Effort **M–L**.

## W4 — Model negative controls / specificity (#8, compute)
Add controls where H/R should NOT peak: AR process (matched autocorrelation); colored noise (matched
slope); non-critical oscillator with a harmonic waveform; shuffled/surrogate branching; subcritical
process with imposed harmonic peaks; chaotic broadband system. Demonstrate H peaks at the critical
point, not at matched-spectrum non-critical signals. Effort **M**.

## W5 — Wilson–Cowan independent indicators (#9, compute + writing)
Justify normalized susceptibility (var/mean²) **a priori** (define before comparing to H/R). Add
indicators independent of the amplitude dynamics: Jacobian-eigenvalue zero-crossing of the
deterministic system, order-parameter scaling, and independently-measured critical slowing. Show H/R
co-locate with these (not circularly with the one chosen marker). Effort **M**.

## W6 — Oscillatory-criticality network: R vs m̂ (the tier-up; #2/#8/#9 + "R sidelined")
A critical-oscillations E/I network (Poil 2012 / di Santo 2018 lineage) where avalanches (m̂) and
sustained oscillations coexist, so **R is validated directly against the branching ratio**. Full plan
in `oscillatory_criticality_plan.md`. Effort **L** (model tuning is the risk; Wilson–Cowan-extension
fallback). Single biggest upgrade — and it doubles as the strongest specificity evidence.

## W7 — Reframing & honest claims (#2, #4, #10, minors 1–14, essential, writing)
- Title → "Spectral harmonicity as a **candidate** observable of proximity to criticality…".
- Abstract → 5-beat honest arc: strong model construct validity; failed naive raw-EEG transfer;
  **conditional** scale-free recovery in sleep; boundary nulls in sedation/anaesthesia; needs further
  validation. Drop "single-channel, model-free window onto how close cortex sits to criticality".
- Position H explicitly as interpretation (2)/(3): "tracks model-defined criticality; a candidate
  marker," not a general diagnostic.
- Use "model-matched observable" not "the correct observable"; qualify "single-channel" (GFP is
  multi-channel-constructed); "barely traverse criticality" → "according to our estimator".
- Standardize notation ($H_{\mathrm{full}}$, $H_{\mathrm{aval}}$, $H_{\max}$…); define
  criticality-proximity once ($-|\hat m-1|$); report **CI + n + exact test** for every EEG effect.
  Effort **S–M**.

## W8 — Prospective / held-out logic for the EEG observable (#3, framing + one analysis)
- Argue the scale-free observable is **model-derived**: in all three models the criticality signal is
  the non-oscillatory population/avalanche activity A(t); GFP is its in-vivo analogue. State this in
  Methods **before** the reversal, so it reads as theory-driven, not post-hoc.
- Report **all** tested observables (raw, GFP/scale-free, low/high band) including those that failed.
- Frame the Sleep-EDF Telemetry cohort as the **fixed-pipeline independent replication** (pipeline
  set on the SC cohort, applied unchanged to ST). Soften to "conditional recovery". Effort **S–M**.

## W9 — Housekeeping (#11, essential, S)
Fix the corrupted arrows/symbols (the slow-band-control rendered as plain `->` in places — make them
proper math `\rightarrow`); **remove the "verify references" note** and check every reference; add a
formal **Code & Data Availability** section; proofread all notation, captions, and stats.

---

## Recommended sequence
1. **Quick, high-impact (S):** W7 reframe + W9 housekeeping + W8 prospective framing — these alone
   change how the whole paper reads and answer the "overstated / post-hoc" thrust.
2. **Writing (M):** W1 definitions + W2 EEG methods — make it self-contained and reproducible.
3. **Compute controls (M):** W3 spectral/confound + W4 model negatives + W5 Wilson–Cowan indicators —
   the specificity evidence the reviewer asks for.
4. **Big science (L):** W6 oscillatory-criticality network — the differentiator that makes it about R.

**Scope note.** W1–W5 + W7–W9 constitute a thorough major-revision response sufficient for the
current venue tier (eLife / Communications Physics / Network Neuroscience). **W6** is the lever to
push a tier higher and simultaneously the strongest answer to the specificity critique — recommended
if aiming above the current tier, optional otherwise.

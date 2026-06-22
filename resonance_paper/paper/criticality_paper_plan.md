# Criticality paper — plan (validated 2026-06-22)

**Paper 2 of 3** (Methods → **Criticality** → Consonance). Discovery paper.
Scope decision: **strengthen the real-data leg** (medium effort) — not the large
new-oscillatory-model upgrade. Mid-to-good venue (Communications Physics / eLife / PRX Life).

## Thesis
Spectral harmonicity **H** is a single-channel, model-free observable of **proximity to
criticality**, validated against the branching ratio m̂ across three model systems and
recovered in human EEG — *once read off the correct (scale-free) observable*. R = H·PC is the
oscillation-gated companion (only placeable where sustained oscillations coexist with the
transition); the title quantity is **H**.

## Narrative arc
1. **Intro** — criticality as an organizing principle; its observables (m̂, DFA, susceptibility)
   need population recordings / long stationarity / many channels and disagree. Ask: is a single
   spectral property (H) itself an observable of proximity to criticality? Pre-register H as the
   candidate, R as the oscillation-gated companion.
2. **Construct (brief)** — define H/PC/R (cite methods paper). Falsifiable prediction: H tracks
   m̂→1 / susceptibility peak in models; R only moves where oscillations + criticality coexist.
3. **Models** — H maxed at / generated approaching criticality in 3 independent systems
   (branching net; noise-driven reservoir; E/I Wilson–Cowan). Synthesis: H is the marker, R needs oscillations.
4. **Real data — the tension** — H/R discriminate state but don't beat band power; the naive test
   **reverses** (raw-EEG H highest in subcritical deep N3, ρ=−0.24, p=0.008).
5. **Resolution** — the reversal is an observable-choice artifact: on the scale-free population
   signal (in-vivo analog of avalanche A(t)), within-state, H **positively** tracks m̂-proximity
   (ρ=+0.11, p=0.03) — recovering the model, and it is **H not R**, as predicted.
6. **Discussion** — H as a cheap single-channel proxy for proximity to criticality; H/R division of
   labor; why observable choice flips the sign; limitations; the named (unbuilt) oscillatory-criticality
   model as future work.

## Figures (6)
- **Fig 1** *(new)* — the falsifiable law schematic (transition; m̂→1 / susceptibility peak; H co-located peak; R only with oscillations). Inset: H phase-blind (Study 17), R=H·PC (Study 23).
- **Fig 2** *(Study 10, needs co-location stat)* — branching net: susceptibility + m̂ vs σ; power-law slope→−1.5; H_max inverted-U peak σ=1.0 [0.97,1.08]; R at 1e-5 floor; **(D) formal H-peak↔criticality-peak co-location test**.
- **Fig 3** *(Study 11, sharpen grid)* — reservoir: Lyapunov edge ρ_c=1.57; H_gain over noise baseline (onset ρ≈1.3, peak +0.15 [0.06,0.25]); dissociation from peakedness + memory capacity (Study 9 as MC reference / negative control).
- **Fig 4** *(Study 12, normalize + null)* — E/I net: order param + normalized susceptibility; E↔I PC 0.43→0.99 (ΔPC=+0.55); R/H peak near onset; surrogate null on cross-PC.
- **Fig 5** *(Studies 13–15, ready)* — real EEG: LOSO state decoding (resonance vs band power vs markers; H/R don't beat band power); by-state H_max showing the reversal (N3 highest; ρ=−0.24).
- **Fig 6** *(Study 16, strengthen — load-bearing)* — resolution: mechanism schematic; H_full (raw, ρ=−0.24) vs H_aval (scale-free, ρ=+0.11) sign flip; boundary cases (sedation/deep-GA don't traverse m̂).

## Solid now
- Study 10 (branching, paper-grade): H peaks at σ=1.0 co-located with susceptibility/m̂.
- Study 11 (reservoir, paper-grade): noise→harmonicity generation near edge of chaos, dissociated from memory.
- Study 12 (E/I, paper-grade): E↔I PC rises at sync onset (ΔPC=+0.55).
- Study 16 (sleep): within-state scale-free H tracks m̂-proximity (ρ=+0.11, p=0.03) + H_aval/H_full sign dissociation.
- H-vs-R division of labor; honest "H/R don't beat band power" framing.

## Execution checklist (chosen scope = strengthen real-data leg)
1. **[Study 16, load-bearing]** Add **propofol (Study 13 data) to the scale-free analysis** (loader exists, no JSON yet); compute H_aval/R_aval there. Formally test the **paired per-subject ρ difference** H_full vs H_aval (one paired test, not two separate). Add a **partial correlation** controlling slow-band power / 1-f slope. Apply multiple-comparison correction across primary correlations. Report within-state positive as the headline recovery across **≥2 datasets**.
2. **[Study 10]** Add the **co-location statistic**: correlate H_max vs m̂/susceptibility across σ with p-value + bootstrap test that the H peak location coincides with the criticality peak.
3. **[Study 11]** Sharpen the ρ grid in **1.15–1.8** to localize the H_gain peak (CI now wide [1.5,3.0]); permutation test on MC-peak (≈0.7) vs H-peak (≈1.5) separation; distinguish integer-harmonicity from generic spectral narrowing.
4. **[Study 12]** Normalize susceptibility (var/mean²); add a **surrogate null** on cross-PC (E,I are coupled by construction); refine g-grid 0.55–0.7; optional di-Santo-style power-law fluctuation check.
5. **[axes]** Report **m̂ and DFA side by side** per dataset; document the **DCC drop** (failed to minimize at σ=1 in ground truth). Keep only validated axes (m̂, DFA).
6. **[leftovers]** Study 9 = one-line **negative control** ("resonance ≠ generic marker of reservoir computational regime") or drop; Studies 13/15 = honest **boundary/null** cases (sedation barely traverses; deep-GA m̂≈0.98 both states), not confirmations.

## Honest negatives (state up front)
- It is **H, not R** that works (R at 1e-5 floor in non-oscillatory criticality; R_aval≈0 in vivo).
- The naive in-vivo test **reverses**; recovery needs the scale-free observable + within-state controls; effect is **modest (ρ=+0.11)**.
- Propofol + deep-GA **do not traverse** criticality → boundary nulls; real-data leg leans on sleep (+ propofol once added).
- H/R **do not beat band power** as classifiers.
- Estimators disagree (m̂ vs DFA flip sign in deep-GA); DCC dropped.

## Open risks
- In-vivo recovery rests on few datasets; if propofol scale-free comes back null, the leg weakens to "suggestive."
- ρ=+0.11 could be a slow-power confound — the partial correlation (step 1) is the make-or-break test.
- Observable-choice argument could read as a degrees-of-freedom escape hatch unless justified by the model (A(t) is non-oscillatory by construction) and the partial control is shown.

## Deferred (out of chosen scope)
- The **new oscillatory-criticality network** where R is directly testable against m̂ (large; the single biggest scientific upgrade — flagged as future work / stretch if reviewers push).
- Deep-GA agent stratification + burst-suppression windows.

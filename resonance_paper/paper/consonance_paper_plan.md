# Consonance paper — plan (validated 2026-06-22)

**Paper 3 of 3** (Methods → Criticality → **Consonance**). Discovery/demonstration paper.
Scope decision: **full paper** — add the brain-behavior link + cross-paradigm synthesis (not a short report).
Venue: eNeuro / European Journal of Neuroscience (the Andermann FFR dataset's home); NeuroImage if the
cross-paradigm/methods angle is sharp enough.

## Thesis
One descriptor (harmonicity **H**, with PC/R) recovers consonance-related structure across **three
paradigms** — just-intonation chords, SSVEP intermodulation (reframed as n:m coupling), and the real
brainstem **FFR** — and in the FFR the structure is **neural, not acoustic** (missing-fundamental dyads
with zero stimulus energy below ~640 Hz still show consonant>dissonant harmonicity in the silent band).

## Honesty hinge (must lead with this)
The FFR-consonance phenomenon **reproduces Bidelman & Krishnan / Andermann et al. 2026** — it is the
*validation anchor*, not the discovery. Novelty = (1) a single descriptor that **ports across** acoustic,
frequency-tagging, and EEG paradigms; (2) **intermodulation reframed as n:m phase coupling**; (3) the
**missing-fundamental neural-generation** logic; (4) — if built — a **brain-behavior** (musicianship) link.

## Narrative arc
1. Hook: consonance↔ratio-simplicity measured by disparate methods; propose one descriptor across paradigms.
2. The descriptor (brief; leans on methods paper): H phase-blind (Study 17); R interpretive not detective; single-signal R dilutes to ~0 (stated, not hidden).
3. **Paradigm 1 — acoustic** (Study 20): H tracks chord consonance (ρ=−0.71/−0.73); combination tones sharpen.
4. **Paradigm 2 — driven cortex** (Study 19): single-flicker H at ceiling (sanity); two-flicker IM index rises with nonlinearity (ρ=+0.62); **reframe IM (n·f1±m·f2) as the n:m phase-coupling face of the descriptor**.
5. **Paradigm 3 — real brainstem** (Study 20b): consonant>dissonant harmonicity; load-bearing **CI>DI in the stimulus-silent band**; own-fundamental reconstruction; full adversarial control stack. Frame as reproducing+extending Bidelman & Krishnan.
6. Synthesis: same descriptor, same direction, across all three → cross-paradigm generality.
7. Conclusion: consonance as a paradigm-invariant harmonic-resonance signature; SR explicitly out of scope.

## Figures (6)
- **Fig 1** *(ready)* — descriptor + consonance logic (Study 17 panels: H phase-blind; R interpretive).
- **Fig 2** *(Study 20, upgrade stats)* — acoustic: H vs chord consonance, linear vs nonlinear.
- **Fig 3** *(Study 19, add PC/R)* — IM-as-n:m-coupling in driven cortex; (C) new panel: PC/R of the driven pair.
- **Fig 4** *(Study 20b, ready)* — FFR more harmonic for consonant dyads; band-split leakage; missing-fundamental reconstruction; control stack. Caption cites reproduction of Bidelman & Krishnan.
- **Fig 5** *(new)* — cross-paradigm synthesis: chord-H, SSVEP-IM, FFR-harmsim on a common standardized-effect axis.
- **Fig 6** *(new, biggest upgrade)* — brain-behavior: per-listener FFR consonance advantage vs musicianship/behavioral sensitivity.

## Solid now
- **Study 20b** — paper-grade, bulletproof (n=36, real EEG, full control stack, Bonferroni-surviving; missing-fundamental CI>DI robust). Headline anchor.
- Study 17 construct (H phase-blind) — justifies H as the single-channel consonance observable.
- Study 19 two-flicker IM rises with nonlinearity (ρ=+0.62) — supporting bridge.
- Study 20 direction correct (ρ=−0.71/−0.73).

## Execution checklist (chosen scope = full paper)
1. **[Study 20b, biggest upgrade]** Compute the **brain-behavior correlation**: per-listener FFR consonance advantage vs musicianship / behavioral consonance sensitivity (`Buttonpress.mat` / `sample.mat` already loaded in `run()` but never analyzed). This is what turns reproduction into novelty.
2. **[Study 19]** Compute and report **PC and R** between the two driven SSVEP components (promised in docstring, absent from JSON); add **IM-lattice-matched ratio pairs** to decouple ratio simplicity from frequency-coincidence (current simple-vs-complex rests on the 2:3/6:9 Hz outlier); add surrogate nulls.
3. **[Study 20]** Paper-grade rerun + **stats**: p-value + bootstrap CI on the Spearman; test that the nonlinear "sharpening" (+0.024) exceeds noise; expand the chord bank; **validate against an external human consonance-rating dataset** (avoid Tenney-height circularity).
4. **[new]** Build the **cross-paradigm synthesis** figure/analysis (Fig 5): chord-H, SSVEP-IM, FFR-harmsim on a common standardized-effect axis.
5. **[honesty]** Foreground the **missing-fundamental CI>DI** as primary; disclose CC>DC fragility (attenuates at large N_PEAKS, p=0.098 at n=20). Drop the broadband recon-fraction framing (null p=0.13). Soften/test the active-vs-passive "attention" claim (passive effect is actually larger).
6. **[positioning]** Explicit novelty-boundary paragraph: what reproduces Bidelman & Krishnan / Andermann vs what is new.

## Honest negatives
- FFR consonance **reproduces** prior work — lead with unification/IM-reframing/cross-paradigm/brain-behavior, not "the brainstem prefers consonance."
- R does **not** out-detect PC (Study 17 AUC R 0.66 < PC 0.78); single-signal R ~0 → only H informative for single-channel/acoustic.
- Study 20 "nonlinearity sharpens" is tiny (+0.024), untested (n=10).
- Study 19 "richer IM for simple ratios" confounded by IM-lattice coincidence.
- Brain-behavior link (the key novelty) is currently **unimplemented**.

## Open risks
- Without the brain-behavior link OR a strong cross-paradigm synthesis it collapses to a short report.
- Circularity in Study 20 (Tenney height = ratio simplicity = what H rewards) → needs external ratings.
- Each paradigm is one dataset/model; "cross-paradigm" is qualitative (same direction), frame carefully.

## Stochastic resonance — SHELVED (not a 4th paper)
Studies 18/18b/18c: 18b go/no-go = NO (`stands_alone=false`; complex>simple ran the wrong direction in
both models); 18c rescue failed (only textbook spectral SR survived). At most a one-line bounded-mechanism
caveat in the methods/criticality paper ("noise does not generically enhance harmonic locking"). Do **not**
include in consonance.

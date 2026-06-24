# Plan — Sound measurement of polyrhythmic (n:m) phase coupling

**Thesis (one sentence).** Before any biological claim, validate that biotuner's phase-coupling
spectrum *soundly retrieves* polyrhythmic (n:m, n≠m) phase locking from controllable ground truth —
diagnose where it fails, fix the code, and show the corrected metric matches an oracle. This is a
self-contained methods paper.

---

## 0. Motivating evidence (already run — `study38` micro-test, 2:3 lock, f1=10 / f2=15)

| config | coupled Δφ | uncoupled | Δ (detection) |
|---|---|---|---|
| default `nm_plv` + `binary` + `stft` | 0.215 | 0.301 | **−0.086** (anti-detects) |
| "sound" `nm_plv_canonical` + `fraction` + `hilbert` | 0.065 | 0.024 | +0.040 (weak) |
| ORACLE (handed 2:3, Hilbert n:m PLV) | **0.988** | 0.229 | **+0.759** |

The lock is genuinely strong (oracle Δ=0.76) yet the framework under-detects it by ~20× (best config)
or anti-detects it (default). The gap is the paper.

---

## 1. Source audit — five concrete soundness defects (with code refs)

1. **Convention defect (CRITICAL).** `nm_plv` (the DEFAULT `coupling_metric`) applies `n·φ_i − m·φ_j`
   with the (n,m) the ratio kernels return under the *legacy* convention `ratio=f_j/f_i≈m/n`. Per
   `coupling.py` docstring this "measures STFT-phase-progression coherence rather than true n:m phase
   locking" — it is non-stationary even for a perfect lock. Only `nm_plv_canonical` swaps back to the
   Tass form `n·φ_i − m·φ_j = const`. → the default cannot retrieve n:m. (`resonance/coupling.py:48,136`)

2. **Estimator defect.** Default `phase_estimator="stft"` is STFT-bin phase — "not true oscillation
   phase," and "misses n:m phase locking that hilbert recovers" (its own docstring; Study 5).
   (`resonance/phase_estimators.py:30,67`; default `orchestrator.py:67`)

3. **Ratio-kernel coverage + fallback defect.** Default `binary` kernel has `max_nm=3`: it can test
   1:2, 2:3, 3:2 … but NOT 3:4, 4:5, 3:5, 5:4 (any index >3), and `fallback_to_1_1=True` silently
   **coerces unmatched pairs to 1:1**, mislabelling a true 3:4 polyrhythm as 1:1 and testing the wrong
   ratio. `fraction` (limit_denominator) handles any ratio. (`resonance/kernels_ratio.py:52`)

4. **Complexity-weight-on-the-entry defect.** The PC matrix entry is `W·PLV`, and the `fraction`
   kernel's `W = exp(−β·log2(n·m))` ⇒ for 2:3, W=exp(−log2 6)≈**0.167**. So a *perfect* 2:3 lock is
   reported as ≤0.167 — the Tenney-height penalty (a *consonance* weight) is multiplied into the
   *coupling* statistic, crushing detection. Detection must read the **unweighted** PLV; the weight
   belongs to the resonance/consonance reading, not the coupling test. (`kernels_ratio.py:181`,
   `coupling.py:251`)

5. **Cross-resonance phase-path gap.** With `phase_estimator="hilbert"` requested, the corrected
   config still recovers only PLV≈0.4 at the entry vs the oracle's 0.99 (bandpass+Hilbert at the known
   ratio). ⇒ `compute_cross_resonance` likely does not realize a clean bandpass+Hilbert n:m phase at
   the targeted bins. Must verify/fix the phase path so the framework can reach oracle parity.
   (`harmonic_connectivity.compute_cross_resonance`)

Plus two design questions the paper should settle:
- **Power-weighting of the reduced PC spectrum** (`reduce_matrix_to_spectrum`: `p_i·Σ_j Φ_ij·p_j`)
  mixes power into a phase statistic and dilutes a low-power coupled partner. For detection, use the
  **targeted entry**, not the reduced spectrum. (`coupling.py:271`)
- **Surrogate regime.** Within-signal IAAFT may preserve a stationary lock (cf. study29 null); the
  cross-signal IAAFT-one-channel null (Study 5) is the clean one. Pin the correct null for polyrhythm.

---

## 2. Code revisions (make the metric sound) — `biotuner.resonance` + `harmonic_connectivity`

- **R1 — targeted polyrhythm detector API.** Add `detect_nm_coupling(a, b, sf, ratios, *, metric,
  bandwidth, n_surrogates, null)` returning, per ratio, the **unweighted** n:m PLV (+ pli/wpli/rrci)
  on bandpass+Hilbert phase, with surrogate z / rank-p. This is the clean instrument the paper
  validates and the one users should call for "is there n:m coupling at ratio r?" (matches the oracle).
- **R2 — separate coupling from complexity weight.** In the cross/within results, expose the raw
  metric value alongside `W·value`, so detection ≠ consonance weighting. Document clearly.
- **R3 — sound defaults / guard for n:m.** Make `fraction` + `nm_plv_canonical` + `hilbert` the
  documented n:m configuration; emit a warning when `nm_plv` (legacy) is used for n≠m. Fix
  `fallback_to_1_1` so non-matching pairs are *excluded* (W=0), not mislabelled 1:1, in detection mode.
- **R4 — fix/verify the cross-resonance Hilbert n:m path** so the targeted entry reaches oracle parity
  on clean locks.
- **R5 — regression tests** asserting: perfect n:m lock ⇒ detector ≈1 at true ratio, ≈0 at false
  ratios, for 1:2, 2:3, 3:4, 3:5, 4:5; uncoupled ⇒ at null.

(All changes additive / behind options; preserve bit-exact legacy path for prior papers.)

## 3. Ground-truth generator (`signals` / new `polyrhythm.py`)

Genuine n:m phase-locked oscillator pairs (not just summed sinusoids): forced/Kuramoto n:m model with
controllable **coupling strength κ** (independent→perfect lock), **phase offset**, **amplitude ratio**,
**frequency drift / instantaneous-frequency jitter**, **observation SNR**, **duration**, and
**non-sinusoidal waveform**. Two emission modes: within-signal (sum, one channel) and cross-signal
(two channels). Uncoupled PSD-matched control per condition.

## 4. Soundness test battery (the Results)

- **T1 Detection × specificity.** Ratio confusion matrix: for each true ratio in {1:1,1:2,2:3,3:4,3:5,
  4:5,1:3,2:5,3:4…}, PC at every candidate ratio ⇒ peaks on the diagonal, off-diagonal at null.
- **T2 Dose–response.** PC vs κ (monotone? detection threshold in κ?).
- **T3 Robustness sweeps.** SNR, amplitude asymmetry, frequency drift, phase-lag, duration ⇒ detection
  AUC curves.
- **T4 Convention ablation.** `nm_plv` vs `nm_plv_canonical` ⇒ the default-fails-vs-sound demonstration.
- **T5 Kernel coverage.** `binary(max_nm=3)` vs `fraction` ⇒ fraction retrieves >3:3; binary mislabels.
- **T6 Estimator.** `stft` vs `hilbert` ⇒ Hilbert retrieves, STFT misses.
- **T7 Metric comparison.** plv / pli / wpli / rrci on polyrhythm: sensitivity, 0-lag behavior for n≠m.
- **T8 Surrogate calibration.** FPR at α under the correct null (uncoupled n:m-frequency pair);
  within- vs cross-signal null; permutation vs Gaussian-z (reuse the methods-paper calibration result).
- **T9 Weighting / reduction.** targeted entry vs W·entry vs reduced power-weighted spectrum.
- **T10 Oracle parity.** corrected detector vs the handed-ratio oracle ⇒ does the fix close the
  0.04→0.76 gap? (the headline figure).
- **T11 (stretch) External reference.** compare to tensorpac / pactools n:m PLV on identical ground
  truth, if importable.

## 5. Paper structure (focused, self-contained)

Intro (n:m / polyrhythm; why soundness; prior pitfalls — Tass; Scheffer-Teixeira & Tort; Aru;
Dellavale) → Methods (metric, configs, ground-truth model, surrogate) → Results (T4/T5/T6 = the three
defects; T1/T2/T3 = detection/robustness of the corrected detector; T7 metric; T8 calibration; T10
oracle parity) → Discussion (recommended config; corrected defaults; within vs cross scope; honest
limits — real macroscopic EEG n≠m is null, cite the companion finding). Title direction: *"Sound
retrieval of polyrhythmic phase coupling: validating and correcting an n:m coupling spectrum against
ground truth."*

## 6. Sequencing

1. R3+R1+R4 (sound detector + defaults + cross-resonance phase fix) → R5 tests.
2. §3 generator.
3. T4/T5/T6 (defect demonstrations) + T10 (oracle parity) — the core.
4. T1/T2/T3/T7/T8/T9 — characterization.
5. Figures + draft.

Out of scope (deliberately): real-EEG discovery (the companion null stands), H/R headline, the
methods-paper reframe (paused).

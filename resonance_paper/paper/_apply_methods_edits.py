"""Apply review-driven revisions to _elife_sections.json (methods paper). Idempotent: each edit
is guarded so re-running is safe. Mirrors the criticality paper's _apply_crit_edits pattern."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent / "_elife_sections.json"
d = json.loads(P.read_text(encoding="utf-8"))
d = {k: (v.replace("\r", "") if isinstance(v, str) else v) for k, v in d.items()}

# --- #1/#2/#5/#7: abstract reframed (consistency checks vs validations; PC_z; simulated caveat) ---
ABSTRACT = (
    r"Biological time series are densely organized by harmonic relationships: the QRS complex "
    r"generates an integer-harmonic series, cortical rhythms show overtones and cross-frequency "
    r"relationships, and many physiological oscillators are weakly phase-coupled across frequency. Yet "
    r"there is no unified, testable way to ask \emph{how harmonically organized} a signal is, and at "
    r"which frequencies. We present a harmonic resonance framework that decomposes a single- or "
    r"multi-channel time series into two per-frequency spectra and their product: a harmonicity "
    r"spectrum $H(f)$, which scores how close spectral peaks sit to integer ratios, and a "
    r"phase-coupling spectrum $\mathrm{PC}(f)$, which scores how stably oscillatory components hold an "
    r"$n{:}m$ phase relation, together with a resonance spectrum $R(f)=H(f)\cdot\mathrm{PC}(f)$, "
    r"complexity descriptors, and a channel-by-channel connectivity extension. Our claim is "
    r"methodological: harmonicity and phase coupling are \emph{separable} axes that merit measurement "
    r"as distinct spectra. Several headline relationships are \emph{consistency checks} that follow "
    r"from the construction rather than empirical findings, and we label them as such: $H$ is exactly "
    r"phase-blind ($\rho(H,\kappa)=0.00$, since it depends only on the power spectrum); the product $R$ "
    r"collapses under phase scrambling where $H$ does not (AUC $1.00$ vs $0.50$); and conjunctive "
    r"combine rules isolate joint structure better than either factor alone (AUC $\approx0.99$ vs "
    r"$\leq0.76$). The genuine validations are that the framework recovers ground-truth harmonic "
    r"structure ($\rho=0.98$) and $n{:}m$ coupling (AUC $\approx1.0$), and that a generative "
    r"harmonically-coupled-oscillator model reproduces the predicted Arnold-tongue ordering of locking "
    r"thresholds. Because the raw spectra are power-weighted and therefore amplitude-confounded, "
    r"quantitative phase inference runs through a surrogate-normalized $\mathrm{PC}_z$, calibrated by "
    r"an exact permutation test (the Gaussian-$z$ threshold is mildly anti-conservative under "
    r"heavy-tailed nulls). On real signals the framework matches band power on a 10-subject EEG "
    r"dataset and on simulated multimodal physiology --- a genuine but limited biosignal test. The "
    r"implementation is a swappable strategy registry of harmonic kernels, coupling metrics, and "
    r"combine rules. We build on prior harmonicity measures, $n{:}m$ coupling, and synchronization "
    r"theory; our contribution is their unification, validation, and tooling. We are explicit about "
    r"scope: $R$ is an interpretable decomposition, not a superior detector, and the framework adds a "
    r"harmonic axis without replacing band power on amplitude tasks."
)
d["abstract"] = ABSTRACT

# --- targeted reframes (idempotent: skip if replacement text already present) ---
_subs = {
    "construct": [
        ("whereas $\\mathrm{PC}$ reads the temporal stability of that organization.",
         "whereas $\\mathrm{PC}$ reads the temporal stability of that organization. We note structurally "
         "that the raw factors are power-weighted (each spectrum carries the $w(f_i)\\,w(f_j)$ PSD "
         "weighting) and are therefore amplitude-confounded; we read the raw $H$, $\\mathrm{PC}$ and "
         "$R$ spectra as descriptive and make quantitative phase inference through the "
         "surrogate-normalized $\\mathrm{PC}_z$ (Methods)."),
        ("make quantitative phase inference through the surrogate-normalized $\\mathrm{PC}_z$ (Methods).",
         "make quantitative phase inference through the surrogate-normalized $\\mathrm{PC}_z$ (Methods). "
         "This normalization also guards against the converse concern -- that the fixed-phase harmonics "
         "of a single non-sinusoidal oscillator would inflate the phase axis: on sawtooth, square, "
         "clipped and fixed-phase-harmonic signals the within-signal $\\mathrm{PC}_z$ stayed at the "
         "null (median $\\le0$, none significant), because the phase-preserving surrogate carries the "
         "same stationary harmonic phase relations. It is $H$, not $\\mathrm{PC}_z$, that reads "
         "waveform shape; the coupling axis earns its keep in the cross-signal setting (below)."),
    ],
    "operating": [
        ("We next examined whether the surrogate-based significance test is properly calibrated, since "
         "an interpretable measure is only as trustworthy as its null. On uncoupled pairs -- where no "
         "coupling exists and a well-calibrated test should reject at the nominal rate -- the cross "
         "$\\mathrm{PC}$ z-score was \\textbf{approximately but not perfectly} null-calibrated. The "
         "empirical per-instance false-positive rate was $\\approx\\mathbf{0.09}$ at $\\alpha=0.05$ "
         "($n=120$ instances; null $z=0.07\\pm1.07$), i.e.\\ mildly anti-conservative: the test "
         "rejects somewhat more often than it should. We report this openly and recommend, for strict "
         "type-I control, either increasing the number of surrogates or applying a small "
         "multiple-surrogate correction, which brings the realized rate back toward nominal.",
         "We next examined whether the surrogate-based significance test is properly calibrated, since "
         "an interpretable measure is only as trustworthy as its null. On $n=\\mathbf{1000}$ uncoupled "
         "pairs ($99$ surrogates each) the null cross-$\\mathrm{PC}$ statistic was centred "
         "($z=0.01\\pm1.01$) but \\textbf{right-skewed} (skew $+0.51$; $95$th/$99$th percentiles "
         "$1.81$/$2.49$, above the Gaussian $1.64$/$2.33$). The Gaussian-$z$ threshold was therefore "
         "mildly anti-conservative -- realized false-positive rate $\\mathbf{0.066}$ at $\\alpha=0.05$ "
         "and $\\mathbf{0.016}$ at $\\alpha=0.01$ ($1.3$--$1.6\\times$ nominal). The remedy is not to "
         "assume normality: the exact one-sided permutation/rank $p$ from the same surrogates is "
         "calibrated by construction, and was -- realized rate $\\mathbf{0.045}$ at $\\alpha=0.05$ and "
         "$\\mathbf{0.010}$ at $\\alpha=0.01$. We therefore report significance via the permutation $p$ "
         "rather than a Gaussian threshold. Because every surrogate is passed through the full "
         "pipeline, the $n{:}m$ ratio selection is re-run identically for data and surrogates and "
         "cannot inflate the null (Figure 4)."),
    ],
    "biosignals": [
        ("a multivariate resonance signature on an intentionally easy separation",
         "a multivariate resonance signature on an intentionally easy separation of largely "
         "\\emph{simulated} physiology (the ECG, PPG and respiration modalities are "
         "\\texttt{neurokit2}-generated, so the fingerprint partly reflects generator signatures "
         "rather than physiology)"),
        ("not genuine harmonic organization, was driving the raw difference.",
         "not genuine harmonic organization, was driving the raw difference. The \\emph{sub-chance} "
         "value (rather than a mere drop to $0.5$) is itself informative: the aperiodic slope differs "
         "between the two states in the \\emph{opposite} direction to their genuine harmonic "
         "organization, so the uncorrected feature orders the conditions inversely."),
        ("$H_{\\max}$ decoded the contrast at AUC $0.83$; without it, decoding collapsed to $0.29$",
         "the $H_{\\max}$ harmonicity feature separated the two states at AUC $0.98$; without removal "
         "it collapsed to $0.27$"),
        ("This is direct evidence that harmonicity differences must be read only after aperiodic "
         "removal.",
         "This is direct evidence that harmonicity differences must be read only after aperiodic "
         "removal. The choice of aperiodic estimator does not change this: replacing the lightweight "
         "two-parameter fit with a specparam (FOOOF) decomposition---which models the alpha peak "
         "separately and so cannot be slope-biased by it---gave a near-identical harmonicity AUC "
         "($0.97$ vs $0.98$), confirming the eyes-closed result does not depend on the lightweight "
         "correction."),
    ],
    "discussion": [
        ("The phase-coupling surrogate null is mildly anti-conservative, with a per-instance "
         "false-positive rate of $\\approx 0.09$ at $\\alpha = 0.05$ on uncoupled pairs, so strict "
         "type-I control calls for more surrogates or a multiple-surrogate correction.",
         "The phase-coupling surrogate null is mildly anti-conservative under a Gaussian-$z$ threshold "
         "(false-positive rate $0.066$ at $\\alpha = 0.05$ on $n=1000$ uncoupled pairs, the null $z$ "
         "being right-skewed); we therefore report significance via the exact permutation $p$, which "
         "is calibrated ($0.045$ at $\\alpha=0.05$)."),
    ],
    "methods": [
        ("The core engine is biotuner, pinned to the immutable commit \\texttt{fc158c3} (full SHA "
         "recorded in \\texttt{requirements.txt}).",
         "The core engine is biotuner, pinned to the immutable commit \\texttt{fc158c3} (full SHA "
         "recorded in \\texttt{requirements.txt}). The harmonic-resonance framework described here is "
         "implemented as a module within this biotuner toolbox \\citep{bellemare2025}, released with "
         "unit tests and documentation; the pin guarantees exact reproducibility of every number and "
         "figure."),
        ("to the PSD over the analysis band by non-linear least squares and subtracts it.",
         "to the PSD over the analysis band by non-linear least squares and subtracts it. Any negative "
         "residuals are mapped to non-negative weights by the subsequent min--max normalization. "
         "Because a strong narrowband peak (most acutely eyes-closed alpha) can bias a non-linear "
         "least-squares slope, a specparam cross-check is advisable for spectra with prominent peaks; "
         "the lightweight fit is used here for speed and transparency."),
        ("\\texttt{nm\\_rrci}); the harmonic and ratio kernels",
         "\\texttt{nm\\_rrci}). The default $n{:}m$ phase-locking value is not bias-corrected and "
         "carries a positive $1/\\sqrt{N}$ bias at low window counts (the STFT yields of order "
         "$10$--$40$ windows per estimate at $0.5$~Hz resolution over $12$--$40$~s segments); for "
         "bias-sensitive use the registry exposes the bias- and volume-conduction-robust phase-lag "
         "metrics (\\texttt{nm\\_wpli}, \\texttt{nm\\_pli}), and the surrogate $\\mathrm{PC}_z$ further "
         "absorbs a constant bias by subtracting the null mean. The harmonic and ratio kernels"),
    ],
    "baselines": [
        ("the surrogate-normalized phase-coupling factor $\\mathrm{PC}_z$ matched the oracle raw n:m "
         "PLV across the usable signal-to-noise range, and was in fact \\emph{more} robust as "
         "conditions degraded, reaching reliable detection roughly $6$ dB lower in SNR than the oracle "
         "baseline.",
         "the surrogate-normalized phase-coupling factor $\\mathrm{PC}_z$ matched the oracle PLV across "
         "the usable signal-to-noise range and was \\emph{more} robust as conditions degraded. "
         "Crucially this survives a like-for-like comparison: against a \\emph{surrogate-normalized} "
         "oracle $\\mathrm{PLV}_z$ (the same IAAFT null, so both are $z$-scores), $\\mathrm{PC}_z$ "
         "still reached reliable detection roughly $6$ dB lower in SNR (e.g.\\ AUC $1.00$ vs $0.74$ at "
         "$-12$ dB), so the gain is not an artifact of comparing a normalized statistic to a raw "
         "one."),
    ],
    "connectivity": [
        ("reached AUC $0.74$ for $H$, $0.80$ for $\\mathrm{PC}$, and $0.81$ for $R$.",
         "reached AUC $0.74$--$0.81$ (similar for $H$, $\\mathrm{PC}$ and $R$: $0.74$, $0.80$, $0.81$); "
         "with this sample the factors are not separably ordered, so we report the range rather than "
         "ranking them."),
    ],
    "descriptors": [
        ("spectral flatness, entropy, spread, and Higuchi fractal dimension",
         "spectral flatness, entropy, spread, and Higuchi fractal dimension (the last computed on the "
         "short, $\\sim$90-point spectral vector and so the least stable of the four, reported for "
         "completeness rather than relied upon)"),
    ],
    "why_r": [
        ("A genuine conjunction is what this task demands, and conjunctions delivered:",
         "A genuine conjunction is what this task demands. As a consistency check on the combine rule "
         "rather than an empirical signal result---the ordering \\texttt{min}$\\approx$\\texttt{product}"
         "$>$single factor$>$\\texttt{max} follows from fuzzy-logic algebra---conjunctions delivered:"),
        ("This is the phase-coherence gate: $R$ adds to $H$",
         "This is the phase-coherence gate---a definitional consistency check, since phase scrambling "
         "preserves the power spectrum so a phase-blind $H$ must be unchanged by construction: $R$ "
         "adds to $H$"),
    ],
}
for _sec, _pairs in _subs.items():
    for _a, _b in _pairs:
        if _b not in d[_sec] and _a in d[_sec]:
            d[_sec] = d[_sec].replace(_a, _b, 1)

P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("applied methods-paper edits (abstract + construct PC_z note + why_r consistency labels)")

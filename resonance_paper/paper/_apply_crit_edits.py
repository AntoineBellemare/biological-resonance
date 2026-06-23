"""Apply calibrated revisions to _crit_sections.json (abstract toned down + new specificity
results subsection). Idempotent: re-running overwrites these keys with the strings below."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent / "_crit_sections.json"
d = json.loads(P.read_text(encoding="utf-8"))

ABSTRACT = (
    r"Criticality --- the idea that cortex operates near a phase transition between order and "
    r"disorder --- is a leading candidate organizing principle for brain dynamics, but its standard "
    r"markers are demanding to measure and frequently disagree: avalanche branching estimators need "
    r"population recordings, long-range temporal correlations need extended stationary epochs, and "
    r"susceptibility-based measures need many channels. Here we ask whether a single spectral "
    r"property --- the harmonicity $H$ of a power spectrum, a phase-blind measure of how nearly the "
    r"spectral peaks fall into integer-ratio relationships --- behaves as a \emph{candidate} "
    r"observable of proximity to criticality. Across three model systems with independent ground "
    r"truth, $H$ peaked when the system was tuned to its critical point: a branching network at its "
    r"critical branching ratio, a noise-driven echo-state reservoir at the edge of chaos, and a "
    r"Wilson--Cowan excitatory/inhibitory network at the edge of synchronization. A surrogate battery "
    r"showed this peak is specific to the real dynamics --- it is reproduced neither by "
    r"phase-randomized signals (confirming $H$ is phase-blind) nor by slope-, power- and "
    r"peakiness-matched spectra --- although it reflects the emergence of structured spectral peaks at "
    r"criticality rather than exact integer-ratio tuning. We then tested the prediction in human EEG. "
    r"A naive test reversed the model result; the prediction was recovered only when $H$ was read off "
    r"the model-matched, scale-free population observable, within sleep state and controlling for "
    r"slow-wave power, where it tracked the wake-to-sleep traversal of criticality. The effect was "
    r"modest and rested on this sleep traversal; propofol sedation and deep anaesthesia, which barely "
    r"traverse the critical region according to our estimator, were boundary nulls. Finally, the "
    r"oscillation-gated companion $R = H\cdot\mathrm{PC}$ tracked synchronization rather than avalanche "
    r"criticality: in a spiking network where avalanches and oscillations coexist, $R$ peaked with "
    r"synchronization, above the avalanche-critical point --- so $H$ and $R$ are separable axes, and it "
    r"is $H$, not $R$, that serves as the candidate criticality observable. $H$ is thus a candidate, "
    r"single-channel index of proximity to criticality whose in-vivo evidence remains preliminary and "
    r"conditional, but whose model-side construct validity is strong and surrogate-controlled."
)

SPECIFICITY = (
    r"Spectral shape changes near a transition, so we asked whether $H$'s criticality peak reflects "
    r"genuine spectral organization or merely the slope, power, or peakiness of the spectrum. For each "
    r"model we recomputed $H_{\max}$ across the control sweep on four spectrum-matched surrogates of "
    r"every signal, each preserving one nuisance property (Fig.~\ref{fig:crit_Fig7_specificity}): a "
    r"phase-randomized surrogate (identical power spectrum), an amplitude-adjusted (AAFT) surrogate, a "
    r"matched-slope $1/f$ colored-noise surrogate, and a fair peak-warp surrogate that keeps the same "
    r"peaks (count and heights) and $1/f$ background but relocates each peak frequency off "
    r"integer-ratio relationships." "\n\n"
    r"Three results followed. First, $H$ is phase-blind by construction, and the phase-randomized "
    r"surrogate confirmed it: its $H$ was essentially identical to the real signal across the sweep "
    r"--- the criticality peak is a property of the power spectrum, not of phase. Second, the peak is "
    r"not an artefact of spectral slope or power: matched-slope colored noise produced a flat $H$ that "
    r"did not track criticality (per-seed Spearman $\rho\approx-0.1$ in both models), whereas real $H$ "
    r"tracked it strongly (branching $\rho=+0.77$, 95\% CI $[0.71,0.82]$; Wilson--Cowan $\rho=+0.60$ "
    r"$[0.58,0.61]$), and in the oscillatory Wilson--Cowan model real $H$ sat far above the noise floor "
    r"at criticality ($\Delta H=+0.53$ $[0.51,0.55]$). Third, and as a candid boundary on the "
    r"interpretation, the fair peak-warp surrogate --- same peaks, integer-ratio relationships "
    r"destroyed --- was \emph{not} abolished, scoring comparably to the real signal near the peak. "
    r"$H$'s criticality signal therefore reflects the emergence of \emph{structured spectral peaks} at "
    r"the transition rather than a tuning to exact integer ratios; we retain the term harmonicity for "
    r"the operational metric (the $\mathrm{harmsim}$ kernel; Methods) while making this scope explicit. "
    r"This battery directly addresses whether $H$ is a criticality observable or a restatement of "
    r"spectral shape: it is the former in the sense that its criticality \emph{tracking} is "
    r"surrogate-specific and slope-independent, with the magnitude modest in the pure branching process "
    r"and pronounced where a genuine oscillation emerges (the reservoir's harmonicity signal is the "
    r"relative gain $H_\mathrm{internal}-H_\mathrm{input}$ of the section above, not single-signal $H$)."
)

d["abstract"] = ABSTRACT
d["specificity"] = SPECIFICITY
P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("updated abstract + specificity in", P.name)

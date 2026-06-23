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
    r"traverse the critical region according to our estimator, were boundary nulls. Finally, in a "
    r"spiking network where avalanches and oscillations coexist, harmonicity read off the population "
    r"signal tracked the emergent synchronization rather than the avalanche-critical point---the same "
    r"observable-mixing that drives the raw-EEG reversal---so the criticality signal in $H$ is exposed "
    r"only on a scale-free observable, while $R=H\cdot\mathrm{PC}$ is its oscillation-gated "
    r"synchronization index. $H$ is thus a candidate, "
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

DEFINITIONS = (
    r"\textbf{Formal definitions.} Let $p(f)$ be the Welch power spectral density restricted to "
    r"$[f_{\min},f_{\max}]$. The aperiodic ($1/f$) component is removed (below) and the residual is "
    r"min--max scaled and renormalized to a non-negative weight $w(f)$ with $\sum_f w(f)=1$. For an "
    r"ordered pair of frequency bins $(f_i,f_j)$ the harmonic kernel scores the rational simplicity of "
    r"their ratio: reducing $f_i/f_j$ to the lowest-terms fraction $a/b$ (continued-fraction "
    r"approximation), "
    r"\[ s(f_i,f_j)=\frac{a+b-1}{a\,b}, \] "
    r"which equals $1$ for a unison/simple ratio and decreases as the integers grow "
    r"($3{:}2\mapsto\tfrac{4}{6}$, $16{:}9\mapsto\tfrac{24}{144}$; Gill \& Purves, 2009; the "
    r"implementation scales $s$ by $100$, immaterial after normalization). The harmonicity spectrum is "
    r"the power-weighted, self-excluded reduction of this matrix, "
    r"\[ H(f_i)=w(f_i)\sum_{j\neq i} s(f_i,f_j)\,w(f_j), \] "
    r"so a bin scores highly when it stands in low-integer ratios with \emph{other} spectral power. "
    r"The phase-coupling spectrum applies the same reduction to an $n{:}m$ phase-locking kernel, "
    r"\[ \mathrm{PC}(f_i)=w(f_i)\sum_{j\neq i} \Big|\tfrac{1}{T}\sum_{t} "
    r"e^{\,i\,(n\varphi_i(t)-m\varphi_j(t))}\Big|\,w(f_j), \] "
    r"where $\varphi_i(t)$ is the short-time-Fourier-transform phase at $f_i$ and $(n,m)$ is the "
    r"integer pair from reducing $f_i/f_j$ with denominator $\le 16$ (the harmonic kernel $s$ uses an "
    r"effectively exact reduction, denominator $\le 1000$). The resonance spectrum is their "
    r"per-frequency product, $R(f)=H(f)\cdot\mathrm{PC}(f)$, and we report the maxima "
    r"$H_{\max}=\max_f H(f)$ (the headline harmonicity readout) and $R_{\max}$. Because $H$ depends "
    r"only on $p(f)$ it is phase-blind and defined whether or not the signal oscillates, whereas "
    r"$\mathrm{PC}$ (hence $R$) requires sustained, phase-stable rhythms. A single fixed configuration "
    r"(harmonic kernel $s$; $n{:}m$ phase-locking value; ratio denominators $\le 16$; aperiodic removal "
    r"on; identical $w$ normalization) was used across every model and EEG analysis, so all reported "
    r"$H$ and $R$ values are directly comparable."
)

_anchor = "the harmonic structure that tracks it lives in the emergent signal."
if "Formal definitions" not in d["methods"] and _anchor in d["methods"]:
    d["methods"] = d["methods"].replace(_anchor, _anchor + "\n\n" + DEFINITIONS, 1)

# --- discussion: report the spiking-network result (was future work) + specificity caveat ---
_disc_corrected = (
    r"Two directions follow, one of which we pursued directly. To place $R$ against a branching ground "
    r"truth we built a spiking E/I network of the Brunel type in which power-law avalanches and a "
    r"sustained population oscillation coexist at low firing rate, and swept its excitatory branching "
    r"gain through the avalanche-critical point. The result cuts against an over-simple reading: the "
    r"resonance observables computed on the population signal---$H$, $\mathrm{PC}$ and $R$ alike---all "
    r"rose with synchronization and peaked \emph{above} the avalanche-critical point (best power-law / "
    r"lowest distance-to-criticality at low gain), with $R$ uncorrelated with the avalanche power-law "
    r"across conditions ($\rho\approx-0.06$). Once a strong oscillation is present in the signal, its "
    r"harmonics dominate harmonicity, so $H$ read off the oscillation-laden population activity follows "
    r"the rhythm rather than the scale-free avalanche structure---precisely the mechanism behind the "
    r"raw-EEG reversal, now reproduced in a model with known ground truth. It is therefore the "
    r"observable (scale-free versus oscillation-laden), not the metric, that determines whether $H$ "
    r"reads the avalanche face of criticality: $H$ indexes avalanche criticality only when computed on "
    r"a scale-free signal---the branching model and the scale-free EEG observable---while $R$, the "
    r"oscillation-gated factor, is a synchronization index throughout. (The network also exposes an "
    r"estimator caveat: a strong superimposed rhythm biases the multistep-regression branching "
    r"estimator, which is why we anchor its criticality axis on the avalanche power-law and "
    r"crackling-scaling statistics rather than on $\hat m$.)"
)
_disc_priors = [
    (r"Two directions follow. The decisive model test is an E/I spiking network that exhibits both "
     r"avalanches and oscillations: the bare branching model is scale-free but non-oscillatory, and the "
     r"reservoir oscillates but is not a spiking neural system, so neither can place $R$ against a "
     r"branching ground truth. A balanced spiking network with both ingredients would let us test $R$ "
     r"directly against $\hat m$, asking whether the oscillation-gated factor adds information about the "
     r"synchronization face of criticality beyond what $H$ captures of the avalanche face."),
    (r"Two directions follow, one of which we pursued directly. To place $R$ against a branching ground "
     r"truth we built a spiking E/I network of the Brunel type in which power-law avalanches and a "
     r"sustained population oscillation coexist at low firing rate, and swept its excitatory branching "
     r"parameter through the avalanche-critical point. The outcome sharpens the division of labour: $R$ "
     r"and its phase-coupling factor $\mathrm{PC}$ peaked not at the avalanche-critical point but higher, "
     r"in the strongly synchronized regime---$R$ tracked synchronization, not the avalanche-critical "
     r"point, and its per-condition correlation with the avalanche power-law was essentially null, while "
     r"$\mathrm{PC}$ saturated as the network synchronized. $H$ and $R$ are thus separable not only in "
     r"principle but within a single system where both faces of criticality are present: the avalanche "
     r"face is indexed by $H$, the synchronization face by $R$, and the two need not co-locate. (This "
     r"network also makes plain why we anchor the model criticality axis on the avalanche power-law and "
     r"crackling-scaling statistics rather than on $\hat m$ there: a strong superimposed rhythm biases "
     r"the multistep-regression branching estimator, an estimator-scope caveat in its own right.)"),
]
if "harmonics dominate harmonicity" not in d["discussion"]:
    for _o in _disc_priors:
        if _o in d["discussion"]:
            d["discussion"] = d["discussion"].replace(_o, _disc_corrected, 1)
            break

_spec_anchor = "a reminder that not every plausible distance metric survives validation."
_spec_caveat = (
    r" Relatedly, a surrogate battery shows that $H$'s criticality-tracking is specific to the real "
    r"dynamics---phase-randomized, amplitude-matched and slope-matched surrogates do not reproduce the "
    r"criticality peak---but also that it reflects the emergence of structured spectral peaks at the "
    r"transition rather than a tuning to exact integer ratios: a fair surrogate that relocates peaks "
    r"off integer ratios is not abolished, and in the pure branching process the harmonicity at "
    r"criticality is only modestly above the matched-slope floor (the effect is most pronounced where a "
    r"genuine oscillation emerges). We therefore read $H$ as an index of structured spectral "
    r"organization maximized near criticality, retaining the term harmonicity for the operational "
    r"metric without claiming exact-harmonic tuning."
)
if _spec_anchor in d["discussion"] and "surrogate battery shows" not in d["discussion"]:
    d["discussion"] = d["discussion"].replace(_spec_anchor, _spec_anchor + _spec_caveat, 1)

OSCILLATORY = (
    r"The branching, reservoir and Wilson--Cowan systems each isolate one ingredient---avalanches "
    r"without oscillations, an emergent mode without spikes, or oscillations whose criticality is read "
    r"through susceptibility---so none lets the oscillation-gated factor $R$ be tested against a "
    r"branching ground truth in a spiking network. We therefore built a current-based "
    r"excitatory--inhibitory spiking network of the Brunel type, tuned to a fluctuation-driven, "
    r"low-rate regime in which power-law avalanches and a sustained $\sim$8--13~Hz population rhythm "
    r"coexist (Fig.~\ref{fig:crit_Fig8_oscillatory}A). Its control parameter is the recurrent "
    r"excitatory branching gain $\sigma$; we located the avalanche-critical point by the avalanche "
    r"power-law goodness of fit and the crackling-scaling deviation (best near $\sigma\approx0.6$, "
    r"before the largest avalanches run away into system-spanning events), and read $H$, $\mathrm{PC}$ "
    r"and $R$ off the population spike-count signal and the E/I cross-spectrum with the same "
    r"configuration as the Wilson--Cowan model." "\n\n"
    r"The result is informative precisely because it is not the tidy one. Across the sweep, $H$, "
    r"$\mathrm{PC}$ and $R$ all rose with synchronization and peaked \emph{above} the avalanche-critical "
    r"point, in the strongly synchronized regime (Fig.~\ref{fig:crit_Fig8_oscillatory}B); $R$ was "
    r"uncorrelated with the avalanche power-law across conditions (per-seed Spearman $\rho=-0.06$, "
    r"95\% CI $[-0.15,+0.05]$, $n=8$), and $H$ on this oscillation-laden signal in fact \emph{anti}"
    r"-tracked avalanche criticality ($\rho=-0.74$). This is the same observable confound that produces "
    r"the raw-EEG reversal, now in a model with known ground truth: when a strong oscillation is "
    r"present, its harmonic series dominates harmonicity, so $H$ computed on the oscillation-laden "
    r"population signal follows the rhythm rather than the scale-free avalanche structure. The contrast "
    r"with the branching network---where $H$ on the non-oscillatory avalanche signal peaked sharply at "
    r"criticality (Fig.~\ref{fig:crit_Fig8_oscillatory}C)---makes the lesson explicit: it is the "
    r"observable, scale-free versus oscillation-laden, not the metric, that sets whether $H$ reads the "
    r"avalanche face of criticality. $R$, the oscillation-gated factor, indexes synchronization "
    r"throughout and sits at the floor in the non-oscillatory branching model; it is thus a marker of "
    r"the synchronization face of criticality, complementary to---not a substitute for---$H$ on a "
    r"scale-free observable. A strong superimposed rhythm also biases the multistep-regression "
    r"branching ratio, so we anchor this model's criticality axis on the avalanche power-law and "
    r"crackling scaling rather than on $\hat m$."
)

d["oscillatory"] = OSCILLATORY
d["abstract"] = ABSTRACT
d["specificity"] = SPECIFICITY
P.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
print("updated abstract + specificity + methods definitions in", P.name)

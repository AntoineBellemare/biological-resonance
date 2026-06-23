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

PARAM_TABLE = (
    r"\begin{table}[t]\centering\small" "\n"
    r"\caption{Resonance parameters, fixed across all model and EEG analyses.}\label{tab:params}" "\n"
    r"\begin{tabular}{lll}\toprule" "\n"
    r"Step & Choice & Role \\ \midrule" "\n"
    r"Power spectrum & Welch, aperiodic-removed & input $p(f)$ \\" "\n"
    r"Frequency band & $2$--$60$/$80$~Hz (system-specific) & analysis range \\" "\n"
    r"Resolution & $0.5$--$1$~Hz & bin spacing \\" "\n"
    r"Harmonic kernel & $\mathrm{harmsim}$, $s=(a{+}b{-}1)/(ab)$ & rational simplicity \\" "\n"
    r"Ratio reduction & lowest terms; $n{:}m$ denominator $\le 16$ & allowed integer ratios \\" "\n"
    r"Spectral weight & min--max-scaled PSD, $\sum_f w=1$ & power weighting $w(f)$ \\" "\n"
    r"Phase estimator & short-time Fourier transform & $\varphi_i(t)$ \\" "\n"
    r"Coupling metric & $n{:}m$ phase-locking value & $\mathrm{PC}$ \\" "\n"
    r"Reduction & power-weighted, self-excluded & pairwise to per-bin \\" "\n"
    r"Combination & product $R=H\cdot\mathrm{PC}$ & resonance \\" "\n"
    r"Summary & maximum over $f$ & $H_{\max}$, $R_{\max}$ \\ \bottomrule" "\n"
    r"\end{tabular}\end{table}"
)

_anchor = "the harmonic structure that tracks it lives in the emergent signal."
if "Formal definitions" not in d["methods"] and _anchor in d["methods"]:
    d["methods"] = d["methods"].replace(_anchor, _anchor + "\n\n" + DEFINITIONS, 1)
if "label{tab:params}" not in d["methods"] and "are directly comparable." in d["methods"]:
    d["methods"] = d["methods"].replace(
        "are directly comparable.", "are directly comparable (Table~\\ref{tab:params}).\n\n" + PARAM_TABLE, 1)

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

# --- soften residual overstatements (#8) + clarify single-channel (#4) ---
_subs = {
    "ei": [
        ("to $g\\approx0.65$ (95\\% interval $[0.45,0.65]$)",
         "to $g\\approx0.60$ (95\\% interval $[0.50,0.60]$)"),
        ("the autocorrelation time, our measure of critical slowing down, peaked at the same gain.",
         "the autocorrelation time, our measure of critical slowing down, peaked at the same gain. "
         "Critically, this critical point is fixed independently by the deterministic system itself: "
         "the leading eigenvalue of the noise-free Wilson--Cowan Jacobian crosses zero---the Hopf "
         "bifurcation---at $g=0.66$, computed with no reference to harmonicity or resonance and "
         "coinciding with the normalized-susceptibility and critical-slowing peaks."),
        ("With the critical region fixed, both $H$ and $R$ tracked it. The harmonicity $H$ and the "
         "oscillation-gated resonance $R$ peaked together at $g\\approx0.65$, co-located with the "
         "normalized susceptibility and the autocorrelation maximum. This is the one model in which "
         "$R$ is placeable, and it lands exactly where the criticality markers say it should.",
         "With the critical region fixed independently, we asked where the resonance observables sit. "
         "This is the one of the three models in which $R$ is non-trivial: harmonicity $H$ rises and "
         "phase coupling switches on at the onset, so $R=H\\cdot\\mathrm{PC}$ becomes placeable rather "
         "than sitting at the floor. $R$ does \\emph{not} sharply peak at the edge, however---it rises "
         "through the onset and stays elevated across the synchronized regime, with its maximum well "
         "into synchrony---foreshadowing the spiking-network result below that $R$ indexes "
         "synchronization rather than the avalanche-critical point per se."),
        ("Taken together, the E/I network shows that when oscillations and criticality genuinely "
         "coexist, $R$ recovers the same critical point that $H$ marks, and that proper normalization "
         "of the susceptibility is essential to read that point off correctly.",
         "Taken together, the E/I network shows that $R$ becomes a placeable, non-trivial quantity "
         "only when oscillations and criticality coexist---switching on at the synchronization onset "
         "that the deterministic eigenvalue, normalized susceptibility, and critical slowing jointly "
         "fix---and that proper normalization of the susceptibility is essential to locate that onset."),
    ],
    "resolution": [
        ("the convergence is exact: read off the correct observable, the human brain reproduces the "
         "model law that $H$, not $R$, is the single-channel marker of proximity to criticality.",
         "the alignment is preliminary: read off the model-matched observable, sleep EEG shows the "
         "predicted ordering within state, with $H$ (not $R$) the candidate single-channel index of "
         "proximity to criticality---the harmonicity computation is single-signal, though here it is "
         "applied to global field power, a population-level observable."),
        ("the correct in-vivo analogue", "the model-matched in-vivo analogue"),
    ],
    "discussion": [
        ("The correct empirical analogue of that signal is not the",
         "The model-matched empirical analogue of that signal is not the"),
        ("$H$ is the spectral criticality factor---it is maximized at the critical point of an "
         "avalanche system that never oscillates.",
         "$H$ behaves as the spectral criticality factor in this division of labour: read off "
         "scale-free avalanche activity, it is maximized at the critical point of a system that never "
         "oscillates."),
    ],
}
for _sec, _pairs in _subs.items():
    for _a, _b in _pairs:
        if _a in d[_sec]:
            d[_sec] = d[_sec].replace(_a, _b, 1)

# --- "What H is and is not" closing paragraph (#10) + exploratory labelling (#9) ---
WHATIS = (
    r"\textbf{What $H$ is and is not.} To prevent over-reading we state the scope compactly. $H$ is "
    r"\emph{not} a direct estimator of the branching ratio $\hat m$; it is a complementary spectral "
    r"readout that co-varies with proximity to criticality. $H$ is \emph{not} a classifier that "
    r"outperforms band power, and we make no decoding claim. $H$ is \emph{not} evidence of exact "
    r"integer-ratio tuning; the surrogate battery shows it indexes structured low-order spectral "
    r"organization that often includes, but is not reducible to, exact integer ratios. What $H$ "
    r"\emph{is}: a cheap, largely model-free spectral index of structured organization that peaks near "
    r"independently defined critical transitions in three model systems and that---read off a "
    r"scale-free population observable---shows preliminary, conditional alignment with estimated "
    r"criticality in human sleep EEG. We did not preregister; the scale-free observable was dictated by "
    r"the model construction rather than chosen to rescue the result, and the independent Sleep-EDF "
    r"Telemetry cohort provides a fixed-pipeline replication of the dissociation."
)
if "What $H$ is and is not" not in d["discussion"]:
    d["discussion"] = d["discussion"].rstrip() + "\n\n" + WHATIS

SUMMARY_TABLE = (
    r"\begin{table}[t]\centering\footnotesize" "\n"
    r"\caption{Each system, its independent (model) or estimated (EEG) criticality marker, the "
    r"behaviour of $H$ and of $R=H\cdot\mathrm{PC}$, and the main caveat.}\label{tab:summary}" "\n"
    r"\begin{tabular}{p{2.3cm}p{2.7cm}p{3.0cm}p{2.6cm}p{2.4cm}}\toprule" "\n"
    r"System & Criticality marker & $H$ & $R$ & Caveat \\ \midrule" "\n"
    r"Branching network & branching ratio $\hat m$ near 1; avalanche power-law & peaks at criticality "
    r"(scale-free signal) & at the floor (no oscillation) & near-ceiling (construct validity) \\" "\n"
    r"Echo-state reservoir & Lyapunov edge of chaos & harmonicity gain peaks near $\rho_c$ & --- "
    r"(non-spiking) & relative measure ($H$ gain over input) \\" "\n"
    r"Wilson--Cowan E/I & normalized susceptibility; Hopf eigenvalue ($g{=}0.66$) & rises at the onset "
    r"& placeable; maximal in the synchronized regime & rate model; $R$ via E--I cross \\" "\n"
    r"Spiking E/I (avalanches + oscillations) & avalanche power-law / crackling & tracks "
    r"synchronization on the population signal & peaks above the avalanche-critical point & "
    r"observable-dependent (oscillation-laden) \\" "\n"
    r"Sleep EEG & $\hat m$, DFA traversal & $H_\mathrm{aval}$ tracks criticality; raw $H$ reverses & "
    r"oscillation-gated & modest; single traversal \\" "\n"
    r"Propofol / deep anaesthesia & $\hat m$ (barely traverses) & null & null & boundary null "
    r"(no traversal) \\ \bottomrule" "\n"
    r"\end{tabular}\end{table}"
)
if "tab:summary" not in d["discussion"]:
    d["discussion"] = SUMMARY_TABLE + "\n\n" + d["discussion"]

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

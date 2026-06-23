"""Assemble the eLife-style CRITICALITY paper LaTeX from drafted section bodies + figures.

Reads _crit_sections.json with keys: abstract, introduction, branching, reservoir, ei,
tension, resolution, discussion, methods, references (list of formatted reference strings).
Emits criticality_paper.tex (compile: tectonic -X compile criticality_paper.tex).

    python -m resonance_paper.paper.build_criticality_tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SECT = json.loads((PAPER / "_crit_sections.json").read_text(encoding="utf-8"))
OUT = PAPER / "criticality_paper.tex"

TITLE = ("Spectral harmonicity as a candidate observable of proximity to criticality "
         "in models and human EEG")
AUTHORS = r"Antoine Bellemare-Pepin\textsuperscript{1,2}, Karim Jerbi\textsuperscript{1}"
AFFIL = (r"\textsuperscript{1}CoCo Lab, Psychology Department, University of Montr\'eal, Canada \\ "
         r"\textsuperscript{2}Music Department, Concordia University, Montr\'eal, Canada")

# Results subsections in order: (json key, heading, figure key)
RESULTS = [
    ("branching",  "Harmonicity peaks at the critical point of a branching network", "crit_Fig2_branching"),
    ("reservoir",  "Criticality generates harmonicity from structureless noise",     "crit_Fig3_reservoir"),
    ("ei",         "At the synchronization onset, phase coupling becomes placeable",  "crit_Fig4_ei_network"),
    ("oscillatory","A spiking network dissociates H from R",                          "crit_Fig8_oscillatory"),
    ("specificity","Harmonicity's criticality peak is surrogate-specific",            "crit_Fig7_specificity"),
    ("tension",    "In human EEG the naive prediction reverses",                       "crit_Fig5_realdata_tension"),
    ("resolution", "The reversal is an observable-choice effect",                      "crit_Fig6_resolution"),
]

CAPS = {
"crit_Fig1_schematic": (
    r"\textbf{The hypothesis.} As a system crosses its critical point, classical markers "
    r"(branching ratio $\hat m\to1$, susceptibility) peak; we predict that spectral harmonicity "
    r"$H$ peaks there too, whereas phase coupling and $R=H\cdot\mathrm{PC}$ rise with synchronization "
    r"(supercritical / synchronized regime) and index the synchronization axis rather than avalanche "
    r"criticality per se."),
"crit_Fig2_branching": (
    r"\textbf{Harmonicity peaks at criticality in a branching network.} \textbf{(A)} Classical "
    r"criticality markers (susceptibility, avalanche power-law $R^2$, branching estimate $\hat m$) "
    r"peak at $\sigma\approx1$. \textbf{(B)} Harmonicity $H_{\max}$ is an inverted-U peaking at the "
    r"critical point (shaded 95\% CI of the peak location); phase-coupling $R$ stays at the floor "
    r"because bare avalanches are scale-free but non-oscillatory. \textbf{(C)} The $H$ peak "
    r"co-locates with the susceptibility peak, and $H$ tracks criticality proximity across $\sigma$."),
"crit_Fig3_reservoir": (
    r"\textbf{Criticality generates harmonicity from noise.} \textbf{(A)} The Lyapunov exponent "
    r"locates the edge of chaos $\rho_c$ (zero-crossing, 95\% CI). \textbf{(B)} Driven by white "
    r"noise, the reservoir's emergent mode gains harmonic structure over the (flat) input near and "
    r"beyond the edge (generation onset marked). \textbf{(C)} Generation (harmonicity gain) "
    r"dissociates from computation (memory capacity): their peaks fall at different $\rho$."),
"crit_Fig4_ei_network": (
    r"\textbf{At the synchronization onset, phase coupling and resonance become placeable.} "
    r"\textbf{(A)} The edge of synchronization, located by the relative-fluctuation susceptibility "
    r"(var/mean$^2$) and the autocorrelation time (critical slowing), which peak together near "
    r"$g\approx0.60$ (shaded), where the deterministic Jacobian's leading eigenvalue also crosses zero "
    r"(Hopf bifurcation, $g=0.66$); raw envelope-variance susceptibility peaks later, an amplitude "
    r"confound. "
    r"\textbf{(B)} The resonance factors against the edge: harmonicity $H$, phase coupling $\mathrm{PC}$, "
    r"and $R=H\cdot\mathrm{PC}$ (normalized); $\mathrm{PC}$ rises through the edge and $R$ becomes "
    r"non-trivial here---the only one of the three models where it does---though $R$ is maximal in the "
    r"synchronized regime above the edge, not exactly at it. \textbf{(C)} The absolute rise of "
    r"E$\leftrightarrow$I phase coupling (PLV) "
    r"above its non-zero asynchronous baseline ($\Delta\mathrm{PC}=+0.39$)."),
"crit_Fig5_realdata_tension": (
    r"\textbf{In human EEG the prediction reverses.} \textbf{(A)} In sleep, raw-EEG harmonicity "
    r"$H_{\max}$ is \emph{highest} in deep N3 --- the most subcritical state (branching $\hat m$ "
    r"overlaid). \textbf{(B)} Per-subject Spearman $\rho$ between $H_{\max}$ and criticality "
    r"proximity is null or negative across datasets. \textbf{(C)} Boundary cases: propofol sedation "
    r"and deep anaesthesia barely traverse criticality (small $\hat m$ range), so they cannot test "
    r"the law; the real-data test rests on the sleep traversal."),
"crit_Fig6_resolution": (
    r"\textbf{The reversal is an observable-choice effect.} Stars denote $p<0.05$. \textbf{(A)} On the "
    r"scale-free population signal ($H_\mathrm{aval}$) harmonicity tracks criticality proximity, while "
    r"on the raw oscillatory signal ($H_\mathrm{full}$) it reverses (sleep paired sign-dissociation "
    r"$p=0.016$); propofol does not traverse criticality and is null on both. \textbf{(B)} The "
    r"scale-free recovery survives controlling for slow-band power (partial $\rho=+0.16$, $p=0.023$), "
    r"whereas the raw reversal does not. \textbf{(C)} Mechanistic control: removing the slow-wave band "
    r"from the raw signal substantially attenuates the reversal and removes its significance "
    r"(Spearman $\rho$ from $-0.24$ to $-0.09$, $p$ from $0.008$ to $0.11$) --- it was largely carried "
    r"by the slow-wave harmonic series. "
    r"\textbf{(D)} Across sleep stages, raw $H$ inflates into deep N3 (the most subcritical state) while "
    r"the scale-free $H$ does not."),
"crit_Fig7_specificity": (
    r"\textbf{Harmonicity's criticality peak is surrogate-specific.} For each model, $H_{\max}$ on the "
    r"real signal is compared, across the control sweep, against spectrum-matched surrogates that each "
    r"preserve one nuisance property. \textbf{(A)} Branching process and \textbf{(B)} Wilson--Cowan "
    r"network: $H$(real) peaks at the critical point and tracks criticality proximity (per-seed "
    r"Spearman $\rho$ inset), whereas the phase-randomized surrogate coincides with the real curve "
    r"(confirming $H$ is phase-blind) and the slope-, power- and peakiness-matched surrogates "
    r"(matched-slope colored noise; inharmonic peak-warp) stay comparatively flat. \textbf{(C)} Only "
    r"the real $H$ tracks criticality (per-seed $\rho$, 95\% CI); matched-slope noise does not. The "
    r"peak-warp null --- same peaks, relocated off integer ratios --- is not abolished, so $H$ indexes "
    r"the emergence of structured spectral peaks at criticality rather than exact integer-ratio tuning."),
"crit_Fig8_oscillatory": (
    r"\textbf{A spiking network dissociates $H$ from $R$.} A current-based E/I spiking network of the "
    r"Brunel type in which power-law avalanches and a sustained population rhythm coexist at low firing "
    r"rate; the control parameter is the excitatory branching gain $\sigma$. \textbf{(A)} Avalanches "
    r"and oscillation coexist: the avalanche power-law goodness of fit (criticality, peaking at low "
    r"$\sigma$) alongside the rising oscillation prominence and the runaway growth of the largest "
    r"avalanche at high $\sigma$ (supercritical). \textbf{(B)} Normalized $H$, $\mathrm{PC}$, and "
    r"$R=H\cdot\mathrm{PC}$ against $\sigma$: $\mathrm{PC}$ and $R$ peak in the synchronized regime, "
    r"\emph{above} the avalanche-critical point (green line) marked by the power-law optimum. "
    r"\textbf{(C)} Per-seed Spearman $\rho$ between each resonance factor and avalanche-criticality "
    r"proximity (95\% CI): $R$ does not track avalanche criticality (null), confirming that $R$ indexes "
    r"oscillatory synchronization rather than the branching/avalanche face of criticality."),
}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{graphicx,xcolor,booktabs}
\usepackage[labelfont={bf,color=critblue},font=small,labelsep=period,justification=justified,singlelinecheck=false]{caption}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\definecolor{critblue}{HTML}{283593}
\definecolor{rulegray}{HTML}{B8B8B8}
\titleformat{\section}{\sffamily\bfseries\large\color{critblue}}{}{0em}{}
\titleformat{\subsection}{\sffamily\bfseries\normalsize\color{black!82}}{}{0em}{}
\titlespacing*{\section}{0pt}{16pt}{5pt}
\titlespacing*{\subsection}{0pt}{11pt}{3pt}
\setlength{\parskip}{1pt}\setlength{\parindent}{1.4em}
\linespread{1.05}\setlength{\emergencystretch}{2em}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.3pt}\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{\footnotesize\itshape Harmonicity tracks proximity to criticality}
\fancyhead[R]{\footnotesize\thepage}
\begin{document}
"""


def _ascii(s):
    for a, b in [("é", "e"), ("è", "e"), ("ë", "e"), ("ü", "u"), ("ö", "o"), ("á", "a"), ("í", "i"), ("ñ", "n")]:
        s = s.replace(a, b)
    return re.sub(r"[^a-zA-Z]", "", s)


def _citemap():
    mp = {}
    for r in SECT.get("references", []):
        ym = re.search(r"\((\d{4})", r)
        if not ym:
            continue
        year = ym.group(1); head = r[:ym.start()].lstrip("*").strip()
        surnames = re.findall(r"([A-Z][a-zA-Z'\-]+),", head)   # "Surname," tokens (skips initials)
        first_full = head.split(",")[0].strip()
        if len(surnames) >= 3 or head.count("&") > 1:
            disp = f"{first_full} et al."
        elif len(surnames) == 2:
            disp = f"{first_full} \\& {surnames[1]}"
        else:
            disp = first_full
        key_tok = re.split(r"[\s,\-]+", head)[0]
        mp[(_ascii(key_tok).lower(), year)] = disp
    return mp


CITEMAP = _citemap()


def _resolve(key):
    km = re.match(r"([a-zA-Z\-]+?)(\d{4})", key.strip())
    if not km:
        return (key.strip(), "")
    a, y = _ascii(km.group(1)).lower(), km.group(2)
    for (b, yy), name in CITEMAP.items():
        if yy == y and (a.startswith(b) or b.startswith(a)):
            return (name, y)
    return (a.capitalize(), y)


def decite(s):
    def rp(paren):
        def f(m):
            res = [_resolve(k) for k in m.group(1).split(",") if k.strip()]
            if paren:
                return "(" + "; ".join(f"{n}, {y}" if y else n for n, y in res) + ")"
            return "; ".join(f"{n} ({y})" if y else n for n, y in res)
        return f
    s = re.sub(r"\\citep\{([^}]*)\}", rp(True), s)
    s = re.sub(r"\\citet\{([^}]*)\}", rp(False), s)
    s = re.sub(r"\\cite\{([^}]*)\}", rp(False), s)
    return s


def references_tex():
    refs = SECT.get("references", [])
    if not refs:
        return ""
    def esc(s):
        s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
        return s
    lines = "\n".join(rf"\item {esc(r)}" for r in refs)
    note = (r"\textit{\small References compiled during a literature-positioning pass; "
            r"verify bibliographic details (page numbers, DOIs) before submission.}\par\vspace{4pt}")
    return ("\\section*{References}\n" + note + "\n\\begingroup\\small\n\\begin{list}{}{\\setlength{\\itemsep}{1pt}"
            "\\setlength{\\leftmargin}{1.2em}\\setlength{\\itemindent}{-1.2em}}\n" + lines +
            "\n\\end{list}\\endgroup\n")


def figure_tex(key):
    return ("\\begin{figure}[t]\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{key}.pdf}}\n"
            f"\\caption{{{CAPS[key]}}}\n\\label{{fig:{key}}}\n\\end{{figure}}\n")


def main():
    s = {k: (decite(v) if isinstance(v, str) else v) for k, v in SECT.items()}
    parts = [PREAMBLE]
    parts.append(rf"""\thispagestyle{{plain}}
\begin{{center}}
{{\LARGE\sffamily\bfseries\color{{critblue}} {TITLE}\par}}
\vspace{{11pt}}{{\large {AUTHORS}\par}}
\vspace{{5pt}}{{\small\itshape {AFFIL}\par}}
\end{{center}}
\vspace{{3pt}}{{\color{{rulegray}}\hrule height 0.6pt}}\vspace{{8pt}}
\noindent{{\sffamily\bfseries\color{{critblue}}Abstract}}\par\vspace{{2pt}}
{{\small {s['abstract']}\par}}
\vspace{{7pt}}{{\color{{rulegray}}\hrule height 0.6pt}}\vspace{{10pt}}
""")
    parts.append(r"\section{Introduction}" + "\n" + s["introduction"] + "\n")
    parts.append(r"\section{Results}" + "\n")
    parts.append(figure_tex("crit_Fig1_schematic"))
    for key, head, figkey in RESULTS:
        parts.append(rf"\subsection{{{head}}}" + "\n" + s[key] + "\n")
        parts.append(figure_tex(figkey))
    parts.append(r"\section{Discussion}" + "\n" + s["discussion"] + "\n")
    parts.append(r"\section{Methods}" + "\n" + s["methods"] + "\n")
    parts.append(references_tex())
    parts.append(r"\end{document}")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()

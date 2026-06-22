"""Assemble the eLife-style methods-paper LaTeX from drafted section bodies + figures.

Reads _elife_sections.json (produced from the prose-drafting workflow) with keys:
  abstract, introduction, construct, why_r, ground_truth, operating, baselines,
  mechanism, biosignals, connectivity, descriptors, methods, discussion
Emits methods_paper.tex (compile with: tectonic -X compile methods_paper.tex).

    python -m resonance_paper.paper.build_elife_tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SECT = json.loads((PAPER / "_elife_sections.json").read_text(encoding="utf-8"))
DRAFT = (PAPER / "resonance_paper_draft.md").read_text(encoding="utf-8")
OUT = PAPER / "methods_paper.tex"

TITLE = ("Harmonic Resonance Spectra of Biosignals: Separable Harmonicity and "
         "Phase-Coupling Axes with a Validated, Swappable-Strategy Toolbox")
# NOTE: author/affiliation are placeholders — edit before submission.
AUTHORS = r"Antoine Bellemare-Pepin\textsuperscript{1,2}, Karim Jerbi\textsuperscript{1}"
AFFIL = (r"\textsuperscript{1}CoCo Lab, Psychology Department, University of Montr\'eal, Canada \\ "
         r"\textsuperscript{2}Music Department, Concordia University, Montr\'eal, Canada")

# Results subsections in order: (json key, heading, figure number)
RESULTS = [
    ("construct",    "Harmonicity and phase coupling are separable axes", 1),
    ("why_r",        "A multiplicative resonance descriptor and what it adds", 2),
    ("ground_truth", "The framework recovers known harmonic structure and coupling", 3),
    ("operating",    "Operating characteristics: sensitivity, calibration, and cost", 4),
    ("baselines",    "Comparison with established measures", 5),
    ("mechanism",    "Method choices and the locking mechanism", 6),
    ("biosignals",   "Application to real biosignals", 7),
    ("connectivity", "Network-level resonance connectivity", 8),
    ("descriptors",  "Spectral-complexity descriptors as features", 9),
]

FIGFILE = {
    1: "method_Fig1_dissociation", 2: "method_Fig2_R_justification", 3: "method_Fig3_ground_truth",
    4: "method_Fig4_operating", 5: "method_Fig5_baselines", 6: "method_Fig6_strategy_mechanism",
    7: "method_Fig7_real_biosignals", 8: "method_Fig8_connectivity", 9: "method_Fig9_descriptors",
}

# Panel-aware figure captions (the manuscript's captions; figures carry no in-image titles).
CAPS = {
1: (r"\textbf{Harmonicity is phase-blind; phase coupling tracks locking.} "
    r"\textbf{(A)} Harmonicity $H$ on a grid that independently varies ratio complexity (Tenney height) "
    r"and phase-locking strength $\kappa$ (one line per complexity level); $H$ is flat across $\kappa$ "
    r"($\rho(H,\kappa)=0.00$), ordered only by complexity. \textbf{(B)} Phase coupling rises with $\kappa$. "
    r"\textbf{(C)} Independence: Spearman $\rho$ of each factor against each axis. \textbf{(D)} Generative "
    r"confirmation with harmonically-coupled phase oscillators: as coupling $K$ rises each pair locks at its "
    r"n:m ratio, but the locking threshold $K^*$ rises with ratio complexity "
    r"($K^*=3.3, 8.7, 23.5$ for 2:3, 3:4, 4:5; $\rho(K^*,n{\cdot}m)=+1.0$; log axis)."),
2: (r"\textbf{Why $R=H\cdot\mathrm{PC}$ is the principled conjunction.} "
    r"\textbf{(A)} With $H$ and $\mathrm{PC}$ varied as independent factors, conjunctive combine rules "
    r"separate a true resonance from single-factor confounds (specificity AUC $\approx0.99$) and beat both "
    r"factors alone ($\approx0.75$) and the OR-disjunction. \textbf{(B)} Phase-scrambling preserves the power "
    r"spectrum, so $H$ is unchanged (AUC $0.50$) while $R$ collapses (AUC $1.00$): $R$ adds the phase-coherence "
    r"gate $H$ lacks. \textbf{(C)} The same gate generatively: in harmonically-coupled 2:3 oscillators, $R$ "
    r"stays low while harmonicity is present but phases are unlocked, rising only once coupling locks the phases "
    r"($K^*\approx3.3$; log axis)."),
3: (r"\textbf{Ground-truth recovery.} \textbf{(A)} Mean harmonicity rises monotonically with known harmonic "
    r"richness across signal classes (Spearman $\rho=0.98$). \textbf{(B)} Surrogate-normalized detection AUC "
    r"for single- and cross-signal n:m phase coupling ($\approx1.0$). \textbf{(C)} Polyrhythm (2:3:4) recovery: "
    r"resonance peak-to-median, locked vs phase-scrambled (rank AUC $=1.00$)."),
4: (r"\textbf{Operating characteristics.} \textbf{(A)} Detection AUC vs signal-to-noise ratio for harmonicity "
    r"$H$ and cross-signal coupling $\mathrm{PC}_z$ (chance dashed); $H$ holds to $\approx-18$~dB, coupling has "
    r"a sharp threshold near $-12$~dB. \textbf{(B)} Null calibration of $\mathrm{PC}_z$ on uncoupled pairs "
    r"(per-instance false-positive rate; $n=120$): mildly anti-conservative ($\approx0.09$ at $\alpha=0.05$). "
    r"\textbf{(C)} Runtime is set by the frequency-grid size, not signal length."),
5: (r"\textbf{Comparison with established measures.} \textbf{(A)} Coupling detection: framework "
    r"$\mathrm{PC}_z$ matches the oracle raw n:m phase-locking value and saturates $\approx6$~dB earlier. "
    r"\textbf{(B,C)} Harmonicity $H$ vs the classical harmonic-to-noise ratio, separating harmonic from "
    r"inharmonic (B) and from noise (C); $H$ matches HNR at usable SNR and adds a per-frequency spectrum."),
6: (r"\textbf{Method choices and the locking mechanism.} \textbf{(A)} Across the strategy registry, detection "
    r"accuracy saturates (all configurations $0.982$--$1.000$), so the practical choice is runtime "
    r"(\texttt{harmsim} $\sim27\times$ faster). \textbf{(B)} Rotation number $\rho(\Omega)$ of a forced van der "
    r"Pol oscillator. \textbf{(C)} Arnold-tongue width vs ratio complexity (colour: harmonicity), recovering "
    r"the classical dependence of lockability on ratio simplicity ($\rho_s=-0.50$)."),
7: (r"\textbf{Application to real biosignals.} \textbf{(A)} Leave-one-subject-out decoding AUC, resonance vs "
    r"relative band power, for eyes-open/closed and a rest/motor contrast. \textbf{(B)} The eyes-closed "
    r"harmonicity effect is aperiodic-dependent ($H$ AUC with vs without 1/f removal). \textbf{(C)} "
    r"Normalized resonance spectra fingerprint seven signal modalities (cross-validated accuracy vs chance)."),
8: (r"\textbf{Network-level resonance connectivity.} \textbf{(A,B)} Cross-resonance connectivity matrices "
    r"(harmonicity, phase coupling) for a montage with a planted coupled cluster (cyan box); $H$ is diffuse, "
    r"$\mathrm{PC}$ sharper. \textbf{(C)} Within-cluster-vs-rest recovery AUC ($H$ 0.74, $\mathrm{PC}$ 0.80, "
    r"$R$ 0.81); recovery is overlap-driven, and isolating pure phase requires the IAAFT surrogate-z variant."),
9: (r"\textbf{Spectral-complexity descriptors of the resonance spectra.} \textbf{(A)} Spectral flatness of "
    r"the $H$ spectrum by signal class. \textbf{(B)} Per-class descriptor fingerprint (z-scored across "
    r"classes). \textbf{(C)} Shape space (flatness $\times$ spread); the descriptors carry information the "
    r"scalar summaries miss."),
}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{graphicx,xcolor,booktabs}
\usepackage[labelfont={bf,color=elifeblue},font=small,labelsep=period]{caption}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\definecolor{elifeblue}{HTML}{1A4F63}
\definecolor{rulegray}{HTML}{B8B8B8}
\titleformat{\section}{\sffamily\bfseries\large\color{elifeblue}}{}{0em}{}
\titleformat{\subsection}{\sffamily\bfseries\normalsize\color{black!82}}{}{0em}{}
\titlespacing*{\section}{0pt}{16pt}{5pt}
\titlespacing*{\subsection}{0pt}{11pt}{3pt}
\setlength{\parskip}{1pt}\setlength{\parindent}{1.4em}
\linespread{1.05}
\setlength{\emergencystretch}{2em}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{\footnotesize\itshape Harmonic resonance spectra of biosignals}
\fancyhead[R]{\footnotesize\thepage}
\captionsetup{justification=justified,singlelinecheck=false}
\begin{document}
"""


def _ascii(s):
    for a, b in [("é", "e"), ("è", "e"), ("ë", "e"), ("ü", "u"), ("ö", "o"), ("á", "a"), ("í", "i")]:
        s = s.replace(a, b)
    return re.sub(r"[^a-zA-Z]", "", s)


def _citemap():
    m = re.search(r"## References\n(.*?)$", DRAFT, flags=re.S)
    refs = re.findall(r"^- (.+)$", m.group(1), flags=re.M) if m else []
    mp = {}
    for r in refs:
        ym = re.search(r"\((\d{4})", r)
        if not ym:
            continue
        year = ym.group(1); head = r[:ym.start()]
        full_surname = head.split(",")[0].strip().lstrip("*").strip()
        first_tok = re.split(r"[\s,\-]+", head.strip().lstrip("*"))[0]
        base = _ascii(first_tok).lower()
        etal = ("&" in head) or (head.count(",") >= 2)
        mp[(base, year)] = f"{full_surname} et al." if etal else full_surname
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
    m = re.search(r"## References\n(.*?)$", DRAFT, flags=re.S)
    if not m:
        return ""
    body = m.group(1)
    items = re.findall(r"^- (.+)$", body, flags=re.M)
    def esc(s):
        s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
        return s
    lines = "\n".join(rf"\item {esc(it)}" for it in items)
    return ("\\section*{References}\n\\begingroup\\small\n"
            "\\begin{list}{}{\\setlength{\\itemsep}{1pt}\\setlength{\\leftmargin}{1.2em}"
            "\\setlength{\\itemindent}{-1.2em}}\n" + lines + "\n\\end{list}\\endgroup\n")


def figure_tex(n):
    return ("\\begin{figure}[t]\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{FIGFILE[n]}.pdf}}\n"
            f"\\caption{{{CAPS[n]}}}\n\\label{{fig:{n}}}\n\\end{{figure}}\n")


def main():
    s = {k: decite(v) for k, v in SECT.items()}
    parts = [PREAMBLE]
    parts.append(rf"""\thispagestyle{{plain}}
\begin{{center}}
{{\LARGE\sffamily\bfseries\color{{elifeblue}} {TITLE}\par}}
\vspace{{11pt}}{{\large {AUTHORS}\par}}
\vspace{{5pt}}{{\small\itshape {AFFIL}\par}}
\end{{center}}
\vspace{{3pt}}{{\color{{rulegray}}\hrule height 0.6pt}}\vspace{{8pt}}
\noindent{{\sffamily\bfseries\color{{elifeblue}}Abstract}}\par\vspace{{2pt}}
{{\small {s['abstract']}\par}}
\vspace{{7pt}}{{\color{{rulegray}}\hrule height 0.6pt}}\vspace{{10pt}}
""")
    parts.append(r"\section{Introduction}" + "\n" + s["introduction"] + "\n")
    parts.append(r"\section{Results}" + "\n")
    for key, head, fign in RESULTS:
        parts.append(rf"\subsection{{{head}}}" + "\n" + s[key] + "\n")
        parts.append(figure_tex(fign))
    parts.append(r"\section{Discussion}" + "\n" + s["discussion"] + "\n")
    parts.append(r"\section{Methods}" + "\n" + s["methods"] + "\n")
    parts.append(references_tex())
    parts.append(r"\end{document}")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()

"""Assemble the eLife-style CONSONANCE paper LaTeX from drafted sections + figures.

Reads _cons_sections.json (keys: abstract, introduction, construct, chords, ssvep, ffr,
synthesis, discussion, methods, references). Emits consonance_paper.tex.

    python -m resonance_paper.paper.build_consonance_tex
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parent
SECT = json.loads((PAPER / "_cons_sections.json").read_text(encoding="utf-8"))
OUT = PAPER / "consonance_paper.tex"

TITLE = ("A unified harmonic-resonance descriptor of consonance across musical chords, "
         "frequency-tagging, and the brainstem frequency-following response")
AUTHORS = r"Antoine Bellemare-Pepin\textsuperscript{1,2}, Karim Jerbi\textsuperscript{1}"
AFFIL = (r"\textsuperscript{1}CoCo Lab, Psychology Department, University of Montr\'eal, Canada \\ "
         r"\textsuperscript{2}Music Department, Concordia University, Montr\'eal, Canada")

RESULTS = [
    ("construct", "One descriptor, applied across paradigms",            "cons_Fig1_construct"),
    ("chords",    "Acoustic consonance is read directly from the spectrum", "cons_Fig2_chords"),
    ("ssvep",     "Frequency-tagging: intermodulation as n:m phase coupling", "cons_Fig3_ssvep"),
    ("ffr",       "The brainstem response is more harmonic for consonant dyads", "cons_Fig4_ffr"),
    ("synthesis", "Cross-paradigm convergence on a single axis",          "cons_Fig5_synthesis"),
]

CAPS = {
"cons_Fig1_construct": (
    r"\textbf{The descriptor.} \textbf{(A)} Harmonicity $H$ is phase-blind: it is flat across the "
    r"imposed phase-locking $\kappa$ and ordered only by ratio complexity (colour). \textbf{(B)} The "
    r"product $R=H\cdot\mathrm{PC}$ is an interpretable decomposition, not a better detector than "
    r"$\mathrm{PC}$ (specificity AUC). This licenses $H$ as the single-channel consonance read-out used "
    r"across all paradigms."),
"cons_Fig2_chords": (
    r"\textbf{Acoustic consonance from the spectrum.} \textbf{(A)} Across just-intonation chords, "
    r"harmonicity $H_{\max}$ decreases with chord complexity (Tenney height; more dissonant), with and "
    r"without an auditory nonlinearity that generates combination tones; the Spearman $\rho$ and its "
    r"permutation $p$ are reported. \textbf{(B)} Whether the nonlinearity \emph{sharpens} the "
    r"relationship (bootstrap of $|\rho_\mathrm{nonlinear}|-|\rho_\mathrm{linear}|$ with 95\% CI)."),
"cons_Fig3_ssvep": (
    r"\textbf{Frequency-tagging: intermodulation as n:m coupling.} \textbf{(A)} Single-flicker "
    r"harmonicity is at ceiling (a sanity check). \textbf{(B)} With two flickers, the intermodulation "
    r"index and the framework's n:m phase coupling between the two driven components both rise with "
    r"neural nonlinearity --- the phase-coupling reading of intermodulation. \textbf{(C)} Coupling is "
    r"stronger for simpler frequency ratios."),
"cons_Fig4_ffr": (
    r"\textbf{The brainstem FFR is more harmonic for consonant dyads.} \textbf{(A)} Neural harmonicity "
    r"by condition (consonant CC/CI vs dissonant DC/DI), including the missing-fundamental conditions "
    r"(CI/DI). \textbf{(B)} The CI$>$DI advantage lives in the stimulus-\emph{silent} low band (neurally "
    r"generated), not the high band where stimulus energy exists (leakage would land there). "
    r"\textbf{(C)} The neural effect exceeds the acoustic-stimulus baseline. \textbf{(D)} Honest null: "
    r"the per-listener neural advantage does not scale with musicianship (AMMA)."),
"cons_Fig5_synthesis": (
    r"\textbf{Cross-paradigm convergence.} A single harmonic-resonance descriptor recovers a "
    r"consonance-aligned effect in all three paradigms --- chord harmonicity, SSVEP n:m coupling, and "
    r"the FFR consonant$>$dissonant advantage --- each significant and in the same direction. Effects "
    r"are shown on a common standardized (0--1) axis; the units differ across paradigms, so the panel "
    r"summarizes direction and significance, not a pooled estimate."),
}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{textcomp}
\usepackage{amsmath}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{graphicx,xcolor,booktabs}
\usepackage[labelfont={bf,color=consblue},font=small,labelsep=period,justification=justified,singlelinecheck=false]{caption}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\definecolor{consblue}{HTML}{6A1B9A}
\definecolor{rulegray}{HTML}{B8B8B8}
\titleformat{\section}{\sffamily\bfseries\large\color{consblue}}{}{0em}{}
\titleformat{\subsection}{\sffamily\bfseries\normalsize\color{black!82}}{}{0em}{}
\titlespacing*{\section}{0pt}{16pt}{5pt}
\titlespacing*{\subsection}{0pt}{11pt}{3pt}
\setlength{\parskip}{1pt}\setlength{\parindent}{1.4em}
\linespread{1.05}\setlength{\emergencystretch}{2em}
\pagestyle{fancy}\fancyhf{}
\renewcommand{\headrulewidth}{0.3pt}\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{\footnotesize\itshape A unified harmonic-resonance descriptor of consonance}
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
        surnames = re.findall(r"([A-Z][a-zA-Z'\-]+),", head)
        first_full = head.split(",")[0].strip()
        if len(surnames) >= 3 or head.count("&") > 1:
            disp = f"{first_full} et al."
        elif len(surnames) == 2:
            disp = f"{first_full} \\& {surnames[1]}"
        else:
            disp = first_full
        mp[(_ascii(re.split(r"[\s,\-]+", head)[0]).lower(), year)] = disp
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
    def rp(mode):   # 'p' parenthetical, 't' textual, 'alp' author-year no parens
        def f(m):
            res = [_resolve(k) for k in m.group(1).split(",") if k.strip()]
            if mode == "p":
                return "(" + "; ".join(f"{n}, {y}" if y else n for n, y in res) + ")"
            if mode == "alp":
                return "; ".join(f"{n}, {y}" if y else n for n, y in res)
            return "; ".join(f"{n} ({y})" if y else n for n, y in res)
        return f
    s = re.sub(r"\\citep\{([^}]*)\}", rp("p"), s)
    s = re.sub(r"\\citealp\{([^}]*)\}", rp("alp"), s)
    s = re.sub(r"\\citealt\{([^}]*)\}", rp("t"), s)
    s = re.sub(r"\\citet\{([^}]*)\}", rp("t"), s)
    s = re.sub(r"\\cite\{([^}]*)\}", rp("t"), s)
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
    note = (r"\textit{\small References compiled during a literature-positioning pass; "
            r"verify bibliographic details before submission.}\par\vspace{4pt}")
    lines = "\n".join(rf"\item {esc(r)}" for r in refs)
    return ("\\section*{References}\n" + note + "\n\\begingroup\\small\n\\begin{list}{}{\\setlength{\\itemsep}{1pt}"
            "\\setlength{\\leftmargin}{1.2em}\\setlength{\\itemindent}{-1.2em}}\n" + lines + "\n\\end{list}\\endgroup\n")


def figure_tex(key):
    return ("\\begin{figure}[t]\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{figures/{key}.pdf}}\n"
            f"\\caption{{{CAPS[key]}}}\n\\label{{fig:{key}}}\n\\end{{figure}}\n")


def main():
    s = {k: (decite(v) if isinstance(v, str) else v) for k, v in SECT.items()}
    parts = [PREAMBLE]
    parts.append(rf"""\thispagestyle{{plain}}
\begin{{center}}
{{\LARGE\sffamily\bfseries\color{{consblue}} {TITLE}\par}}
\vspace{{11pt}}{{\large {AUTHORS}\par}}
\vspace{{5pt}}{{\small\itshape {AFFIL}\par}}
\end{{center}}
\vspace{{3pt}}{{\color{{rulegray}}\hrule height 0.6pt}}\vspace{{8pt}}
\noindent{{\sffamily\bfseries\color{{consblue}}Abstract}}\par\vspace{{2pt}}
{{\small {s['abstract']}\par}}
\vspace{{7pt}}{{\color{{rulegray}}\hrule height 0.6pt}}\vspace{{10pt}}
""")
    parts.append(r"\section{Introduction}" + "\n" + s["introduction"] + "\n")
    parts.append(r"\section{Results}" + "\n")
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

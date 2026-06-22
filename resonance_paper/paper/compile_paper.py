"""Compile the methods-paper markdown draft + figures into a single PDF.

Embeds each composite figure inline after its Results subsection (3.1->Fig1 ... 3.9->Fig9),
pulls the one-line captions from the draft's "## Figures" list, renders via weasyprint.

    python -m resonance_paper.paper.compile_paper
Output: paper/methods_paper_compiled.pdf
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown
from weasyprint import HTML

PAPER = Path(__file__).resolve().parent
SRC = PAPER / "resonance_paper_draft.md"
OUT = PAPER / "methods_paper_compiled.pdf"

FIGFILE = {
    1: "method_Fig1_dissociation", 2: "method_Fig2_R_justification", 3: "method_Fig3_ground_truth",
    4: "method_Fig4_operating", 5: "method_Fig5_baselines", 6: "method_Fig6_strategy_mechanism",
    7: "method_Fig7_real_biosignals", 8: "method_Fig8_connectivity", 9: "method_Fig9_descriptors",
}

CSS = """
@page { size: A4; margin: 1.8cm 1.9cm; @bottom-center { content: counter(page); font: 9px Georgia; color:#666; } }
body { font: 10.5px/1.5 Georgia, 'Times New Roman', serif; color:#111; }
h1 { font-size: 19px; text-align:center; line-height:1.25; margin:0 0 4px; }
h2 { font-size: 14px; border-bottom:1px solid #ccc; padding-bottom:2px; margin:16px 0 6px; }
h3 { font-size: 11.5px; margin:12px 0 4px; color:#222; }
p, li { text-align: justify; }
em { color:#333; }
code { font-family:'Consolas','DejaVu Sans Mono',monospace; font-size:9.2px; background:#f4f4f4; padding:0 2px; }
pre { background:#f4f4f4; border:1px solid #e2e2e2; border-radius:3px; padding:6px 8px; font-size:9px; white-space:pre-wrap; }
blockquote { border-left:3px solid #bbb; margin:8px 0; padding:2px 10px; color:#444; background:#fafafa; font-size:9.8px; }
table { border-collapse:collapse; width:100%; font-size:9.2px; margin:6px 0; }
th, td { border:1px solid #ccc; padding:3px 5px; text-align:left; vertical-align:top; }
th { background:#f0f0f0; }
figure { margin:10px 0 14px; text-align:center; page-break-inside:avoid; }
figure img { max-width:100%; }
figcaption { font-size:8.8px; color:#333; text-align:justify; margin-top:3px; }
.subtitle { text-align:center; font-size:9px; color:#666; margin:0 0 10px; }
hr { border:none; border-top:1px solid #ddd; margin:10px 0; }
a { color:#2a4d7a; text-decoration:none; }
"""


def main():
    md = SRC.read_text(encoding="utf-8")

    # captions from the "## Figures" bullet list: "- **Fig N — ...** (`file`): caption"
    caps = {}
    for m in re.finditer(r"- \*\*Fig (\d+)[^\n]*?\*\*[^:\n]*:\s*(.+)", md):
        caps[int(m.group(1))] = m.group(2).strip()

    # drop the now-redundant "## Figures" list (figures are embedded inline); keep References
    md = re.sub(r"\n## Figures\n.*?(?=\n## References)", "\n", md, flags=re.S)

    def fig_block(n):
        cap = caps.get(n, "")
        return (f'\n\n<figure><img src="figures/{FIGFILE[n]}.png" alt="Figure {n}"/>'
                f'<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>\n\n')

    # insert each composite figure at the end of its Results subsection (### 3.N)
    out, cur = [], None
    for ln in md.split("\n"):
        if (ln.startswith("### ") or ln.startswith("## ")) and cur is not None:
            out.append(fig_block(cur)); cur = None
        out.append(ln)
        m = re.match(r"### 3\.(\d)\b", ln)
        if m:
            cur = int(m.group(1))
    if cur is not None:
        out.append(fig_block(cur))
    md = "\n".join(out)

    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    HTML(string=html, base_url=str(PAPER)).write_pdf(str(OUT))
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()

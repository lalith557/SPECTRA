"""
Build a modular Overleaf project (paper_final/) from the validated single-file
paper_ieee/main.tex: main.tex + sections/*.tex + figures/ + references.bib.
Content is identical to the validated paper; only the file layout changes.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "paper_ieee" / "main.tex").read_text(encoding="utf-8")
OUT = ROOT / "paper_final"
(OUT / "sections").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# --- extract pieces from the source ---
abstract = re.search(r"(\\begin\{abstract\}.*?\\end\{abstract\})", SRC, re.S).group(1)
keywords = re.search(r"(\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\})", SRC, re.S).group(1)
body_start = SRC.index(r"\section{Introduction}")
body_end = SRC.index(r"\begin{thebibliography}")
body = SRC[body_start:body_end].rstrip()

# --- split body at \section / \section* boundaries ---
parts = re.split(r"(?=^\\section\*?\{)", body, flags=re.M)
parts = [p for p in parts if p.strip()]


def slug(title):
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s[:34]


inputs = []
for i, part in enumerate(parts, 1):
    m = re.match(r"\\section\*?\{([^}]*)\}", part)
    title = m.group(1) if m else f"section{i}"
    fname = f"{i:02d}_{slug(title)}.tex"
    (OUT / "sections" / fname).write_text(part.rstrip() + "\n", encoding="utf-8")
    inputs.append(f"\\input{{sections/{fname[:-4]}}}")

(OUT / "sections" / "00_abstract.tex").write_text(abstract + "\n", encoding="utf-8")

# --- copy figures the paper uses ---
for f in (ROOT / "paper_ieee" / "figures").glob("*.pdf"):
    (OUT / "figures" / f.name).write_bytes(f.read_bytes())

# --- main.tex ---
main = r"""%==============================================================================
% SPECTRA -- modular Overleaf project (main.tex).
% Compiles out-of-the-box in Overleaf: pdfLaTeX -> BibTeX -> pdfLaTeX x2.
% IEEEtran + IEEEtran.bst are bundled with Overleaf; no external style needed.
%==============================================================================
\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{xcolor}
\usepackage{textcomp}
\usepackage{url}
\usepackage[hidelinks]{hyperref}

\graphicspath{{figures/}}

\newcommand{\spectra}{\textsc{Spectra}}
\newcommand{\ofcv}{\textsc{ofcv}}
\newcommand{\brf}{\textsc{brf}}
\newcommand{\dinov}{DINOv2}
\newcommand{\iou}{IoU}
\newcommand{\best}[1]{\textbf{#1}}

\begin{document}

\title{SPECTRA: Representation over Refraction --- A Controlled Study of\\
Physics-Motivated Cues for Transparent-Object Segmentation}

\author{\IEEEauthorblockN{Lalith Gona}
\IEEEauthorblockA{\textit{Department of Computer Science and Engineering}\\
\textit{Lovely Professional University}\\
Punjab, India\\
gonalalith2005@gmail.com}}

\maketitle

\input{sections/00_abstract}

""" + keywords + "\n\n" + "\n".join(inputs) + r"""

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""
(OUT / "main.tex").write_text(main, encoding="utf-8")

print(f"wrote {OUT/'main.tex'}")
print(f"sections: {len(inputs)+1}  figures: {len(list((OUT/'figures').glob('*.pdf')))}")
for x in inputs:
    print("  ", x)

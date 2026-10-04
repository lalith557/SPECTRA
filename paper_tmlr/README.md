# SPECTRA — TMLR submission

`main.tex` is the complete paper in TMLR format (`\documentclass{article}` +
`\usepackage{tmlr}`), with all tables, figures, and the embedded 45-reference
bibliography. Validated error-free.

## How to compile (one requirement: the TMLR style file)
TMLR mandates `tmlr.sty` (and `tmlr.bst`), which ship with the official
template, **not** this folder. Do this once:

1. Open the official **TMLR** Overleaf template (the TMLR site links it; it
   contains `tmlr.sty`/`tmlr.bst`).
2. Replace the template's `main.tex` with **this** `main.tex`.
3. Upload the `figures/` folder.
4. Compiler = **pdfLaTeX**. Compile once — the bibliography is embedded, so no
   bibtex pass is needed.

For the camera-ready (de-anonymised) version, change the preamble to
`\usepackage[accepted]{tmlr}`.

## Why TMLR is the recommended venue
TMLR's acceptance criteria are (1) *are the claims supported by accurate,
convincing evidence* and (2) *would some of TMLR's audience be interested* —
**novelty and SOTA are explicitly out of scope**. This is a rigorously
controlled negative result, which fits those criteria far better than a
novelty-gated venue (CVPR/ICCV). See `../paper_ieee/COVER_LETTER_TMLR.md` for the
submission letter mapped to both criteria.

- TMLR has **no page limit**; the full paper (all analyses, appendices) is
  appropriate as-is.
- All numbers are real and traceable to released artifacts.

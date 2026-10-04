# SPECTRA — BMVC submission

`main.tex` is the complete paper in BMVC format (`\documentclass{bmvc2k}`),
with all tables, figures, and the embedded 45-reference bibliography. It is
validated error-free (balanced braces, all citations resolved, no undefined
references).

## How to compile (one requirement: the BMVC class file)
BMVC mandates `bmvc2k.cls`, which is **not** in this folder (it ships with the
official template). Do this once:

1. Open the official **BMVC** template on Overleaf (search the Overleaf
   template gallery for "BMVC", or download `bmvc2k.cls` from the BMVC author
   kit and drop it next to `main.tex`).
2. Replace the template's `main.tex` with **this** `main.tex`.
3. Upload the `figures/` folder.
4. Compiler = **pdfLaTeX**. Compile once — no bibtex pass needed (the
   bibliography is embedded).

For the anonymous review copy, uncomment `\def\bmvcreviewcopy{}` in the preamble.

## Notes
- BMVC has a **9-page limit** (excluding references). This draft is written to
  fit; if it runs long, the Robustness table and one of the qualitative figures
  are the safest cuts.
- All numbers are real and traceable to released artifacts (see
  `../paper_ieee/README.md` for the provenance table).

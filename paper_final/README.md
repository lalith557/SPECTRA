# SPECTRA — modular Overleaf project (`paper_final/`)

Publication-ready, **modular** LaTeX source. Compiles out-of-the-box in Overleaf
(IEEEtran + IEEEtran.bst are bundled there — no external style file needed).

## Layout
```
paper_final/
  main.tex                 preamble + \input of all sections + \bibliography
  references.bib           46 real references, chronological order (2012–2024)
  sections/                one .tex per section (00_abstract … 20_code_availability)
  figures/                 the 8 figure PDFs used by the paper
  README.md
```

## Build (Overleaf)
Upload the folder → set compiler to **pdfLaTeX** → Recompile.
Overleaf runs the full `pdflatex → bibtex → pdflatex → pdflatex` cycle automatically.

## Build (local)
```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

## Notes
- **Every number is real** and traceable to the consolidated `../EXPERIMENTS/`
  folder (see `../EXPERIMENTS/STATUS.md` for the provenance map).
- Tables and the OFCV algorithm are kept inline within their section files
  (standard practice); the `sections/` split is the modular unit.
- This is the **IEEE-conference** layout (compiles immediately). For a TMLR
  submission — the recommended venue — swap the preamble for the TMLR template
  (see `../paper_tmlr/`); the `sections/` and `references.bib` are reusable
  unchanged.
- Validated: braces balanced, 46/46 citations resolve, 0 undefined references,
  0 environment mismatches.

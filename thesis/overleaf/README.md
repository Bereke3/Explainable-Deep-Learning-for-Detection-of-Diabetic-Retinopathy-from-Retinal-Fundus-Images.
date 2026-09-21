# DR Thesis: Overleaf Writing Scaffold

This is a writing scaffold, not a finished thesis. Every `Writing note`,
`Figure placeholder`, and `Table placeholder` must be replaced before submission.
No completed experimental chapter or fabricated literature review is included.

## Preserved Material

`front/frontespice.tex` and `unime-logo.jpg` are copied unchanged from the supplied
Overleaf ZIP. The cover still reads **Diabetic Retinopathy Detection using CNN**,
with **Baimuratov Bereke**, **Prof. Ravi Daniele**, and **2025/2026**. No correction
of the title or names has been assumed. PDF metadata in `main.tex` matches the
cover; update both if the supervisor later approves a different title.

The original A4/report/12 pt layout and spacing settings are retained. Front
matter uses Roman page numbers; Chapter 1 starts Arabic numbering at 1.
Abstract and acknowledgement precede the contents. The unused list of code
listings is not included. This is not a claim that a particular front-matter
order is mandatory: follow the supervisor's current requirements.

## Opening in Overleaf

Upload the prepared ZIP as a new project to keep the original project as a backup.
Select `main.tex` as the main document and use pdfLaTeX. Recompile until contents,
references, figure/table lists, and bibliography have updated. Overleaf normally
runs BibTeX automatically. The starter bibliography includes three verified
method references cited in the writing notes, not a completed literature review.

## Where to Write

| File | Purpose |
| --- | --- |
| `main.tex` | Document assembly, metadata, and shared formatting |
| `front/frontespice.tex` | Your existing title page |
| `front/abstract.tex` | Abstract, written after the main chapters |
| `front/acknowledgement.tex` | Personal acknowledgement |
| `body/chapter1.tex` | Introduction; agreed starting point |
| `body/chapter2.tex` | Background and Related Work |
| `body/chapter3.tex` | Materials and Methods |
| `body/chapter4.tex` | Experimental Results |
| `body/chapter5.tex` | Discussion |
| `body/chapter6.tex` | Conclusion and Future Work |
| `appendices/data-and-reproducibility.tex` | Detailed preparation and reproduction |
| `appendices/exploratory-experiments.tex` | Historical binary and multiclass work |
| `blibliography.bib` | BibTeX database; original template spelling retained |
| `figures/` | Thesis-ready figures, not screenshots of notebook cells |
| `MATERIALS_MAP.md` | Map from thesis slots to existing project evidence |
| `writing-tools.tex` | Clearly marked writing-note and placeholder helpers |

Write one chapter at a time, from Introduction through Conclusion, as agreed
with the author. First agree the chapter outline, then draft and review its
sections; do not draft later chapters before the author is ready to move on.
Leave the Abstract as a placeholder until the main text is settled. The
six-chapter plan is adaptable, not a formal university requirement.

## Replacing Placeholders

Replace each `\writingnote{...}` with your own finished paragraphs. The notes
are intentionally visible so the scaffold cannot be mistaken for a final draft.
Do not merely hide the notes and submit empty sections.

Replace `\figureplaceholder{caption}{label}{instructions}` with a standard
`figure` containing an exported image, retaining its `\caption` and `\label`.
Replace table placeholders with a `table` using `tabular`, `tabularx`, or another
suitable table environment. Labels follow captions. Refer to figures and tables
using `\ref{...}`, and cite sources using `\cite{...}`.

Use a single Figure sequence for plots, photographs, diagrams, and Grad-CAM
panels, and a separate Table sequence. LaTeX numbers them by chapter. Existing
notebook numbers such as 22.1 are provenance identifiers, not thesis numbers.
Re-export plots without an embedded notebook number to avoid double numbering.
Do not copy image panels so small that their labels cannot be read.

## Evidence Rules

- Use one dataset name, APTOS 2019, but distinguish exploratory and final splits.
- Keep the final preparation description short; put technical checks in Appendix A.
- Use only the final 354-image evaluation for headline final B0 results.
- Historical model scores are not a final-protocol architecture ranking.
- Describe Grad-CAM as qualitative attribution, not verified lesion detection.
- Include only actual experiments and checked references; mark unfinished work.
- Keep patient independence, external validation, and selection-history limitations.

## Scaffold Validation

The scaffold was compiled locally with Tectonic 0.17.0, including BibTeX and
cross-reference passes. The preview contains 23 pages of structure and visibly
marked writing notes, not 23 pages of completed thesis prose. All referenced
files, labels, and citation keys were checked. The personalized cover and logo
were verified byte-for-byte against the supplied ZIP.

The original cover produces a small overfull-line warning (2.608 pt) on the
supervisor/candidate row; the inspected preview remains readable. That row was
not changed because the cover is intentionally preserved. Overleaf compilation
itself has not been run from this workspace.

## Template Attribution

Adapted from Lorenzo Carnevale's *UNIME - Computer Science's Bachelors Thesis
Template*, distributed under Creative Commons CC BY 4.0:
https://www.overleaf.com/latex/templates/unime-computer-sciences-bachelors-thesis-template/nyjjqxrnhzzs

License: https://creativecommons.org/licenses/by/4.0/

Changes: six-chapter outline, writing prompts, figure/table slots, appendices,
starter bibliography, front-matter numbering, PDF metadata, and writing guidance.
The supplied personalized cover and university logo are unchanged. Template
licensing does not imply rights to redistribute datasets or third-party figures.

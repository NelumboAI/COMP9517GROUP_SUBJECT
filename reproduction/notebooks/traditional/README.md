# Traditional-method notebooks

This directory contains notebook interfaces for reproducing, monitoring, and
analysing the traditional computer-vision experiments.

The teammates' original Python implementation remains unchanged under:

`reproduction/working_code/src/traditional/`

Notebooks in this directory should import or invoke those modules rather than
duplicate their algorithm implementations. Generated models, feature caches,
predictions, tables, and figures must be written under a unique directory in:

`reproduction/outputs/traditional/`

Planned notebook:

- `traditional_reproduction.ipynb`: data overview, experiment monitoring,
  results tables, confusion matrices, error examples, and comparisons.

Do not launch a second copy of an experiment while the same run is active in
the background.

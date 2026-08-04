# Reproduction workspace

This directory keeps the code, configuration, manifests, and outputs produced
while reproducing the teammates' COMP9517 experiments. It is intentionally
separate from the original implementation under `src/`.

## Layout

- `configs/`: reproduction-specific paths and experiment settings.
- `scripts/`: safe command-line scripts used for reproduction.
- `notebooks/`: reproduction notebooks, including Grad-CAM analysis.
- `manifests/`: exact selected categories and image lists.
- `outputs/`: reproduced models, metrics, plots, and Grad-CAM images.
- `logs/`: terminal and training logs.

The large image datasets live under `data/inat2021/reproduced/` and are ignored
by Git. The `outputs/` and `logs/` directories are also ignored so that large
generated artifacts are not committed accidentally.

## Safety rule

Reproduction scripts must refuse to overwrite a non-empty dataset or experiment
directory unless an explicit, separately reviewed overwrite option is provided.

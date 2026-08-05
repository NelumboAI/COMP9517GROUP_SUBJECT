# Deep-model explainability notebooks

`gradcam_analysis.ipynb` presents the isolated outputs produced by
`reproduction/scripts/run_gradcam_analysis.py`.

The formal run compares six teammate checkpoints on the same deterministic
set of confident correct and incorrect examples. The resulting case-set
accuracy is descriptive only: the examples were deliberately selected with
the pretrained ResNet50 and are not an unbiased test sample.

Large checkpoints and generated figures remain under ignored `data/` and
`reproduction/outputs/` directories and must not be committed.

`robustness_analysis.ipynb` presents the fixed-model test-time degradation
study for Gaussian noise, Gaussian blur, motion blur, and JPEG compression.
It reads the full 5,000-image results and report-ready curves without
retraining any model or modifying any dataset image on disk.

The CUDA packages used for these experiments are pinned in
`reproduction/requirements-deep-cu128.txt`. Install them through the project
environment with:

```powershell
.\.9517venv\Scripts\python.exe -m pip install -r reproduction\requirements-deep-cu128.txt
```

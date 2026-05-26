# Reproduction status for arXiv:2602.13008

Date: 2026-05-26

This is the local audit record from pulling the paper and public notebook repo. The referenced `paper/`, `code/`, and `artifacts/` paths are intentionally not tracked in this repository because they are either upstream/copyrighted artifacts, cloned third-party code, generated logs, or large local run products.

## Pulled artifacts

- Paper PDF: `paper/2602.13008.pdf`
- Extracted paper text: `paper/2602.13008.txt`
- arXiv API metadata: `paper/arxiv_api_2602.13008.xml`
- Public code repository: `code/MRP`
  - Remote: `https://github.com/MIntelligence-Group/MRP.git`
  - Commit checked out: `84e1ea3 Add files via upload`

## Public code contents

The public repository contains only:

- `1_Step1_SMOTE_Up-sampling.ipynb`
- `2_Step2_Binary_Classification.ipynb`
- `3_Step3_Results_Plots.ipynb`
- `ReadMe`

No `Data/`, `Results/`, `Plotting_Archive/`, releases, or large-file assets are present in the public repo.

## Data/code availability findings

The paper's data availability statement says the supporting data are available from the corresponding author upon reasonable request, subject to Mass General Brigham IRB approval and institutional data-sharing policy.

The paper's code availability statement points to `https://github.com/MIntelligence-Group`. The matching public repo is `MIntelligence-Group/MRP`.

Public searches for the required filenames did not find copies of:

- `reho_output.csv`
- `reho_output_upsampled.csv`
- `reho_residuals.csv`
- `important ROI ranking v3_hypothesized-rois.csv`
- `results_after_RA.txt`
- `results_before_RA.txt`

## Local environment

Created `.venv` and installed the notebook dependencies. The exact package freeze is in:

- `artifacts/requirements.freeze.txt`

Notebook code was extracted for inspection/run attempts into:

- `artifacts/mrp_scripts/1_Step1_SMOTE_Up-sampling.py`
- `artifacts/mrp_scripts/2_Step2_Binary_Classification.py`
- `artifacts/mrp_scripts/3_Step3_Results_Plots.py`

## Run attempts

Commands were run from `code/MRP` so the notebooks' `../Data/...` paths resolve the same way they would in Jupyter.

Step 1:

```bash
. ../../.venv/bin/activate
python ../../artifacts/mrp_scripts/1_Step1_SMOTE_Up-sampling.py
```

Result: failed because `../Data/1ReHo/reho_output.csv` is not present.

Step 2:

```bash
. ../../.venv/bin/activate
python ../../artifacts/mrp_scripts/2_Step2_Binary_Classification.py
```

Result: failed because `../Data/1ReHo/reho_output_upsampled.csv` is not present.

Step 3:

```bash
. ../../.venv/bin/activate
python ../../artifacts/mrp_scripts/3_Step3_Results_Plots.py
```

Result: failed because `1No-upsampled_NoCV.txt` is not present.

Logs:

- `artifacts/step1_run.log`
- `artifacts/step2_run.log`
- `artifacts/step3_run.log`

## Required files to complete the reported reproduction

Place these under `code/Data` relative to the checked-out notebook directory:

- `code/Data/1ReHo/reho_output.csv`
- `code/Data/important ROI ranking v3_hypothesized-rois.csv`

For residual analysis and plot reproduction, also provide:

- `code/Data/1ReHo/reho_residuals.csv`
- result text outputs such as `results_before_RA.txt` and `results_after_RA.txt`, or regenerate them by rerunning Step 2 on raw and residual inputs.

## Reproducibility notes from code inspection

- The public repo does not include an environment file or dependency versions.
- The README references plotting notebooks under `Plotting_Archive/src/`, but that folder is not in the public repo.
- The notebook contains both "v4.2 without CV" and "v4.3 with 5-fold Stratified CV" cells. Running the extracted script runs both sections sequentially; in Jupyter, the intended path appears to be running only the recommended v4.3 section.
- The public Step 2 code selects the ensemble's top three models by held-out case-study test accuracy. The paper's pseudocode says top models are selected by validation metric, e.g. Cohen's kappa.
- The public Step 2 code uses ordinary `StratifiedKFold` over rows in the upsampled group dataset. The paper's pseudocode describes subject-wise stratified folds.
- Step 1 performs upsampling before classification and writes a global `reho_output_upsampled.csv`. The paper's pseudocode describes SMOTE inside training folds only.
- The public Step 2 code defines 24 binary pairs, while the paper's extended results summarize 19 pairs. This may be due to skipped pairs in the unavailable data, but it cannot be verified without the inputs.

## Current conclusion

The paper and public code have been pulled, and the local environment can import the required libraries. Full numerical reproduction is currently blocked by missing non-public data and ancillary CSV/text artifacts.

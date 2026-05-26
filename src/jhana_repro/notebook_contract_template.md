# MRP Notebook Contract

This contract is derived from the public `MIntelligence-Group/MRP` notebooks and the extracted scripts in `artifacts/mrp_scripts`.

## `reho_output.csv`

- Producer missing from public repo.
- Consumer: `1_Step1_SMOTE_Up-sampling.ipynb`.
- One row represents one subject/run/segment sample.
- Required metadata columns:
  - `subid`: subject identifier.
  - `run`: run identifier used as a temporary class label during the notebook's per-subject/per-segment SMOTE step.
  - `segment`: condition/class label.
- All non-metadata numeric columns are ROI-level ReHo features.

## `reho_output_upsampled.csv`

- Producer: `1_Step1_SMOTE_Up-sampling.ipynb`.
- Consumer: `2_Step2_Binary_Classification.ipynb`.
- Same schema as `reho_output.csv`.
- The notebook splits `sub-001` out as case-study data before augmentation.
- The notebook augments only `subid != sub-001`, per subject and per segment, then concatenates unmodified `sub-001` rows back into one CSV.
- Target rows per subject/segment in the notebook are `27` for `j1`-`j8`, `16` for `counting`, and `16` for `memory`.

## ROI Ranking CSV

- Expected path: `../Data/important ROI ranking v3_hypothesized-rois.csv`.
- Required columns:
  - `Label Name`
  - `Ranking`
  - `Atlas`
- The classifier keeps rows with `Ranking == 1`, replaces `-` with `_` in labels, then intersects those labels with numeric ReHo columns.
- This ranking file is specific to the paper's 498-ROI atlas stack. It is not used for the public Schaefer-100 run.

## Classification Outputs

- `output_{timestamp}.txt`: console text mirrored to file.
- `scores_{timestamp}/feature_importance_{ModelName}_{label1}_vs_{label2}.csv`.
- `scores_{timestamp}/average_feature_importance_{label1}_vs_{label2}.csv`.

## Verified Methodological Ordering

- Step 2 loads `reho_output_upsampled.csv` before cross-validation.
- Step 2 applies RFE to the full pair-specific training matrix before row-level CV.
- Step 2 uses `StratifiedKFold` over already-upsampled rows in the group dataset.
- Step 2 selects the top three ensemble members by held-out case-study `Test Accuracy`.
- The public rebuild keeps this behavior only in `compat` mode and defaults to a leak-safe mode.

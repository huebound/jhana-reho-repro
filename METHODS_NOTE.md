# Methods Note

This note records verified implementation details from the public `MIntelligence-Group/MRP` notebooks and how this rebuild handles them.

## Verified Public Notebook Behavior

The original public code has three notebooks:

- `1_Step1_SMOTE_Up-sampling.ipynb`
- `2_Step2_Binary_Classification.ipynb`
- `3_Step3_Results_Plots.ipynb`

The first notebook loads `../Data/1ReHo/reho_output.csv`, splits out `sub-001` as the case-study subject, upsamples only `subid != sub-001`, then writes one combined `../Data/1ReHo/reho_output_upsampled.csv`.

The second notebook loads `reho_output_upsampled.csv`, splits group rows from case rows, applies RFE to the full pair-specific training matrix, then runs row-level `StratifiedKFold` over that already-upsampled matrix. It fits models on the full group training data, evaluates on the held-out case subject, then selects the top three ensemble members by held-out test accuracy.

So the public code does not directly SMOTE the held-out `sub-001` case data. The concern is pre-CV augmentation/RFE within the group dataset and model-family selection using the held-out test result.

## Rebuild Behavior

This project provides two modes:

- `leak-safe`: split subjects first, fit RFE inside training folds, apply SMOTE only to fold-train data, select ensemble members by validation Cohen's kappa, and evaluate once on untouched test subjects.
- `compat`: preserve the public notebook's broad behavior for comparison by augmenting training rows before row-level CV, fitting RFE before CV, and selecting ensemble members by held-out test accuracy.

The outputs `method_comparison.csv` and `method_diagnostics.csv` report both modes on public `ds002748` data. This quantifies the effect of implementation choices on a public stand-in dataset. It is not a statement about the private ACAM-J data or the reported 66.82% result.

## Public Stand-In Result

The local public smoke run used 20 balanced `ds002748` subjects: `sub-01` through `sub-10` for depression and `sub-52` through `sub-61` for controls.

Held-out test accuracy was `0.40` in both modes on this small public split. Compat mode raised held-out AUC from `0.50` to `0.667`, but the larger diagnostic was the validation/test gap:

- `leak-safe`: top-model CV accuracy mean `0.689`, held-out accuracy `0.40`.
- `compat`: top-model CV accuracy mean `0.994`, held-out accuracy `0.40`.

The neutral interpretation is that row-level CV over augmented rows can produce highly optimistic validation estimates on this public stand-in dataset. This does not prove the private ACAM-J result is inflated; it shows why the leak-safe mode is included and why subject-level validation should be preferred for any rerun on the private data.

## Atlas and ROI Selection

The paper reports a 498-ROI feature space: Schaefer-400, Tian-32, Bianciardi-54, and MDTB-10. The public run intentionally uses Schaefer-100 only. This keeps the proof of machinery small and reproducible.

The original ROI ranking file is tied to the unpublished 498-ROI feature space. Public mode therefore uses all Schaefer-100 ROI columns before RFE. This validates the pipeline path from ReHo to classification metrics, not the paper's hypothesis-driven ROI-selection scheme.

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

The completed public run used all 72 `ds002748` subjects: 51 depression subjects and 21 healthy controls. Public mode used 100 Schaefer-100 ReHo ROI columns.

On the single full-dataset split, the leak-safe mode reached accuracy `0.667`, AUC `0.769`, and F1 `0.518`. Compat mode reached accuracy `0.722`, AUC `0.677`, and F1 `0.557`.

The stronger diagnostic is the 50-seed benchmark:

- `leak-safe`: test accuracy `0.640 +/- 0.099`, top-model CV accuracy `0.663 +/- 0.046`, CV-test accuracy gap `0.023 +/- 0.111`.
- `compat`: test accuracy `0.693 +/- 0.083`, top-model CV accuracy `0.971 +/- 0.044`, CV-test accuracy gap `0.278 +/- 0.086`.
- `compat - leak-safe`: test accuracy `+0.053 +/- 0.097`, top-model CV accuracy `+0.308 +/- 0.071`, CV-test accuracy gap `+0.255 +/- 0.107`.

The neutral interpretation is that row-level CV over augmented rows can produce highly optimistic validation estimates on this public stand-in dataset. Compat mode had a modest mean held-out accuracy lift, but a much larger validation/test gap. This does not prove the private ACAM-J result is inflated; it shows why the leak-safe mode is included and why subject-level validation should be preferred for any rerun on the private data.

The ReHo front-end was checked against AFNI `3dReHo` on four mixed public subjects: `sub-01`, `sub-25`, `sub-52`, and `sub-70`. All four passed the repository threshold of `0.90`; voxelwise Pearson correlations ranged from `0.9722` to `0.9833`, with mean `0.9765`.

## Atlas and ROI Selection

The paper reports a 498-ROI feature space: Schaefer-400, Tian-32, Bianciardi-54, and MDTB-10. The public run intentionally uses Schaefer-100 only. This keeps the proof of machinery small and reproducible.

The original ROI ranking file is tied to the unpublished 498-ROI feature space. Public mode therefore uses all Schaefer-100 ROI columns before RFE. This validates the pipeline path from ReHo to classification metrics, not the paper's hypothesis-driven ROI-selection scheme.

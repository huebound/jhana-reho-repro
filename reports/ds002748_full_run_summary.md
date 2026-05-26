# ds002748 Full Public Run Summary

- Date: 2026-05-26
- Dataset: OpenNeuro ds002748 fMRIPrep derivatives
- Subjects: 72 (51 depression, 21 control)
- Features: 100 Schaefer-100 ReHo ROI columns
- Finite feature check: True

## Single Split

| mode | accuracy | AUC | F1 | kappa | CV-test accuracy gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| leak-safe | 0.667 | 0.769 | 0.518 | 0.053 | 0.000 |
| compat | 0.722 | 0.677 | 0.557 | 0.151 | 0.265 |

## 50-Seed Benchmark

| mode | test accuracy mean +/- std | test AUC mean +/- std | top-model CV accuracy mean +/- std | CV-test accuracy gap mean +/- std |
| --- | ---: | ---: | ---: | ---: |
| leak-safe | 0.640 +/- 0.099 | 0.645 +/- 0.116 | 0.663 +/- 0.046 | 0.023 +/- 0.111 |
| compat | 0.693 +/- 0.083 | 0.656 +/- 0.104 | 0.971 +/- 0.044 | 0.278 +/- 0.086 |

## Compat Minus Leak-Safe

- Test accuracy: 0.053 +/- 0.097
- Top-model CV accuracy: 0.308 +/- 0.071
- CV-test accuracy gap: 0.255 +/- 0.107

## AFNI ReHo Validation

- Subjects: sub-01, sub-25, sub-52, sub-70
- Passed: 4 / 4
- Pearson r mean: 0.9765
- Pearson r range: 0.9722 to 0.9833
- Threshold: 0.90

# Public Benchmark Report

- ReHo CSV: `runs/ds002748/Data/1ReHo/reho_output_full.csv`
- Seeds: `0` through `49`

## Mode Summary

| mode | n | test accuracy | test AUC | CV-test accuracy gap |
| --- | ---: | ---: | ---: | ---: |
| compat | 50 | 0.693 +/- 0.083 | 0.656 +/- 0.104 | 0.278 +/- 0.086 |
| leak-safe | 50 | 0.640 +/- 0.099 | 0.645 +/- 0.116 | 0.023 +/- 0.111 |

## Compat Minus Leak-Safe

| metric | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| cv_accuracy_mean_all_models | 0.320 | 0.039 | 0.248 | 0.388 |
| cv_accuracy_mean_top_models | 0.308 | 0.071 | 0.142 | 0.410 |
| test_accuracy | 0.053 | 0.097 | -0.222 | 0.222 |
| test_auc | 0.012 | 0.085 | -0.246 | 0.185 |
| test_f1 | -0.009 | 0.112 | -0.366 | 0.174 |
| test_kappa | -0.018 | 0.222 | -0.743 | 0.343 |
| cv_minus_test_accuracy | 0.255 | 0.107 | 0.057 | 0.534 |

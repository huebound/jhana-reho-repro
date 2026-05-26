#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${1:-runs/ds002748}"
N_PERMUTATIONS="${N_PERMUTATIONS:-100}"
SUBJECTS=(
  sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10
  sub-52 sub-53 sub-54 sub-55 sub-56 sub-57 sub-58 sub-59 sub-60 sub-61
)

jhana-repro prepare-ds002748 --work-dir "$WORK_DIR" --clone-derivatives

DERIVATIVES_DIR="$WORK_DIR/ds002748-fmriprep"
paths=()
for subject in "${SUBJECTS[@]}"; do
  paths+=("$subject/func/${subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
  paths+=("$subject/func/${subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz")
  paths+=("$subject/func/${subject}_task-rest_desc-confounds_timeseries.tsv")
done

(
  cd "$DERIVATIVES_DIR"
  datalad get "${paths[@]}"
)

jhana-repro run-public \
  --work-dir "$WORK_DIR" \
  --subjects "${SUBJECTS[@]}" \
  --mode both \
  --n-permutations "$N_PERMUTATIONS"

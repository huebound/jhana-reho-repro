#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${1:-runs/ds002748}"
N_PERMUTATIONS="${N_PERMUTATIONS:-100}"

jhana-repro prepare-ds002748 --work-dir "$WORK_DIR" --clone-derivatives

DERIVATIVES_DIR="$WORK_DIR/ds002748-fmriprep"
PARTICIPANTS="$WORK_DIR/openneuro_ds002748/participants.tsv"
PATH_LIST="$(mktemp)"
python - "$DERIVATIVES_DIR" "$PARTICIPANTS" > "$PATH_LIST" <<'PY'
from pathlib import Path
import sys
import pandas as pd

derivatives = Path(sys.argv[1])
participants = pd.read_csv(sys.argv[2], sep="\t")
for subject in participants["participant_id"].astype(str):
    print(f"{subject}/func/{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    print(f"{subject}/func/{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz")
    print(f"{subject}/func/{subject}_task-rest_desc-confounds_timeseries.tsv")
PY

(
  cd "$DERIVATIVES_DIR"
  datalad get $(cat "$PATH_LIST")
)

rm -f "$PATH_LIST"

jhana-repro run-public \
  --work-dir "$WORK_DIR" \
  --mode both \
  --n-permutations "$N_PERMUTATIONS"

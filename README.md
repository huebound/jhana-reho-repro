# jhana-reho-repro

Public-data reproduction harness for **Machine Learning-Based Classification of Jhana Advanced Concentrative Absorption Meditation (ACAM-J) using 7T fMRI**.

This is not a numerical reproduction of the paper's reported 66.82% accuracy. The paper's data are not public. This project rebuilds the missing front of the public pipeline, runs it on public resting-state fMRI, and produces sane end-to-end outputs so the machinery is ready to point at the private data if access is granted.

## What This Adds

- A ReHo front-end that the public `MIntelligence-Group/MRP` repo does not include.
- A runnable public-data pipeline using OpenNeuro `ds002748`, a closed-eyes resting-state depression/control dataset.
- A leak-safe classifier mode plus a compatibility mode that mirrors key ordering choices in the public notebooks.
- Output files in the same spirit as the notebooks: result text, metrics tables, and ROI feature-importance CSVs.

## Dataset

Default public run:

- Dataset: OpenNeuro `ds002748`
- Description: 51 mild depression subjects and 21 healthy controls, closed-eyes resting-state fMRI
- DOI: `10.18112/openneuro.ds002748.v1.0.5`
- Derivatives: `OpenNeuroDerivatives/ds002748-fmriprep`

This validates the pipeline mechanics, not the paper's ACAM-J ROI hypothesis. Public mode uses Schaefer-100 ROIs and all generated ROI features before RFE. The original `important ROI ranking v3_hypothesized-rois.csv` is specific to the paper's unpublished 498-ROI feature space and is therefore preserved only as a private-data compatibility concept.

## Install

Use Python 3.11-3.13. Neuroimaging wheels are less reliable on newer Python versions.

```bash
python3.13 -m venv .venv313
. .venv313/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For full derivative download, install DataLad/git-annex separately:

```bash
conda install -c conda-forge git-annex
python -m pip install -e ".[data]"
```

AFNI is optional but recommended for ReHo validation:

```bash
# macOS Apple Silicon example
brew install openmotif libxpm gsl
curl -O https://afni.nimh.nih.gov/pub/dist/bin/misc/@update.afni.binaries
tcsh @update.afni.binaries -package macos_13_ARM -bindir "$HOME/abin"
export PATH="$HOME/abin:$PATH"
```

The AFNI installation route above follows the official AFNI macOS 12+ Apple Silicon instructions.

## Run

Prepare metadata and derivative paths:

```bash
jhana-repro prepare-ds002748 --work-dir runs/ds002748 --clone-derivatives
```

If derivatives are already available:

```bash
jhana-repro prepare-ds002748 \
  --work-dir runs/ds002748 \
  --derivatives-dir /path/to/ds002748-fmriprep
```

If using DataLad, get the files needed for the subjects you plan to run:

```bash
cd runs/ds002748/ds002748-fmriprep
datalad get sub-*/func/*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
datalad get sub-*/func/*space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
datalad get sub-*/func/*desc-confounds_timeseries.tsv
```

Run the exact 20-subject public smoke test used for this repository's checked outputs:

```bash
scripts/run_ds002748_20.sh
```

That script downloads fMRIPrep derivatives for:

- depression: `sub-01` through `sub-10`
- control: `sub-52` through `sub-61`

Or run a smaller/manual subject set:

```bash
jhana-repro run-public \
  --work-dir runs/ds002748 \
  --subjects sub-01 sub-02 sub-03 sub-04 sub-52 sub-53 sub-54 sub-55 \
  --mode both \
  --n-permutations 100
```

Run the full public pipeline over all 72 `ds002748` subjects:

```bash
scripts/run_ds002748_full.sh
```

Or run the same stages manually:

```bash
jhana-repro run-public \
  --work-dir runs/ds002748 \
  --mode both \
  --n-permutations 1000
```

Run a repeated-seed benchmark from an existing ReHo CSV:

```bash
jhana-repro benchmark-public \
  --reho runs/ds002748/Data/1ReHo/reho_output.csv \
  --output-dir reports/ds002748_full_benchmark_50seeds \
  --seeds 50 \
  --mode both
```

Outputs are written under:

- `runs/ds002748/Data/1ReHo/reho_output.csv`
- `runs/ds002748/classification/leak-safe/`
- `runs/ds002748/classification/compat/`
- `runs/ds002748/classification/method_comparison.csv`
- `runs/ds002748/classification/method_diagnostics.csv`

## Verified Local Run

On 2026-05-26, the full 72-subject public run completed end-to-end locally:

- ReHo CSV: 72 rows, 100 Schaefer-100 ROI feature columns, 51 depression and 21 control subjects.
- Single split leak-safe metrics: accuracy `0.667`, AUC `0.769`, F1 `0.518`.
- Single split compat metrics: accuracy `0.722`, AUC `0.677`, F1 `0.557`.
- 50-seed leak-safe benchmark: test accuracy `0.640 +/- 0.099`; top-model CV accuracy `0.663 +/- 0.046`; CV-test gap `0.023 +/- 0.111`.
- 50-seed compat benchmark: test accuracy `0.693 +/- 0.083`; top-model CV accuracy `0.971 +/- 0.044`; CV-test gap `0.278 +/- 0.086`.
- Compat minus leak-safe across 50 seeds: test accuracy `+0.053 +/- 0.097`; top-model CV accuracy `+0.308 +/- 0.071`; CV-test gap `+0.255 +/- 0.107`.
- AFNI validation passed for `sub-01`, `sub-25`, `sub-52`, and `sub-70`: mean Python-vs-AFNI voxelwise Pearson `r = 0.9765`, range `0.9722` to `0.9833`.

Tracked summary reports:

- `reports/ds002748_full_run_summary.md`
- `reports/ds002748_full_run_summary.json`
- `reports/ds002748_full_benchmark_50seeds/`
- `reports/afni_reho_validation_mixed.json`

These numbers are a machinery check on public depression/control data. They are not evidence for or against the paper's private ACAM-J accuracy.

## ReHo Validation

Validate one subject against AFNI's `3dReHo`:

```bash
jhana-repro validate-reho-afni \
  --work-dir runs/ds002748 \
  --subject sub-01
```

The validation writes `runs/ds002748/afni_reho_validation.json`. A Pearson correlation below `0.90` should block release until investigated. If AFNI is not installed, validation is explicitly marked as skipped.

Validate multiple subjects and write a tracked-size summary:

```bash
jhana-repro validate-reho-afni \
  --work-dir runs/ds002748 \
  --subjects sub-01 sub-25 sub-52 sub-70 \
  --output reports/afni_reho_validation_mixed.json
```

## Methods Note

The public MRP notebooks do not upsample the held-out `sub-001` case subject. The verified ordering issue is narrower:

- the classifier loads a globally upsampled group dataset before cross-validation,
- RFE is fitted before row-level cross-validation,
- the top three ensemble members are selected by held-out case-study test accuracy.

This project keeps that behavior in `compat` mode and provides a leak-safe default for comparison. The comparison is reported neutrally as a public-data methods check, not as a claim about the private ACAM-J result.

See `METHODS_NOTE.md` and `docs/notebook_contract.md`.

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
# macOS example
brew install --cask xquartz
brew install afni
```

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

Run the full public pipeline:

```bash
jhana-repro run-public \
  --work-dir runs/ds002748 \
  --mode both \
  --n-permutations 1000
```

Outputs are written under:

- `runs/ds002748/Data/1ReHo/reho_output.csv`
- `runs/ds002748/classification/leak-safe/`
- `runs/ds002748/classification/compat/`
- `runs/ds002748/classification/method_comparison.csv`
- `runs/ds002748/classification/method_diagnostics.csv`

## Verified Local Run

On 2026-05-26, the 20-subject public smoke run completed end-to-end locally:

- ReHo CSV: 20 rows, 100 Schaefer-100 ROI feature columns, 10 depression and 10 control subjects.
- Leak-safe held-out metrics: accuracy `0.40`, AUC `0.50`, F1 `0.40`.
- Compat held-out metrics: accuracy `0.40`, AUC `0.667`, F1 `0.40`.
- Leak-safe top-model CV accuracy mean: `0.689`; held-out accuracy: `0.40`.
- Compat top-model CV accuracy mean: `0.994`; held-out accuracy: `0.40`.
- AFNI validation was attempted and explicitly skipped because `3dReHo` is not installed on this machine.

These numbers are a machinery check on public depression/control data. They are not evidence for or against the paper's private ACAM-J accuracy.

## ReHo Validation

Validate one subject against AFNI's `3dReHo`:

```bash
jhana-repro validate-reho-afni \
  --work-dir runs/ds002748 \
  --subject sub-01
```

The validation writes `runs/ds002748/afni_reho_validation.json`. A Pearson correlation below `0.90` should block release until investigated. If AFNI is not installed, validation is explicitly marked as skipped.

## Methods Note

The public MRP notebooks do not upsample the held-out `sub-001` case subject. The verified ordering issue is narrower:

- the classifier loads a globally upsampled group dataset before cross-validation,
- RFE is fitted before row-level cross-validation,
- the top three ensemble members are selected by held-out case-study test accuracy.

This project keeps that behavior in `compat` mode and provides a leak-safe default for comparison. The comparison is reported neutrally as a public-data methods check, not as a claim about the private ACAM-J result.

See `METHODS_NOTE.md` and `docs/notebook_contract.md`.

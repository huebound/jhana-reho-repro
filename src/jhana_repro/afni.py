from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import pearsonr

from .confounds import load_confounds
from .datasets import fmriprep_subject_files, read_repetition_time
from .reho import RehoConfig, compute_reho_image


def validate_against_afni(
    work_dir: Path,
    derivatives_dir: Path,
    subject: str = "sub-01",
    config: RehoConfig = RehoConfig(),
) -> dict:
    afni = shutil.which("3dReHo")
    if afni is None:
        result = {"subject": subject, "status": "skipped", "reason": "AFNI 3dReHo not found on PATH"}
        _write_result(work_dir, result)
        return result

    files = fmriprep_subject_files(derivatives_dir, subject)
    tr = read_repetition_time(work_dir / "openneuro_ds002748" / "task-rest_bold.json")
    confounds = load_confounds(files["confounds"])
    from nilearn import image

    validation_dir = work_dir / "afni_validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    cleaned = image.clean_img(
        str(files["bold"]),
        confounds=confounds,
        mask_img=str(files["mask"]),
        t_r=tr,
        low_pass=config.low_pass,
        high_pass=config.high_pass,
        detrend=True,
        standardize="zscore_sample",
    )
    cleaned = image.math_img("np.nan_to_num(img)", img=cleaned)
    cleaned_path = validation_dir / f"{subject}_desc-cleaned_bold.nii.gz"
    python_reho_path = validation_dir / f"{subject}_desc-python_reho.nii.gz"
    afni_reho_path = validation_dir / f"{subject}_desc-afni_reho.nii.gz"
    for path in (cleaned_path, python_reho_path, afni_reho_path):
        path.unlink(missing_ok=True)
    cleaned.to_filename(cleaned_path)

    mask_img = nib.load(str(files["mask"]))
    python_reho = compute_reho_image(cleaned, mask_img)
    python_reho.to_filename(python_reho_path)

    subprocess.run(
        [
            afni,
            "-prefix",
            str(afni_reho_path),
            "-inset",
            str(cleaned_path),
            "-mask",
            str(files["mask"]),
            "-nneigh",
            "27",
        ],
        check=True,
    )

    afni_img = nib.load(str(afni_reho_path))
    mask = np.asarray(mask_img.get_fdata()) > 0
    python_values = np.asarray(python_reho.get_fdata())[mask]
    afni_values = np.asarray(afni_img.get_fdata())[mask]
    finite = np.isfinite(python_values) & np.isfinite(afni_values)
    corr = float(pearsonr(python_values[finite], afni_values[finite]).statistic)
    result = {
        "subject": subject,
        "status": "passed" if corr >= 0.90 else "failed",
        "pearson_r": corr,
        "threshold": 0.90,
        "cleaned_bold": str(cleaned_path),
        "python_reho": str(python_reho_path),
        "afni_reho": str(afni_reho_path),
    }
    _write_result(work_dir, result)
    return result


def validate_subjects_against_afni(
    work_dir: Path,
    derivatives_dir: Path,
    subjects: list[str],
    output_path: Path | None = None,
    config: RehoConfig = RehoConfig(),
) -> dict:
    results = [validate_against_afni(work_dir, derivatives_dir, subject, config) for subject in subjects]
    passed = [result for result in results if result["status"] == "passed"]
    correlations = [float(result["pearson_r"]) for result in passed if "pearson_r" in result]
    summary = {
        "subjects": subjects,
        "n_subjects": len(subjects),
        "n_passed": len(passed),
        "n_failed": sum(result["status"] == "failed" for result in results),
        "n_skipped": sum(result["status"] == "skipped" for result in results),
        "threshold": 0.90,
        "pearson_r_mean": float(np.mean(correlations)) if correlations else None,
        "pearson_r_min": float(np.min(correlations)) if correlations else None,
        "pearson_r_max": float(np.max(correlations)) if correlations else None,
        "results": results,
    }
    if output_path is None:
        output_path = work_dir / "afni_reho_validation_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_result(work_dir: Path, result: dict) -> None:
    output = work_dir / "afni_reho_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

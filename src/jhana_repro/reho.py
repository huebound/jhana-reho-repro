from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image
from scipy.stats import rankdata

from .confounds import load_confounds
from .datasets import fmriprep_subject_files, read_participants, read_repetition_time


@dataclass(frozen=True)
class RehoConfig:
    high_pass: float = 0.01
    low_pass: float = 0.1
    smoothing_fwhm: float = 2.0
    schaefer_rois: int = 100
    schaefer_networks: int = 7
    atlas_resolution_mm: int = 2


def compute_reho_dataset(
    work_dir: Path,
    derivatives_dir: Path,
    output_csv: Path,
    max_subjects: int | None = None,
    subject_ids: list[str] | None = None,
    config: RehoConfig = RehoConfig(),
) -> pd.DataFrame:
    participants_path = work_dir / "openneuro_ds002748" / "participants.tsv"
    task_json = work_dir / "openneuro_ds002748" / "task-rest_bold.json"
    participants = read_participants(participants_path)
    tr = read_repetition_time(task_json)

    atlas = fetch_schaefer_atlas(work_dir, config)
    rows: list[dict[str, float | str]] = []
    if subject_ids:
        selected = participants[participants["participant_id"].isin(subject_ids)]
        missing = sorted(set(subject_ids) - set(selected["participant_id"]))
        if missing:
            raise ValueError(f"Requested subjects not found in participants.tsv: {missing}")
    else:
        selected = participants.head(max_subjects) if max_subjects else participants
    for record in selected.to_dict(orient="records"):
        subject = str(record["participant_id"])
        group = str(record["group"])
        files = fmriprep_subject_files(derivatives_dir, subject)
        row = compute_reho_subject_row(
            subject=subject,
            segment=group,
            run="run-01",
            bold_path=files["bold"],
            mask_path=files["mask"],
            confounds_path=files["confounds"],
            repetition_time=tr,
            atlas_path=atlas["maps"],
            atlas_labels=atlas["labels"],
            output_dir=work_dir / "reho_maps",
            config=config,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    metadata = {
        "dataset": "ds002748",
        "repetition_time": tr,
        "atlas": "Schaefer2018",
        "n_rois": config.schaefer_rois,
        "yeo_networks": config.schaefer_networks,
        "smoothing_fwhm": config.smoothing_fwhm,
        "high_pass": config.high_pass,
        "low_pass": config.low_pass,
        "rows": len(df),
    }
    output_csv.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return df


def compute_reho_subject_row(
    subject: str,
    segment: str,
    run: str,
    bold_path: Path,
    mask_path: Path,
    confounds_path: Path,
    repetition_time: float,
    atlas_path: Path,
    atlas_labels: list[str],
    output_dir: Path,
    config: RehoConfig = RehoConfig(),
    save_cleaned: bool = False,
) -> dict[str, float | str]:
    confounds = load_confounds(confounds_path)
    cleaned = image.clean_img(
        str(bold_path),
        confounds=confounds,
        mask_img=str(mask_path),
        t_r=repetition_time,
        low_pass=config.low_pass,
        high_pass=config.high_pass,
        detrend=True,
        standardize="zscore_sample",
    )
    cleaned = image.math_img("np.nan_to_num(img)", img=cleaned)
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_cleaned:
        cleaned.to_filename(output_dir / f"{subject}_task-rest_desc-cleaned_bold.nii.gz")

    mask_img = nib.load(str(mask_path))
    reho_img = compute_reho_image(cleaned, mask_img)
    z_img = zscore_in_mask(reho_img, mask_img)
    smoothed_img = image.smooth_img(z_img, config.smoothing_fwhm)
    smoothed_path = output_dir / f"{subject}_task-rest_desc-reho_zsmooth.nii.gz"
    smoothed_img.to_filename(smoothed_path)

    roi_values = summarize_reho_by_roi(smoothed_img, atlas_path, atlas_labels)
    row: dict[str, float | str] = {"subid": subject, "run": run, "segment": segment}
    row.update(roi_values)
    return row


def fetch_schaefer_atlas(work_dir: Path, config: RehoConfig = RehoConfig()) -> dict:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=config.schaefer_rois,
        yeo_networks=config.schaefer_networks,
        resolution_mm=config.atlas_resolution_mm,
        data_dir=str(work_dir / "atlases"),
        verbose=0,
    )
    labels = [decode_label(label) for label in atlas.labels]
    labels = [label for label in labels if label.lower() not in {"background", "b'background'"}]
    return {"maps": Path(atlas.maps), "labels": labels}


def decode_label(label: bytes | str) -> str:
    text = label.decode("utf-8") if isinstance(label, bytes) else str(label)
    text = text.replace("7Networks_", "")
    return text.replace("-", "_")


def compute_reho_image(cleaned_img: nib.spatialimages.SpatialImage, mask_img: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    data = np.asarray(cleaned_img.get_fdata(dtype=np.float32))
    mask = np.asarray(mask_img.get_fdata()) > 0
    if data.ndim != 4:
        raise ValueError("BOLD image must be 4D")
    if data.shape[:3] != mask.shape:
        raise ValueError("BOLD and mask spatial shapes do not match")

    n_time = data.shape[3]
    if n_time < 4:
        raise ValueError("Need at least 4 time points to compute ReHo")

    coords = np.argwhere(mask)
    flat_index = -np.ones(mask.shape, dtype=np.int32)
    flat_index[mask] = np.arange(coords.shape[0], dtype=np.int32)
    series = data[mask].astype(np.float32)
    ranks = rankdata(series, axis=1).astype(np.float32)

    reho_values = np.zeros(coords.shape[0], dtype=np.float32)
    denominator_base = float(n_time**3 - n_time)
    offsets = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int16)
    shape = np.array(mask.shape, dtype=np.int32)

    for row_index, coord in enumerate(coords):
        neighbors = coord + offsets
        valid = np.all((neighbors >= 0) & (neighbors < shape), axis=1)
        neighbors = neighbors[valid]
        neighbor_rows = flat_index[neighbors[:, 0], neighbors[:, 1], neighbors[:, 2]]
        neighbor_rows = neighbor_rows[neighbor_rows >= 0]
        k_neighbors = int(neighbor_rows.size)
        if k_neighbors < 2:
            continue
        rank_sums = ranks[neighbor_rows].sum(axis=0)
        mean_rank_sum = k_neighbors * (n_time + 1) / 2.0
        ss = np.square(rank_sums - mean_rank_sum).sum()
        reho_values[row_index] = (12.0 * ss) / ((k_neighbors**2) * denominator_base)

    reho_data = np.zeros(mask.shape, dtype=np.float32)
    reho_data[mask] = reho_values
    return nib.Nifti1Image(reho_data, cleaned_img.affine, cleaned_img.header)


def zscore_in_mask(img: nib.spatialimages.SpatialImage, mask_img: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    data = np.asarray(img.get_fdata(dtype=np.float32))
    mask = np.asarray(mask_img.get_fdata()) > 0
    values = data[mask]
    mean = float(values.mean())
    std = float(values.std())
    if std == 0.0:
        raise ValueError("ReHo map has zero variance inside mask")
    z_data = np.zeros(data.shape, dtype=np.float32)
    z_data[mask] = (values - mean) / std
    return nib.Nifti1Image(z_data, img.affine, img.header)


def summarize_reho_by_roi(
    reho_img: nib.spatialimages.SpatialImage,
    atlas_path: Path,
    atlas_labels: list[str],
) -> dict[str, float]:
    atlas_img = image.resample_to_img(
        str(atlas_path),
        reho_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    atlas_data = np.asarray(atlas_img.get_fdata()).astype(np.int32)
    reho_data = np.asarray(reho_img.get_fdata(dtype=np.float32))
    values: dict[str, float] = {}
    for roi_index, label in enumerate(atlas_labels, start=1):
        roi_mask = atlas_data == roi_index
        if not roi_mask.any():
            values[label] = np.nan
        else:
            values[label] = float(np.nanmean(reho_data[roi_mask]))
    if not np.isfinite(list(values.values())).all():
        missing = [label for label, value in values.items() if not np.isfinite(value)]
        raise ValueError(f"Atlas ROIs missing from ReHo map after resampling: {missing[:10]}")
    return values

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd


OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org/ds002748"
DERIV_REPO = "https://github.com/OpenNeuroDerivatives/ds002748-fmriprep"


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def prepare_ds002748(work_dir: Path, derivatives_dir: Path | None = None, clone_derivatives: bool = False) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = work_dir / "openneuro_ds002748"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in ("participants.tsv", "dataset_description.json", "task-rest_bold.json", "README"):
        download_url(f"{OPENNEURO_S3}/{name}", raw_dir / name)

    deriv_dir = derivatives_dir or (work_dir / "ds002748-fmriprep")
    if clone_derivatives and not deriv_dir.exists():
        _clone_derivatives(deriv_dir)

    info = {
        "dataset": "ds002748",
        "participants": str(raw_dir / "participants.tsv"),
        "dataset_description": str(raw_dir / "dataset_description.json"),
        "task_json": str(raw_dir / "task-rest_bold.json"),
        "derivatives_dir": str(deriv_dir),
        "derivatives_present": deriv_dir.exists(),
    }
    (work_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _clone_derivatives(deriv_dir: Path) -> None:
    if shutil.which("datalad"):
        subprocess.run(["datalad", "clone", DERIV_REPO, str(deriv_dir)], check=True)
    else:
        subprocess.run(["git", "clone", DERIV_REPO, str(deriv_dir)], check=True)


def read_participants(path: Path) -> pd.DataFrame:
    participants = pd.read_csv(path, sep="\t")
    required = {"participant_id", "group"}
    missing = required - set(participants.columns)
    if missing:
        raise ValueError(f"participants.tsv is missing columns: {sorted(missing)}")
    return participants


def read_repetition_time(task_json: Path) -> float:
    metadata = json.loads(task_json.read_text())
    try:
        return float(metadata["RepetitionTime"])
    except KeyError as exc:
        raise ValueError(f"{task_json} does not define RepetitionTime") from exc


def fmriprep_subject_files(derivatives_dir: Path, subject: str) -> dict[str, Path]:
    func_dir = derivatives_dir / subject / "func"
    bold = func_dir / f"{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    mask = func_dir / f"{subject}_task-rest_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
    confounds = func_dir / f"{subject}_task-rest_desc-confounds_timeseries.tsv"
    missing = [path for path in (bold, mask, confounds) if not _is_real_file(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing fMRIPrep files for {subject}: {missing}. "
            "If this is a DataLad/git-annex checkout, run `datalad get` for these paths."
        )
    return {"bold": bold, "mask": mask, "confounds": confounds}


def _is_real_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 1024

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_REHO_COLUMNS = ("subid", "run", "segment")
REQUIRED_RANKING_COLUMNS = ("Label Name", "Ranking", "Atlas")


@dataclass(frozen=True)
class RehoSchema:
    metadata_columns: tuple[str, ...]
    roi_columns: tuple[str, ...]
    segments: tuple[str, ...]
    subjects: tuple[str, ...]


def read_reho_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_reho_frame(df)
    return df


def validate_reho_frame(df: pd.DataFrame) -> RehoSchema:
    missing = [col for col in REQUIRED_REHO_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"ReHo CSV is missing required columns: {missing}")

    roi_columns = [
        col for col in df.columns if col not in REQUIRED_REHO_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not roi_columns:
        raise ValueError("ReHo CSV must contain at least one numeric ROI feature column")

    non_numeric_features = [
        col
        for col in df.columns
        if col not in REQUIRED_REHO_COLUMNS and col not in roi_columns
    ]
    if non_numeric_features:
        raise ValueError(
            "ReHo CSV contains nonnumeric non-metadata columns; all ROI features must be numeric: "
            f"{non_numeric_features[:10]}"
        )

    values = df[roi_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("ReHo ROI feature matrix contains NaN or infinite values")

    if df["subid"].isna().any() or df["segment"].isna().any() or df["run"].isna().any():
        raise ValueError("ReHo CSV metadata columns must not contain missing values")

    return RehoSchema(
        metadata_columns=REQUIRED_REHO_COLUMNS,
        roi_columns=tuple(roi_columns),
        segments=tuple(sorted(df["segment"].astype(str).unique())),
        subjects=tuple(sorted(df["subid"].astype(str).unique())),
    )


def validate_ranking_frame(ranking: pd.DataFrame, roi_columns: list[str]) -> list[str]:
    missing = [col for col in REQUIRED_RANKING_COLUMNS if col not in ranking.columns]
    if missing:
        raise ValueError(f"ROI ranking CSV is missing required columns: {missing}")

    rank_1 = ranking.loc[ranking["Ranking"] == 1, "Label Name"].astype(str).str.replace("-", "_", regex=False)
    selected = [roi for roi in rank_1.tolist() if roi in roi_columns]
    if not selected:
        raise ValueError(
            "ROI ranking did not match any ReHo ROI columns. "
            "For public Schaefer mode, use all generated ROI columns instead of the original 498-ROI ranking file."
        )
    return selected


def public_roi_columns(df: pd.DataFrame) -> list[str]:
    schema = validate_reho_frame(df)
    return list(schema.roi_columns)

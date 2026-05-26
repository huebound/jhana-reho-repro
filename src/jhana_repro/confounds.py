from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CONFOUND_CANDIDATES = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "csf",
    "white_matter",
)


def load_confounds(path: Path, candidates: tuple[str, ...] = DEFAULT_CONFOUND_CANDIDATES) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    selected = [col for col in candidates if col in table.columns]
    if not selected:
        raise ValueError(f"No usable confound columns found in {path}. Available columns: {list(table.columns)[:30]}")
    confounds = table[selected].replace(["n/a", "NA", "nan"], np.nan).astype(float)
    return confounds.fillna(0.0)

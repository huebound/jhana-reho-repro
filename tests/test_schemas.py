import pandas as pd
import pytest

from jhana_repro.schemas import public_roi_columns, validate_ranking_frame, validate_reho_frame


def test_reho_schema_accepts_wide_roi_rows():
    df = pd.DataFrame(
        {
            "subid": ["sub-01", "sub-02"],
            "run": ["run-01", "run-01"],
            "segment": ["depr", "control"],
            "ROI_1": [0.1, 0.2],
            "ROI_2": [0.3, 0.4],
        }
    )

    schema = validate_reho_frame(df)

    assert schema.roi_columns == ("ROI_1", "ROI_2")
    assert public_roi_columns(df) == ["ROI_1", "ROI_2"]


def test_reho_schema_rejects_missing_metadata():
    df = pd.DataFrame({"subid": ["sub-01"], "segment": ["depr"], "ROI_1": [0.1]})

    with pytest.raises(ValueError, match="missing required"):
        validate_reho_frame(df)


def test_ranking_requires_matching_roi_labels():
    ranking = pd.DataFrame({"Label Name": ["Other_ROI"], "Ranking": [1], "Atlas": ["Test"]})

    with pytest.raises(ValueError, match="did not match"):
        validate_ranking_frame(ranking, ["ROI_1"])

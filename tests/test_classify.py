import pandas as pd

from jhana_repro.classify import ClassificationConfig, run_modes


def test_public_classification_outputs_both_modes(tmp_path):
    rows = []
    for index in range(12):
        label = "depr" if index < 6 else "control"
        signal = 1.0 if label == "depr" else -1.0
        rows.append(
            {
                "subid": f"sub-{index:02d}",
                "run": "run-01",
                "segment": label,
                "ROI_1": signal + index * 0.01,
                "ROI_2": signal * 0.5,
                "ROI_3": -signal,
                "ROI_4": signal * 0.25,
            }
        )
    reho = tmp_path / "reho.csv"
    pd.DataFrame(rows).to_csv(reho, index=False)

    results = run_modes(
        reho,
        tmp_path / "out",
        mode="both",
        config=ClassificationConfig(n_splits=2, n_permutations=10, compat_target_per_subject=2),
    )

    assert set(results) == {"leak-safe", "compat"}
    assert (tmp_path / "out" / "method_comparison.csv").exists()
    assert (tmp_path / "out" / "method_diagnostics.csv").exists()
    assert set(results["leak-safe"]["train_subjects"]).isdisjoint(results["leak-safe"]["test_subjects"])

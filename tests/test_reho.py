import numpy as np
import nibabel as nib

from jhana_repro.reho import compute_reho_image


def test_reho_detects_local_synchrony():
    rng = np.random.default_rng(42)
    shared = np.sin(np.linspace(0, 4 * np.pi, 20))
    data = np.zeros((3, 3, 3, 20), dtype=np.float32)
    data[:] = shared
    data += rng.normal(0, 0.01, size=data.shape)
    mask = np.ones((3, 3, 3), dtype=np.uint8)

    reho = compute_reho_image(nib.Nifti1Image(data, np.eye(4)), nib.Nifti1Image(mask, np.eye(4)))
    values = reho.get_fdata()[mask.astype(bool)]

    assert np.nanmean(values) > 0.9


def test_reho_random_series_are_lower_than_synchronized_series():
    rng = np.random.default_rng(42)
    random_data = rng.normal(size=(3, 3, 3, 20)).astype(np.float32)
    mask = np.ones((3, 3, 3), dtype=np.uint8)

    random_reho = compute_reho_image(
        nib.Nifti1Image(random_data, np.eye(4)),
        nib.Nifti1Image(mask, np.eye(4)),
    )

    shared = np.sin(np.linspace(0, 4 * np.pi, 20))
    sync_data = np.zeros((3, 3, 3, 20), dtype=np.float32)
    sync_data[:] = shared
    sync_data += rng.normal(0, 0.01, size=sync_data.shape)
    sync_reho = compute_reho_image(
        nib.Nifti1Image(sync_data, np.eye(4)),
        nib.Nifti1Image(mask, np.eye(4)),
    )

    assert np.nanmean(sync_reho.get_fdata()) > np.nanmean(random_reho.get_fdata())

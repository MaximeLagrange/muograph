from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking
from muograph.utils.save import muograph_path

from typing import Tuple, Optional
from pathlib import Path
import torch
import numpy as np

# Test data file path
TEST_HIT_FILE = Path(__file__).parent / "../data/iron_barrel/barrel_and_cubes_scattering.csv"
OUTPUT_DIR = "../output_test/"


def get_tracks(hits_file: str, output_dir: Optional[str] = None) -> Tuple[Tracking, Tracking]:
    r"""
    Generates `Tracking` instances for input and output hits.

    Args:
        hits_file (str): Path to the hits data CSV file.
        output_dir (Optional[str]): Directory where the output tracking files will be saved.

    Returns:
        Tuple[Tracking, Tracking]: A tuple containing the `Tracking` objects for the incoming
        (above) and outgoing (below) hits.
    """
    hits_in = Hits(
        plane_labels=(0, 1, 2),
        data=hits_file,
        spatial_res=(1.0, 1.0, 0.0),
        energy_range=(0.0, 1_000_000),
        efficiency=0.98,
        input_unit="mm",
    )

    hits_out = Hits(
        plane_labels=(3, 4, 5),
        data=hits_file,
        spatial_res=(1.0, 1.0, 0.0),
        energy_range=(0.0, 1_000_000),
        efficiency=0.98,
        input_unit="mm",
    )

    tracks_in = Tracking(label="above", hits=hits_in, output_dir=output_dir)
    tracks_out = Tracking(label="below", hits=hits_out, output_dir=output_dir)

    return tracks_in, tracks_out


def test_tracks_output() -> None:
    r"""
    Tests if the output tracking files are correctly created.
    """
    output_dir = str(Path(muograph_path) / OUTPUT_DIR)
    _ = get_tracks(hits_file=str(TEST_HIT_FILE), output_dir=output_dir)

    output_tracks_in_path = Path(output_dir) / "tracks_above.hdf5"
    output_tracks_out_path = Path(output_dir) / "tracks_below.hdf5"

    assert output_tracks_in_path.exists(), f"File not created: {output_tracks_in_path}"
    assert output_tracks_out_path.exists(), f"File not created: {output_tracks_out_path}"


def test_tracks_loading() -> None:
    r"""
    Tests if the loaded tracking instances match the original instances.
    """
    output_dir = str(Path(muograph_path) / OUTPUT_DIR)

    tracks_in, tracks_out = get_tracks(hits_file=str(TEST_HIT_FILE), output_dir=output_dir)

    tracks_in_loaded = Tracking(label="above", tracks_hdf5=str(Path(output_dir) / "tracks_above.hdf5"))
    tracks_out_loaded = Tracking(label="below", tracks_hdf5=str(Path(output_dir) / "tracks_below.hdf5"))

    def compare_attr(attr: str, ref_instance: Tracking, loaded_instance: Tracking) -> bool:
        r"""
        Compares a specific attribute between a reference and a loaded Tracking instance.

        Args:
            attr (str): The attribute to compare.
            ref_instance (Tracking): The reference Tracking instance.
            loaded_instance (Tracking): The loaded Tracking instance.

        Returns:
            bool: True if the attributes match, False otherwise.
        """
        ref_value = getattr(ref_instance, attr)
        loaded_value = getattr(loaded_instance, attr)

        if isinstance(ref_value, torch.Tensor):
            # Ensure both tensors have the same dtype before comparison. TO BE CHANGED.
            if ref_value.dtype != loaded_value.dtype:
                loaded_value = loaded_value.to(ref_value.dtype)
            return torch.allclose(ref_value, loaded_value)
        elif isinstance(ref_value, (bool, np.bool_)):
            return ref_value == loaded_value
        else:
            return np.array_equal(ref_value, loaded_value)

    comparison_results = [
        compare_attr(attr, ref, loaded) for attr in Tracking._vars_to_save for ref, loaded in [(tracks_in, tracks_in_loaded), (tracks_out, tracks_out_loaded)]
    ]

    assert all(comparison_results), "Mismatch between reference and loaded Tracking instances."


def test_tracks_zenith() -> None:
    r"""
    Validates that the zenith angle (`theta`) of muon tracks is within expected bounds.

    The test ensures:
    - All `theta` values are below 0.71 radians (40 degrees) with a tolerance.
    - The mean `theta` value is close to 0.25 radians (14 degrees) with a tolerance.

    Raises:
        AssertionError: If any of the conditions are not met.
    """
    output_dir = str(Path(muograph_path) / OUTPUT_DIR)
    tracks_in, tracks_out = get_tracks(hits_file=str(TEST_HIT_FILE), output_dir=output_dir)

    tol = 0.05

    for tracks in [tracks_in, tracks_out]:
        assert (tracks.theta < 0.71 + tol).all(), f"Muon zenith angle exceeds 0.71 rad (40 deg) for {tracks.label} tracks."
        assert abs(tracks.theta.mean() - 0.25) < tol, f"Mean muon zenith angle deviates from 0.25 rad (14 deg) for {tracks.label} tracks."

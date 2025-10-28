from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking, TrackingMST
from typing import Tuple, Optional
from pathlib import Path

# Test data file path
TEST_HIT_FILE = str(Path(__file__).parent / "../data/iron_barrel/barrel_and_cubes_scattering.csv")
EFFICIENCY = 0.98
NPANELS = 6  # the number of detection planes


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
        efficiency=EFFICIENCY,
        input_unit="mm",
    )

    hits_out = Hits(
        plane_labels=(3, 4, 5),
        data=hits_file,
        spatial_res=(1.0, 1.0, 0.0),
        energy_range=(0.0, 1_000_000),
        efficiency=EFFICIENCY,
        input_unit="mm",
    )

    tracks_in = Tracking(label="above", hits=hits_in, output_dir=output_dir)
    tracks_out = Tracking(label="below", hits=hits_out, output_dir=output_dir)

    return tracks_in, tracks_out


def get_tracking_mst(hits_file: str, output_dir: Optional[str] = None) -> TrackingMST:
    r"""
    Generates `Tracking` instances for input and output hits.

    Args:
        hits_file (str): Path to the hits data CSV file.
        output_dir (Optional[str]): Directory where the output tracking files will be saved.

    Returns:
        TrackingMST: Instance of the TrackingMST class.
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

    return TrackingMST(trackings=(tracks_in, tracks_out))


def test_efficiency() -> None:
    r"""
    Tests the efficiency of the muon tracking system.

    The test ensures that the fraction of rejected events matches the expected
    efficiency based on the detector's panel efficiency and the number of panels.

    Raises:
        AssertionError: If the rejection fraction deviates from the expected value
        beyond the allowed tolerance.
    """
    tracks_in, tracks_out = get_tracks(TEST_HIT_FILE)

    # Total number of muon events
    nevent = tracks_in.n_mu

    # Perform tracking using MST
    mst = TrackingMST(trackings=(tracks_in, tracks_out))

    # Number of muon events after MST processing
    nevent_mst = mst.n_mu

    # Tolerance for rejection fraction comparison
    tol = 0.01

    # Calculate actual and expected rejection fractions
    rejection_fraction = nevent_mst / nevent
    expected_rejection_fraction = EFFICIENCY**NPANELS

    assert abs(rejection_fraction - expected_rejection_fraction) < tol, (
        f"The fraction of rejected events {rejection_fraction:.4f} must be close to the "
        f"detector efficiency {expected_rejection_fraction:.4f} (panel_eff * n_panels)."
    )


def test_scattering_angle() -> None:
    mst = get_tracking_mst(TEST_HIT_FILE)

    tol = 0.05
    tol_mean = 0.001

    assert (mst.dtheta >= 0.0).all(), "Sattering anle values must be positive."
    assert (mst.dtheta.max() <= 0.5 + tol).all(), "Sattering angle values must be below 0.50 rad (~30 deg)."
    assert abs(mst.dtheta.mean() - 0.0076) < tol_mean, "Mean scattering angle value must be close to 0.0076 rad (~0.5 deg)."

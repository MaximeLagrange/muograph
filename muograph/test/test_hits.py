import os
from muograph.hits.hits import Hits
import torch

# Test data file path
test_hit_file = os.path.dirname(__file__) + "/../data/iron_barrel/barrel_and_cubes_scattering.csv"


def get_hits(hits_file: str) -> Hits:
    """
    Initialize and return a Hits object based on the provided file.

    Args:
        hits_file (str): Path to the CSV file with hit data.

    Returns:
        Hits: Initialized Hits object.
    """
    hits = Hits(
        plane_labels=(0, 1, 2),
        data=hits_file,
        spatial_res=(1.0, 1.0, 0.0),
        energy_range=(0.0, 1_000_000),
        efficiency=0.90,
        input_unit="mm",
    )
    return hits


def test_hits_shape() -> None:
    """
    Test that the shape of gen_hits matches the number of detector panels.
    """
    hits = get_hits(test_hit_file)

    expected_num_panels = len(hits.plane_labels)
    actual_num_panels = hits.gen_hits.size(1)

    assert actual_num_panels == expected_num_panels, f"Expected {expected_num_panels} panels but got {actual_num_panels}."


def test_hits_spatial_res() -> None:
    """
    Test spatial resolution effect on gen_hits and reco_hits data.
    """
    event_tol = 100  # Tolerance for similar gen_hits and reco_hits events
    hits = get_hits(test_hit_file)

    # Check zero spatial resolution case
    if hits.spatial_res.sum() == 0.0:
        assert torch.equal(hits.gen_hits, hits.reco_hits), "gen_hits and reco_hits should match when spatial resolution is zero."

    # Check non-zero spatial resolution per dimension
    for i, res in enumerate(hits.spatial_res):
        if res != 0.0:
            identical_count = (hits.gen_hits[i] == hits.reco_hits[i]).sum().item()
            total_events = hits.gen_hits[i].numel()

            assert identical_count < total_events - event_tol, (
                f"Expected fewer identical events in gen_hits[{i}] and reco_hits[{i}] due to non-zero "
                f"spatial resolution ({res}), but found {identical_count} identical events."
            )


def test_hits_efficiency() -> None:
    """
    Test hit efficiency in the Hits object.
    """
    tolerance = 0.01  # Allowed difference in effective efficiency
    hits = get_hits(test_hit_file)

    if hits.efficiency == 1.0:
        assert (hits.hits_eff == 1.0).all(), "All efficiencies should be 1.0 when efficiency is set to 1.0."
    else:
        effective_efficiency = hits.hits_eff.float().mean().item()
        expected_efficiency = torch.tensor(hits.efficiency).item()

        assert abs(effective_efficiency - expected_efficiency) <= tolerance, (
            f"Expected efficiency close to {hits.efficiency}, but got {effective_efficiency} " f"(tolerance: {tolerance})."
        )

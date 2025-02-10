from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking, TrackingMST
from muograph.reconstruction.poca import POCA
from muograph.volume.volume import Volume
from muograph.utils.save import muograph_path

import os
from pathlib import Path
import torch
import numpy as np

# Test data file path
TEST_HIT_FILE = os.path.dirname(__file__) + "/../data/iron_barrel/barrel_and_cubes_scattering.csv"
VOI = Volume(position=(0, 0, -1200), dimension=(1000, 600, 600), voxel_width=(20, 10, 20))


def get_mst(hits_file: str) -> TrackingMST:
    hits_in = Hits(
        plane_labels=(0, 1, 2),
        csv_filename=hits_file,
        spatial_res=(1.0, 1.0, 0.0),
        energy_range=(0.0, 1_000_000),
        efficiency=0.98,
        input_unit="mm",
    )

    hits_out = Hits(
        plane_labels=(3, 4, 5),
        csv_filename=hits_file,
        spatial_res=(1.0, 1.0, 0.0),
        energy_range=(0.0, 1_000_000),
        efficiency=0.98,
        input_unit="mm",
    )

    tracks_in = Tracking(label="above", hits=hits_in)
    tracks_out = Tracking(label="below", hits=hits_out)

    return TrackingMST(trackings=(tracks_in, tracks_out))


def test_poca_loading() -> None:
    mst = get_mst(TEST_HIT_FILE)

    output_dir = str(Path(muograph_path) / "../output_test/")

    poca = POCA(tracking=mst, voi=VOI, output_dir=output_dir)

    def compare_attr(attr: str, ref_instance: POCA, loaded_instance: POCA) -> bool:
        r"""
        Compares a specific attribute between a reference and a loaded POCA instance.

        Args:
            attr (str): The attribute to compare.
            ref_instance (POCA): The reference POCA instance.
            loaded_instance (POCA): The loaded POCA instance.

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

    poca_loaded = POCA(poca_file=output_dir + "/poca.hdf5")

    comparison_results = [compare_attr(attr, poca, poca_loaded) for attr in POCA._vars_to_load]

    assert all(comparison_results), "Mismatch between reference and loaded POCA instances."


def test_poca_predictions() -> None:
    mst = get_mst(TEST_HIT_FILE)

    poca = POCA(mst, voi=VOI)

    n_poca_uranium_x_region = poca.n_poca_per_vox[23:28].float().mean()
    n_poca_empty_x_region = poca.n_poca_per_vox[-5:].float().mean()

    assert (
        n_poca_uranium_x_region > n_poca_empty_x_region
    ), "The average number of POCA points per voxel in the uranium x region {n_poca_uranium_x_region} must be higher than in the empty region {n_poca_empty_x_region}"

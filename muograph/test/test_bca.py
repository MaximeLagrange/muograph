from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking, TrackingMST
from muograph.reconstruction.binned_clustered import BCA
from muograph.volume.volume import Volume
from muograph.utils.save import muograph_path

import torch

import os
from pathlib import Path
from functools import partial
import math

# Test data file path
TEST_HIT_FILE = os.path.dirname(__file__) + "/../data/iron_barrel/barrel_and_cubes_scattering.csv"
VOI = Volume(position=(0, 0, -1200), dimension=(1000, 600, 600), voxel_width=(20, 10, 20))
OUPUT_DIR = str(Path(muograph_path) / "../output_test/")


def get_mst(hits_file: str) -> TrackingMST:
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

    tracks_in = Tracking(label="above", hits=hits_in)
    tracks_out = Tracking(label="below", hits=hits_out)

    return TrackingMST(trackings=(tracks_in, tracks_out))


def test_bca_predictions() -> None:
    mst = get_mst(TEST_HIT_FILE)

    bca = BCA(voi=VOI, tracking=mst)

    bca.params = {
        "n_max_per_vox": 20,
        "n_min_per_vox": 3,
        "score_method": partial(torch.quantile, q=0.5),
        "metric_method": partial(torch.log),  # type: ignore
        "p_range": (0, 1000000),
        "dtheta_range": (0.05 * math.pi / 180, 20 * math.pi / 180),
    }

    n_poca_uranium_x_region = bca.xyz_voxel_pred[23:28].float().mean()
    n_poca_empty_x_region = bca.xyz_voxel_pred[-5:].float().mean()

    assert (
        n_poca_uranium_x_region > n_poca_empty_x_region
    ), "The voxel scattering density in the uranium x region {n_poca_uranium_x_region} must be higher than in the empty region {n_poca_empty_x_region}"

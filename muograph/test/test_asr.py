from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking, TrackingMST
from muograph.reconstruction.asr import ASR, ASRParams
from muograph.volume.volume import Volume
import os
from pathlib import Path
from muograph.utils.save import muograph_path
import numpy as np
from functools import partial
import math

# Test data file path
TEST_HIT_FILE = os.path.dirname(__file__) + "/../data/iron_barrel/barrel_and_cubes_scattering.csv"
VOI = Volume(position=(0, 0, -1200), dimension=(1000, 600, 600), voxel_width=(50, 100, 50))
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


def test_asr_predictions() -> None:
    mst = get_mst(TEST_HIT_FILE)

    asr = ASR(voi=VOI, tracking=mst)

    asr.params = ASRParams(
        score_method=partial(np.quantile, q=0.8),
        p_range=(0.0, 10000000),  # MeV
        dtheta_range=(0.1 * math.pi / 180, 10 * math.pi / 180),
        use_p=False,
    )

    n_poca_uranium_x_region = asr.xyz_voxel_pred[14:16].float().mean()
    n_poca_empty_x_region = asr.xyz_voxel_pred[-3:].float().mean()

    assert (
        n_poca_uranium_x_region > n_poca_empty_x_region
    ), "The voxel scattering density in the uranium x region {n_poca_uranium_x_region} must be higher than in the empty region {n_poca_empty_x_region}"

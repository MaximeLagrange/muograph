from __future__ import annotations
from typing import Union, Tuple, List, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

from muograph.volume.volume import Volume
from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking, TrackingMST
from muograph.reconstruction.asr import ASR
from muograph.reconstruction.poca import POCA
from muograph.reconstruction.binned_clustered import BCA
from muograph.utils.tools import print_memory_usage

ALGO_REGISTRY = {"ASR": ASR, "POCA": POCA, "BCA": BCA}


@dataclass
class Algorithm:
    class_ref: type
    params: Any
    preds: Optional[np.ndarray] = None
    name: Optional[str] = None


class Scan:
    """
    High-level interface for running tomographic reconstruction algorithms
    (ASR, POCA, BCA, etc.) on muon tracking data within a defined volume of interest (VOI).
    """

    def __init__(
        self,
        input_data: Union[str, Path, pd.DataFrame],
        voi: Volume,
        algorithms: List[Algorithm],
        plane_labels_in_out: Tuple[Tuple[int, ...], Tuple[int, ...]],
        energy_range: Optional[Tuple[float, float]] = None,
        spatial_res: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        n_mu_max: Optional[int] = None,
    ) -> None:
        """The `Scan` class orchestrates the complete workflow:
                - Converts raw or tabular hit data into `Hits` and `TrackingMST` objects.
                - Initializes and executes one or more reconstruction algorithms.
                - Optionally saves algorithm predictions to disk.

        Args:
            input_data (Union[str, Path, pd.DataFrame]): The input data source containing muon hits. Can be a path to a CSV, ROOT file
                or an in-memory pandas DataFrame.
            voi (Volume): The volume of interest defining the 3D region to be reconstructed.
            algorithms (List[Algorithm]): List of reconstruction algorithms to execute. Each `Algorithm` wraps the
                algorithm class reference, its parameter object, and stores results.
            plane_labels_in_out (Tuple[Tuple[int, ...], Tuple[int, ...]]): Two tuples specifying the detector plane indices for incoming and outgoing hits.
            energy_range (Optional[Tuple[float, float]], optional): Allowed energy range (min, max) for filtering muons. Default is None (no filtering).
            spatial_res (Tuple[float, float, float], optional): Spatial resolution (sigma_x, sigma_y, sigma_z) in millimeters for hit uncertainty modeling. Defaults to (0.0, 0.0, 0.0).
            n_mu_max (Optional[int], optional): Optional maximum number of muons to process. Useful for limiting dataset size
            during testing or benchmarking. Defaults to None.
        """
        self.input_data = input_data
        self.voi = voi
        self.algorithms = algorithms
        self.plane_label_in, self.plane_label_out = plane_labels_in_out
        self.energy_range = energy_range
        self.spatial_res = spatial_res
        self.n_mu_max = n_mu_max

        print_memory_usage("Preparing hits for Scan")
        self._hits_in = self._make_hits(self.plane_label_in)

        self._hits_out = self._make_hits(self.plane_label_out)

        self._tracking = TrackingMST((Tracking(hits=self._hits_in, label="above"), Tracking(hits=self._hits_out, label="below")))
        print_memory_usage("Tracking prepared")

    def _make_hits(self, plane_labels: Tuple[int, ...]) -> Hits:
        return Hits(
            data=self.input_data,
            plane_labels=plane_labels,
            spatial_res=self.spatial_res,
            energy_range=self.energy_range,
            n_mu_max=self.n_mu_max,
        )

    def get_preds(self, algo: Algorithm) -> np.ndarray:
        """Execute a reconstruction algorithm and return its voxel-level predictions.


        Args:
            algo (Algorithm): reconstruction algorithm.

        Returns:
            np.ndarray: Volume predictions.
        """

        reconstructor = algo.class_ref(tracking=self._tracking, voi=self.voi)
        print_memory_usage(f"Reconstructor {algo.class_ref.__name__} initialized")

        reconstructor.params = algo.params

        algo.name = reconstructor.name
        preds = reconstructor.xyz_voxel_pred.detach().cpu().numpy()
        return preds.copy()

    def scan_all_algos(self, save_dir: Optional[Path] = None) -> None:
        """Run all reconstruction algorithms defined in `self.algorithms`.

        Args:
            save_dir (Optional[Path], optional): If provided, each algorithm's voxel predictions are saved as `.npy` files
            in this directory. File names are based on algorithm names. Defaults to None.
        """

        n_voxels = np.prod(self.voi.n_vox_xyz)
        print(f"Scanning volume with {n_voxels} voxels using {self._tracking.n_mu} muons")

        for algo in self.algorithms:
            try:
                preds = self.get_preds(algo)
            except Exception as e:
                print(f"[ERROR] Failed to compute predictions for {algo.name or algo.class_ref.__name__}: {e}")
                continue

            if save_dir:
                try:
                    # Ensure directory exists
                    save_dir.mkdir(parents=True, exist_ok=True)

                    save_path = save_dir / f"{algo.name}_preds.npy"
                    np.save(save_path, preds)

                    print_memory_usage(f"Saved predictions for {algo.name} → {save_path}")
                except OSError as e:
                    print(f"[WARNING] Could not save predictions for {algo.name or algo.class_ref.__name__} " f"to '{save_dir}': {e}")
                except Exception as e:
                    print(f"[ERROR] Unexpected error while saving predictions for {algo.name or algo.class_ref.__name__}: {e}")

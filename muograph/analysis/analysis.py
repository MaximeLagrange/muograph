from __future__ import annotations
from typing import Union, Tuple, List, Optional, Any, Dict
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from torch import Tensor
import math
import h5py
import torch
import shutil

from muograph.volume.volume import Volume
from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking, TrackingMST
from muograph.reconstruction.asr import ASR
from muograph.reconstruction.poca import POCA
from muograph.reconstruction.binned_clustered import BCA
from muograph.utils.tools import print_memory_usage, apply_gaussian_filter, normalize
from muograph.plotting.voxel import VoxelPlotting as vp

ALGO_REGISTRY = {"ASR": ASR, "POCA": POCA, "BCA": BCA}


@dataclass
class Algorithm:
    class_ref: type
    params: Any
    preds: Optional[np.ndarray] = None
    name: str = ""
    params_str: str = ""


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
        mask: Optional[Tensor] = None,
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
        self.mask = mask

        print_memory_usage("Preparing hits for Scan")
        self._hits_in = self._make_hits(self.plane_label_in)

        self._hits_out = self._make_hits(self.plane_label_out)

        self._tracking = TrackingMST((Tracking(hits=self._hits_in, label="above"), Tracking(hits=self._hits_out, label="below")))

        if mask is not None:
            assert len(mask) == self._tracking.n_mu, f"Provided mask has size {len(mask):,d} but tracking has {self._tracking.n_mu:,d} muons"
            self._tracking._filter_muons(mask)
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
        algo.params_str = reconstructor.get_string_params()
        preds = reconstructor.xyz_voxel_pred.detach().cpu().numpy()
        return preds.copy()

    def scan_all_algos(self, save_dir: Optional[Path] = None) -> None:
        """
        Run all reconstruction algorithms defined in `self.algorithms` and save their voxel predictions
        as HDF5 files with parameter metadata.
        """
        import traceback

        n_voxels = np.prod(self.voi.n_vox_xyz)
        print(f"Scanning volume with {n_voxels} voxels using {self._tracking.n_mu} muons")

        for i, algo in enumerate(self.algorithms):
            algo_name = algo.class_ref.__name__ + f"_{i}"
            print(f"\n[INFO] Running reconstruction with {algo_name}")

            try:
                preds = self.get_preds(algo)
            except Exception as e:
                print(f"[ERROR] Failed to compute predictions for {algo_name}: {e}")
                traceback.print_exc()
                continue

            if save_dir:
                try:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{algo_name}_preds.h5"

                    # --- Save predictions + metadata
                    with h5py.File(save_path, "w") as hf:
                        hf.create_dataset("preds", data=preds, compression="gzip")
                        hf.attrs["algorithm_name"] = algo.class_ref.__name__
                        hf.attrs["n_voxels"] = str(n_voxels)
                        hf.attrs["n_muons"] = str(self._tracking.n_mu)
                        hf.attrs["params"] = algo.params_str

                    print(f"[OK] Saved predictions for {algo_name} - {save_path}")
                    print(f"      Params: {algo.params_str[:120]}{'...' if len(algo.params_str) > 120 else ''}")

                except OSError as e:
                    print(f"[WARNING] Could not save predictions for {algo_name} to '{save_dir}': {e}")
                except Exception as e:
                    print(f"[ERROR] Unexpected error while saving predictions for {algo_name}: {e}")
                    traceback.print_exc()


class Visualizer:
    _gauss_ker: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(
        self,
        input_data: Union[str, Path, np.ndarray, torch.Tensor],
        voi: Volume,
        output_dir: Union[Optional[str], Optional[Path]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initializes the reconstruction or analysis object.

        Args:
            input_data: Path, array, or tensor representing reconstruction predictions.
            output_dir: Directory where output results will be saved.
            overwrite: Whether to overwrite an existing output directory (default: False)
        """
        # --- Load prediction data
        self.preds, self.params = self.load_data(input_data)
        self.preds_norm = normalize(self.preds)
        self.voi = voi

        # --- Prepare output directory using helper
        self.output_dir = self._prepare_output_dir(output_dir, overwrite)
        self.overwrite = overwrite

    @staticmethod
    def load_data(input_data: Union[str, Path, np.ndarray, Tensor]) -> Tuple[Tensor, Dict]:
        """
        Loads input data into a torch.Tensor along with metadata.

        Supports:
        - torch.Tensor - returned directly
        - np.ndarray - converted to Tensor
        - str or Path to HDF5 (.h5/.hdf5) file - loads predictions + attributes

        Returns:
            Tuple[Tensor, Dict]: (data tensor, metadata dict)
        """

        # --- Case 1: Already a Tensor
        if isinstance(input_data, Tensor):
            return input_data, {"params": None, "source": "Tensor"}

        # --- Case 2: NumPy array
        elif isinstance(input_data, np.ndarray):
            return torch.as_tensor(input_data), {"params": None, "source": "numpy"}

        # --- Case 3: String path
        elif isinstance(input_data, str):
            input_data = Path(input_data)

        # --- Case 4: Path (to HDF5 file)
        if isinstance(input_data, Path):
            if not input_data.exists():
                raise FileNotFoundError(f"[ERROR] File not found: {input_data}")

            # --- HDF5 file
            if input_data.suffix.lower() in [".h5", ".hdf5"]:
                with h5py.File(input_data, "r") as hf:
                    if "preds" not in hf:
                        raise KeyError(f"[ERROR] 'preds' dataset not found in {input_data}")

                    preds = hf["preds"][:]
                    # Convert attrs (HDF5 attributes) to a normal Python dict
                    metadata = {k: v.decode() if isinstance(v, bytes) else v for k, v in hf.attrs.items()}
                    metadata["source"] = str(input_data)
                return torch.as_tensor(preds), metadata

            else:
                raise ValueError(f"[ERROR] Unsupported file type: '{input_data.suffix}'. " "Expected '.h5' or '.hdf5'.")

        # --- Otherwise unsupported type
        raise TypeError(f"[ERROR] Unsupported input_data type: {type(input_data).__name__}. " "Expected Tensor, np.ndarray, str, or Path.")

    def _prepare_output_dir(
        self,
        output_dir: Union[Optional[str], Optional[Path]],
        overwrite: bool = False,
    ) -> Path:
        """
        Validates and prepares the output directory.

        - If None - defaults to 'output/'
        - If not exists, creates it
        - If exists:
            • overwrite=False, raises FileExistsError
            • overwrite=True , deletes and recreates it

        Returns:
            Path: The prepared output directory
        """
        # --- Normalize type
        if output_dir is None:
            output_dir = Path("output")
            print("[INFO] No output directory specified, using default: 'output/'")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not isinstance(output_dir, Path):
            raise TypeError(f"[ERROR] Unsupported output_dir type: {type(output_dir).__name__}. " "Expected str, Path, or None.")

        # --- Existence handling
        if output_dir.exists():
            if overwrite:
                print(f"[WARNING] Output directory already exists: {output_dir}")
                print("          - Overwriting existing directory and its contents.")
                try:
                    shutil.rmtree(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=False)
                except Exception as e:
                    raise OSError(f"[ERROR] Failed to overwrite directory '{output_dir}': {e}")
            else:
                raise FileExistsError(f"[ERROR] Output directory already exists: {output_dir}. " "Use overwrite=True to replace it.")
        else:
            try:
                output_dir.mkdir(parents=True, exist_ok=False)
                print(f"[OK] Created output directory - {output_dir}")
            except Exception as e:
                raise OSError(f"[ERROR] Could not create output directory '{output_dir}': {e}")

        return output_dir

    def plot_slice(
        self,
        smooth: bool = True,
        extension: str = "pdf",
    ) -> None:
        preds = self.smoothed_preds if smooth else self.preds_norm

        for dim in [0, 1, 2]:
            nslices = self.voi.n_vox_xyz[dim]
            ncols = max(int(math.sqrt(nslices)), int(math.sqrt(nslices)) + 1)
            algo_name = self.params["algorithm_name"] + "_" if self.params is not None else ""
            pred_label = self.params["algorithm_name"] + " score" if self.params is not None else ""
            figname = str(self.output_dir) + "/" + algo_name
            vp.plot_pred_by_slice(
                voi=self.voi, xyz_voxel_preds=preds, dim=dim, ncols=ncols, figname=figname, pred_label=pred_label, pred_unit="[a.u.]", extension=extension
            )

    @property
    def smoothed_preds(self) -> Tensor:
        return normalize(apply_gaussian_filter(self.preds, self._gauss_ker))  # type: ignore

    @property
    def gauss_ker(self) -> Tuple[float, float, float]:
        return self._gauss_ker

    @gauss_ker.setter
    def gauss_ker(self, value: Tuple[float, float, float]) -> None:
        self._gauss_ker = value

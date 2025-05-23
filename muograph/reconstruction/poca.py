import torch
from torch import Tensor
from copy import deepcopy
from typing import Optional, Dict, Union
import matplotlib
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import numpy as np
import seaborn as sns
import math

from muograph.utils.save import AbsSave
from muograph.utils.device import DEVICE
from muograph.utils.datatype import dtype_track, dtype_n
from muograph.volume.volume import Volume
from muograph.tracking.tracking import TrackingMST
from muograph.plotting.voxel import VoxelPlotting
from muograph.plotting.params import font


r"""
Provides class for computing POCA locations and POCA-based voxelized scattering density predictions
"""


def are_parallel(v1: Tensor, v2: Tensor, tol: float = 1e-5) -> bool:
    cross_prod = torch.linalg.cross(v1, v2)
    return bool(torch.all(torch.abs(cross_prod) < tol).detach().cpu().item())


class POCA(AbsSave, VoxelPlotting):
    r"""
    A class for Point Of Closest Approach computation in the context of a Muon Scattering Tomography analysis.
    """

    _parallel_mask: Optional[Tensor] = None  # (mu)
    _poca_points: Optional[Tensor] = None  # (mu, 3)
    _n_poca_per_vox: Optional[Tensor] = None  # (nx, ny, nz)
    _dtheta_mean_per_vox: Optional[Tensor] = None  # (nx, ny, nz)
    _dtheta_rms_per_vox: Optional[Tensor] = None  # (nx, ny, nz)
    _poca_indices: Optional[Tensor] = None  # (mu, 3)
    _mask_in_voi: Optional[Tensor] = None  # (mu)

    _batch_size: int = 2048

    _vars_to_save = [
        "poca_points",
        "n_poca_per_vox",
        "dtheta_mean_per_vox",
        "dtheta_rms_per_vox",
        "poca_indices",
    ]

    _vars_to_load = [
        "poca_points",
        "n_poca_per_vox",
        "dtheta_mean_per_vox",
        "dtheta_rms_per_vox",
        "poca_indices",
    ]

    def __init__(
        self,
        tracking: Optional[TrackingMST] = None,
        voi: Optional[Volume] = None,
        poca_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        r"""
        Initializes the POCA object with either an instance of the TrackingMST class or a
        poca.hdf5 file.

        Args:
            - tracking (Optional[TrackingMST]): Instance of the TrackingMST class.
            - voi (Optional[Volume]): Instance of the Volume class. If provided, muon events with
            poca locations outside the voi will be filtered out, the number of poca locations per voxel
            `n_poca_per_vox` as well as the voxel indices of each poca location will be computed.
            poca_file (Optional[str]): The path to the poca.hdf5 to load attributes from.
            - output_dir (Optional[str]): Path to a directory where to save POCA attributes
            in a hdf5 file.
            - filename (Optional[str]): name of the hdf5 file where POCA attributes will be saved if output_dir is filled
        """
        AbsSave.__init__(self, output_dir=output_dir)
        VoxelPlotting.__init__(self, voi)

        self.filename = filename if filename else "poca"

        if tracking is None and poca_file is None:
            raise ValueError("Provide either poca.hdf5 file of TrackingMST instance.")

        # Compute poca attributes if TrackingMST is provided
        elif (tracking is not None) and (voi is not None):
            self.tracks = deepcopy(tracking)
            self.voi = voi

            # Remove parallel events
            self.tracks._filter_muons(self.parallel_mask)

            # Remove POCAs outside voi
            self.tracks._filter_muons(self.mask_in_voi)
            self._filter_pocas(self.mask_in_voi)

            # Save attributes to hdf5
            if output_dir is not None:
                self.save_attr(self._vars_to_save, self.output_dir, filename=self.filename)

        # Load poca attributes from hdf5 if poca_file is provided
        elif tracking is None and poca_file is not None:
            self.load_attr(self._vars_to_load, poca_file)

    def __repr__(self) -> str:
        return f"Collection of {self.n_mu} POCA locations."

    @staticmethod
    def compute_parallel_mask(tracks_in: Tensor, tracks_out: Tensor, tol: float = 1e-7) -> Tensor:
        """
        Compute a mask to filter out events with parallel or nearly parallel tracks.

        Arguments:
            tracks_in: Tensor containing the incoming tracks, with size (n_mu, 3).
            tracks_out: Tensor containing the outgoing tracks, with size (n_mu, 3).
            tol: Tolerance value for determining if tracks are parallel.

        Returns:
            mask: Boolean tensor with size (n_mu), where True indicates the event is not parallel.
        """
        # Compute the cross product for all pairs at once
        cross_prod = torch.linalg.cross(tracks_in, tracks_out, dim=1)

        # Compute the mask by checking if the cross product magnitude is below the tolerance
        mask = torch.all(torch.abs(cross_prod) >= tol, dim=1)

        return mask

    def _filter_pocas(self, mask: Tensor) -> None:
        r"""
        Remove poca points specified as False in `mask`.

        Arguments:
            mask: (N,) Boolean tensor. poca points with False elements will be removed.
        """
        self.poca_points = self.poca_points[mask]

    @staticmethod
    def compute_poca_points(points_in: Tensor, points_out: Tensor, tracks_in: Tensor, tracks_out: Tensor) -> Tensor:
        """
        @MISC {3334866,
        TITLE = {Closest points between two lines},
        AUTHOR = {Brian (https://math.stackexchange.com/users/72614/brian)},
        HOWPUBLISHED = {Mathematics Stack Exchange},
        NOTE = {URL:https://math.stackexchange.com/q/3334866 (version: 2019-08-26)},
        EPRINT = {https://math.stackexchange.com/q/3334866},
        URL = {https://math.stackexchange.com/q/3334866}
        }

        Compute POCA points.

        Arguments:
            points_in: xyz coordinates of a point on the incomming track, with size (n_mu, 3).
            points_out: xyz coordinates of a point on the outgoing track, with size (n_mu, 3).
            tracks_in: The incomming track, with size (n_mu, 3).
            tracks_out: The outgoing track, with size (n_mu, 3).

        Returns:
            POCA points' coordinate(n_mu, 3)

        Given 2 lines L1, L2 aka incoming and outgoing tracks with parametric equation:
        L1 = P1 + t*V1

        1- A segment of shortest length between two 3D lines L1 L2 is perpendicular to both lines (if L1 L2 are neither parallele or in the same plane). One must compute V3, vector perpendicular to L1 and L2
        2- Search for points where L3 = P1 + t1*V1 +t3*V3 crosses L2. One must find t1 and t2 for which:
        L3 = P1 + t1*V1 +t3*V3 = P2 + t2*V2

        3- Then POCA location M is the middle of the segment Q1-Q2 where Q1,2 = P1,2 +t1,2*V1,2
        """
        from numpy import cross

        P1, P2 = points_in[:], points_out[:]
        V1, V2 = tracks_in[:], tracks_out[:]

        V3 = torch.tensor(
            cross(V2.detach().cpu().numpy(), V1.detach().cpu().numpy()),
            dtype=dtype_track,
            device=DEVICE,
        )

        if are_parallel(V1, V2):
            raise ValueError("Tracks are parallel or nearly parallel")

        RES = P2 - P1
        LES = torch.transpose(torch.stack([V1, -V2, V3]), 0, 1)
        LES = torch.transpose(LES, -1, 1)

        if LES.dtype != RES.dtype:
            LES = torch.ones_like(LES, dtype=torch.float32) * LES
            RES = torch.ones_like(RES, dtype=torch.float32) * RES

        try:
            ts = torch.linalg.solve(LES, RES)
        except RuntimeError as e:
            if "singular" in str(e):
                raise ValueError(f"Singular matrix encountered for points: {P1}, {P2} with tracks: {V1}, {V2}")
            else:
                raise e

        t1 = torch.stack([ts[:, 0], ts[:, 0], ts[:, 0]], -1)
        t2 = torch.stack([ts[:, 1], ts[:, 1], ts[:, 1]], -1)

        Q1s, Q2s = P1 + t1 * V1, P2 + t2 * V2
        M = (Q2s - Q1s) / 2 + Q1s

        return M

    @staticmethod
    def assign_voxel_to_pocas(poca_points: Tensor, voi: Volume, batch_size: int) -> Tensor:
        """
        Get the indinces of the voxel corresponding to each poca point.

        Arguments:
            - poca_points: Tensor with size (n_mu, 3).
            - voi: An instance of the VolumeInterest class.

        Returns:
            - poca points voxel indices as List[List[int]] with length n_mu.
        """
        indices = torch.ones((len(poca_points), 3), dtype=dtype_n, device=poca_points.device) * -1

        # Process POCA points in batches
        for start in progress_bar(range(0, len(poca_points), batch_size)):
            end = min(start + batch_size, len(poca_points))
            poca_batch = poca_points[start:end]  # Batch size is (batch_size, 3)

            voxel_index = torch.full((poca_batch.size(0), 3), -1, dtype=dtype_n, device=poca_batch.device)

            for dim in range(3):  # Loop over x, y, z dimensions
                # Extract lower and upper bounds for voxels along the current dimension
                lower_bounds = voi.voxel_edges[..., 0, dim]  # Shape: (vox_x, vox_y, vox_z)
                upper_bounds = voi.voxel_edges[..., 1, dim]  # Shape: (vox_x, vox_y, vox_z)

                # Broadcast comparison across the batch
                mask_lower = poca_batch[:, dim].unsqueeze(1).unsqueeze(1).unsqueeze(1) >= lower_bounds  # Shape: (batch_size, vox_x, vox_y, vox_z)
                mask_upper = poca_batch[:, dim].unsqueeze(1).unsqueeze(1).unsqueeze(1) <= upper_bounds  # Shape: (batch_size, vox_x, vox_y, vox_z)

                valid_voxels = mask_lower & mask_upper  # Shape: (batch_size, vox_x, vox_y, vox_z)

                # Find the first valid voxel index for each POCA point in the batch
                valid_indices = valid_voxels.nonzero(as_tuple=False)  # Shape: (num_valid_points, 4) - (batch_idx, x_idx, y_idx, z_idx)

                # Remove the loop by assigning voxel indices for the whole batch at once
                # Get the first valid index for each POCA point using a mask
                first_valid_indices = valid_indices[:, 1 + dim].view(-1)  # Extract indices for current dim (x_idx, y_idx, z_idx)

                # Ensure that each POCA point gets a valid index
                batch_indices = valid_indices[:, 0]  # Get the batch indices corresponding to each POCA point

                # Use advanced indexing to assign voxel indices
                voxel_index[batch_indices, dim] = first_valid_indices.to(dtype_n)

            indices[start:end] = voxel_index
        return indices

    @staticmethod
    def compute_n_poca_per_vox(poca_points: Tensor, voi: Volume) -> Tensor:
        """
        Computes the number of POCA points per voxel, given a voxelized volume VOI.

        Arguments:
         - voi:VolumeIntrest, an instance of the VOI class.
         - poca_indices: Tensor containing the poca points location, with size (n_mu, 3).

        Returns:
         - n_poca_per_vox: torch.tensor(dtype=int64) with size (nvox_x,nvox_y,nvox_z),
         the number of poca points per voxel.
        """

        n_poca_per_vox = torch.zeros(tuple(voi.n_vox_xyz), device=DEVICE, dtype=dtype_n)

        for i in range(voi.n_vox_xyz[2]):
            z_min = voi.xyz_min[2] + i * voi.vox_width[2]
            z_max = z_min + voi.vox_width[2]
            mask_slice = (poca_points[:, 2] >= z_min) & ((poca_points[:, 2] <= z_max))

            H, _, __ = np.histogram2d(
                poca_points[mask_slice, 0].detach().cpu().numpy(),
                poca_points[mask_slice, 1].detach().cpu().numpy(),
                bins=(int(voi.n_vox_xyz[0]), int(voi.n_vox_xyz[1])),
                range=(
                    (
                        voi.xyz_min[0].detach().cpu().numpy(),
                        voi.xyz_max[0].detach().cpu().numpy(),
                    ),
                    (
                        voi.xyz_min[1].detach().cpu().numpy(),
                        voi.xyz_max[1].detach().cpu().numpy(),
                    ),
                ),
            )

            n_poca_per_vox[:, :, i] = torch.tensor(H, dtype=dtype_n)

        return n_poca_per_vox

    @staticmethod
    def compute_dtheta_mean_per_vox(self, poca_indices: Tensor, voi: Volume) -> Tensor:
        """
        Computes the POCA points per voxel, given a voxelized volume VOI.

        Arguments:
         - voi:VolumeIntrest, an instance of the VOI class.
         - poca_indices: Tensor containing the poca points location, with size (n_mu, 3).

        Returns:
         - dtheta_mean_per_vox: torch.tensor(dtype=int64) with size (nvox_x,nvox_y,nvox_z),
         the average scattering angle per voxel.
        """
        from statistics import mean
        from torch import where
        from numpy import sqrt

        dtheta_mean_per_vox = torch.zeros(tuple(voi.n_vox_xyz), device=DEVICE, dtype=torch.float64)

        for i in range(voi.n_vox_xyz[2]):
            mask_slice_z = (poca_indices[:, 2] >= i) & ((poca_indices[:, 2] <= (i + 1)))
            for j in range(voi.n_vox_xyz[1]):
                mask_slice_y = (poca_indices[:, 1] >= j) & ((poca_indices[:, 1] <= (j + 1)))
                for k in range(voi.n_vox_xyz[0]):
                    mask_slice_x = (poca_indices[:, 0] >= k) & ((poca_indices[:, 0] <= (k + 1)))
                    part_mask = (mask_slice_x) & (mask_slice_z)
                    total_mask = (part_mask) & (mask_slice_y)
                    poca_indices_where = torch.where(total_mask)
                    if poca_indices_where[0].size() != torch.Size([0]):
                        dtheta_in_voxel = []
                        for index in poca_indices_where[0]:
                            dtheta_in_voxel.append(float(self.tracks.dtheta[index]))

                        dtheta_mean_per_vox[k, j, i] = float(mean(dtheta_in_voxel))

        return dtheta_mean_per_vox

    @staticmethod
    def compute_dtheta_rms_per_vox(self, poca_indices: Tensor, voi: Volume) -> Tensor:
        """
        Computes the rms of the scattering angle per voxel, given a voxelized volume VOI.

        Arguments:
         - voi:VolumeIntrest, an instance of the VOI class.
         - poca_indices: Tensor containing the poca points location, with size (n_mu, 3).

        Returns:
         - dtheta_rms_per_vox: torch.tensor(dtype=int64) with size (nvox_x,nvox_y,nvox_z),
         the rms of the  scattering angle per voxel.
        """

        from statistics import mean
        from torch import where
        from numpy import sqrt

        dtheta_rms_per_vox = torch.zeros(tuple(voi.n_vox_xyz), device=DEVICE, dtype=torch.float64)

        for i in range(voi.n_vox_xyz[2]):
            mask_slice_z = (poca_indices[:, 2] >= i) & ((poca_indices[:, 2] <= (i + 1)))
            for j in range(voi.n_vox_xyz[1]):
                mask_slice_y = (poca_indices[:, 1] >= j) & ((poca_indices[:, 1] <= (j + 1)))
                for k in range(voi.n_vox_xyz[0]):
                    mask_slice_x = (poca_indices[:, 0] >= k) & ((poca_indices[:, 0] <= (k + 1)))

                    part_mask = (mask_slice_x) & (mask_slice_z)

                    total_mask = (part_mask) & (mask_slice_y)

                    poca_indices_where = torch.where(total_mask)

                    if poca_indices_where[0].size() != torch.Size([0]):
                        dtheta_square_in_voxel = []

                        for index in poca_indices_where[0]:
                            dtheta_square_in_voxel.append((float(self.tracks.dtheta[index])) ** 2)

                        dtheta_rms_per_vox[k, j, i] = np.sqrt(mean(dtheta_square_in_voxel))

        return dtheta_rms_per_vox

    @staticmethod
    def compute_mask_in_voi(poca_points: Tensor, voi: Volume) -> Tensor:
        """
        Compute the mask of POCA points located within the volumne of interest.

        Arguments:
            - poca_points: Tensor with size (n_mu, 3).
            - voi: An instance of the VolumeInterest class.

        Returns:
            - mask_in_voi: Tensor of bool with size (n_mu) with True if
            the POCA point is within the voi, False otherwise.
        """

        masks_xyz = [(poca_points[:, i] >= voi.xyz_min[i]) & (poca_points[:, i] <= voi.xyz_max[i]) for i in range(3)]
        return masks_xyz[0] & masks_xyz[1] & masks_xyz[2]

    @staticmethod
    def compute_full_mask(mask_in_voi: Tensor, parallel_mask: Tensor) -> Tensor:
        """Combines the mask_in_voi and parallel_mask to create a full mask. Usefull for comparing
        instances of the POCA class obatined from the same hits.

        Args:
            mask_in_voi (Tensor): The mask of POCA points located within the volume of interest.
            parallel_mask (Tensor): The mask of parallel tracks.

        Returns:
            Tensor: The full mask.
        """

        full_mask = torch.zeros_like(parallel_mask, dtype=torch.bool, device=DEVICE)
        full_mask[torch.where(parallel_mask)[0][mask_in_voi]] = True

        return full_mask

    def plot_poca_event(self, event: int, proj: str = "XZ", voi: Optional[Volume] = None, figname: Optional[str] = None) -> None:
        matplotlib.rc("font", **font)

        sns.set_theme(
            style="darkgrid",
            rc={
                "font.family": font["family"],
                "font.size": font["size"],
                "axes.labelsize": font["size"],  # Axis label font size
                "axes.titlesize": font["size"],  # Axis title font size
                "xtick.labelsize": font["size"],  # X-axis tick font size
                "ytick.labelsize": font["size"],  # Y-axis tick font size
            },
        )

        dim_map: Dict[str, Dict[str, Union[str, int]]] = {
            "XZ": {"x": 0, "y": 2, "xlabel": r"$x$ [mm]", "ylabel": r"$z$ [mm]", "proj": "XZ"},
            "YZ": {"x": 1, "y": 2, "xlabel": r"$y$ [mm]", "ylabel": r"$z$ [mm]", "proj": "YZ"},
        }

        dim_xy = (int(dim_map[proj]["x"]), int(dim_map[proj]["y"]))

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(
            f"Tracking of event {event:,d}"
            + "\n"
            + f"{dim_map[proj]['proj']} projection, "
            + r"$\delta\theta$ = "
            + f"{self.tracks.dtheta[event] * 180 / math.pi:.2f} deg",
            fontweight="bold",
        )

        points_in_np = self.tracks.points_in.detach().cpu().numpy()
        points_out_np = self.tracks.points_out.detach().cpu().numpy()
        track_in_np = self.tracks.tracks_in.detach().cpu().numpy()[event]
        track_out_no = self.tracks.tracks_out.detach().cpu().numpy()[event]

        # Get plot xy span
        y_span = np.abs(points_in_np[event, 2] - points_out_np[event, 2])

        # Get x min max
        x_min = min(np.min(points_in_np[:, dim_map[proj]["x"]]), np.min(points_out_np[:, dim_map[proj]["x"]]))  # type: ignore
        x_max = max(np.max(points_in_np[:, dim_map[proj]["x"]]), np.max(points_out_np[:, dim_map[proj]["x"]]))  # type: ignore

        # Set plot x span
        ax.set_xlim(xmin=x_min, xmax=x_max)

        # Plot fitted point
        for point, label, color in zip((points_in_np, points_out_np), ("in", "out"), ("red", "green")):
            self.tracks.plot_point(ax=ax, point=point[event], dim_xy=dim_xy, color=color, label=label)

        # plot fitted track
        for point, track, label, pm, color in zip(
            (points_in_np[event], points_out_np[event]), (track_in_np, track_out_no), ("in", "out"), (1, -1), ("red", "green")
        ):
            ax.plot(
                [point[dim_map[proj]["x"]], point[dim_map[proj]["x"]] + track[dim_map[proj]["x"]] * y_span * pm],  # type: ignore
                [point[dim_map[proj]["y"]], point[dim_map[proj]["y"]] + track[dim_map[proj]["y"]] * y_span * pm],  # type: ignore
                alpha=0.4,
                color=color,
                linestyle="--",
                label=f"Fitted track {label}",
            )
        # Plot POCA point
        self.tracks.plot_point(ax=ax, point=self.poca_points[event].detach().cpu().numpy(), dim_xy=dim_xy, color="purple", label="POCA", size=100)

        # Plot volume of interest (if provided)
        if voi is not None:
            self.tracks.plot_voi(voi=voi, ax=ax, dim_xy=dim_xy)  # type: ignore

        plt.tight_layout()

        ax.legend(
            bbox_to_anchor=(1.0, 0.7),
        )
        ax.set_aspect("equal")
        ax.set_xlabel(f"{dim_map[proj]['xlabel']}", fontweight="bold")
        ax.set_ylabel(f"{dim_map[proj]['ylabel']}", fontweight="bold")

        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    @property
    def n_mu(self) -> int:
        r"""The number of muons."""
        return self.poca_points.shape[0]

    @property
    def poca_points(self) -> Tensor:
        r"""Tensor: The POCA points computed from the incoming and outgoing tracks."""
        if self._poca_points is None:
            self._poca_points = self.compute_poca_points(
                points_in=self.tracks.points_in,
                points_out=self.tracks.points_out,
                tracks_in=self.tracks.tracks_in,
                tracks_out=self.tracks.tracks_out,
            )
        return self._poca_points

    @poca_points.setter
    def poca_points(self, value: Tensor) -> None:
        r"""Set the POCA points."""
        self._poca_points = value

    @property
    def mask_in_voi(self) -> Tensor:
        r"""Tensor: The mask indicating which POCA points are within the volume of interest (VOI)."""
        if self._mask_in_voi is None:
            self._mask_in_voi = self.compute_mask_in_voi(poca_points=self.poca_points, voi=self.voi)
        return self._mask_in_voi

    @property
    def n_poca_per_vox(self) -> Tensor:
        r"""Tensor: The number of POCA points per voxel."""
        if self._n_poca_per_vox is None:
            self._n_poca_per_vox = self.compute_n_poca_per_vox(poca_points=self.poca_points, voi=self.voi)
        return self._n_poca_per_vox

    @n_poca_per_vox.setter
    def n_poca_per_vox(self, value: Tensor) -> None:
        r"""Set the number of POCA points per voxel."""
        self._n_poca_per_vox = value

    @property
    def dtheta_mean_per_vox(self) -> Tensor:
        r"""Tensor: The POCA points per voxel."""
        if self._dtheta_mean_per_vox is None:
            self._dtheta_mean_per_vox = self.compute_dtheta_mean_per_vox(self, poca_indices=self.poca_indices, voi=self.voi)
        return self._dtheta_mean_per_vox

    @dtheta_mean_per_vox.setter
    def dtheta_mean_per_vox(self, value: Tensor) -> None:
        r"""Set the mean scatter angle per voxel."""
        self._dtheta_mean_per_vox = value

    @property
    def dtheta_rms_per_vox(self) -> Tensor:
        r"""Tensor: The POCA points per voxel."""
        if self._dtheta_rms_per_vox is None:
            self._dtheta_rms_per_vox = self.compute_dtheta_rms_per_vox(self, poca_indices=self.poca_indices, voi=self.voi)
        return self._dtheta_rms_per_vox

    @dtheta_rms_per_vox.setter
    def dtheta_rms_per_vox(self, value: Tensor) -> None:
        r"""Set the rms of the scatter angle per voxel."""
        self._dtheta_rms_per_vox = value

    @property
    def poca_indices(self) -> Tensor:
        r"""Tensor: The indices of the POCA points assigned to each voxel."""
        if self._poca_indices is None:
            self._poca_indices = self.assign_voxel_to_pocas(poca_points=self.poca_points, voi=self.voi, batch_size=self._batch_size)
        return self._poca_indices

    @poca_indices.setter
    def poca_indices(self, value: Tensor) -> None:
        r"""Set the indices of the POCA points assigned to each voxel."""
        self._poca_indices = value

    @property
    def parallel_mask(self) -> Tensor:
        r"""Tensor: The mask indicating which tracks are parallel."""
        if self._parallel_mask is None:
            self._parallel_mask = self.compute_parallel_mask(self.tracks.tracks_in, self.tracks.tracks_out)
        return self._parallel_mask

    @property
    def full_mask(self) -> Tensor:
        r"""The full mask of POCA points."""
        full_mask = self.compute_full_mask(mask_in_voi=self.mask_in_voi, parallel_mask=self.parallel_mask)

        return full_mask

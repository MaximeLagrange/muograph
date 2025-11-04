import torch
from torch import Tensor
from copy import deepcopy
from typing import Optional, Dict, Union, Tuple, Generic, TypeVar, cast
import matplotlib.pyplot as plt
from fastprogress import progress_bar
import numpy as np
import math
from dataclasses import dataclass

from muograph.utils.save import AbsSave
from muograph.utils.device import DEVICE
from muograph.utils.datatype import dtype_track, dtype_n
from muograph.volume.volume import Volume
from muograph.tracking.tracking import TrackingMST
from muograph.plotting.style import set_plot_style
from muograph.utils.tools import normalize
from muograph.reconstruction.voxel_inferer import AbsVoxelInferer


r"""
Provides class for computing POCA locations and POCA-based voxelized scattering density predictions
"""


def are_parallel(v1: Tensor, v2: Tensor, tol: float = 1e-5) -> bool:
    cross_prod = torch.linalg.cross(v1, v2)
    return bool(torch.all(torch.abs(cross_prod) < tol).detach().cpu().item())


@dataclass
class POCAParams:
    use_p: bool = False
    p_clamp: float = 0.999
    dtheta_clamp: float = 0.999
    preds_clamp: float = 0.999


P = TypeVar("P", bound="POCAParams")


class POCA(AbsSave, AbsVoxelInferer, Generic[P]):
    r"""
    A class for Point Of Closest Approach computation in the context of a Muon Scattering Tomography analysis.
    """

    _parallel_mask: Optional[Tensor] = None  # (mu)
    _poca_points: Optional[Tensor] = None  # (mu, 3)
    _n_poca_per_vox: Optional[Tensor] = None  # (nx, ny, nz)
    _poca_indices: Optional[Tensor] = None  # (mu, 3)
    _mask_in_voi: Optional[Tensor] = None  # (mu)
    _poca_xyz_voxel_preds: Optional[Tensor] = None  # (nvox_x, nvox_y, nvox_z)

    _batch_size: int = 2048

    _vars_to_save = [
        "poca_points",
        "n_poca_per_vox",
        "poca_indices",
    ]

    _vars_to_load = [
        "poca_points",
        "n_poca_per_vox",
        "poca_indices",
    ]

    # _params: POCAParams = POCAParams()
    _params: Optional[P] = None

    _recompute_preds: bool = True

    def __init__(
        self,
        voi: Volume,
        tracking: TrackingMST,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        """
        Initializes the POCA object with either a TrackingMST instance.

        Args:
            tracking (Optional[TrackingMST]): Muon tracking data.
            voi (Optional[Volume]): Volume of interest. If provided, POCA points outside the VOI will be filtered.
            output_dir (Optional[str]): Directory to save POCA attributes.
            filename (Optional[str]): Filename for saving if output_dir is provided.
        """
        AbsSave.__init__(self, output_dir=output_dir)
        AbsVoxelInferer.__init__(self, voi=voi, tracking=tracking)

        self.filename = filename if filename else "poca"

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
        Tensor: Boolean mask of shape (n_mu,), True where tracks are **not parallel**.
        """
        # Compute the cross product for all pairs at once
        cross_prod = torch.linalg.cross(tracks_in, tracks_out, dim=1)

        # Compute the mask by checking if the cross product magnitude is below the tolerance
        mask = torch.all(torch.abs(cross_prod) >= tol, dim=1)

        return mask

    def _filter_pocas(self, mask: Tensor) -> None:
        r"""
        Removes POCA points where `mask` is False.

        Args:
            mask (Tensor): Boolean tensor of shape (n_mu). Only POCA points where mask is True are kept.
        """
        self.poca_points = self.poca_points[mask]

        if self._poca_indices is not None:
            self.poca_indices = self.poca_indices[mask]

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
        Assign voxel indices to each POCA point.

        Args:
            poca_points (Tensor): POCA locations, shape (n_mu, 3).
            voi (Volume): Volume of interest.
            batch_size (int): Number of POCA points to process per batch.

        Returns:
            Tensor: Voxel indices for each POCA point, shape (n_mu, 3), with -1 for out-of-bounds points.
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
    def compute_n_poca_per_vox(poca_indices: Tensor, voi: Volume) -> Tensor:
        nx, ny, nz = voi.n_vox_xyz

        flat_indices = poca_indices[:, 0] * ny * nz + poca_indices[:, 1] * nz + poca_indices[:, 2]

        return torch.bincount(flat_indices, minlength=nx * ny * nz).reshape(voi.n_vox_xyz)

    @staticmethod
    def compute_mask_in_voi(poca_points: Tensor, voi: Volume) -> Tensor:
        """
        Compute a boolean mask indicating which POCA points lie inside the VOI.

        Args:
            poca_points (Tensor): POCA locations, shape (n_mu, 3).
            voi (Volume): Volume of interest.

        Returns:
            Tensor: Boolean mask, shape (n_mu,).
        """

        masks_xyz = [(poca_points[:, i] >= voi.xyz_min[i]) & (poca_points[:, i] <= voi.xyz_max[i]) for i in range(3)]
        return masks_xyz[0] & masks_xyz[1] & masks_xyz[2]

    @staticmethod
    def compute_full_mask(mask_in_voi: Tensor, parallel_mask: Tensor) -> Tensor:
        """
        Combine `mask_in_voi` and `parallel_mask` to get a final valid-event mask.

        Args:
            mask_in_voi (Tensor): POCA-in-VOI mask.
            parallel_mask (Tensor): Non-parallel-track mask.

        Returns:
            Tensor: Combined boolean mask.
        """

        full_mask = torch.zeros_like(parallel_mask, dtype=torch.bool, device=DEVICE)
        full_mask[torch.where(parallel_mask)[0][mask_in_voi]] = True

        return full_mask

    def get_name_from_params(self) -> str:
        """
        Generate a string name for the POCA instance based on its parameters.

        Args:
            asr_params (POCAParams): POCA parameters.
        """
        method = "POCA_"
        name = self.get_string_params()
        name = name.replace(", ", "_")
        return method + name

    def plot_poca_event(self, event: int, proj: str = "XZ", voi: Optional[Volume] = None, figname: Optional[str] = None) -> None:
        """
        Plot a single muon event and its POCA location in 2D (XZ or YZ projection).

        Args:
            event (int): Index of the event to plot.
            proj (str): "XZ" or "YZ". Defaults to "XZ".
            voi (Optional[Volume]): If provided, VOI will be plotted.
            figname (Optional[str]): If provided, saves the figure to this path.
        """

        set_plot_style()

        dim_map: Dict[str, Dict[str, Union[str, int]]] = {
            "XZ": {"x": 0, "y": 2, "xlabel": r"$x$ [mm]", "ylabel": r"$z$ [mm]", "proj": "XZ"},
            "YZ": {"x": 1, "y": 2, "xlabel": r"$y$ [mm]", "ylabel": r"$z$ [mm]", "proj": "YZ"},
        }

        dim_xy = (int(dim_map[proj]["x"]), int(dim_map[proj]["y"]))

        fig, ax = plt.subplots()
        fig.suptitle(
            f"Tracking of event {event:,d}"
            + "\n"
            + f"{dim_map[proj]['proj']} projection, "
            + r"$\delta\theta$ = "
            + f"{self.tracks.dtheta[event] * 180 / math.pi:.2f} deg",
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
        ax.set_xlabel(f"{dim_map[proj]['xlabel']}")
        ax.set_ylabel(f"{dim_map[proj]['ylabel']}")

        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    def get_xyz_voxel_pred(self) -> Tensor:
        r"""
        Compute the per-voxel prediction of scattering strength as the root mean square (RMS)
        of scattering angles, with optional momentum weighting.

        For each track, the scattering angle (``dtheta``) and, if enabled, the momentum (``p``)
        are clamped at quantile thresholds defined in ``self.params``. These values are
        then combined into a per-track score, which is accumulated voxel-wise according to
        the track's POCA-assigned voxel index. The RMS of the scores is computed per voxel.

        Steps:
            1. Clamp ``dtheta`` values at the quantile threshold ``dtheta_clamp``.
            2. If ``use_p`` is True:
                - Clamp ``p`` values at ``p_clamp``.
                - Compute the per-track score as:
                ``score = (dtheta ** 2) * (p ** 2) / (p.mean() ** 2)``
            3. If ``use_p`` is False:
                - The score is simply ``dtheta ** 2``.
            4. For each voxel, aggregate track scores assigned to that voxel, and compute the
            RMS value: ``sqrt(sum(score) / count)``.
            5. Clamp the final voxel RMS values at the quantile threshold ``preds_clamp``.

        Returns:
            Tensor: A 3D tensor of shape ``(nx, ny, nz)``, where each element is the clamped RMS
            scattering score of tracks assigned to the corresponding voxel. Voxels without
            assigned tracks are set to 0.
        """

        dtheta_max = torch.quantile(self.tracks.dtheta, q=self.params.dtheta_clamp)
        dtheta = torch.clamp(self.tracks.dtheta, max=dtheta_max)

        if self.params.use_p:
            p_max = torch.quantile(self.tracks.p, q=self.params.p_clamp)
            p = torch.clamp(self.tracks.p, max=p_max)
            dtheta = torch.clamp(self.tracks.dtheta, max=dtheta_max)
            # score = (dtheta ** 2) * (torch.log(p ** 2)) / torch.log(p.mean() ** 2)
            score = (dtheta**2) * ((p**2)) / (p.mean() ** 2)

        else:
            score = dtheta**2

        nx, ny, nz = self.voi.n_vox_xyz

        flat_indices = self.poca_indices[:, 0] * ny * nz + self.poca_indices[:, 1] * nz + self.poca_indices[:, 2]

        # Count number of entries per voxel
        counts = torch.bincount(flat_indices, minlength=nx * ny * nz)

        sums = torch.bincount(flat_indices, weights=score, minlength=nx * ny * nz)

        # Avoid division by zero
        mask = counts > 0
        rms_values = torch.zeros_like(sums)
        rms_values[mask] = torch.sqrt(sums[mask] / counts[mask])

        voxel_rms = rms_values.reshape(nx, ny, nz)

        self._recompute_preds = False

        voxel_rms_max = torch.quantile(voxel_rms, q=self.params.preds_clamp)
        voxel_rms = torch.clamp(voxel_rms, max=voxel_rms_max)

        return voxel_rms

    def get_tracks_in_voxel(self, voxel_idx: Tuple[int, int, int]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Given a voxel index (i, j, k), return:
        - track indices contributing to that voxel
        - corresponding dtheta values
        - corresponding p values (if used)
        """
        i, j, k = voxel_idx
        nx, ny, nz = self.voi.n_vox_xyz

        # Compute flat voxel index
        voxel_flat = i * ny * nz + j * nz + k

        flat_indices = self.poca_indices[:, 0] * ny * nz + self.poca_indices[:, 1] * nz + self.poca_indices[:, 2]

        # Mask for tracks in that voxel
        mask = flat_indices == voxel_flat

        # Get indices of contributing tracks
        track_indices = torch.nonzero(mask, as_tuple=True)[0]

        # dtheta values (clamped as in voxel_rms computation)
        dtheta_max = torch.quantile(self.tracks.dtheta, q=self.params.dtheta_clamp)
        dtheta = torch.clamp(self.tracks.dtheta, max=dtheta_max)[mask]

        # p values (clamped if enabled)
        if self.params.use_p:
            p_max = torch.quantile(self.tracks.p, q=self.params.p_clamp)
            p = torch.clamp(self.tracks.p, max=p_max)[mask]
        else:
            p = None

        return track_indices, dtheta, p

    def get_string_params(self) -> str:
        """
        Converts a dataclass-like params object into a readable string, including partial functions.
        """

        parts = []
        for key, val in vars(self.params).items():
            if isinstance(val, (tuple, list)):
                val_str = "_".join([str(v) for v in val])
            else:
                val_str = str(val)
            parts.append(f"{key}={val_str}")
        return ", ".join(parts)

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
            self._n_poca_per_vox = self.compute_n_poca_per_vox(poca_indices=self.poca_indices, voi=self.voi)
        return self._n_poca_per_vox

    @n_poca_per_vox.setter
    def n_poca_per_vox(self, value: Tensor) -> None:
        r"""Set the number of POCA points per voxel."""
        self._n_poca_per_vox = value

    @property
    def n_poca_per_vox_norm(self) -> Tensor:
        """
        Normalized count of POCA points per voxel.
        """
        return normalize(self.n_poca_per_vox)  # type: ignore

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

    @property
    def params(self) -> POCAParams:
        """Return the typed params instance for this POCA variant."""
        if self._params is None:
            self._params = cast(P, POCAParams())
        return self._params

    @params.setter
    def params(self, value: P) -> None:
        """Safely update POCA parameters while preserving non-overwritten defaults."""
        if not isinstance(value, POCAParams):
            raise TypeError("params must be an instance of POCAParams")

        if self._params is None:
            # initialize with a default POCAParams and let the loop below overwrite fields
            self._params = cast(P, POCAParams())

        for key, val in value.__dict__.items():
            if val is not None:
                setattr(self._params, key, val)

        self._recompute_preds = True

    @property
    def name(self) -> str:
        r"""
        The name of the POCA configuration based on its parameters.
        """
        return self.get_name_from_params()

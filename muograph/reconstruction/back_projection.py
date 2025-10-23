from muograph.utils.save import AbsSave
from muograph.volume.volume import Volume
from muograph.tracking.tracking import Tracking
from muograph.reconstruction.asr import ASR
from muograph.plotting.voxel import VoxelPlotting
from muograph.plotting.plotting import plot_2d_vector

import math
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Union
from functools import partial
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from fastprogress import progress_bar

value_type = Union[partial, Tuple[float, float]]


class BackProjection(AbsSave, VoxelPlotting):
    r"""
    Class for computing voxel-wise muon counts through a voxelized volume.

    First, muons outside of the desired energy range are removed. By default, all muons are included.

    Each voxel receives an individual empty list V(ix, iy, iz), where ix, iy, iz are the voxel's indices
    along the x, y and z axis respectively.

    For each muon event, the ix, iy, iz indices of voxels traversed by the muon track are saved in
    the `triggered_voxels` list.

    The voxel-wise muon counts is then computed looping over the (ix, iy, iz) elements
    of the `triggered_voxels` list where one appends `score` to the associated individual
    voxel list V(ix, iy, iz). The resulting voxel-wise muon counts is a Tensor with size $(Vx, Vy, Vz)$
    where Vx, Vy, Vz is the number of voxels along the x,y,z axis.

    By default, `score` is set to 1 and we are simply counting how many times a muon passes through a given voxel.
    """

    # BackProjection  params
    _score_method: partial = partial(torch.sum)
    _score: Optional[Tensor] = None  # (mu, )

    _back_proj_params: Dict[str, value_type] = {
        "energy_range": (0.0, 1000000.0),  #
        "score_method": partial(torch.sum),
    }

    # Muon position entering / exiting volume
    _xyz_in_out: Optional[Tensor] = None
    _xyz_entry_point: Optional[Tensor] = None

    # Triggered voxels
    _triggered_voxels: Optional[List[np.ndarray]] = None

    # Density prediction
    _voxel_xyz_muon_count: Optional[Tensor] = None  # (Vx, Vy, Vz)
    _voxel_xyz_muon_count_uncs: Optional[Tensor] = None  # (Vx, Vy, Vz)

    _recompute: bool = True

    def __init__(
        self,
        voi: Volume,
        tracking: Tracking,
        label: str,
        output_dir: Optional[str] = None,
        triggered_vox_file: Optional[str] = None,
        energy_range: Tuple[float, float] = (0.0, 1000000.0),  # MeV
    ) -> None:
        r"""
        Instanciates the `BackProjection` class.

        Args:
            - voi (`Volume`) Instance of the Volume class.
            - tracks (`Tracking`) Instance of the Tracking class.
            - label (`str`) String defining the type of measurement campaign. Either
            `freesky` or `absorption`. This label is used as a tag when saving/loading
            hdf5 files.
            - output_dir (`str`) Path to the directory where to save
            the list triggered voxels as hdf5 file.
            - triggered_vox_file (`str`) Path to the hdf5 file to load the list of
            triggered voxels from.
            - energy_range (`Tuple[float, float]`) The muon energy range to consider, in MeV.
            Muon events outside of energy range are filtered out from the `Tracking` instance
            before running the algorithm.
        """

        AbsSave.__init__(self, output_dir=output_dir)
        VoxelPlotting.__init__(self, voi=voi)

        # Set volume of interest
        self.voi = voi

        # Set tracking
        self.tracks = tracking

        # Set energy range
        self.back_projection_params = {"energy_range": energy_range}

        # Check if label is correct
        if label in ["freesky", "absorption"]:
            self.label = label
        else:
            raise ValueError("Provide either 'freesky' or 'absorption' as label")

        # Filter muons outside energy range
        mask = (tracking.E > energy_range[0]) & (tracking.E < energy_range[-1])
        tracking._filter_muons(mask=mask)

        if triggered_vox_file is None:
            # Computes and saves the list of triggered voxels to hdf5 file.
            ASR.save_triggered_vox(
                self.triggered_voxels,
                self.output_dir,
                "triggered_voxels_" + label + ".hdf5",
            )

        elif triggered_vox_file is not None:
            # Loads the list of triggered voxels from hdf5 file.
            self.triggered_voxels = ASR.load_triggered_vox(triggered_vox_file)

    def __repr__(self) -> str:
        return "Back Projection algorithm for {} measurement campaign,\nwith {} muons between {:.1f} and {:.1f} GeV.".format(
            self.label,
            self.tracks.n_mu,
            self.back_projection_params["energy_range"][0] / 1000,  # type: ignore
            self.back_projection_params["energy_range"][-1] / 1000,  # type: ignore
        )

    @staticmethod
    def _get_back_projection_name(back_projection_params: Dict[str, value_type]) -> str:
        r"""
        Get the backprojection parameters names and values as string.

        Args:
            - back_projection_params (`Dict`) The parameters of the back projection algo.
        """

        def get_partial_name_args(func: Optional[partial]) -> str:
            if func is None:
                return ""

            func_name = func.func.__name__
            args, values = list(func.keywords.keys()), list(func.keywords.values())

            for i, arg in enumerate(args):
                func_name += "_{}{}_".format(arg, values[i])
            return func_name

        method = "method_{}".format(get_partial_name_args(back_projection_params["score_method"]))  # type: ignore
        energy_range = "_energy_range_{:.1f}_{:.1f}_GeV_".format(
            back_projection_params["energy_range"][0],  # type: ignore
            back_projection_params["energy_range"][1],  # type: ignore
        )
        return method + energy_range

    @staticmethod
    def _compute_xyz_out(voi: Volume, points: Tensor, theta_xy: Tensor) -> Tensor:
        r"""
        Compute muons position (x,y,z) when entering/exiting the volume.

        Args:
            - points (`Tensor`) Points along the fited muon track, with size (mu, 3).
            - voi (`Volume`)  The instance of the `Volume` class.
            - theta_xy (`Tensor`) The muon projected zenith angle in the XZ and YZ plane, with size (2, mu).

        Returns:
             - xyz_out_voi (`Tensor`): the muon position when entering/exiting the VOI, with size (n_event,2,3).
        """

        xyz_out_voi = torch.zeros((points.size()[0], 2, 3))

        dz = (abs(points[:, 2] - voi.xyz_max[2]), abs(points[:, 2] - voi.xyz_min[2]))

        for coord in [0, 1]:
            xyz_out_voi[:, 0, coord] = points[:, coord] + dz[1] * torch.tan(theta_xy[coord])
            xyz_out_voi[:, 1, coord] = points[:, coord] + dz[0] * torch.tan(theta_xy[coord])

        xyz_out_voi[:, 0, 2], xyz_out_voi[:, 1, 2] = voi.xyz_min[2], voi.xyz_max[2]

        return xyz_out_voi

    @staticmethod
    def _compute_discrete_tracks(
        voi: Volume,
        xyz_out_voi: Tensor,
        theta_xy: Tensor,
        n_points_per_z_layer: int = 3,
    ) -> torch.Tensor:
        r"""
        Computes a discretized version of the muon tracks within the voi.
        The discretized tracks contain `n_points`, with `n_points` = `n_points_per_z_layer` * `n_z_layer`,
        where `n_z_layer` is the number of voxels layers along z.

        Args:
            - voi (`Volume`) The instance of the `Volume` class.
            - xyz_out_voi (`Tensor`) The location of muons when entering/exiting the volume for outgoing tracks.
            - theta_xy (`Tensor`) The muon zenith angle projections in the XZ and YZ plane, with size (2, mu).
            - n_points_per_z_layer (`int`) The number of locations per voxel. Must be not too small (all the voxels are not triggered),
            nor too large (computationaly expensive). Default value is set at 3  point per voxel.

        Returns:
            - The discretized tracks with size (3, n_points, n_mu).
        """

        # Number of muons
        n_mu = theta_xy.size(-1)

        # Number of discrete points along the z axis
        n_points = (voi.n_vox_xyz[2] + 1) * n_points_per_z_layer

        # Get the z boundaries for the discretization
        z_min = torch.min(voi.voxel_edges[0, 0, :, :, 2])
        z_max = torch.max(voi.voxel_edges[0, 0, :, :, 2])

        # Discretize the z locations
        z_discrete = torch.linspace(z_min, z_max, n_points).unsqueeze(1).expand(-1, n_mu)

        # Precompute some terms for performance optimization
        z_diff = (z_discrete - xyz_out_voi[:, 0, 2]).unsqueeze(0)  # Shape: (1, n_points, n_mu)

        # Precompute the tangent of the angles for each muon
        tan_theta_xy = torch.tan(theta_xy)  # Shape: (2, n_mu)

        # Initialize the output tensor
        xyz_discrete_out = torch.empty((3, n_points, n_mu), dtype=z_discrete.dtype, device=z_discrete.device)

        # Compute x and y positions based on the z positions and angles
        xyz_discrete_out[0] = xyz_out_voi[:, 0, 0] + z_diff[0] * tan_theta_xy[0]  # x component
        xyz_discrete_out[1] = xyz_out_voi[:, 0, 1] + z_diff[0] * tan_theta_xy[1]  # y component

        # Assign z positions directly
        xyz_discrete_out[2] = z_discrete  # z component is the same as z_discrete

        return xyz_discrete_out

    @staticmethod
    def _find_sub_volume(voi: Volume, xyz_out_voi: torch.Tensor) -> List[List[Tensor]]:
        r"""
        Find the voxels xy indices of the sub-volume which contains the track.

        Args:
            - voi (`Volume`) Instance of the `Volume` class.
            - xyz_out_voi (`Tensor`) The location of muons when entering/exiting the volume.

        Returns:
            - sub_vol_indices_min_max (`List[Tensor]`) List containing the voxel indices.
        """

        print("\nSub-volumes")
        sub_vol_indices_min_max = []

        for event in progress_bar(range(xyz_out_voi.size()[0])):
            x_min = torch.min(torch.min(xyz_out_voi[event, :, 0]))
            x_max = torch.max(torch.max(xyz_out_voi[event, :, 0]))

            y_min = torch.min(torch.min(xyz_out_voi[event, :, 1]))
            y_max = torch.max(torch.max(xyz_out_voi[event, :, 1]))

            sub_vol_indices = (
                (voi.voxel_edges[:, :, 0, 1, 0] > x_min)
                & (voi.voxel_edges[:, :, 0, 1, 1] > y_min)
                & (voi.voxel_edges[:, :, 0, 0, 0] < x_max)
                & (voi.voxel_edges[:, :, 0, 0, 1] < y_max)
            ).nonzero()

            if len(sub_vol_indices) != 0:
                sub_vol_indices_min_max.append([sub_vol_indices[0], sub_vol_indices[-1]])

            else:
                sub_vol_indices_min_max.append([])

        return sub_vol_indices_min_max

    @staticmethod
    def _find_triggered_voxels(voi: Volume, sub_vol_indices_min_max: List[List[Tensor]], xyz_discrete_out: Tensor) -> List[np.ndarray]:
        r"""
        For each muon track, find the associated triggered voxels.

        Args:
            - voi (`Volume`) Instance of the `Volume` class.
            - sub_vol_indices_min_max (`List[Tensor]`) List containing the voxel indices.
            - xyz_discrete_out (`Tensor`) The discretized tracks, with size (3, n_points, n_mu).

        Returns:
            - triggererd_voxels (List[np.ndarray]) List containing the indices of triggered voxels as a Tensor with size (mu, n_triggered_vox, 3).
        """

        triggered_voxels = []

        print("\nVoxel triggering")
        for event in progress_bar(range(xyz_discrete_out.size()[-1])):
            if len(sub_vol_indices_min_max[event]) != 0:
                ix_min, iy_min = (
                    sub_vol_indices_min_max[event][0][0],
                    sub_vol_indices_min_max[event][0][1],
                )
                ix_max, iy_max = (
                    sub_vol_indices_min_max[event][1][0],
                    sub_vol_indices_min_max[event][1][1],
                )

                sub_voi_edges = voi.voxel_edges[ix_min : ix_max + 1, iy_min : iy_max + 1]
                sub_voi_edges = sub_voi_edges[:, :, :, :, None, :].expand(-1, -1, -1, -1, xyz_discrete_out.size()[1], -1)

                sub_mask = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_discrete_out[0, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_discrete_out[0, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_discrete_out[1, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_discrete_out[1, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_discrete_out[2, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_discrete_out[2, :, event])
                )

                vox_list = (sub_mask).nonzero()[:, :-1].unique(dim=0)
                vox_list[:, 0] += ix_min
                vox_list[:, 1] += iy_min
                triggered_voxels.append(vox_list.numpy())
            else:
                triggered_voxels.append([])

        return triggered_voxels

    @staticmethod
    def _get_triggered_voxels(voi: Volume, theta_xy: Tensor, points: Tensor) -> List[np.ndarray]:
        r"""
        For each muon track, find the associated triggered voxels. The computation is done in 4 steps:
         - Compute muon position when entering/exiting the `Volume`, based on the fitted muon tracks.
         - Compute discretized tracks.
         - Find the sub volumes traversed by the tracks.
         - For each sub-volume, find the triggered voxels.

        Args:
            - voi (`Volume`) Instance of the `Volume` class.
            - theta_xy (`Tensor`) The muon zenith angle projections in the XZ and YZ plane, with size (2, mu).
            - points (`Tensor`) Points along the fited muon track, with size (mu, 3).

        Returns:
            - triggererd_voxels (`List[Tensor]`) List containing the indices of triggered voxels as a Tensor with size (mu, n_triggered_vox, 3).
        """

        xyz_out_voi = BackProjection._compute_xyz_out(
            voi=voi,
            points=points,
            theta_xy=theta_xy,
        )

        xyz_discrete_out = BackProjection._compute_discrete_tracks(
            voi=voi,
            theta_xy=theta_xy,
            xyz_out_voi=xyz_out_voi,
        )

        sub_vol_indices_min_max = BackProjection._find_sub_volume(
            voi=voi,
            xyz_out_voi=xyz_out_voi,
        )

        return BackProjection._find_triggered_voxels(
            voi=voi,
            sub_vol_indices_min_max=sub_vol_indices_min_max,
            xyz_discrete_out=xyz_discrete_out,
        )

    def _get_voxel_xyz_muon_counts_old(self) -> Tensor:
        r"""
        Computes the density predictions per voxel.

        Returns:
            - vox_density_pred (`Tensor`) voxelwise density predictions.
        """

        score_list: List[List[List[List]]] = [
            [[[] for _ in range(self.voi.n_vox_xyz[2])] for _ in range(self.voi.n_vox_xyz[1])] for _ in range(self.voi.n_vox_xyz[0])
        ]

        print("\nAssigning voxels score")
        for i, vox_list in enumerate(progress_bar(self.triggered_voxels)):
            if self.back_projection_params["energy_range"] is not None:
                if (self.tracks.E[i] > self.back_projection_params["energy_range"][0]) & (  # type: ignore
                    self.tracks.E[i] < self.back_projection_params["energy_range"][1]  # type: ignore
                ):
                    for vox in vox_list:
                        score_list[vox[0]][vox[1]][vox[2]].append(self.score[i])
            else:
                for vox in vox_list:
                    score_list[vox[0]][vox[1]][vox[2]].append(self.score[i])

        vox_density_preds = torch.zeros(tuple(self.voi.n_vox_xyz))

        print("\nComputing final score")
        for i in progress_bar(range(self.voi.n_vox_xyz[0])):
            for j in range(self.voi.n_vox_xyz[1]):
                for k in range(self.voi.n_vox_xyz[2]):
                    if score_list[i][j][k] != []:
                        vox_density_preds[i, j, k] = self.back_projection_params["score_method"](torch.tensor(score_list[i][j][k]))  # type: ignore

        if vox_density_preds.isnan().any():
            raise ValueError("Prediction contains NaN values")

        vox_density_preds = torch.where(vox_density_preds == 0, 1.0, vox_density_preds)

        return vox_density_preds

    def _get_voxel_xyz_muon_counts(self) -> Tensor:
        """
        Computes the density predictions per voxel using a memory-efficient vectorized approach.

        Returns:
            - vox_density_pred (`Tensor`) voxelwise density predictions.
        """
        Vx, Vy, Vz = self.voi.n_vox_xyz
        flat_indices_list = []
        scores_list = []

        for i, vox_list in enumerate(self.triggered_voxels):
            if len(vox_list) == 0:
                continue
            vox_tensor = torch.tensor(vox_list, dtype=torch.long)
            flat = vox_tensor[:, 0] * Vy * Vz + vox_tensor[:, 1] * Vz + vox_tensor[:, 2]
            flat_indices_list.append(flat)
            scores_list.append(torch.full((flat.shape[0],), self.score[i].item()))

        if not flat_indices_list:
            return torch.ones((Vx, Vy, Vz))  # fallback
        flat_indices = torch.cat(flat_indices_list)
        scores = torch.cat(scores_list)

        # Accumulate with bincount
        voxel_counts_flat = torch.bincount(flat_indices, weights=scores, minlength=Vx * Vy * Vz)
        voxel_counts = voxel_counts_flat.view(Vx, Vy, Vz)

        # Avoid zero-count voxels (replace with 1.0 to match previous behavior)
        voxel_counts = torch.where(voxel_counts == 0, torch.tensor(1.0, device=voxel_counts.device), voxel_counts)

        return voxel_counts

    @staticmethod
    def _get_voxel_xyz_muon_count_uncs(xyz_muon_count: Tensor) -> Tensor:
        r"""
        Compute voxel-wise muon counts Poissonian uncertainties.

        Args:
            - xyz_muon_count (Tensor) The voxel-wise xyz muon counts with size (Vx, Vy, Vz) where
            Vi the number of voxels along the i axis.

        Returns:
            - xyz_muon_count_uncs (Tensor) The uncertainty on the voxel-wise xyz muon counts, with size (Vx, Vy, Vz) where
            Vi the number of voxels along the i axis.
        """
        print("xyz muon count uncertainty computation NOT IMPLEMENTED YET!")

        return 1 / torch.sqrt(xyz_muon_count)

    @staticmethod
    def _compute_muon_entry_point(
        xyz_in_out: Tensor,
        theta_xy: Tensor,
        voi: Volume,
    ) -> Tensor:
        # Distance between muon xy pos at the bottom of the voi
        #  and the xy edge of the voi
        dx = voi.xyz_min[0] - xyz_in_out[:, 0, 0]
        dy = voi.xyz_min[1] - xyz_in_out[:, 0, 1]

        # Distance between bottom of the voi and muon z pos when entering voi
        dz = torch.tan(math.pi / 2 - theta_xy[0]) * dx

        xyz_enters_voi = deepcopy(xyz_in_out[:, 0])

        # Muon entering from left side
        xyz_enters_voi[:, 0] = torch.where(dx > 0, xyz_in_out[:, 0, 0] + dx, xyz_enters_voi[:, 0])
        xyz_enters_voi[:, 1] = torch.where(dy > 0, xyz_in_out[:, 0, 1] + dy, xyz_enters_voi[:, 1])

        # Muons entering from the right side
        xyz_enters_voi[:, 0] = torch.where(
            dx < -voi.dxyz[0],
            xyz_in_out[:, 0, 0] - (torch.abs(xyz_in_out[:, 0, 0] - voi.xyz_max[0])),
            xyz_enters_voi[:, 0],
        )

        xyz_enters_voi[:, 1] = torch.where(
            dy < -voi.dxyz[1],
            xyz_in_out[:, 0, 1] - (torch.abs(xyz_in_out[:, 0, 1] - voi.xyz_max[1])),
            xyz_enters_voi[:, 1],
        )
        mask_xy_out = (dx > 0) | (dy > 0) | (dx < -voi.dxyz[0]) | (dy < -voi.dxyz[1])

        xyz_enters_voi[:, 2] = torch.where(mask_xy_out, xyz_in_out[:, 0, 2] + dz, xyz_enters_voi[:, 2])

        return xyz_enters_voi

    def plot_event_display(
        self,
        dim: int = 2,
        event: Optional[int] = None,
        filename: Optional[str] = None,
    ) -> None:
        from muograph.plotting.style import set_plot_style

        set_plot_style()

        # Compute figure size based on XY ratio
        figsize = self.get_fig_size(
            voi=self.voi,
            nrows=1,
            ncols=1,
            dims=tuple(np.delete([0, 1, 2], dim)),
            scale=6,
        )

        fig, ax = plt.subplots(figsize=figsize)

        # Plot voxel grid
        self.plot_voxel_grid(dim=dim, voi=self.voi, ax=ax)

        # If event is not provided, pick a random event
        event = np.random.randint(self.tracks.n_mu) if event is None else event

        # Indices of the triggered voxels
        voxels = self.triggered_voxels[event]
        n_triggered_voxel = len(voxels)

        # Compute triggered voxels position
        if n_triggered_voxel > 0:
            ixs, iys, izs = voxels[:, 0], voxels[:, 1], voxels[:, 2]

            # positions of the triggered voxels
            vx = [self.voi.voxel_centers[ix, 0, 0, 0].item() for ix in ixs]
            vy = [self.voi.voxel_centers[0, iy, 0, 1].item() for iy in iys]
            vz = [self.voi.voxel_centers[0, 0, iz, 2].item() for iz in izs]
        else:
            vx, vy, vz = None, None, None

        mapping = {
            2: {"vox_x": vx, "vox_y": vy, "projection": "XY"},
            1: {"vox_x": vx, "vox_y": vz, "projection": "XZ"},
            0: {"vox_x": vy, "vox_y": vz, "projection": "YZ"},
        }

        # Plot voxel position
        ax.scatter(
            mapping[dim]["vox_x"],  # type: ignore
            mapping[dim]["vox_y"],  # type: ignore
            color="red",
            marker=".",
            alpha=0.5,
        )

        # Muon's position entering the voi
        point_2D = np.delete(self.xyz_entry_point[event].numpy(), dim)

        track_2D = np.delete(self.tracks.tracks[event].numpy(), dim)
        track_2D_norm = -track_2D / np.linalg.norm(track_2D)

        # Plot muon direction
        plot_2d_vector(
            ax,
            vector=track_2D_norm,
            origin=point_2D,
            scale=1 / (float(self.voi.vox_width[0]) * 5),
        )

        # Plot muon position entering volume
        ax.scatter(
            point_2D[0],
            point_2D[1],
            color="green",
            marker="x",
            s=200,
            label=r"$\mu$ enters volume",
        )

        # Plot triggered voxels
        ax.scatter(
            mapping[dim]["vox_x"],  # type: ignore
            mapping[dim]["vox_y"],  # type: ignore
            color="red",
            marker=".",
            alpha=0.5,
            label="triggered voxels",
        )

        # Muon's direction legend
        ax.scatter(
            [],
            [],
            marker=r"$\longrightarrow$",
            c="green",
            s=120,
            label=r"$\mu$ direction",
        )

        if n_triggered_voxel > 0:
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False)
            title = f"Muon event display in {mapping[dim]['projection']} for event {event:,d}\n{n_triggered_voxel} triggered voxels"
        else:
            title = f"Muon event display in {mapping[dim]['projection']} for event {event:,d}\nno triggered voxels"

        # Plot title
        ax.set_title(
            title,
            fontweight="bold",
        )

        # Save figure
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")

        plt.show()

    def _filter_events(self, mask: Tensor) -> None:
        r"""
        Filter out events with `mask = False`.
        """

        self.triggered_voxels = [vox for vox, keep in zip(self.triggered_voxels, mask.tolist()) if keep]
        self.tracks._filter_muons(mask)

    @property
    def voxel_xyz_muon_count(self) -> Tensor:
        r"""
        The voxel-wise xyz muon counts, with size (Vx, Vy, Vz),
        where Vi the number of voxels along the i axis.
        """
        if self._voxel_xyz_muon_count is None or self._recompute:
            self._voxel_xyz_muon_count = self._get_voxel_xyz_muon_counts()
            self._recompute = False
        return self._voxel_xyz_muon_count

    @property
    def voxel_xyz_muon_count_uncs(self) -> Tensor:
        r"""
        The uncertainty on the voxel-wise xyz muon counts, with size (Vx, Vy, Vz),
        where Vi the number of voxels along the i axis.
        """
        if self._voxel_xyz_muon_count_uncs is None:
            self._voxel_xyz_muon_count_uncs = self._get_voxel_xyz_muon_count_uncs(self.voxel_xyz_muon_count)
        return self._voxel_xyz_muon_count_uncs

    @property
    def triggered_voxels(self) -> List[np.ndarray]:
        r"""
        The event-wise list of triggered voxels indices.
        """
        if self._triggered_voxels is None:
            self._triggered_voxels = self._get_triggered_voxels(voi=self.voi, theta_xy=self.tracks.theta_xy, points=self.tracks.points)
        return self._triggered_voxels

    @triggered_voxels.setter
    def triggered_voxels(self, value: List[np.ndarray]) -> None:
        self._triggered_voxels = value

    @property
    def xyz_in_out(self) -> Tensor:
        if self._xyz_in_out is None:
            self._xyz_in_out = self._compute_xyz_out(self.voi, self.tracks.points, self.tracks.theta_xy)
        return self._xyz_in_out

    @property
    def xyz_entry_point(self) -> Tensor:
        if self._xyz_entry_point is None:
            self._xyz_entry_point = self._compute_muon_entry_point(self.xyz_in_out, self.tracks.theta_xy, self.voi)
        return self._xyz_entry_point

    # Parameters
    @property
    def back_projection_params(self) -> Dict[str, value_type]:
        r"""
        The back projection algorithm parameters.
        """
        return self._back_proj_params

    @back_projection_params.setter
    def back_projection_params(self, value: Dict[str, value_type]) -> None:
        r"""
        Sets the parameters of the back projection algorithm.

        Args:
            - Dict containing the parameters name and value. Only parameters with
            valid name and non `None` values will be updated.
        """
        self._recompute = True
        for key in value.keys():
            if key in self._back_proj_params.keys():
                if value[key] is not None:
                    self._back_proj_params[key] = value[key]

    @property
    def score(self) -> Tensor:
        r"""
        The score to append to a voxel score list when it's triggered by a muon.
        By default, voxels receive 1 for each muons traversing them.
        """
        if self._score is None:
            self._score = torch.ones(self.tracks.n_mu)
            # self._score = (torch.cos(self.tracks.theta)) ** (-1)
        return self._score

    @score.setter
    def score(self, value: Tensor) -> None:
        self._score = value

from typing import Optional, Tuple, Dict, Union, List
import numpy as np
from functools import partial
import math
import torch
from torch import Tensor
from pathlib import Path
from fastprogress import progress_bar
import h5py
import matplotlib.pyplot as plt

from muograph.utils.save import AbsSave
from muograph.tracking.tracking import TrackingMST
from muograph.volume.volume import Volume
from muograph.reconstruction.voxel_inferer import AbsVoxelInferer
from muograph.plotting.params import configure_plot_theme, font, tracking_figsize

value_type = Union[partial, Tuple[float, float], bool]


r"""
Provides class for computing voxelized scattering density predictions
based on the Binned clustered algorithm (Angle Statistics Reconstruction:
a robust reconstruction algorithm for Muon Scattering Tomography,
M. Stapleton et al 2014 JINST 9 P11019,
https://iopscience.iop.org/article/10.1088/1748-0221/9/11/P11019)).
"""


class ASR(AbsSave, AbsVoxelInferer):
    _triggered_voxels: Optional[List[np.ndarray]] = None
    _n_mu_per_vox: Optional[Tensor] = None  # (Nx, Ny, Nz)
    _recompute_preds = True

    _asr_params: Dict[str, value_type] = {
        "score_method": partial(np.quantile, q=0.5),
        "p_range": (0.0, 10000000),  # MeV
        "dtheta_range": (0.0, math.pi / 3),
        "use_p": False,
    }

    _vars_to_save = ["triggered_voxels"]

    _vars_to_load = ["trigerred_voxels"]

    def __init__(
        self,
        voi: Volume,
        tracking: TrackingMST,
        output_dir: Optional[str] = None,
        triggered_vox_file: Optional[str] = None,
    ) -> None:
        r"""Initializes the ASR object with either instances of the `Volume` and `TrackingMST` class

        Args:
            voi (Volume): The volume of interest, as an instance of the `Volume` class.
            tracking (TrackingMST): The muon tracks, as an instance of the `TrackingMST` class.
            output_dir (Optional[str], optional): Path to a directory where to save the triggered voxels
            as a hdf5 file. Defaults to None.
            triggered_vox_file (Optional[str], optional): Path to a hdf5 file where to load the triggered voxels
            from. Defaults to None.
        """

        AbsSave.__init__(self, output_dir=output_dir)
        AbsVoxelInferer.__init__(self, voi=voi, tracking=tracking)

        if triggered_vox_file is None:
            if self.output_dir is not None:
                self.save_triggered_vox(self.triggered_voxels, self.output_dir, "triggered_voxels.hdf5")

        elif triggered_vox_file is not None:
            self.tracks = tracking
            self.triggered_voxels = self.load_triggered_vox(triggered_vox_file)

    def __repr__(self) -> str:
        description = "ASR algorithm using a c"
        description_tracks = self.tracks.__repr__()[1:]
        return description + description_tracks

    @staticmethod
    def save_triggered_vox(triggered_voxels: List[np.ndarray], directory: Path, filename: str) -> None:
        r"""
        Method for saving triggered voxel as a hdf5 file.
        """
        with h5py.File(directory / filename, "w") as f:
            print("Saving trigerred voxels to {}".format(directory / filename))
            for i, indices in enumerate(progress_bar(triggered_voxels)):
                f.create_dataset("{}".format(i), data=indices)
        f.close()

    @staticmethod
    def load_triggered_vox(triggered_vox_file: str) -> List[np.ndarray]:
        r"""
        Method for loading triggered voxel from hdf5 file.
        """
        with h5py.File(triggered_vox_file, "r") as f:
            print("Loading trigerred voxels from {}".format(triggered_vox_file))
            triggered_voxels = [f["{}".format(i)][:] for i, _ in enumerate(progress_bar(f.keys()))]
        f.close()
        return triggered_voxels

    @staticmethod
    def _compute_xyz_in_out(
        points_in: Tensor,
        points_out: Tensor,
        voi: Volume,
        theta_xy_in: Tuple[Tensor, Tensor],
        theta_xy_out: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Compute muon position (x,y,z) when enters/exits the volume,
        both for the incoming and outgoing tracks.

        Args:
             - points_in (Tensor): Points on incoming muon tracks.
             - points_out (Tensor): Points on outgoing muon tracks.
             - voi (Volume): Instance of the volume class.
             - theta_xy_in (Tensor): The incoming projected zenith angle in XZ and YZ plane.
             - theta_xy_out (Tensor): The outgoing projected zenith angle in XZ and YZ plane.

        Returns:
             - xyz_in_VOI [3, 2, mu] the muon position when entering/exiting the volume,
            for the INCOMING tracks.
             - xyz_out_VOI [3, 2, mu] the muon position when entering/exiting the volume,
            for the OUTGOING tracks.
        """
        n_mu = theta_xy_in[0].size(0)
        xyz_in_voi, xyz_out_voi = torch.zeros((n_mu, 2, 3), device=theta_xy_in[0].device), torch.zeros((n_mu, 2, 3), device=theta_xy_in[0].device)

        for point, theta_xy, pm, xyz in zip(
            [points_in, points_out],
            [theta_xy_in, theta_xy_out],
            [1, -1],
            [xyz_in_voi, xyz_out_voi],
        ):
            dz = (abs(point[:, 2] - voi.xyz_max[2]), abs(point[:, 2] - voi.xyz_min[2]))

            for coord, theta in zip([0, 1], theta_xy):
                xyz[:, 0, coord] = point[:, coord] - dz[1] * torch.tan(theta) * (pm)
                xyz[:, 1, coord] = point[:, coord] - dz[0] * torch.tan(theta) * (pm)
            xyz[:, 0, 2], xyz[:, 1, 2] = voi.xyz_min[2], voi.xyz_max[2]

        return xyz_in_voi, xyz_out_voi

    @staticmethod
    def _compute_discrete_tracks(
        voi: Volume,
        xyz_in_out_voi: Tuple[Tensor, Tensor],
        theta_xy_in: Tuple[Tensor, Tensor],
        theta_xy_out: Tuple[Tensor, Tensor],
        n_points_per_z_layer: int = 7,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Computes a discretized version of the muon incoming and outgoing track
        within the volume.
        The number of points is defined as n_points = n_points_per_z_layer * n_z_layer,
        where n_z_layer is the number of voxels along z.

        Args:
             - voi (Volume): Instance of the volume class.
             - xyz_in_out_voi (Tensor): The location of muons when entering/exiting the volume for incoming and outgoing tracks
             - theta_xy_in (Tuple[Tensor, Tensor]):  The incoming muon zenith angle projections in the XZ and YZ plane.
             - theta_xy_out (Tuple[Tensor, Tensor]):  The outcoming muon zenith angle projections in the XZ and YZ plane.
             - n_points_per_z_layer (int): The number of locations per voxel. Must be not too small (all the voxels are not triggered),
            nor too large (computationaly expensive). Default value is set at 3  point per voxel.

        Returns:
             - The discretized incoming and outgoing tracks with size (3, n_points, n_mu)
        """
        n_mu = theta_xy_in[0].size(0)
        n_points = (voi.n_vox_xyz[2] + 1) * n_points_per_z_layer

        # Compute the z locations cross the voi
        z_discrete = (
            torch.linspace(torch.min(voi.voxel_edges[0, 0, :, :, 2]), torch.max(voi.voxel_edges[0, 0, :, :, 2]), n_points, device=theta_xy_out[0].device)[  # type: ignore[call-overload]
                :, None
            ]
        ).expand(-1, n_mu)

        xyz_discrete_in, xyz_discrete_out = torch.ones((3, n_points, n_mu), device=theta_xy_out[0].device), torch.ones(
            (3, n_points, n_mu), device=theta_xy_out[0].device
        )

        for xyz_discrete, theta_in_out, xyz_in_out in zip(
            [xyz_discrete_in, xyz_discrete_out],
            [theta_xy_in, theta_xy_out],
            xyz_in_out_voi,
        ):
            for dim, theta in zip([0, 1], theta_in_out):
                xyz_discrete[dim] = torch.abs(z_discrete - xyz_in_out[:, 0, 2]) * torch.tan(theta) + xyz_in_out[:, 0, dim]

            xyz_discrete[2] = z_discrete

        return xyz_discrete_in, xyz_discrete_out

    @staticmethod
    def _find_sub_volume(
        voi: Volume,
        xyz_in_voi: Tensor,
        xyz_out_voi: Tensor,
    ) -> List[List[Tensor]]:
        r"""
        Find the xy voxel indices of the sub-volume which contains both incoming and outgoing tracks.

        Args:
             - voi (Volume): Instance of the volume class.
             - xyz_in_voi (Tensor): The location of muons when entering/exiting the volume for the incoming track.
             - xyz_out_voi (Tensor): The location of muons when entering/exiting the volume for the outgoing track.

        Returns:
             - sub_vol_indices_min_max (List[Tensor]): List containing the voxel indices.
        """
        # Precompute voxel boundaries for simpler condition checks
        voxel_edges_x_min = voi.voxel_edges[:, :, 0, 0, 0]
        voxel_edges_x_max = voi.voxel_edges[:, :, 0, 1, 0]
        voxel_edges_y_min = voi.voxel_edges[:, :, 0, 0, 1]
        voxel_edges_y_max = voi.voxel_edges[:, :, 0, 1, 1]

        # Get the min, max x and y coordinnates of the tracks entering the voi
        xyz_min = torch.min(torch.cat([xyz_in_voi, xyz_out_voi], dim=1), dim=1).values
        xyz_max = torch.max(torch.cat([xyz_in_voi, xyz_out_voi], dim=1), dim=1).values

        x_min, y_min = xyz_min[:, 0], xyz_min[:, 1]
        x_max, y_max = xyz_max[:, 0], xyz_max[:, 1]

        print("\nSub-volumes")
        sub_vol_indices_min_max = []
        n_mu = xyz_in_voi.size(0)

        for event in progress_bar(range(n_mu)):
            # Filter valid voxels for this event
            condition = (
                (voxel_edges_x_min < x_max[event])
                & (voxel_edges_x_max > x_min[event])
                & (voxel_edges_y_min < y_max[event])
                & (voxel_edges_y_max > y_min[event])
            )

            # Get the valid voxels indices
            sub_vol_indices = condition.nonzero()

            # If sub_vol_indices is non-empty, determine bounds
            if sub_vol_indices.numel() > 0:
                sub_vol_indices_min_max.append([sub_vol_indices[0], sub_vol_indices[-1]])

            else:
                sub_vol_indices_min_max.append([])

        return sub_vol_indices_min_max

    @staticmethod
    def _find_triggered_voxels(
        voi: Volume,
        sub_vol_indices_min_max: List[List[Tensor]],
        xyz_discrete_in: Tensor,
        xyz_discrete_out: Tensor,
    ) -> List[np.ndarray]:
        r"""
        For each muon incoming and outgoing tracks, find the associated triggered voxels.
        Only voxels triggered by both INCOMING and OUTGOING tracks are kept.

        Args:
             - voi (Volume): Instance of the volume class.
             - sub_vol_indices_min_max (List[Tensor]):  the xy voxel indices of the sub-volume
             which contains both incoming and outgoing tracks.
             - xyz_discrete_in (Tensor): The discretized incoming tracks with size (3, n_points, n_mu)
             - xyz_discrete_out (Tensor): The discretized outgoing tracks with size (3, n_points, n_mu)

        Returns:
             - triggererd_voxels (List[Tensor]): List with len() = n_mu, containing the indices
             of the triggered voxels as a Tensor (with size [n_triggered_vox,3])
        """

        triggered_voxels = []
        n_mu = xyz_discrete_in.size(2)

        print("\nVoxel triggering")
        for event in progress_bar(range(n_mu)):
            event_indices = sub_vol_indices_min_max[event]

            if len(sub_vol_indices_min_max[event]) != 0:
                ix_min, iy_min = event_indices[0][0], event_indices[0][1]
                ix_max, iy_max = event_indices[1][0], event_indices[1][1]

                sub_voi_edges = voi.voxel_edges[ix_min : ix_max + 1, iy_min : iy_max + 1]
                sub_voi_edges = sub_voi_edges[:, :, :, :, None, :].expand(-1, -1, -1, -1, xyz_discrete_out.size()[1], -1)

                xyz_in_event = xyz_discrete_in[:, :, event]
                xyz_out_event = xyz_discrete_out[:, :, event]

                sub_mask_in = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_in_event[0])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_in_event[0])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_in_event[1])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_in_event[1])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_in_event[2])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_in_event[2])
                )

                sub_mask_out = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_out_event[0])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_out_event[0])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_out_event[1])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_out_event[1])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_out_event[2])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_out_event[2])
                )

                vox_list = (sub_mask_in & sub_mask_out).nonzero()[:, :-1].unique(dim=0)
                vox_list[:, 0] += ix_min
                vox_list[:, 1] += iy_min
                triggered_voxels.append(vox_list.detach().cpu().numpy())
            else:
                triggered_voxels.append(np.empty([0, 3]))
        return triggered_voxels

    @staticmethod
    def get_asr_name(
        asr_params: Dict[str, value_type],
    ) -> str:
        r"""
        Returns the name of the bca given its parameters.
        """

        def get_partial_name_args(func: partial) -> str:
            r"""
            Returns the name, arguments and their value of a partial method as a string.
            """
            func_name = func.func.__name__
            args, values = list(func.keywords.keys()), list(func.keywords.values())
            for i, arg in enumerate(args):
                func_name += "_{}={}".format(arg, values[i])
            return func_name

        method = "method_{}_".format(get_partial_name_args(asr_params["score_method"]))  # type: ignore
        dtheta = "{:.2f}_{:.2f}_rad_".format(asr_params["dtheta_range"][0], asr_params["dtheta_range"][1])  # type: ignore
        dp = "{:.0f}_{:.0f}_MeV_".format(asr_params["p_range"][0], asr_params["p_range"][1])  # type: ignore
        use_p = "use_p_{}".format(asr_params["use_p"])

        asr_name = method + dtheta + dp + use_p

        return asr_name

    @staticmethod
    def get_triggered_voxels(
        points_in: Tensor,
        points_out: Tensor,
        voi: Volume,
        theta_xy_in: Tuple[Tensor, Tensor],
        theta_xy_out: Tuple[Tensor, Tensor],
    ) -> List[np.ndarray]:
        """
        Gets the `xyz` indices of the voxels along each muon path, as a list of np.ndarray.
        e.g triggered_voxels[i] is an np.ndarray with shape (n, 3) where n is the number
        of voxels along the muon path for the muon event i.

        Args:
             - points_in (Tensor): Points on incoming muon tracks.
             - points_out (Tensor): Points on outgoing muon tracks.
             - voi (Volume): Instance of the volume class.
             - theta_xy_in (Tensor): The incoming projected zenith angle in XZ and YZ plane.
             - theta_xy_out (Tensor): The outgoing projected zenith angle in XZ and YZ plane.

        Returns:
             - triggered_voxels (List[np.ndarray]): the list of triggered voxels.
        """
        xyz_in_voi, xyz_out_voi = ASR._compute_xyz_in_out(
            points_in=points_in,
            points_out=points_out,
            voi=voi,
            theta_xy_in=theta_xy_in,
            theta_xy_out=theta_xy_out,
        )

        xyz_discrete_in, xyz_discrete_out = ASR._compute_discrete_tracks(
            voi=voi,
            xyz_in_out_voi=(xyz_in_voi, xyz_out_voi),
            theta_xy_in=theta_xy_in,
            theta_xy_out=theta_xy_out,
            n_points_per_z_layer=7,
        )

        sub_vol_indices_min_max = ASR._find_sub_volume(voi=voi, xyz_in_voi=xyz_in_voi, xyz_out_voi=xyz_out_voi)

        return ASR._find_triggered_voxels(
            voi=voi,
            sub_vol_indices_min_max=sub_vol_indices_min_max,
            xyz_discrete_in=xyz_discrete_in,
            xyz_discrete_out=xyz_discrete_out,
        )

    def get_xyz_voxel_pred(self) -> Tensor:
        r"""
        Computes the density predictions per voxel.

        Returns:
            vox_density_pred (Tensor): voxelwise density predictions
        """

        score_list: List[List[List[List]]] = [
            [[[] for _ in range(self.voi.n_vox_xyz[2])] for _ in range(self.voi.n_vox_xyz[1])] for _ in range(self.voi.n_vox_xyz[0])
        ]

        if self._asr_params["use_p"]:
            score = np.log(self.tracks.dtheta.detach().cpu().numpy() * self.tracks.E.detach().cpu().numpy())
        else:
            score = self.tracks.dtheta.detach().cpu().numpy()

        mask_E = (self.tracks.E > self.asr_params["p_range"][0]) & (  # type: ignore
            self.tracks.E < self.asr_params["p_range"][1]  # type: ignore
        )
        mask_theta = (self.tracks.dtheta > self.asr_params["dtheta_range"][0]) & (  # type: ignore
            self.tracks.dtheta < self.asr_params["dtheta_range"][1]  # type: ignore
        )

        if self.asr_params["use_p"]:  # type: ignore
            mask = mask_E & mask_theta
        else:
            mask = mask_E

        print("\nAssigning voxels score")
        for i, vox_list in enumerate(progress_bar(self.triggered_voxels)):
            for vox in vox_list:
                if mask[i]:
                    score_list[vox[0]][vox[1]][vox[2]].append(score[i])

        vox_density_preds = torch.zeros(tuple(self.voi.n_vox_xyz))

        print("Compute final score")
        for i in progress_bar(range(self.voi.n_vox_xyz[0])):
            for j in range(self.voi.n_vox_xyz[1]):
                for k in range(self.voi.n_vox_xyz[2]):
                    if score_list[i][j][k] != []:
                        vox_density_preds[i, j, k] = self.asr_params["score_method"](score_list[i][j][k])  # type: ignore

        if vox_density_preds.isnan().any():
            raise ValueError("Prediction contains NaN values")

        self._recompute_preds = False

        if self.asr_params["use_p"]:  # type: ignore
            return torch.exp(vox_density_preds)
        else:
            return vox_density_preds

    def get_n_mu_per_vox(
        self,
    ) -> Tensor:
        n_mu_per_vox = torch.zeros(self.voi.n_vox_xyz)

        all_voxels = np.vstack([ev for ev in self.triggered_voxels if len(ev) > 0])
        unique_voxels, counts = np.unique(all_voxels, axis=0, return_counts=True)
        for (x, y, z), count in zip(unique_voxels, counts):
            n_mu_per_vox[x, y, z] += count

        return n_mu_per_vox

    def plot_asr_event(self, event: int, proj: str = "XZ", figname: Optional[str] = None) -> None:
        configure_plot_theme(font=font)  # type: ignore

        # Inidices and labels
        dim_map: Dict[str, Dict[str, Union[str, int]]] = {
            "XZ": {"x": 0, "y": 2, "xlabel": r"$x$ [mm]", "ylabel": r"$z$ [mm]"},
            "YZ": {"x": 1, "y": 2, "xlabel": r"$y$ [mm]", "ylabel": r"$z$ [mm]"},
        }

        # Data numpy
        points_in_np = self.tracks.points_in.detach().cpu().numpy()
        points_out_np = self.tracks.points_out.detach().cpu().numpy()
        track_in_np = self.tracks.tracks_in.detach().cpu().numpy()[event]
        track_out_no = self.tracks.tracks_out.detach().cpu().numpy()[event]

        # Y span
        y_span = abs(points_in_np[event, 2] - points_out_np[event, 2])

        fig, ax = plt.subplots(figsize=tracking_figsize)
        if self.triggered_voxels[event].shape[0] > 0:
            n_trig_vox = f"# triggered voxels = {self.triggered_voxels[event].shape[0]}"
        else:
            n_trig_vox = "no voxels triggered"
        fig.suptitle(
            f"Tracking of event {event:,d}" + "\n" + r"$\delta\theta$ = " + f"{self.tracks.dtheta[event] * 180 / math.pi:.2f} deg, " + n_trig_vox,
            fontweight="bold",
            y=1.05,
        )

        # Plot voxel grid
        self.plot_voxel_grid(
            dim=1,
            voi=self.voi,
            ax=ax,
        )

        # Plot triggered voxels
        if self.triggered_voxels[event].shape[0] > 0:
            for i, vox_idx in enumerate(self.triggered_voxels[event]):
                ix, iy = vox_idx[dim_map[proj]["x"]], vox_idx[2]
                vox_x = self.voi.voxel_centers[ix, 0, 0, 0] if proj == "XZ" else self.voi.voxel_centers[0, ix, 0, 1]
                label = "Triggered voxel" if i == 0 else None
                ax.scatter(
                    x=vox_x.detach().cpu(),
                    y=self.voi.voxel_centers[0, 0, iy, 2].detach().cpu(),
                    color="blue",
                    label=label,
                    alpha=0.3,
                )

        # Plot tracks
        for point, track, label, pm, color in zip(
            (points_in_np[event], points_out_np[event]), (track_in_np, track_out_no), ("in", "out"), (1, -1), ("red", "green")
        ):
            ax.plot(
                [point[dim_map[proj]["x"]], point[dim_map[proj]["x"]] + track[dim_map[proj]["x"]] * y_span * pm],  # type: ignore
                [point[dim_map[proj]["y"]], point[dim_map[proj]["y"]] + track[dim_map[proj]["y"]] * y_span * pm],  # type: ignore
                alpha=0.6,
                color=color,
                linestyle="--",
                label=f"Fitted track {label}",
            )

        # Legend
        ax.legend(
            bbox_to_anchor=(1.0, 0.7),
        )

        # Save figure
        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    @property
    def theta_xy_in(self) -> Tuple[Tensor, Tensor]:
        r"""
        Returns:
            Tuple[(mu, )] tensors of muons incoming projected zenith angles in XZ and YZ plane.
        """
        return (self.tracks.theta_xy_in[0], self.tracks.theta_xy_in[1])

    @property
    def theta_xy_out(self) -> Tuple[Tensor, Tensor]:
        r"""
        Returns:
            Tuple[(mu, )] tensors of muons outgoing projected zenith angles in XZ and YZ plane.
        """
        return (self.tracks.theta_xy_out[0], self.tracks.theta_xy_out[1])

    @property
    def asr_params(self) -> Dict[str, value_type]:
        r"""
        The parameters of the ASR algorithm.
        """
        return self._asr_params

    @asr_params.setter
    def asr_params(self, value: Dict[str, value_type]) -> None:
        r"""
        Sets the parameters of the ASR algorithm.
        Args:
            - Dict containing the parameters name and value. Only parameters with
            valid name and non `None` values wil be updated.
        """
        for key in value.keys():
            if key in self._asr_params.keys():
                if value[key] is not None:
                    self._asr_params[key] = value[key]

        self._recompute_preds = True

    @property
    def triggered_voxels(self) -> List[np.ndarray]:
        r"""
        The list of triggered voxels.
        """
        if self._triggered_voxels is None:
            self._triggered_voxels = self.get_triggered_voxels(
                self.tracks.points_in,
                self.tracks.points_out,
                self.voi,
                self.theta_xy_in,
                self.theta_xy_out,
            )
        return self._triggered_voxels

    @triggered_voxels.setter
    def triggered_voxels(self, value: List[np.ndarray]) -> None:
        self._triggered_voxels = value

    @property
    def n_mu_per_vox(self) -> Tensor:
        if self._n_mu_per_vox is None:
            self._n_mu_per_vox = self.get_n_mu_per_vox()
        return self._n_mu_per_vox

    @n_mu_per_vox.setter
    def n_mu_per_vox(self, value: Tensor) -> None:
        self._n_mu_per_vox = value

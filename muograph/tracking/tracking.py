import torch
from torch import Tensor
from typing import Tuple, Optional, Dict, Union
import math
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np

from muograph.utils.save import AbsSave
from muograph.hits.hits import Hits
from muograph.volume.volume import Volume
from muograph.plotting.style import set_plot_style, subplots_4_figsize, subplots_2_figsize, n_bins, alpha
from muograph.utils.device import DEVICE

MU_MASS = 105.66  # MeV/c

r"""
Provides class for converting muon hits of the `Hits` class into
muon tracks usable for image reconstruction purposes.
"""


class Tracking(AbsSave):
    r"""
    A class for tracking muons based on hits data.

    The muon hits on detector planes are plugged into a linear fit
    to compute a track T(tx, ty, tz) and a point on that track P(px, py, pz).

    From T(tx, ty, tz), one computes the muons' zenith angles labeled theta, and defined as the
    angle between the vertical axis z and the muon track. A vertical muon has
    a 0 [rad] zenith angle.

    The projections of the zenith angle in the XZ and YZ planes,
    theta_x and theta_y respectively, are also computed.

    If the detector spatial resolution is simulated, the tracking angular resolution is compted as
    the standard deviation of the distribution of the error on theta. The error on theta,
    is computed by comparing the values of theta computed from the generated hits (Hits.gen_hits) and
    from the smeared hits (Hits.reco_hits).
    """

    _tracks: Optional[Tensor] = None  # (mu, 3)
    _points: Optional[Tensor] = None  # (mu, 3)
    _theta: Optional[Tensor] = None  # (mu)
    _theta_xy: Optional[Tensor] = None  # (mu)
    _angular_error: Optional[Tensor] = None  # (mu)
    _angular_res: Optional[float] = None
    _E: Optional[Tensor] = None  # (mu)
    _tracks_eff: Optional[Tensor] = None  # (mu)
    _vars_to_save = [
        "tracks",
        "points",
        "angular_res",
        "angular_error",
        "E",
        "label",
        "measurement_type",
        "tracks_eff",
    ]
    _maskable = ["_tracks", "_points", "_theta", "_theta_xy", "_angular_error", "_E", "tracks_eff"]

    def __init__(
        self,
        label: str,
        hits: Optional[Hits] = None,
        output_dir: Optional[str] = None,
        tracks_hdf5: Optional[str] = None,
        tracks_df: Optional[pd.DataFrame] = None,
        measurement_type: Optional[str] = None,
    ) -> None:
        r"""
        Initializes the Tracking object.

        The instantiation can be done in three ways:
        - By providing `hits`: Computes tracks and saves them as HDF5 files in `output_dir`.
        - By providing `tracks_hdf5`: Loads tracking features from the specified HDF5 file.
        - By providing `tracks_df`: Loads tracking features from the specified Pandas DataFrame.


        Args:
            label (str): The position of the hits relative to the passive volume ('above' or 'below').
            hits (Optional[Hits]): An instance of the Hits class, required if `tracks_hdf5` is not provided.
            output_dir (Optional[str]): Directory to save Tracking attributes.
            tracks_hdf5 (Optional[str]): Path to an HDF5 file with previously saved Tracking data.
            tracks_df (Optional[pd.DataFrame]): Pandas DataFrame with previously saved Tracking data.
            measurement_type (Optional[str]): Type of measurement campaign, either 'absorption' or 'freesky'.
        """

        self._label = self._validate_label(label)
        self._measurement_type = self._validate_measurement_type(measurement_type)

        super().__init__(output_dir=output_dir)

        if (hits is not None) & (tracks_hdf5 is None) & (tracks_df is None):
            self.hits = hits

            if self.output_dir is not None:
                filename = "tracks_" + self.label if self._measurement_type == "" else "tracks_" + self.label + "_" + self._measurement_type
                self.save_attr(
                    attributes=self._vars_to_save,
                    directory=self.output_dir,
                    filename=filename,
                )

        elif tracks_hdf5 is not None:
            self.hits = None
            self.load_attr(attributes=self._vars_to_save, filename=tracks_hdf5)

        elif tracks_df is not None:
            self.hits = None
            self.load_from_df(df=tracks_df)

    def __repr__(self) -> str:
        description = f"Collection of tracks from {self.n_mu:,d} muons "
        if self.angular_res == 0.0:
            description += "\n with perfect angular resolution."
        else:
            description += f"\n with angular resolution = {self.angular_res*180/math.pi:.2f} deg"
        return description

    @staticmethod
    def _validate_label(label: str) -> str:
        if label not in ["above", "below"]:
            raise ValueError("Label must be either 'above' or 'below'.")
        return label

    @staticmethod
    def _validate_measurement_type(measurement_type: Optional[str]) -> str:
        valid_types = ["absorption", "freesky", None]
        if measurement_type not in valid_types:
            raise ValueError("Measurement type must be 'absorption', 'freesky', or None.")
        return measurement_type or ""

    @staticmethod
    def get_tracks_points_from_hits(hits: Tensor, chunk_size: int = 200_000) -> Tuple[Tensor, Tensor]:
        r"""Compute muon tracks and points based on the number of the number of detection planes.

        Args:
            hits (Tensor): The hits data with shape (3, n_plane, mu).
            chunk_size (int, optional): Size of chunks for processing in case n_mu is very large. Defaults to 200_000.

        Raises:
            ValueError: If there are less than 2 hits per event, cannot proceed to tracking.

        Returns:
            Tuple[Tensor, Tensor]: The points and tracks tensors
            with respective size (mu, 3).
        """

        n_panels = hits.size()[1]

        if n_panels > 2:
            return Tracking.get_tracks_points_from_multi_hits(hits=hits, chunk_size=chunk_size)
        elif n_panels == 2:
            return Tracking.get_tracks_points_from_two_hits(hits=hits)
        else:
            raise ValueError("Hits have less than 2 hits per event, cannot proceed to tracking.")

    @staticmethod
    def get_tracks_points_from_multi_hits(hits: Tensor, chunk_size: int = 200_000) -> Tuple[Tensor, Tensor]:
        r"""
        The muon hits on detector planes are plugged into a linear fit
        to compute a track T(tx, ty, tz) and a point on that track P(px, py, pz).

        Args:
            - hits (Tensor): The hits data with shape (3, n_plane, mu).
            - chunk_size (int): Size of chunks for processing in case n_mu is very large.

        Returns:
            - tracks, points (Tuple[Tensor, Tensor]): The points and tracks tensors
            with respective size (mu, 3).
        """

        _, __, mu = hits.shape

        tracks = torch.empty((mu, 3), dtype=hits.dtype, device=hits.device)
        points = torch.empty((mu, 3), dtype=hits.dtype, device=hits.device)

        # Process in chunks to manage memory
        for start in range(0, mu, chunk_size):
            end = min(start + chunk_size, mu)

            hits_chunk = hits[:, :, start:end]  # Shape: (3, n_plane, chunk_size)

            # Calculate the mean point for each set of hits in the chunk
            points_chunk = hits_chunk.mean(dim=1)  # Shape: (3, chunk_size)

            # Center the data
            centered_hits_chunk = hits_chunk - points_chunk.unsqueeze(1)  # Shape: (3, n_plane, chunk_size)

            # Perform SVD in batch mode
            centered_hits_chunk = centered_hits_chunk.permute(2, 1, 0)  # Shape: (chunk_size, n_plane, 3)
            _, _, vh = torch.linalg.svd(centered_hits_chunk, full_matrices=False)  # vh shape: (chunk_size, 3, 3)

            # Extract the principal direction (first right singular vector) for each set
            tracks_chunk = vh[:, 0, :]  # Shape: (chunk_size, 3)

            # Store the chunk results in the main output tensors
            tracks[start:end, :] = tracks_chunk
            points[start:end, :] = points_chunk.T

        tracks[:, 2] = torch.where(tracks[:, 2] > 0, -tracks[:, 2], tracks[:, 2])
        return tracks, points

    @staticmethod
    def get_tracks_points_from_two_hits(hits: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        The muon hits on detector planes are used
        to compute a track T(tx, ty, tz) and a point on that track P(px, py, pz).

        Args:
            - hits (Tensor): The hits data with shape (3, n_plane, mu).

        Returns:
            - tracks, points (Tuple[Tensor, Tensor]): The points and tracks tensors
            with respective size (mu, 3).
        """

        points = torch.mean(hits, dim=1)
        tracks = (hits[:, -1] - hits[:, 0]) / torch.norm(hits[:, -1] - hits[:, 0], dim=0, keepdim=True)

        return tracks.T, points.T

    @staticmethod
    def get_theta_from_tracks(tracks: Tensor) -> Tensor:
        r"""
        Compute muons' zenith angle in radians from the direction vector of the track.

        Args:
            tracks (Tensor): Direction vector of the tracks.

        Returns:
            theta (Tensor): Muons' zenith angle.
        """
        x, y, z = tracks[:, 0], tracks[:, 1], tracks[:, 2]

        # Compute theta using arctan of the transverse component over z
        theta = math.pi - torch.atan2(torch.sqrt(x**2 + y**2), z)

        return theta

    @staticmethod
    def get_theta_xy_from_tracks(tracks: Tensor) -> Tensor:
        r"""
        Compute muons' projected zenith angle in XZ and YZ planes in radians,
        from the direction vector of the track.

        Args:
            tracks (Tensor): Direction vector of the tracks.

        Returns:
            theta_xy (Tensor): Muons' zenith angle in XZ and YZ planes.
        """
        theta_xy = torch.empty((2, tracks.size(0)), dtype=tracks.dtype, device=tracks.device)

        theta_xy[0] = torch.atan(tracks[:, 0] / tracks[:, 2])
        theta_xy[1] = torch.atan(tracks[:, 1] / tracks[:, 2])

        theta_xy = torch.where(theta_xy > math.pi / 2, math.pi - theta_xy, theta_xy)

        return theta_xy

    @staticmethod
    def get_tracks_eff_from_hits_eff(hits_eff: Tensor) -> Tensor:
        r"""
        Computes the tracks efficiency.
        """
        n_expected_hit_per_track = hits_eff.size()[0]
        tracks_eff = torch.where(hits_eff.sum(dim=0) == n_expected_hit_per_track, 1, 0)
        return tracks_eff

    @staticmethod
    def tracks_points_to_df(tracks: Tensor, points: Tensor, E: Tensor, angular_error: Tensor) -> pd.DataFrame:
        """
        Convert tracks and points into a pandas DataFrame.

        Args:
            tracks (Tensor): Tensor of shape (n_mu, 3), with track information as px, py, pz.
            points (Tensor): Tensor of shape (n_mu, 3), with point information as x, y, z.

        Returns:
            pd.DataFrame: A DataFrame with columns ["x", "y", "z", "px", "py", "pz", "E"].
        """
        # Ensure the tensors have the correct shape
        assert tracks.size(1) == 3 and points.size(1) == 3, "Input tensors must have shape (n_mu, 3)"
        assert tracks.size(0) == points.size(0), "Tracks and points must have the same number of rows"

        # Convert tensors to numpy arrays and create the DataFrame
        data = {
            "x": points[:, 0].detach().cpu().numpy(),
            "y": points[:, 1].detach().cpu().numpy(),
            "z": points[:, 2].detach().cpu().numpy(),
            "px": tracks[:, 0].detach().cpu().numpy(),
            "py": tracks[:, 1].detach().cpu().numpy(),
            "pz": tracks[:, 2].detach().cpu().numpy(),
            "E": E.detach().cpu().numpy(),
            "angular_error": angular_error.detach().cpu().numpy(),
        }
        return pd.DataFrame(data)

    def load_from_df(self, df: pd.DataFrame) -> None:
        """
        Load tracks and points from a pandas DataFrame into PyTorch tensors.

        Args:
            df (pd.DataFrame): A DataFrame with columns ["x", "y", "z", "px", "py", "pz"].

        Sets:
            self.tracks: Tensor of shape (n_mu, 3) with columns ["px", "py", "pz"].
            self.points: Tensor of shape (n_mu, 3) with columns ["x", "y", "z"].
        """
        # Validate DataFrame columns
        required_columns = ["x", "y", "z", "px", "py", "pz"]
        assert all(col in df.columns for col in required_columns), f"DataFrame must contain columns {required_columns}"

        # Convert DataFrame columns to PyTorch tensors
        self.tracks = torch.tensor(df[["px", "py", "pz"]].values, dtype=torch.float32, device=DEVICE)
        self.points = torch.tensor(df[["x", "y", "z"]].values, dtype=torch.float32, device=DEVICE)
        self.E = torch.tensor(df[["E"]].values, dtype=torch.float32, device=DEVICE)
        if "angular_error" in df.keys():
            self.angular_error = torch.tensor(df["angular_error"].values, dtype=torch.float32, device=DEVICE)

    def get_angular_error(self, reco_theta: Tensor) -> Tensor:
        r"""
        Compute the angular error between the generated and reconstructed tracks.
        Args:
            reco_theta (Tensor): Zenith angle of the reconstructed tracks.

        Returns:
            (Tensor): angular error with size (mu).
        """

        gen_tracks, _ = self.get_tracks_points_from_hits(hits=self.hits.gen_hits)  # type: ignore
        gen_theta = self.get_theta_from_tracks(tracks=gen_tracks)
        return gen_theta - reco_theta

    def plot_muon_features(
        self,
        figname: Optional[str] = None,
    ) -> None:
        r"""
        Plot the zenith angle and energy of the reconstructed tracks using Seaborn for improved visualization.
        Args:
            figname (Tensor): If provided, save the figure as `figname`.
        """

        # Extract data
        zenith_angle_deg = self.theta.detach().cpu().numpy() * 180 / math.pi
        energy_gev = self.E.detach().cpu().numpy() / 1000
        zenith_mean = zenith_angle_deg.mean()
        energy_mean = energy_gev.mean()

        set_plot_style()

        # Create subplots
        fig, axs = plt.subplots(ncols=2, figsize=subplots_2_figsize)
        fig.suptitle(f"Batch of {self.n_mu:,d} muons")

        # Zenith angle plot
        axs[0].hist(zenith_angle_deg, bins=n_bins, color="blue", edgecolor="black", alpha=alpha)
        axs[0].axvline(zenith_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {zenith_mean:.2f}°")
        axs[0].set_xlabel(r"Zenith Angle $\theta$ [deg]")
        axs[0].set_ylabel("Frequency [a.u]")
        axs[0].legend()

        # Energy plot
        axs[1].hist(energy_gev, bins=n_bins, color="purple", edgecolor="black", log=True, alpha=alpha)

        axs[1].axvline(energy_mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {energy_mean:.2f} GeV")
        axs[1].set_xlabel(
            "Energy [GeV]",
        )
        axs[1].set_ylabel(
            "Frequency [a.u]",
        )
        axs[1].legend()

        # Adjust layout
        plt.tight_layout()

        # Save the plot if required
        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    def plot_angular_error(
        self,
        figname: Optional[str] = None,
    ) -> None:
        """Plot the angular error of the tracks using Seaborn.

        Args:
            figname (Optional[str], optional): Path to a file where to save the figure. Defaults to None.
        """
        set_plot_style()

        # Extract data
        angular_error_deg = self.angular_error.detach().cpu().numpy() * 180 / math.pi
        mean = angular_error_deg.mean()
        std = angular_error_deg.std()

        # Create the plot
        plt.figure()
        plt.hist(angular_error_deg, bins=n_bins, color="blue", alpha=alpha, edgecolor="black")

        # Add mean line
        plt.axvline(mean, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean:.2f}°")

        # Add ±1σ region
        plt.axvline(mean - std, color="green", linestyle=":", linewidth=1.5)
        plt.axvline(mean + std, color="green", linestyle=":", linewidth=1.5, label=r"$\pm 1\sigma$")

        # Add labels, title, and legend
        plt.title(f"Batch of {self.n_mu:,d} muons\nAngular Resolution = {std:.3f}°")
        plt.xlabel(
            r"Angular Error $\delta\theta$ [deg]",
        )
        plt.ylabel(
            "Frequency [a.u.]",
        )
        plt.legend()

        # Adjust layout
        plt.tight_layout()

        # Save the plot if required
        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    def plot_tracking_event(self, event: int, proj: str = "XZ", hits: Optional[Hits] = None, figname: Optional[str] = None) -> None:
        """Plot the hits and the fitted tracks for a given event.

        Args:
            event (int): The event to plot.
            proj (str): The projection to plot along. Either XZ or YZ.
            hits (Hits): An instance of the Hits class. Must be provided if self.hits is None.
            figname (Optional[str], optional): Path to a file where to save the figure. Defaults to None.
        """

        set_plot_style()

        if proj not in ("XZ", "YZ"):
            raise ValueError("proj argument must be XZ or YZ.")

        dim_map = {
            "XZ": {"x": 0, "y": 2, "xlabel": r"$x$ [mm]", "ylabel": r"$z$ [mm]", "proj": "XZ"},
            "YZ": {"x": 1, "y": 2, "xlabel": r"$y$ [mm]", "ylabel": r"$z$ [mm]", "proj": "YZ"},
        }

        fig, axs = plt.subplots(ncols=3, gridspec_kw={"width_ratios": [1, 1, 0.25]})
        axs[2].axis("off")  # Turn off the legend axis
        fig.suptitle(f"Tracking of event {event:,d}\n{dim_map[proj]['proj']} projection, " r"$\theta$ = " + f"{self.theta[event] * 180 / math.pi:.2f} deg")

        if (self.hits is None) and (hits is None):
            raise ValueError("Provide hits as argument.")
        elif self.hits is not None:
            n_panels = self.hits.n_panels
            reco_hits_np = self.hits.reco_hits.detach().cpu().numpy()
            gen_hits_np = self.hits.gen_hits.detach().cpu().numpy()
        elif hits is not None:
            n_panels = hits.n_panels
            reco_hits_np = hits.reco_hits.detach().cpu().numpy()
            gen_hits_np = hits.gen_hits.detach().cpu().numpy()

        points_np = self.points.detach().cpu().numpy()
        tracks_np = self.tracks.detach().cpu().numpy()

        hits_x = reco_hits_np[dim_map[proj]["x"], :, event]  # type: ignore

        # Calculate spans
        x_span = abs(np.min(hits_x) - np.max(hits_x))
        y_span = abs(np.min(reco_hits_np[dim_map[proj]["y"]]) - np.max(reco_hits_np[dim_map[proj]["y"]]))  # type: ignore

        # Axis limits
        x_extent = max(x_span / 1.5, 100)  # Enforce minimum extent of 100

        axs[0].set_xlim(
            points_np[event, dim_map[proj]["x"]] - x_extent,  # type: ignore
            points_np[event, dim_map[proj]["x"]] + x_extent,  # type: ignore
        )
        axs[1].set_xlim(
            points_np[event, dim_map[proj]["x"]] - x_span / 1.5,  # type: ignore
            points_np[event, dim_map[proj]["x"]] + x_span / 1.5,  # type: ignore
        )
        for ax in axs[:2]:
            ax.set_ylim(
                points_np[event, dim_map[proj]["y"]] - y_span / 1.8,  # type: ignore
                points_np[event, dim_map[proj]["y"]] + y_span / 1.8,  # type: ignore
            )

        # Detector panels
        for ax in axs[:2]:
            for i in range(n_panels):
                label = "Detector panel" if i == 0 else None
                ax.axhline(y=np.mean(reco_hits_np[dim_map[proj]["y"], i]), label=label, alpha=0.4)  # type: ignore

        # Plot hits, points, and tracks
        for ax in axs[:2]:
            ax.scatter(
                x=reco_hits_np[dim_map[proj]["x"], :, event],  # type: ignore
                y=reco_hits_np[dim_map[proj]["y"], :, event],  # type: ignore
                label="Reco. hits",
                color="red",
                marker="+",
                alpha=0.5,
                s=80,
            )
            ax.scatter(
                x=gen_hits_np[dim_map[proj]["x"], :, event],  # type: ignore
                y=gen_hits_np[dim_map[proj]["y"], :, event],  # type: ignore
                label="Gen. hits",
                color="green",
                marker="+",
                alpha=0.5,
            )
            ax.scatter(
                x=points_np[event, dim_map[proj]["x"]],  # type: ignore
                y=points_np[event, dim_map[proj]["y"]],  # type: ignore
                label="Fitted point",
                color="red",
                marker="x",
                s=100,
            )
            ax.plot(
                [
                    points_np[event, dim_map[proj]["x"]] - tracks_np[event, dim_map[proj]["x"]] * 1000,  # type: ignore
                    points_np[event, dim_map[proj]["x"]] + tracks_np[event, dim_map[proj]["x"]] * 1000,  # type: ignore
                ],
                [
                    points_np[event, dim_map[proj]["y"]] - tracks_np[event, dim_map[proj]["y"]] * 1000,  # type: ignore
                    points_np[event, dim_map[proj]["y"]] + tracks_np[event, dim_map[proj]["y"]] * 1000,  # type: ignore
                ],
                alpha=0.4,
                color="red",
                linestyle="--",
                label="Fitted track",
            )
            ax.set_xlabel(
                dim_map[proj]["xlabel"],
            )
            ax.set_ylabel(
                dim_map[proj]["ylabel"],
            )

        axs[0].set_aspect("equal")

        # Collect unique labels and handles
        handles, labels = axs[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))  # Remove duplicates while preserving order

        # Show the legend on the third axis
        axs[2].legend(unique.values(), unique.keys(), loc="center left", frameon=False)

        plt.tight_layout()

        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    def _reset_vars(self) -> None:
        r"""
        Reset attributes to None.
        """
        self._theta = None  # (mu)
        self._theta_xy = None  # (2, mu)

    def _filter_muons(self, mask: Tensor) -> None:
        r"""
        Remove muons specified as False in `mask`.

        Args:
            - mask (Boolean tensor) Muons with False elements will be removed.
        """

        # Set attributes without setter method to None
        self._reset_vars()
        n_muons = self.tracks.size(0)

        for var in self._maskable:
            data = getattr(self, var, None)
            if isinstance(data, Tensor) and data.size(0) == n_muons:
                setattr(self, var, data[mask])

        if self.hits is not None:
            self.hits._filter_events(mask)

    @property
    def df(self) -> pd.DataFrame:
        r"""
        DataFrame containing tracks, points and energy data.
        """
        return self.tracks_points_to_df(tracks=self.tracks, points=self.points, E=self.E, angular_error=self.angular_error)

    @property
    def tracks(self) -> Tensor:
        r"""
        The muons' direction
        """
        if self._tracks is None:
            self._tracks, self._points = self.get_tracks_points_from_hits(hits=self.hits.reco_hits)  # type: ignore
        return self._tracks

    @tracks.setter
    def tracks(self, value: Tensor) -> None:
        self._tracks = value

    @property
    def points(self) -> Tensor:
        r"""
        Point on muons' trajectory.
        """
        if self._points is None:
            self._tracks, self._points = self.get_tracks_points_from_hits(hits=self.hits.reco_hits)  # type: ignore
        return self._points

    @points.setter
    def points(self, value: Tensor) -> None:
        self._points = value

    @property
    def tracks_eff(self) -> Tensor:
        r"""
        The tracks efficiency: 1 -> tracks is detcted, 0 -> tracks not detected.
        """
        if self._tracks_eff is None:
            if self.hits is not None:
                self._tracks_eff = self.get_tracks_eff_from_hits_eff(self.hits.hits_eff)  # type: ignore
            else:
                self._tracks_eff = torch.ones_like(self.theta)
        return self._tracks_eff

    @tracks_eff.setter
    def tracks_eff(self, value: Tensor) -> None:
        self._tracks_eff = value

    @property
    def tracking_eff(self) -> float:
        r"""
        The tracking efficiency, defined as the number of detected events
        divided by the total number of events.
        """
        return self.tracks_eff.sum().detach().cpu().item() / self.n_mu

    @property
    def theta_xy(self) -> Tensor:
        r"""
        The muons' projected zenith angle in XZ and YZ plane.
        """
        if self._theta_xy is None:
            self._theta_xy = self.get_theta_xy_from_tracks(self.tracks)
        return self._theta_xy

    @property
    def theta(self) -> Tensor:
        r"""
        The muons' zenith angle.
        """

        if self._theta is None:
            self._theta = self.get_theta_from_tracks(self.tracks)
        return self._theta

    @property
    def E(self) -> Tensor:
        r"""The muons' energy."""
        if self._E is None:
            self._E = self.hits.E  # type: ignore
        return self._E

    @E.setter
    def E(self, value: Tensor) -> None:
        self._E = value

    @property
    def n_mu(self) -> int:
        r"""
        The number of muons.
        """
        return len(self.theta)

    @property
    def angular_error(self) -> Tensor:
        r"""
        The angular error between the generated and reconstructed tracks.
        """
        if self._angular_error is None:
            if self.hits is None or self.hits.spatial_res.sum() == 0.0:
                self._angular_error = torch.zeros_like(self.theta)
            else:
                self._angular_error = self.get_angular_error(self.theta)
        return self._angular_error

    @angular_error.setter
    def angular_error(self, value: Tensor) -> None:
        self._angular_error = value

    @property
    def angular_res(self) -> float:
        r"""
        The angular resolution, computed as the standard deviation of the
        angular error distribution.
        """
        if self._angular_res is None:
            self._angular_res = self.angular_error.std().item()
        return self._angular_res

    @angular_res.setter
    def angular_res(self, value: float) -> None:
        self._angular_res = value

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def measurement_type(self) -> str:
        return self._measurement_type

    @measurement_type.setter
    def measurement_type(self, value: str) -> None:
        self._measurement_type = value


class TrackingMST:
    r"""
    A class for tracking muons in the context of a Muon Scattering Tomography analysis.
    """

    _theta_in: Optional[Tensor] = None  # (mu)
    _theta_out: Optional[Tensor] = None  # (mu)
    _theta_xy_in: Optional[Tensor] = None  # (2, mu)
    _theta_xy_out: Optional[Tensor] = None  # (2, mu)
    _dtheta: Optional[Tensor] = None  # (mu)
    _muon_eff: Optional[Tensor] = None  # (mu)

    _vars_to_load = ["tracks", "points", "angular_res", "E", "tracks_eff", "hits"]

    def __init__(self, trackings: Tuple[Tracking, Tracking]) -> None:
        r"""
        Initializes the TrackingMST object with 2 instances of the Tracking class
        (with tags 'above' and 'below').

        Args:
            - trackings (Tuple[Tracking, Tracking]): instances of the Tracking class
            for the incoming muon tracks (Tracking.label = 'above') and outgoing tracks
            (Tracking.label = 'below')
        """
        # Load data from Tracking instances
        for tracking, tag in zip(trackings, ["_in", "_out"]):
            self.load_attr_from_tracking(tracking, tag)

        # Filter muon event due to detector efficiency
        self.n_mu_removed = (self.n_mu - self.muon_eff.sum()).detach().cpu().item()
        self._tracking_eff = 1 - (self.n_mu_removed / self.n_mu)
        self._filter_muons(self.muon_eff)

    def __repr__(self) -> str:
        description = f"Collection of tracks from {self.n_mu:,d} muons "
        if (self.angular_res_in == 0.0) & (self.angular_res_out == 0.0):
            description += "\n with perfect angular resolution."
        else:
            average_res_deg = (self.angular_res_in + self.angular_res_out) * 180 / (2 * math.pi)
            description += f"\n with average angular resolution = {average_res_deg:.2f} deg"
        if self.tracking_eff < 1.0:
            description += f"\n with tracking efficiency = {self.tracking_eff * 100:.2f} %"
        else:
            description += "\n with perfect tracking efficiency"
        return description

    def load_attr_from_tracking(self, tracking: Tracking, tag: str) -> None:
        r"""
        Load class attributes in TrackingMST._vars_to_load from the input Tracking class.
        Attributes name are modified according to the tag as `attribute_name` + `tag`,
        so that incoming and outgoing muon features can be treated independently (except for the kinetic energy).

        Args:
            - tracking (Tracking): Instance of the Tracking class.
            - tag (str): tag to add to the attribuites name (either `_in` or `_out`)
        """

        for attr in self._vars_to_load:
            data = getattr(tracking, attr)
            if attr != "E":
                attr += tag
            setattr(self, attr, data)

    @staticmethod
    def compute_dtheta_from_tracks(tracks_in: Tensor, tracks_out: Tensor, tol: float = 1.0e-12) -> Tensor:
        r"""
        Computes the scattering angle between the incoming and outgoing muon tracks.

        Args:
            - tracks_in (Tensor): The incoming muon tracks with size (mu, 3)
            - tracks_out (Tensor): The outgoing muon tracks with size (mu, 3)
            - tol (float): A tolerance parameter to avoid errors when computing acos(dot_prod).
            the dot_prod is clamped between (-1 + tol, 1 - tol). Default value is 1.e12.

        Returns:
            - dtheta (Tensor): The scattering angle between the incoming and outgoing muon
            tracks in [rad], with size (mu).
        """

        def norm(x: Tensor) -> Tensor:
            return torch.sqrt((x**2).sum(dim=-1))

        dot_prod = torch.abs(tracks_in * tracks_out).sum(dim=-1) / (norm(tracks_in) * norm(tracks_out))
        dot_prod = torch.clamp(dot_prod, -1.0 + tol, 1.0 - tol)
        dtheta = torch.acos(dot_prod)
        return dtheta

    @staticmethod
    def get_muon_eff(tracks_eff_in: Tensor, tracks_eff_out: Tensor) -> Tensor:
        """Computes muon-wise efficiency through all detector panels, based on the
        muon-wise efficiency through the set of panels before and after the object.
        Muon is detected => efficency = 1, muon not detected => efficiency = 0.

        Args:
            tracks_eff_in (Tensor): muon-wise efficiency through the set of panels before the object.
            tracks_eff_out (Tensor): muon-wise efficiency through the set of panels after the object.

        Returns:
            muon_wise_eff: muon-wise efficiency through all detector panels.
        """
        muon_wise_eff = (tracks_eff_in + tracks_eff_out) == 2
        return muon_wise_eff

    def _filter_muons(self, mask: Tensor) -> None:
        r"""
        Remove muons specified as False in `mask`.

        Arguments:
            mask: (N,) Boolean tensor. Muons with False elements will be removed.
        """

        # Set attributes without setter method to None
        self._reset_vars()

        n_muons = self.tracks_in.size()[0]
        # Loop over class attributes and apply the mask is Tensor
        for var in vars(self).keys():
            data = getattr(self, var)
            if isinstance(data, Tensor):
                if data.size()[0] == n_muons:
                    setattr(self, var, data[mask])
            elif isinstance(data, Hits):
                data._filter_events(mask=mask)

    def _reset_vars(self) -> None:
        r"""
        Reset attributes to None.
        """

        self._theta_in = None  # (mu)
        self._theta_out = None  # (mu)
        self._theta_xy_in = None  # (2, mu)
        self._theta_xy_out = None  # (2, mu)
        self._dtheta = None  # (mu)

    def plot_muon_features(
        self,
        figname: Optional[str] = None,
    ) -> None:
        r"""
        Plot the zenith angle and energy of the reconstructed tracks.

        Args:
            figname (str): If provided, save the figure at figname.
        """

        set_plot_style()

        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=subplots_4_figsize)
        axs = axs.ravel()

        fig.suptitle(f"Batch of {self.n_mu:,d} muons")

        # Zenith angle
        data_theta = self.theta_in.detach().cpu().numpy() * 180 / math.pi
        axs[0].hist(data_theta, bins=n_bins, color="blue", edgecolor="black", alpha=alpha)
        axs[0].axvline(
            x=data_theta.mean(),
            label=f"mean = {data_theta.mean():.1f} deg",
            color="red",
        )
        axs[0].set_xlabel(
            r" Zenith angle $\theta$ [deg]",
        )

        # Energy
        data_E = self.E.detach().cpu().numpy() / 1000
        axs[1].hist(data_E, bins=n_bins, log=True, color="purple", edgecolor="black", alpha=alpha)
        axs[1].axvline(
            x=data_E.mean(),
            label=f"mean = {data_E.mean():.2f} GeV",
            color="red",
        )
        axs[1].set_xlabel(
            r" Energy [GeV]",
        )

        # Scattering angle
        data_dtheta = self.dtheta.detach().cpu().numpy() * 180 / math.pi
        axs[2].hist(data_dtheta, bins=n_bins, log=True, color="green", edgecolor="black", alpha=alpha)
        axs[2].axvline(
            x=data_dtheta.mean(),
            label=f"mean = {data_dtheta.mean():.3f} deg",
            color="red",
        )
        axs[2].set_xlabel(r" Scattering angle $\delta\theta$ [deg]")

        # Style formatting
        for ax in axs[:-1]:
            ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5)
            ax.set_ylabel("Frequency [a.u]")
            ax.tick_params(axis="both")
            ax.legend()

        # Remove unused subplot
        axs[-1].remove()
        axs[-1] = None

        plt.tight_layout()

        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")

        plt.show()

    @staticmethod
    def plot_voi(
        voi: Volume,
        ax: matplotlib.axes._axes.Axes,
        dim_xy: Tuple[int, int],
    ) -> None:
        r"""
        Plot the volume of interest along the desired projection on provided matplolib axis.

        Args:
            voi (Volume): Instance of the volume class.
            ax (matplotlib.axes._axes.Axes): The axis to plot on.
            dim_xy (Tuple[int, int]): The index of the x and y dimension on the plot.
            e.g for XZ projection, dim_xy = (0, 2), for YZ projection, dim_xy = (1, 2).
        """
        from matplotlib.patches import Rectangle

        ax.add_patch(
            Rectangle(
                xy=(  # type: ignore
                    voi.xyz_min[dim_xy[0]].detach().cpu().numpy(),
                    voi.xyz_min[dim_xy[1]].detach().cpu().numpy(),
                ),
                width=voi.dxyz[dim_xy[0]].detach().cpu().numpy(),  # type: ignore
                height=voi.dxyz[dim_xy[1]].detach().cpu().numpy(),  # type: ignore
                fill=True,
                edgecolor="black",
                facecolor="blue",
                alpha=0.3,
                label="VOI",
            )
        )

    @staticmethod
    def plot_point(
        ax: matplotlib.axes._axes.Axes,
        point: np.ndarray,
        dim_xy: Tuple[int, int],
        color: str,
        label: str,
        size: int = 100,
    ) -> None:
        r"""
        Plot a point on the provided matplotlib axes.

        Args:
            point (np.ndarray): The point to plot, with shape (3,).
            ax (matplotlib.axes._axes.Axes): The axis to plot on.
            dim_xy (Tuple[int, int]): The index of the x and y dimension on the plot.
            e.g for XZ projection, dim_xy = (0, 2), for YZ projection, dim_xy = (1, 2).
            color (str): The desired point color.
            label (str): The label to put on the legend.
            size (int): The size of the point.
        """
        ax.scatter(x=point[dim_xy[0]], y=point[dim_xy[1]], label=f"Fitted point {label}", color=color, marker="x", s=size)

    def plot_tracking_event(
        self,
        event: int,
        proj: str = "XZ",
        voi: Optional[Volume] = None,
        figname: Optional[str] = None,
    ) -> None:
        """Plot the fitted incoming and outgoing tracks and point for the desired event.

        Args:
            event (int): The desired event.
            proj (str, optional): The desired projection, either `XZ` or `YZ`. Defaults to "XZ".
            voi (Optional[Volume], optional): An instance of the Volume class. If provided, gets represented on the plot. Defaults to None.
            figname (Optional[str], optional): If provided, the figure is saved as `figname`. Defaults to None.
        """
        set_plot_style()

        # Numpy data
        points_in_np = self.points_in.detach().cpu().numpy()
        points_out_np = self.points_out.detach().cpu().numpy()
        track_in_np = self.tracks_in.detach().cpu().numpy()[event]
        track_out_no = self.tracks_out.detach().cpu().numpy()[event]

        dim_map: Dict[str, Dict[str, Union[str, int]]] = {
            "XZ": {"x": 0, "y": 2, "xlabel": r"$x$ [mm]", "ylabel": r"$z$ [mm]", "proj": "XZ"},
            "YZ": {"x": 1, "y": 2, "xlabel": r"$y$ [mm]", "ylabel": r"$z$ [mm]", "proj": "YZ"},
            "XY": {
                "x": 0,
                "y": 1,
                "xlabel": r"$x$ [mm]",
                "ylabel": r"$y$ [mm]",
                "proj": "XY",
            },
        }

        dim_xy = (int(dim_map[proj]["x"]), int(dim_map[proj]["y"]))

        fig, ax = plt.subplots()
        fig.suptitle(
            f"Tracking of event {event:,d}"
            + "\n"
            + f"{dim_map[proj]['proj']} projection, "
            + r"$\delta\theta$ = "
            + f"{self.dtheta[event] * 180 / math.pi:.2f} deg",
        )

        # Get plot xy span
        y_min = min(np.min(points_in_np[:, dim_map[proj]["y"]]), np.min(points_out_np[:, dim_map[proj]["y"]]))  # type: ignore
        y_max = max(np.max(points_in_np[:, dim_map[proj]["y"]]), np.max(points_out_np[:, dim_map[proj]["y"]]))  # type: ignore
        y_span = abs(y_min - y_max)

        x_min = min(np.min(points_in_np[:, dim_map[proj]["x"]]), np.min(points_out_np[:, dim_map[proj]["x"]]))  # type: ignore
        x_max = max(np.max(points_in_np[:, dim_map[proj]["x"]]), np.max(points_out_np[:, dim_map[proj]["x"]]))  # type: ignore
        x_span = abs(x_min - x_max)

        # Set plot x span
        ax.set_xlim(xmin=x_min - x_span / 10, xmax=x_max + x_span / 10)
        ax.set_ylim(ymin=y_min - y_span / 10, ymax=y_max + y_span / 10)

        # Plot fitted point
        for point, label, color in zip((points_in_np, points_out_np), ("in", "out"), ("red", "green")):
            self.plot_point(ax=ax, point=point[event], dim_xy=dim_xy, color=color, label=label)
        # plot fitted track
        for point, track, label, pm, color in zip(
            (points_in_np[event], points_out_np[event]), (track_in_np, track_out_no), ("in", "out"), (1, -1), ("red", "green")
        ):
            ax.plot(
                [point[dim_map[proj]["x"]], point[dim_map[proj]["x"]] + track[dim_map[proj]["x"]] * 2 * y_span * pm],  # type: ignore
                [point[dim_map[proj]["y"]], point[dim_map[proj]["y"]] + track[dim_map[proj]["y"]] * 2 * y_span * pm],  # type: ignore
                alpha=0.4,
                color=color,
                linestyle="--",
                label=f"Fitted track {label}",
            )
        # Plot volume of interest (if provided)
        if voi is not None:
            self.plot_voi(voi=voi, ax=ax, dim_xy=dim_xy)  # type: ignore

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

    # Number of muons
    @property
    def n_mu(self) -> int:
        """The number of muons."""
        return self.dtheta.size()[0]

    # Hits
    @property
    def hits_in(self) -> Hits:
        r"""
        The hits of the incoming muons.
        """
        return self._hits_in

    @hits_in.setter
    def hits_in(self, value: Hits) -> None:
        self._hits_in = value

    @property
    def hits_out(self) -> Hits:
        r"""
        The hits of the outgoing muons.
        """
        return self._hits_out

    @hits_out.setter
    def hits_out(self, value: Hits) -> None:
        self._hits_out = value

    # Energy
    @property
    def E(self) -> Tensor:
        r"""
        Muons kinetic energy in MeV/c2.
        """
        return self._E

    @E.setter
    def E(self, value: Tensor) -> None:
        self._E = value

    @property
    def p(self) -> Tensor:
        r"""Muon momentum in MeV/c."""
        return MU_MASS * torch.sqrt(((self.E / MU_MASS) + 1) ** 2 - 1)

    # Scattering angle
    @property
    def dtheta(self) -> Tensor:
        r"""Muon scattering angle measured between the incoming and outgoing tracks"""
        if self._dtheta is None:
            self._dtheta = self.compute_dtheta_from_tracks(self.tracks_in, self.tracks_out)
        return self._dtheta

    @dtheta.setter
    def dtheta(self, value: Tensor) -> None:
        self._dtheta = value

    # Tracks
    @property
    def tracks_in(self) -> Tensor:
        r"""Incoming muon tracks, with size (mu, 3)"""
        return self._tracks_in

    @tracks_in.setter
    def tracks_in(self, value: Tensor) -> None:
        self._tracks_in = value

    @property
    def tracks_out(self) -> Tensor:
        r"""Outgoing muon tracks, with size (mu, 3)"""
        return self._tracks_out

    @tracks_out.setter
    def tracks_out(self, value: Tensor) -> None:
        self._tracks_out = value

    @property
    def tracks_eff_in(self) -> Tensor:
        r"""
        The incoming tracks efficiency: 1 -> tracks is detcted, 0 -> tracks not detected.
        """
        return self._tracks_eff_in

    @tracks_eff_in.setter
    def tracks_eff_in(self, value: Tensor) -> None:
        self._tracks_eff_in = value

    @property
    def tracks_eff_out(self) -> Tensor:
        r"""
        The outgoing tracks efficiency: 1 -> tracks is detcted, 0 -> tracks not detected.
        """
        return self._tracks_eff_out

    @tracks_eff_out.setter
    def tracks_eff_out(self, value: Tensor) -> None:
        self._tracks_eff_out = value

    # Muon efficiency
    @property
    def muon_eff(self) -> Tensor:
        """The muon efficiencies."""
        if self._muon_eff is None:
            self._muon_eff = self.get_muon_eff(self.tracks_eff_in, self.tracks_eff_out)
        return self._muon_eff

    @property
    def tracking_eff(self) -> float:
        return self._tracking_eff

    # Points
    @property
    def points_in(self) -> Tensor:
        r"""Points on the incoming muon tracks, with size (mu, 3)"""
        return self._points_in

    @points_in.setter
    def points_in(self, value: Tensor) -> None:
        self._points_in = value

    @property
    def points_out(self) -> Tensor:
        r"""Points on the outgoing muon tracks, with size (mu, 3)"""
        return self._points_out

    @points_out.setter
    def points_out(self, value: Tensor) -> None:
        self._points_out = value

    @property
    def theta_in(self) -> Tensor:
        r"""
        Zenith angle of the incoming tracks.
        """
        if self._theta_in is None:
            self._theta_in = Tracking.get_theta_from_tracks(self.tracks_in)
        return self._theta_in

    @property
    def theta_out(self) -> Tensor:
        r"""
        Zenith angle of the outgoing tracks.
        """
        if self._theta_out is None:
            self._theta_out = Tracking.get_theta_from_tracks(self.tracks_out)
        return self._theta_out

    @property
    def theta_xy_in(self) -> Tensor:
        r"""
        Projected zenith angles of the incoming tracks.
        """
        if self._theta_xy_in is None:
            self._theta_xy_in = Tracking.get_theta_xy_from_tracks(self.tracks_in)
        return self._theta_xy_in

    @property
    def theta_xy_out(self) -> Tensor:
        r"""
        Projected zenith angles of the outgoing tracks.
        """
        if self._theta_xy_out is None:
            self._theta_xy_out = Tracking.get_theta_xy_from_tracks(self.tracks_out)
        return self._theta_xy_out

    # Resolutions
    @property
    def angular_res_in(self) -> float:
        r"""
        Angular resolution of the incoming tracks.
        """
        return self._angular_res_in

    @angular_res_in.setter
    def angular_res_in(self, value: float) -> None:
        self._angular_res_in = value

    @property
    def angular_res_out(self) -> float:
        r"""
        Angular resolution of the outgoing tracks.
        """
        return self._angular_res_out

    @angular_res_out.setter
    def angular_res_out(self, value: float) -> None:
        self._angular_res_out = value

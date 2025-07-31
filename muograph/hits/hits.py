from pathlib import Path
import pandas as pd
import torch
from torch import Tensor
from typing import Optional, Tuple, Dict
import matplotlib
import matplotlib.pyplot as plt

from muograph.plotting.plotting import get_n_bins_xy_from_xy_span
from muograph.plotting.params import (
    font,
    d_unit,
    n_bins_2D,
    hist_figsize,
    labelsize,
    configure_plot_theme,
    cmap,
)
from muograph.utils.datatype import dtype_hit, dtype_E
from muograph.utils.device import DEVICE

allowed_d_units = ["m", "cm", "mm", "dm"]

r"""
Provides class for handling muon hits and simulating detector response.
"""


class Hits:
    r"""
    A class to handle and process muon hit data from a CSV file.
    """

    # Muon hits
    _gen_hits = None  # (3, n_plane, mu)
    _reco_hits = None  # (3, n_plane, mu)

    # Muon energy
    _E = None  # (mu)

    # Hits efficiency
    _hits_eff: Optional[Tensor] = None

    # Units
    _unit_coef: Dict[str, float] = {
        "mm": 1.0,
        "cm": 10.0,
        "dm": 100.0,
        "m": 1000.0,
    }

    def __init__(
        self,
        csv_filename: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        plane_labels: Optional[Tuple[int, ...]] = None,
        spatial_res: Optional[Tuple[float, float, float]] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        efficiency: float = 1.0,
        input_unit: str = "mm",
        n_mu_max: Optional[int] = None,
    ) -> None:
        r"""
        Initializes a Hits object to represent particle hit data from a detector, with input from either
        a CSV file path or an existing DataFrame.

        Args:
            csv_filename (Optional[str]): The file path to the CSV containing hit and energy data.
                Either `csv_filename` or `df` must be provided, but not both.
            df (Optional[pd.DataFrame]): A DataFrame containing hit and energy data. Use this instead of
                loading data from a CSV file.
            plane_labels (Optional[Tuple[int, ...]]): Specifies the plane labels to include from the data,
                as a tuple of integers. Only hits from these planes will be loaded if provided.
            spatial_res (Optional[Tuple[float, float, float]]): The spatial resolution of detector panels
                along the x, y, and z axes, in units specified by `input_unit`. Assumes uniform resolution
                across all panels if provided.
            energy_range (Optional[Tuple[float, float]]): A tuple specifying the minimum and maximum energy
                range for hits to be included. Only hits within this range will be processed if provided.
            efficiency (float): The efficiency factor of the detector panels, applied uniformly across all panels.
                Defaults to 1.0, representing full efficiency.
            input_unit (str): The unit of measurement for the input data (e.g., "mm", "cm"). Data will be rescaled to
                millimeters if another unit is specified. Defaults to "mm".
        """

        # Detector panel parameters
        self.spatial_res = (
            torch.tensor(spatial_res, dtype=dtype_hit, device=DEVICE) if spatial_res is not None else torch.zeros(3, dtype=dtype_hit, device=DEVICE)
        )

        self.efficiency = efficiency  # in %
        if (efficiency > 1.0) | (efficiency < 0.0):
            raise ValueError("Efficency must be positive and < 1.")

        # Energy range
        self.energy_range = energy_range

        # Load or create hits DataFrame
        if csv_filename is None and df is None:
            raise ValueError("Provide either csv_filename or df, not both.")

        if csv_filename is not None:
            self.input_unit = input_unit
            if input_unit not in allowed_d_units:
                raise ValueError("Input unit must be mm, cm, dm or m")

            self._df = self.get_data_frame_from_csv(csv_filename, n_mu_max)

        elif df is not None:
            self.input_unit = input_unit
            if input_unit not in allowed_d_units:
                raise ValueError("Input unit must be mm, cm, dm or m")

            self._df = df

        else:
            raise ValueError("Either csv_filename or df must be provided.")

        # Panels label
        self.plane_labels = plane_labels if plane_labels is not None else self.get_panels_labels_from_df(self._df)

        # Filter events with E out of energy_range
        if self.energy_range is not None:
            energy_mask = (self.E > self.energy_range[0]) & (self.E < self.energy_range[-1])
            self._filter_events(energy_mask)

        # Detector efficiency
        if (self.efficiency < 0.0) | (self.efficiency > 1.0):
            raise ValueError("Panels efficiency must be in [0., 1.].")

    def __repr__(self) -> str:
        description = f"Collection of hits from {self.n_mu:,d} muons " f"on {self.n_panels} detector panels"

        if self.spatial_res.sum() > 0:
            res = ", ".join(f"{value:.2f}" for value in self.spatial_res.detach().cpu().numpy())
            description += f",\n with spatial resolution [{res}] mm along x, y, z"

        if self.efficiency < 1.0:
            description += f", with panel efficiency of {self.efficiency * 100:.1f}%"

        description += "."

        return description

    @staticmethod
    def get_data_frame_from_csv(csv_filename: str, n_mu_max: Optional[int] = None) -> pd.DataFrame:
        r"""
        Reads a CSV file into a DataFrame.

        Args:
            csv_filename (Path): The path to the CSV file containing
            hit and energy data.

        Returns:
            pd.DataFrame: The DataFrame containing the data from the CSV file.
        """
        if not Path(csv_filename).exists():
            raise FileNotFoundError(f"The file {csv_filename} does not exist.")

        return pd.read_csv(csv_filename) if n_mu_max is None else pd.read_csv(csv_filename, nrows=n_mu_max)

    @staticmethod
    def get_hits_from_df(df: pd.DataFrame, plane_labels: Optional[Tuple[int, ...]] = None) -> Tensor:
        r"""
        Extracts hits data from a DataFrame and returns it as a Tensor.

        IMPORTANT:
            The DataFrame must have the following columns:
            "X0, Y0, Z0, ..., Xi, Yi, Zi", where Xi is the muon hit x position on plane i.

        Args:
            df (pd.DataFrame): DataFrame containing the hit data.

        Returns:
            hits (Tensor): Hits, with size (3, n_plane, n_mu)
        """

        # Normalize column mapping to lowercase
        col_map = {col.lower(): col for col in df.columns}

        # Extract plane count and validate columns
        n_plane = len(plane_labels)  # type: ignore
        hits = torch.zeros((3, n_plane, len(df)), dtype=dtype_hit, device=DEVICE)

        for i, plane in enumerate(plane_labels):  # type: ignore
            x_key = f"x{plane}"
            y_key = f"y{plane}"
            z_key = f"z{plane}"

            try:
                x_col = col_map[x_key]
                y_col = col_map[y_key]
                z_col = col_map[z_key]
            except KeyError as e:
                raise KeyError(f"Missing columns for plane {plane}: {x_key.upper()}, {y_key.upper()}, {z_key.upper()}") from e

            hits[0, i, :] = torch.tensor(df[x_col].values, dtype=dtype_hit, device=DEVICE)
            hits[1, i, :] = torch.tensor(df[y_col].values, dtype=dtype_hit, device=DEVICE)
            hits[2, i, :] = torch.tensor(df[z_col].values, dtype=dtype_hit, device=DEVICE)

        return hits

    @staticmethod
    def get_energy_from_df(df: pd.DataFrame) -> Tensor:
        r"""
        Extracts energy data from a DataFrame and returns it as a Tensor.

        IMPORTANT:
            The DataFrame must have the following column:
            "E"

        Args:
            df (pd.DataFrame): DataFrame where the hits and muons energy are saved.

        Returns:
            E (Tensor): Muons energy, with size (n_mu)
        """
        if "E" not in df:
            raise KeyError("Column 'E' not found in the DataFrame. Muon energy set to 0.")
        return torch.tensor(df["E"].values, dtype=dtype_E, device=DEVICE)

    @staticmethod
    def get_panels_labels_from_df(df: pd.DataFrame) -> Tuple[int, ...]:
        r"""
        Get the labels of ALL detector panels from the csv file.

        IMPORTANT:
            The DataFrame must have the following column 'X':

        Args:
            df (pd.DataFrame): DataFrame where the hits and muons energy are saved.

        Returns:
            plane_labels (Tuple[int, ...]): The labels of the detector panels.
        """

        planes = [col for col in df.columns if col.startswith("X")]
        plane_labels = tuple([int(s[1:]) for s in planes])

        return plane_labels

    @staticmethod
    def get_reco_hits_from_gen_hits(gen_hits: Tensor, spatial_res: Tensor) -> Tensor:
        r"""
        Smear the gen_hits position using a Normal distribution centered at 0,
        and with standard deviation equal to the spatial resolution along a given dimension

        Args:
            gen_hits (Tensor): The generated level hits, with size (3, n_plane, mu).
            spatial_res (Tensor): The spatial resolution along x,y,z with size (3).

        Returns:
            reco_hits (Tensor): The reconstructed hits, with size (3, n_plane, mu)
        """
        reco_hits = torch.ones_like(gen_hits, dtype=dtype_hit, device=DEVICE) * gen_hits

        for i in range(spatial_res.size()[0]):
            if spatial_res[i] != 0.0:
                reco_hits[i] += torch.normal(
                    mean=0.0,
                    std=torch.ones_like(reco_hits[i]) * spatial_res[i],
                )

        return reco_hits

    @staticmethod
    def get_muon_wise_eff(efficiency: float, gen_hits: Tensor) -> Tensor:
        r"""
        Returns a muon-wise efficiency based on the panels' efficiency.
        The muon-wise efficiency is either 0 (muon not detected) or 1 (muon detected).

        IMPORTANT: Currently, all panels are assumed to have the same efficency.
        Can be improved by setting the effieicy as a tensor with size n_panels instead of a float.

        Args:
            efficiency (float): The panels' efficency. Must be between 0 and 1.
            gen_hits (Tensor): The generated_hits.

        Returns:
            muon_wise_eff (Tensor): The muon-wise efficiency.
        """

        # Probability for a muon to leave hit on a detector panel
        p = torch.rand(gen_hits.size()[1:], device=DEVICE, dtype=dtype_hit)

        muon_wise_eff = torch.where(p < efficiency, 1, 0)

        return muon_wise_eff

    def _filter_events(self, mask: Tensor) -> None:
        r"""
        Remove muons specified as False in `mask`.

        Args:
            mask: (N,) Boolean tensor. Muons with False elements will be removed.
        """

        self.reco_hits = self.reco_hits[:, :, mask]
        self.gen_hits = self.gen_hits[:, :, mask]
        self.E = self.E[mask]

    def plot(
        self,
        plane_label: int = 0,
        reco_hits: bool = True,
        n_bins: int = n_bins_2D,
        cmap: str = cmap,
        filename: Optional[str] = None,
    ) -> None:
        """Plot the XY coordinates of the muon hits on a given detector plane, as a 2D histogram.


        Args:
            plane_label (int, optional): The label of the panel to plot. Defaults to 0.
            reco_hits (bool, optional): Plot the reconstructed hits `rec_hits` if True, else the generated hits `gen_hits`. Defaults to True.
            n_bins (int, optional): The number of bins of the 2D histogram. Defaults to n_bins_2D.
            filename (Optional[str], optional): Path to a filename where to save the figure. Defaults to None.
        """
        # Set default font
        matplotlib.rc("font", **font)

        configure_plot_theme(font)  # type: ignore

        # Create figure
        fig, ax = plt.subplots(figsize=hist_figsize)

        # Get true hits or real hits
        hits = self.reco_hits if reco_hits is True else self.gen_hits

        # The span of the detector in x and y
        dx = (hits[0, plane_label].max() - hits[0, plane_label].min()).item()
        dy = (hits[1, plane_label].max() - hits[1, plane_label].min()).item()

        # Get the number of bins as function of the xy ratio
        bins_x, bins_y, pixel_size = get_n_bins_xy_from_xy_span(dx=dx, dy=dy, n_bins=n_bins)

        # Plot hits as 2D histogram
        h = ax.hist2d(
            hits[0, plane_label].detach().cpu().numpy(),
            hits[1, plane_label].detach().cpu().numpy(),
            bins=(bins_x, bins_y),
            cmap=cmap,
        )

        ax.set_aspect("equal")

        # Set axis labels
        ax.set_xlabel(f"x [{d_unit}]", fontweight="bold")
        ax.set_ylabel(f"y [{d_unit}]", fontweight="bold")
        ax.tick_params(axis="both", labelsize=labelsize)

        # Set figure title
        fig.suptitle(
            f"Muon hits on plane {plane_label} \nat z = {hits[2,plane_label,:].mean(dim=-1):.0f} [{d_unit}]",
            fontweight="bold",
            y=1,
        )

        # Add colorbar
        cbar_ax = fig.add_axes(rect=(0.85, 0.15, 0.05, 0.7))
        fig.colorbar(h[3], cax=cbar_ax, label=f"# hits / {pixel_size**2:.0f} {d_unit}$^2$")

        # Save plot
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        plt.show()

    @property
    def n_mu(self) -> int:
        r"""
        The number of muon events.
        """
        return self.gen_hits.size()[-1]

    @property
    def n_panels(self) -> int:
        return len(self.plane_labels)

    @property
    def E(self) -> Tensor:
        r"""
        Muon's energy as a Tensor. If is not provided in the input csv/DataFrame,
        it is automatically set to zero.
        """
        if self._E is None:
            try:
                self._E = self.get_energy_from_df(self._df)
            except Exception as e:
                print(f"An error occurred: {e}")
                self._E = torch.zeros(self.reco_hits.size()[-1])  # Setting _E to zero
        return self._E

    @E.setter
    def E(self, value: Tensor) -> None:
        self._E = value

    @property
    def gen_hits(self) -> Tensor:
        r"""
        Hits data as a Tensor with size (3, n_plane, mu).
        """
        if self._gen_hits is None:
            self._gen_hits = self.get_hits_from_df(self._df, self.plane_labels) * self._unit_coef[self.input_unit]
        return self._gen_hits

    @gen_hits.setter
    def gen_hits(self, value: Tensor) -> None:
        self._gen_hits = value

    @property
    def reco_hits(self) -> Tensor:
        r"""
        Reconstructed hits data as Tensor with size (3, n_plane, mu).
        """
        if self.spatial_res is None:
            return self.gen_hits
        elif self._reco_hits is None:
            self._reco_hits = self.get_reco_hits_from_gen_hits(gen_hits=self.gen_hits, spatial_res=self.spatial_res)
        return self._reco_hits

    @reco_hits.setter
    def reco_hits(self, value: Tensor) -> None:
        self._reco_hits = value

    @property
    def hits_eff(self) -> Tensor:
        if self._hits_eff is None:
            self._hits_eff = self.get_muon_wise_eff(self.efficiency, self.gen_hits)
        return self._hits_eff


def filter_nans(hits_in: Hits, hits_out: Hits) -> None:
    """
    Filters out events containing NaN values in the input and output `Hits` objects.

    This function checks for NaN values across all events in the `gen_hits` attribute
    of both input `hits_in` and output `hits_out` objects. Events with NaN values
    in either object are excluded by applying a combined mask.

    Args:
        hits_in (Hits): An instance of the `Hits` class representing input events.
                        Must have a `gen_hits` attribute containing event data as
                        a tensor.
        hits_out (Hits): An instance of the `Hits` class representing output events.
                         Must have a `gen_hits` attribute containing event data as
                         a tensor.

    Side Effects:
        Both `hits_in` and `hits_out` are modified in-place by filtering out events
        that contain NaN values in their `gen_hits`.

    Example Usage:
        >>> filter_nans(input_hits, output_hits)
    """

    mask_nan_in = ~torch.any(torch.isnan(hits_in.gen_hits), dim=(0, 1))
    mask_nan_out = ~torch.any(torch.isnan(hits_out.gen_hits), dim=(0, 1))

    mask_nan = mask_nan_in & mask_nan_out

    hits_in._filter_events(mask_nan)
    hits_out._filter_events(mask_nan)

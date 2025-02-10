import matplotlib.axis
import torch
from torch import Tensor
from typing import Tuple, Optional, Union, Dict, List
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from muograph.volume.volume import Volume
from muograph.plotting.params import font, d_unit, scale, cmap, fontsize, hist_figsize, n_bins, labelsize, titlesize, configure_plot_theme, colors


class VoxelPlotting:
    def __init__(self, voi: Volume) -> None:
        self.voi = voi

    @staticmethod
    def get_n_rows(nplots: int, ncols: int) -> Tuple[int, int]:
        """
        Compute the number of rows and empty subplots of the figure, given the total
        number of plots and the desired number of columns.

        Args:
            - nplots (`int`) the total number of plots.
            - ncols (`int`) the number of columns.

        Returns:
            - nrows (`int`) the number of rows.
            - extra (`int`) the number of subplots left empty when nplots%ncols!=0.
        """
        # Compute the number of full rows and extra subplots
        nrows, extra_plots = divmod(nplots, ncols)

        # If there are extra plots, add an extra row and compute the empty subplots
        extra = 0 if extra_plots == 0 else ncols - extra_plots
        if extra_plots > 0:
            nrows += 1

        return nrows, extra

    @staticmethod
    def get_fig_size(
        voi: Volume,
        nrows: int,
        ncols: int = 3,
        dims: Tuple[int, int] = (0, 1),
        scale: float = scale,
    ) -> Tuple[float, float]:
        """
        Compute matplotlib.pyplot.figure figsize, given the xy subplot ratio,
        the number of columns and rows, and the total number of subplots.

        Args:
            voi (Volume): An instance of the volume class.
            nrows (int): The number of rows.
            ncols (int): The number of columns.
            dims (Tuple[int, int]): The Volume dimensions to consider, e.g. dims=(0,1) for XY plane,
                                    dims=(0,2) for XZ plane, dims=(1,2) for YZ plane.
            scale (float): Scale factor of the figure, e.g., figsize with scale=2 is twice as big as scale=1.

        Returns:
            Tuple[float, float]: The figure size (width, height).
        """

        dx, dy = (
            voi.dxyz[dims[0]].detach().cpu().numpy(),
            voi.dxyz[dims[1]].detach().cpu().numpy(),
        )

        # Compute xy_ratio as a list: larger dimension normalized to 1, and the other scaled accordingly
        xy_ratio = [min(dx / dy, 1.0), min(dy / dx, 1.0)]

        # Precompute the adjustment for subplot spacing
        adj_xy_ratio = [r + 0.25 for r in xy_ratio]

        return scale * ncols * adj_xy_ratio[0], scale * nrows * adj_xy_ratio[1]

    @staticmethod
    def get_2D_slice_from_3D(dim: int, xyz_voxel_preds: Tensor, voi_slice: Union[int, Tuple[int, int]]) -> Tensor:
        r"""
        Get a 2D from a 3D array.

        e.g Given a 3D array with size V = (Vx, Vy, Vz), the i-th 2D slice along
        the z dimension is V[:, :, i]. Setting voi_slice to (i, j) will return
        the mean of the slices with indices ranging from i to j included.


        Args:
            - dim (`int`) The dimension to slice along (2 = z, 1 = y, 0 = x).
            - xyz_voxel_preds (`Tensor`) The input 3D array.
            - voi_slice (`Union[int, Tuple[int, int]]`) The range of voxel indices corresponding
            to the desired slice. Setting voi_slice to `i`, will return the i-th slice along the
            desired dimension. Setting `voi_slice` to `(i, j)` will return the mean of all slices
            with indices between `i` and `j` included.
        Returns:
            - (`Tensor`) The 2D slice.
        """

        if xyz_voxel_preds.dtype not in [torch.float16, torch.float32, torch.float64]:
            xyz_voxel_preds = torch.ones_like(xyz_voxel_preds, dtype=torch.float32) * xyz_voxel_preds

        if isinstance(voi_slice, int):  # type: ignore
            if dim == 2:
                preds = xyz_voxel_preds[:, :, voi_slice]
            elif dim == 1:
                preds = xyz_voxel_preds[:, voi_slice, :]
            elif dim == 0:
                preds = xyz_voxel_preds[voi_slice, :, :]

        elif isinstance(voi_slice, tuple) and len(voi_slice) == 2:
            start, end = voi_slice
            if dim == 2:
                preds = xyz_voxel_preds[:, :, start : end + 1].mean(dim=2)
            elif dim == 1:
                preds = xyz_voxel_preds[:, start : end + 1, :].mean(dim=1)
            elif dim == 0:
                preds = xyz_voxel_preds[start : end + 1, :, :].mean(dim=0)

        return preds

    @staticmethod
    def get_voi_slice(dim: int, voi: Volume, voi_slice: Union[int, Tuple[int, int]]) -> Tuple[float, float]:
        """
        Get the range of positions in the slice along the desired dimension.

        e.g  Setting dim to 2 and voi_slice to (i, j) will return the z position
        of voxels with indices i and j along the z dimension:
        voi.voxel_edges[0, 0, i, 0, dim] and voi.voxel_edges[0, 0, j, 1, dim].

        Args:
            - dim (`int`) The dimension to slice along (2 = z, 1 = y, 0 = x).
            - voi (`Volume`) Instance of the volume class, a voxelized volume with Nx, Ny, Nz
            voxels along the x, y, z axis.
            - voi_slice (`Union[int, Tuple[int, int]]`) The range of voxel indices corresponding
            to the desired slice.
        Returns:
            Tuple[float, float]: the range of positions in the slice along the desired dimension.
        """

        if isinstance(voi_slice, int):
            if dim == 2:
                return (
                    voi.voxel_edges[0, 0, voi_slice, 0, dim].detach().cpu().item(),
                    voi.voxel_edges[0, 0, voi_slice, 1, dim].detach().cpu().item(),
                )

            elif dim == 1:
                return (
                    voi.voxel_edges[0, voi_slice, 0, 0, dim].detach().cpu().item(),
                    voi.voxel_edges[0, voi_slice, 0, 1, dim].detach().cpu().item(),
                )

            elif dim == 0:
                return (
                    voi.voxel_edges[voi_slice, 0, 0, 0, dim].detach().cpu().item(),
                    voi.voxel_edges[voi_slice, 0, 0, 1, dim].detach().cpu().item(),
                )

        elif isinstance(voi_slice, tuple) and len(voi_slice) == 2:
            if dim == 2:
                return (
                    voi.voxel_edges[0, 0, voi_slice[0], 0, 2].detach().cpu().item(),
                    voi.voxel_edges[0, 0, voi_slice[-1], 1, 2].detach().cpu().item(),
                )

            elif dim == 1:
                return (
                    voi.voxel_edges[0, voi_slice[0], 0, 0, 1].detach().cpu().item(),
                    voi.voxel_edges[0, voi_slice[-1], 0, 1, 1].detach().cpu().item(),
                )

            elif dim == 0:
                return (
                    voi.voxel_edges[voi_slice[0], 0, 0, 0, 0].detach().cpu().item(),
                    voi.voxel_edges[voi_slice[-1], 0, 0, 1, 0].detach().cpu().item(),
                )
        return (-0.666, -0.666)

    @staticmethod
    def plot_pred_slice(
        voi: Volume,
        xyz_voxel_preds: Tensor,
        xyz_voxel_pred_uncs: Optional[Tensor] = None,
        voi_slice: Optional[Tuple[int, int]] = None,
        dim: int = 2,
        figname: Optional[str] = None,
        reverse: bool = False,
        pred_label: str = "Scattering density",
        fig_suptitle: Optional[str] = None,
        pred_unit: str = "[a.u]",
        scale: float = 7.0,
        cmap: str = cmap,
        proj1d: bool = True,
        reference_value: Optional[float] = None,
        v_min_max: Optional[Tuple[float, float]] = None,
    ) -> None:
        r"""
        Plot a 2D slice from 3D voxel-wise predictions with optional uncertainty estimates.

        This method generates a plot of voxel-wise predictions from a 3D volume along a specified
        dimension (`dim`). It allows for optional visualization of uncertainty in the predictions
        and provides the ability to average the predictions along x and y axes. Additional
        customization options include setting colormaps, axis labels, and saving the plot to a file.

        Args:
            - voi (`Volume`): An instance of the volume class.
            - xyz_voxel_preds (`Tensor`): Tensor containing voxel predictions for the volume. (shape: [n_vox_x, n_vox_y, n_vox_z]).
            - xyz_voxel_pred_uncs : (`Optional[Tensor]`): Optional 3D tensor of the same shape as `xyz_voxel_preds` containing prediction
            uncertainties, by default None.
            - voi_slice  (`Tuple[int, int]`):  Start and end indices for slicing along the specified dimension.
                Defaults to None, which uses the full range of the dimension.
            - dim (`int`): The dimension along which to slice the 3D voxel-wise predictions.
            Must be 0, 1, or 2 corresponding to x, y, or z axes respectively.
            - figname (`Optional[str]`): If provided, saves the plot to the given file name, by default None.
            - reverse  (`bool`) Whether to reverse the colormap, by default False.
            - fig_suptitle (`str`): Title for the figure, by default None.
            - pred_label (`Optional[str]`): Label for the prediction data, used in axis labels and title, by default 'Prediction'.
            - pred_unit (`Optional[str]`): Unit of the predictions to be displayed in the colorbar and axis labels, by default '[unit]'.
            - scale (`Optional[float]`): Scale factor for determining figure size, by default 1.0.
            - cmap (`Optional[str]`): Colormap to use for the voxel prediction plot, by default 'jet'.
            - proj1d (`Optional[bool]`): Whether to project the 3D predictions onto 1D plots, by default True.
            - reference_value (`Optional[float]`): Optional reference value to display as a horizontal/vertical line in the averaged
            predictions plot, by default None.
             - v_min_max ('Optional[Tuple]'): Sets the min and max values of the 2D histogram.
        """
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

        # Set default font
        matplotlib.rc("font", **font)

        # Define colormap
        cmap = cmap + "_r" if reverse else cmap

        if voi_slice is None:
            voi_slice = (0, voi.n_vox_xyz[dim] - 1)

        # Get 2D slice from 3D xyz voxel-wise predictions
        xy_data = VoxelPlotting.get_2D_slice_from_3D(
            dim=dim,
            xyz_voxel_preds=xyz_voxel_preds,
            voi_slice=voi_slice,
        )

        # Get the associated uncertainties if available
        if xyz_voxel_pred_uncs is not None:
            xy_data_uncs = VoxelPlotting.get_2D_slice_from_3D(
                dim=dim,
                xyz_voxel_preds=xyz_voxel_pred_uncs,
                voi_slice=voi_slice,
            )
        else:
            xy_data_uncs = torch.zeros_like(xy_data)

        # Get the range of the slice position along the reduced dimnsion
        voi_slice_coord = VoxelPlotting.get_voi_slice(dim=dim, voi=voi, voi_slice=voi_slice)

        # Get xy dimensions
        dim_mapping = {
            2: {
                "xy_dims": (0, 1),  # The xy dims (on the plot)
                "dim_label": "z",  # The dim along wich the 3D xyz voxel-wise predictions are sliced
                "x_label": "x",  # The label of the x axis (on the plot)
                "y_label": "y",  # The label of the y axis (on the plot)
                "extent": (
                    voi.xyz_min[0].detach().cpu().numpy(),
                    voi.xyz_max[0].detach().cpu().numpy(),
                    voi.xyz_min[1].detach().cpu().numpy(),
                    voi.xyz_max[1].detach().cpu().numpy(),
                ),  # The left, right, bottom, top extent of the 2D slice.
                "x_data": xy_data.mean(dim=1).detach().cpu().numpy(),  # The 2D slice predictions averaged along the x axis (on the plot)
                "y_data": xy_data.mean(dim=0).detach().cpu().numpy(),  # The 2D slice predictions averaged along the y axis (on the plot)
                "x_data_uncs": xy_data_uncs.mean(dim=1).detach().cpu().numpy(),  # Same for uncs
                "y_data_uncs": xy_data_uncs.mean(dim=0).detach().cpu().numpy(),  # Same for uncs
                "x_vox_pos": voi.voxel_centers[:, 0, 0, 0].detach().cpu().numpy(),  # Voxels position along the x axis (on the plot)
                "y_vox_pos": voi.voxel_centers[0, :, 0, 1].detach().cpu().numpy(),  # Voxels position along the y axis (on the plot)
                "plane": "XY",
            },
            1: {
                "xy_dims": (0, 2),
                "dim_label": "y",
                "x_label": "x",
                "y_label": "z",
                "extent": (
                    voi.xyz_min[0].detach().cpu().numpy(),
                    voi.xyz_max[0].detach().cpu().numpy(),
                    voi.xyz_min[2].detach().cpu().numpy(),
                    voi.xyz_max[2].detach().cpu().numpy(),
                ),
                "x_data": xy_data.mean(dim=1).detach().cpu().numpy(),
                "y_data": xy_data.mean(dim=0).detach().cpu().numpy(),
                "x_data_uncs": xy_data_uncs.mean(dim=1).detach().cpu().numpy(),
                "y_data_uncs": xy_data_uncs.mean(dim=0).detach().cpu().numpy(),
                "x_vox_pos": voi.voxel_centers[:, 0, 0, 0].detach().cpu().numpy(),
                "y_vox_pos": voi.voxel_centers[0, 0, :, 2].detach().cpu().numpy(),
                "plane": "XZ",
            },
            0: {
                "xy_dims": (1, 2),
                "dim_label": "x",
                "x_label": "y",
                "y_label": "z",
                "extent": (
                    voi.xyz_min[1].detach().cpu().numpy(),
                    voi.xyz_max[1].detach().cpu().numpy(),
                    voi.xyz_min[2].detach().cpu().numpy(),
                    voi.xyz_max[2].detach().cpu().numpy(),
                ),
                "x_data": xy_data.mean(dim=1).detach().cpu().numpy(),
                "y_data": xy_data.mean(dim=0).detach().cpu().numpy(),
                "x_data_uncs": xy_data_uncs.mean(dim=1).detach().cpu().numpy(),
                "y_data_uncs": xy_data_uncs.mean(dim=0).detach().cpu().numpy(),
                "x_vox_pos": voi.voxel_centers[0, :, 0, 1].detach().cpu().numpy(),
                "y_vox_pos": voi.voxel_centers[0, 0, :, 2].detach().cpu().numpy(),
                "plane": "YZ",
            },
        }

        # Compute the figure size based on the plot xy ratio
        figsize = VoxelPlotting.get_fig_size(voi=voi, nrows=1, ncols=1, dims=dim_mapping[dim]["xy_dims"], scale=scale)  # type: ignore

        # Compute figure size based on the main axis size
        figsize = VoxelPlotting.get_fig_size(voi=voi, nrows=1, ncols=1, dims=dim_mapping[dim]["xy_dims"], scale=scale)  # type: ignore

        # Create the main figure
        fig, ax = plt.subplots(figsize=figsize)

        # Set title
        if fig_suptitle is None:
            fig_suptitle = (
                f"Voxel predictions\nfor volume slice {dim_mapping[dim]['dim_label']} "
                + r"$\in$"
                + f"[{voi_slice_coord[0]:.0f}, {voi_slice_coord[-1]:.0f}] {d_unit}"
            )

        # Set 2D map contrast
        vmin = xy_data.ravel().min().detach().cpu().item() if v_min_max is None else v_min_max[0]
        vmax = xy_data.ravel().max().detach().cpu().item() if v_min_max is None else v_min_max[1]

        # Plot the 2D slice predictions
        im = ax.imshow(
            xy_data.T.detach().cpu().numpy(),
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            extent=dim_mapping[dim]["extent"],  # type: ignore
        )

        # Remove grid
        ax.grid(False)

        # Set axis labels
        ax.set_xlabel(f"Voxel ${dim_mapping[dim]['x_label']}$ location [mm]", fontweight="bold")
        ax.set_ylabel(f"Voxel ${dim_mapping[dim]['y_label']}$ location [mm]", fontweight="bold")
        ax.tick_params(axis="both", labelsize=labelsize)

        if proj1d:
            divider = make_axes_locatable(ax)
            ax_histx = divider.append_axes("top", 1.0, pad=0.1, sharex=ax)
            ax_histy = divider.append_axes("right", 1.0, pad=0.1, sharey=ax)

            # Remove axis labels for histograms
            ax_histx.tick_params(axis="x", labelbottom=False, labelsize=labelsize)
            ax_histy.tick_params(axis="y", labelleft=False, labelsize=labelsize)

            # Set ticks position to top
            ax_histy.xaxis.set_ticks_position("top")
            ax_histy.xaxis.set_label_position("top")

            # Set figure title
            ax_histx.set_title(
                fig_suptitle,
                fontweight="bold",
                fontsize=titlesize,
                y=1.05,
            )

            # Plot the predictions averaged along the x and y axis
            if xyz_voxel_pred_uncs is None:
                ax_histx.scatter(
                    dim_mapping[dim]["x_vox_pos"],
                    dim_mapping[dim]["x_data"],
                    marker=".",
                )
                ax_histy.scatter(
                    dim_mapping[dim]["y_data"],
                    dim_mapping[dim]["y_vox_pos"],
                    marker=".",
                )

            else:
                # Plot uncertainties if available
                ax_histx.errorbar(
                    x=dim_mapping[dim]["x_vox_pos"],
                    y=dim_mapping[dim]["x_data"],
                    xerr=0,
                    yerr=dim_mapping[dim]["x_data_uncs"],
                    marker=".",
                    alpha=0.6,
                )
                ax_histy.errorbar(
                    x=dim_mapping[dim]["y_data"],
                    y=dim_mapping[dim]["y_vox_pos"],
                    xerr=dim_mapping[dim]["y_data_uncs"],
                    yerr=0,
                    marker=".",
                    alpha=0.6,
                )

            # Set same range for x and y histograms
            if reference_value is not None:
                min_pred_xy = min(
                    np.min(dim_mapping[dim]["x_data"]),  # type: ignore
                    np.min(dim_mapping[dim]["y_data"]),  # type: ignore
                )
                max_pred_xy = max(
                    np.max(dim_mapping[dim]["x_data"]),  # type: ignore
                    np.min(dim_mapping[dim]["y_data"]),  # type: ignore
                    reference_value,
                )
                ax_histx.set_ylim(min_pred_xy * 0.98, max_pred_xy * 1.02)
                ax_histy.set_xlim(min_pred_xy * 0.98, max_pred_xy * 1.02)

                # Plot reference value
                ax_histx.axhline(y=reference_value, color="red", alpha=0.5)
                ax_histy.axvline(x=reference_value, color="red", alpha=0.5)

            # Display grid
            ax_histx.grid(visible=True, color="grey", linestyle="--", linewidth=0.5)
            ax_histy.grid(visible=True, color="grey", linestyle="--", linewidth=0.5)

            # Set axis labels
            ax_histx.set_ylabel(pred_label + " " + pred_unit, fontsize=fontsize, fontweight="bold")
            ax_histy.set_xlabel(pred_label + " " + pred_unit, fontsize=fontsize, fontweight="bold")

        else:
            # Set figure title
            fig.suptitle(
                fig_suptitle,
                fontweight="bold",
                fontsize=titlesize,
            )

        # Add colorbar
        cbar_ax = fig.add_axes(rect=(0.95, 0.1, 0.03, 0.6))  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)  # Attach colorbar to the custom axis
        cbar.set_label(pred_label + " " + pred_unit, fontweight="bold")  # Colorbar label

        # Save plot
        if figname is not None:
            plt.savefig(
                figname + "_" + dim_mapping[dim]["plane"] + "_view",  # type: ignore
                bbox_inches="tight",
            )
        plt.show()

    @staticmethod
    def plot_pred_by_slice(
        voi: Volume,
        xyz_voxel_preds: Tensor,
        voi_slice: Optional[Tuple[int, int]] = None,
        dim: int = 2,
        ncols: int = 4,
        nslice_per_plot: int = 1,
        figname: Optional[str] = None,
        reverse: bool = False,
        fig_suptitle: str = "Voxels predictions",
        pred_label: str = "Scattering density",
        pred_unit: str = "",
        scale: float = scale,
        cmap: str = cmap,
    ) -> None:
        """
        Plots voxel predictions by slicing through a specified dimension of the volume of interest.

        Args:
            - voi (Volume): The volume of interest containing voxel data and metadata.
            - xyz_voxel_preds (Tensor): Tensor containing voxel predictions for the volume. (shape: [n_vox_x, n_vox_y, n_vox_z]).
            - voi_slice (Optional[Tuple[int, int]]): Start and end indices for slicing along the specified dimension.
                Defaults to None, which uses the full range of the dimension.
            - dim (int): The dimension along which to slice the 3D voxel-wise predictions.
            Must be 0, 1, or 2 corresponding to x, y, or z axes respectively. Defaults to 2.
            - ncols (int): Number of columns for the subplot grid. Defaults to 4.
            - nslice_per_plot (int): Number of slices to aggregate per subplot. Defaults to 1.
            - figname (Optional[str]): Name of the file to save the generated plots. Defaults to None.
            - reverse (bool): Whether to reverse the colormap. Defaults to False.
            - fig_suptitle (str): Title for the figure. Defaults to "Voxels predictions".
            - pred_label (str): Label for the prediction values in the colorbar. Defaults to "Scattering density".
            - pred_unit (str): Unit of the prediction values. Defaults to an empty string.
            - scale (float): Scaling factor for the figure size. Defaults to a predefined `scale`.
            - cmap (str): Colormap to use for the plot. Defaults to a predefined `cmap`.

        Returns:
            - None: Displays the generated plots and optionally saves them to a file.
        """

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

        # Set default font
        matplotlib.rc("font", **font)

        # Define colormap
        cmap = cmap + "_r" if reverse else cmap

        # The range of volume slice to plot
        if voi_slice is None:
            voi_slice = (0, voi.n_vox_xyz[dim] - 1)

        # The number of volume slice to take into account
        nplots = int(voi_slice[1] - voi_slice[0]) + 1

        # If multiple slices per plot, adjust nplots accordingly
        if nslice_per_plot > 1:
            assert nplots % nslice_per_plot == 0, f"Make sure that the number of plots ({nplots}) can be divided by nslice_per_plot ({nslice_per_plot})"
            nplots //= nslice_per_plot

        # Compute the number of columns and blank plots in the figure
        nrows, extra = VoxelPlotting.get_n_rows(nplots=nplots, ncols=ncols)

        # Get xy dimensions
        if dim == 2:
            dims = (0, 1)
        elif dim == 1:
            dims = (0, 2)
        elif dim == 0:
            dims = (1, 2)

        # Compute the figure size
        figsize = VoxelPlotting.get_fig_size(voi=voi, dims=dims, ncols=ncols, nrows=nrows, scale=scale)

        # Get figure and axis
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharex=True, sharey=True)

        axs = axs.ravel()

        if nslice_per_plot == 1:
            vmin, vmax = torch.min(xyz_voxel_preds), torch.max(xyz_voxel_preds)
        else:
            # Define the dimension-specific slicing based on dim
            slice_fn = {
                2: lambda p, s: p[:, :, s : s + nslice_per_plot].sum(dim=2) / nslice_per_plot,
                1: lambda p, s: p[:, s : s + nslice_per_plot].sum(dim=1) / nslice_per_plot,
                0: lambda p, s: p[s : s + nslice_per_plot].sum(dim=0) / nslice_per_plot,
            }[dim]

            # Precompute the range for slicing
            sliced_preds = [
                slice_fn(xyz_voxel_preds, slice)
                for slice in range(
                    voi_slice[0],
                    voi_slice[0] + (nplots * nslice_per_plot),
                    nslice_per_plot,
                )
            ]

            # Calculate vmin and vmax
            vmin = float("inf")  # type: ignore
            vmax = float("-inf")  # type: ignore
            for slice_pred in sliced_preds:
                vmin = min(vmin, torch.min(slice_pred))  # type: ignore
                vmax = max(vmax, torch.max(slice_pred))  # type: ignore

        # Mapping for dimension-specific attributes
        dim_mapping = {
            2: {
                "slice_fn": lambda p, s: p[:, :, s : s + nslice_per_plot].sum(dim=2).detach().cpu().numpy() / nslice_per_plot,
                "z_min_fn": lambda v, s: v.voxel_edges[0, 0, s, 0, 2].detach().cpu().numpy(),
                "z_max_fn": lambda v, s: v.voxel_edges[0, 0, s + nslice_per_plot - 1, 1, 2].detach().cpu().numpy(),
                "left_right_extent": (
                    voi.voxel_edges[0, 0, 0, 0, 0],
                    voi.voxel_edges[-1, 0, 0, 1, 0],
                ),
                "bottom_top_extent": (
                    voi.voxel_edges[0, 0, 0, 0, 1],
                    voi.voxel_edges[0, -1, 0, 1, 1],
                ),
                "dim_label": "z",
                "x_label": f"x [{d_unit}]",
                "y_label": f"y [{d_unit}]",
                "plane": "XY",
            },
            1: {
                "slice_fn": lambda p, s: p[:, s : s + nslice_per_plot].sum(dim=1).detach().cpu().numpy() / nslice_per_plot,
                "z_min_fn": lambda v, s: v.voxel_edges[0, s, 0, 0, 1].detach().cpu().numpy(),
                "z_max_fn": lambda v, s: v.voxel_edges[0, s + nslice_per_plot - 1, 0, 1, 1].detach().cpu().numpy(),
                "left_right_extent": (
                    voi.voxel_edges[0, 0, 0, 0, 0],
                    voi.voxel_edges[-1, 0, 0, 1, 0],
                ),
                "bottom_top_extent": (
                    voi.voxel_edges[0, 0, 0, 0, 2],
                    voi.voxel_edges[0, 0, -1, 1, 2],
                ),
                "dim_label": "y",
                "x_label": f"x [{d_unit}]",
                "y_label": f"z [{d_unit}]",
                "plane": "XZ",
            },
            0: {
                "slice_fn": lambda p, s: p[s : s + nslice_per_plot].sum(dim=0).detach().cpu().numpy() / nslice_per_plot,
                "z_min_fn": lambda v, s: v.voxel_edges[s, 0, 0, 0, 0].detach().cpu().numpy(),
                "z_max_fn": lambda v, s: v.voxel_edges[s + nslice_per_plot - 1, 0, 0, 1, 0].detach().cpu().numpy(),
                "left_right_extent": (
                    voi.voxel_edges[0, 0, 0, 0, 1],
                    voi.voxel_edges[0, -1, 0, 1, 1],
                ),
                "bottom_top_extent": (
                    voi.voxel_edges[0, 0, 0, 0, 2],
                    voi.voxel_edges[0, 0, -1, 1, 2],
                ),
                "dim_label": "x",
                "x_label": f"y [{d_unit}]",
                "y_label": f"z [{d_unit}]",
                "plane": "YZ",
            },
        }

        # Extract the necessary functions and labels based on the `dim`
        slice_fn = dim_mapping[dim]["slice_fn"]  # type: ignore
        z_min_fn = dim_mapping[dim]["z_min_fn"]
        z_max_fn = dim_mapping[dim]["z_max_fn"]
        dim_label = dim_mapping[dim]["dim_label"]
        x_label = dim_mapping[dim]["x_label"]
        y_label = dim_mapping[dim]["y_label"]

        # Loop over the number of voxels along `dim` dimension
        for i, slice in enumerate(
            range(
                voi_slice[0],
                voi_slice[0] + (nplots * nslice_per_plot),
                nslice_per_plot,
            )
        ):
            preds_slice = slice_fn(xyz_voxel_preds, slice)
            z_min = z_min_fn(voi, slice)  # type: ignore
            z_max = z_max_fn(voi, slice)  # type: ignore

            im = axs[i].imshow(
                preds_slice.T,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                extent=dim_mapping[dim]["left_right_extent"] + dim_mapping[dim]["bottom_top_extent"],  # type: ignore
            )

            axs[i].set_title(
                r"{} $\in$ [{:.0f},{:.0f}] {}".format(dim_label, z_min, z_max, d_unit),
                fontsize=fontsize - 1,
            )
            axs[i].set_aspect("equal")
            axs[i].grid(False)

            # set axes x label
            if (i % ncols == 0) | (i == 0):
                axs[i].set_ylabel(y_label, fontweight="bold")
            # set axes y label
            if i >= nplots - ncols:
                axs[i].set_xlabel(x_label, fontweight="bold")

            axs[i].tick_params(axis="both", labelsize=labelsize)

        # remove empty subplots
        for i in range(1, extra + 1):
            axs[-i].remove()
            axs[-i] = None

        # Add color bar
        cbar_ax = fig.add_axes(rect=(1.01, 0.15, 0.05, 0.7))
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(pred_label + " " + pred_unit, fontweight="bold")  # Colorbar label

        plt.subplots_adjust(right=0.99)

        # Set title
        fig.suptitle(
            fig_suptitle + f"\nvoxel size = {voi.vox_width} {d_unit}",
            x=0.58,
            y=1,
            fontweight="bold",
            fontsize=titlesize,
            va="top",
        )

        # Save file
        if figname is not None:
            fig.savefig(
                figname + "_" + dim_mapping[dim]["plane"] + "_view_slice.png",  # type: ignore
                bbox_inches="tight",
            )

        plt.show()

    @staticmethod
    def plot_pred_1D(
        xyz_voxel_preds: Tensor,
        title: str = "Voxel-wise predictions",
        x_label: str = "Density predition [a.u]",
        figname: Optional[str] = None,
        log: bool = False,
        n_bins: int = n_bins,
    ) -> None:
        # Set default font
        matplotlib.rc("font", **font)

        # Create figure
        fig, ax = plt.subplots(figsize=hist_figsize)

        # Set figure title
        fig.suptitle(title, fontweight="bold", fontsize=titlesize)

        # Get data range
        range = (xyz_voxel_preds.min().item(), xyz_voxel_preds.max().item())

        # Plot flattened predictions
        ax.hist(
            xyz_voxel_preds.detach().cpu().numpy().ravel(),
            bins=n_bins,
            range=range,
            histtype="step",
        )

        # Plot predictions mean
        mean = xyz_voxel_preds.mean().detach().cpu().item()
        ax.axvline(x=mean, label=f"Mean = {mean:.3f}", color="red")

        # Highlight 1 sigma region
        std = xyz_voxel_preds.std().detach().cpu().numpy()
        mask_1sigma = (xyz_voxel_preds > mean - std) & (xyz_voxel_preds < mean + std)
        ax.hist(
            xyz_voxel_preds[mask_1sigma],
            bins=n_bins,
            alpha=0.3,
            range=range,
            color="green",
            label=r"$\sigma$" + f" = {std:.3f}",
            log=log,
        )

        # Grid
        ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5)

        # Axis labels
        ax.set_ylabel("Frequency", fontweight="bold")
        ax.set_xlabel(x_label, fontweight="bold")
        ax.tick_params(axis="both", labelsize=labelsize)

        ax.legend()
        plt.tight_layout()
        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_voxel_grid(
        dim: int,
        voi: Volume,
        ax: Optional[matplotlib.axes._axes.Axes] = None,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
    ) -> None:
        r"""
        Plot the voxel volume as a grid given the desired projection.
        """
        # Configure plot theme
        configure_plot_theme(font=font)  # type: ignore

        if ax is None:
            fig, ax = plt.subplots()

        # The voxels edges in x,y,z
        voi_x_edges = voi.voxel_edges[:, 0, 0, 0, 0].tolist()
        voi_x_edges.append(voi.voxel_edges[-1, 0, 0, 1, 0].item())

        voi_y_edges = voi.voxel_edges[0, :, 0, 0, 1].tolist()
        voi_y_edges.append(voi.voxel_edges[0, -1, 0, 1, 1].item())

        voi_z_edges = voi.voxel_edges[0, 0, :, 0, 2].tolist()
        voi_z_edges.append(voi.voxel_edges[0, 0, -1, 1, 2].item())

        mapping = {
            2: {  # XY plane
                "x_edges": voi_x_edges,
                "y_edges": voi_y_edges,
                "xy_min_max": (
                    voi.xyz_min[0],
                    voi.xyz_min[1],
                    voi.xyz_max[0],
                    voi.xyz_max[1],
                ),
                "x_label": "x",
                "y_label": "y",
                "x_lim": (
                    (voi.xyz_min[0] - 2 * voi.vox_width[0]).detach().cpu().item(),
                    (voi.xyz_max[0] + 2 * voi.vox_width[0]).detach().cpu().item(),
                ),
                "y_lim": (
                    (voi.xyz_min[1] - 2 * voi.vox_width[1]).detach().cpu().item(),
                    (voi.xyz_max[1] + 2 * voi.vox_width[1]).detach().cpu().item(),
                ),
                "extent_x": (voi.xyz_min[0].detach().cpu().item(), voi.xyz_max[0].detach().cpu().item()),
                "extent_y": (voi.xyz_min[1].detach().cpu().item(), voi.xyz_max[1].detach().cpu().item()),
            },
            1: {  # XZ plane
                "x_edges": voi_x_edges,
                "y_edges": voi_z_edges,
                "xy_min_max": (
                    voi.xyz_min[0],
                    voi.xyz_min[2],
                    voi.xyz_max[0],
                    voi.xyz_max[2],
                ),
                "x_label": "x",
                "y_label": "z",
                "x_lim": (
                    (voi.xyz_min[0] - 2 * voi.vox_width[0]).detach().cpu().item(),
                    (voi.xyz_max[0] + 2 * voi.vox_width[0]).detach().cpu().item(),
                ),
                "y_lim": (
                    (voi.xyz_min[2] - 2 * voi.vox_width[2]).detach().cpu().item(),
                    (voi.xyz_max[2] + 2 * voi.vox_width[2]).detach().cpu().item(),
                ),
                "extent_x": (voi.xyz_min[0].detach().cpu().item(), voi.xyz_max[0].detach().cpu().item()),
                "extent_y": (voi.xyz_min[2].detach().cpu().item(), voi.xyz_max[2].detach().cpu().item()),
            },
            0: {  # YZ plane
                "x_edges": voi_y_edges,
                "y_edges": voi_z_edges,
                "xy_min_max": (
                    voi.xyz_min[1],
                    voi.xyz_min[2],
                    voi.xyz_max[1],
                    voi.xyz_max[2],
                ),
                "x_label": "y",
                "y_label": "z",
                "x_lim": (
                    (voi.xyz_min[1] - 2 * voi.vox_width[1]).detach().cpu().item(),
                    (voi.xyz_max[1] + 2 * voi.vox_width[1]).detach().cpu().item(),
                ),
                "y_lim": (
                    (voi.xyz_min[2] - 2 * voi.vox_width[2]).detach().cpu().item(),
                    (voi.xyz_max[2] + 2 * voi.vox_width[2]).detach().cpu().item(),
                ),
                "extent_x": (voi.xyz_min[1].detach().cpu().item(), voi.xyz_max[1].detach().cpu().item()),
                "extent_y": (voi.xyz_min[2].detach().cpu().item(), voi.xyz_max[2].detach().cpu().item()),
            },
        }

        # Set axis limits
        x_lim = mapping[dim]["x_lim"] if x_lim is None else x_lim  # type: ignore
        y_lim = mapping[dim]["y_lim"] if y_lim is None else y_lim  # type: ignore

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal")

        # Set axis labels
        ax.set_xlabel(mapping[dim]["x_label"] + f" [{d_unit}]", fontweight="bold")  # type: ignore
        ax.set_ylabel(mapping[dim]["y_label"] + f" [{d_unit}]", fontweight="bold")  # type: ignore

        def get_norm_extend(extent: Tuple[float, float], min_max: Tuple[float, float]) -> Tuple[float, float]:
            e_min = (extent[0] - min_max[0]) / (min_max[1] - min_max[0])
            e_max = (extent[1] - min_max[0]) / (min_max[1] - min_max[0])
            return e_min, e_max

        extent_x = get_norm_extend(mapping[dim]["extent_x"], min_max=x_lim)  # type: ignore
        extent_y = get_norm_extend(mapping[dim]["extent_y"], min_max=y_lim)  # type: ignore

        # Plot voxel grid
        for x in mapping[dim]["x_edges"]:
            ax.axvline(
                x=x,
                alpha=0.5,
                ymin=extent_y[0],
                ymax=extent_y[1],
            )

        for y in mapping[dim]["y_edges"]:
            ax.axhline(
                y=y,
                alpha=0.5,
                xmin=extent_x[0],
                xmax=extent_x[1],
            )

    @staticmethod
    def plot_3D_to_1D(
        data_3D: Union[Tensor, List[Tensor]],
        data_labels: Optional[List[str]] = None,
        voi: Optional[Volume] = None,
        dim: int = 2,
        ylabel: str = "value",
        figname: Optional[str] = None,
        title: Optional[str] = None,
        plot_mean: bool = False,
    ) -> None:
        """
        Plots a 1D projection of 3D tensor data along a specified dimension.

        Args:
            data_3D (Tensor): The 3D tensor containing the data to plot.
            voi (Optional[Volume]): Volume of interest, defining voxel centers. If None, a default is created.
            dim (int): The dimension to project onto (0, 1, or 2).
            ylabel (str): Label for the Y-axis.

        Raises:
            ValueError: If `dim` is not in {0, 1, 2}.
        """

        # Set theme
        configure_plot_theme(font)

        fig, ax = plt.subplots(figsize=hist_figsize)

        if voi is None:
            nx, ny, nz = data_3D.size() if isinstance(data_3D, Tensor) else data_3D[0].size()
            voi = Volume(position=(0.0, 0.0, 0.0), dimension=(nx, ny, nz), voxel_width=(1, 1, 1))

        if dim not in {0, 1, 2}:
            raise ValueError(f"Invalid dimension {dim}. Must be one of 0, 1, or 2.")

        def plot_data(ax: matplotlib.axes._axes.Axes, voi: Volume, data: Tensor, color: str = "blue", data_label: Optional[str] = None) -> None:
            dim_mapping: Dict[int, Dict[str, Union[str, np.ndarray]]] = {
                0: {
                    "x_pos": voi.voxel_centers[:, 0, 0, 0].detach().cpu().numpy(),
                    "data": data.mean(dim=2).mean(dim=1).detach().cpu().numpy(),
                },
                1: {
                    "x_pos": voi.voxel_centers[0, :, 0, 1].detach().cpu().numpy(),
                    "data": data.mean(dim=2).mean(dim=0).detach().cpu().numpy(),
                },
                2: {
                    "x_pos": voi.voxel_centers[0, 0, :, 2].detach().cpu().numpy(),
                    "data": data.mean(dim=0).mean(dim=0).detach().cpu().numpy(),
                },
            }

            mean = np.mean(dim_mapping[dim]["data"])  # type: ignore
            label = data_label if data_label is not None else f"Mean = {mean:.3f}"

            # Plot 1D data
            ax.scatter(
                dim_mapping[dim]["x_pos"],
                dim_mapping[dim]["data"],
                marker="+",
                s=50,
                label=label,
                alpha=0.8,
                color=color,
            )

        if isinstance(data_3D, list):
            for i, data in enumerate(data_3D):
                data_label = data_labels[i] if data_labels is not None else None
                plot_data(ax=ax, voi=voi, data=data, color=colors[i], data_label=data_label)

        elif isinstance(data_3D, Tensor):
            plot_data(ax=ax, voi=voi, data=data_3D, color=colors[0])

        xlabel: Dict[int, str] = {0: "x [" + d_unit + "]", 1: "y [" + d_unit + "]", 2: "z [" + d_unit + "]"}

        # Axis label
        ax.set_xlabel(xlabel[dim], fontweight="bold")  # type: ignore
        ax.set_ylabel(ylabel, fontweight="bold")

        # Title
        if title is not None:
            fig.suptitle(title, fontweight="bold")

        # Legend
        if plot_mean:
            ax.legend(bbox_to_anchor=(1.0, 0.7))
        else:
            ax.legend()

        # Save figure
        if figname is not None:
            plt.savefig(figname + "_" + xlabel[dim][0], bbox_inches="tight")
        plt.show()

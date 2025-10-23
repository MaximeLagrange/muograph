import matplotlib
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple, Optional, Union
import numpy as np

from muograph.volume.volume import Volume
from muograph.plotting.params import d_unit, n_bins_2D, n_bins
from muograph.plotting.style import set_plot_style


def plot_n_poca_per_voxel(n_poca_per_vox: Tensor, dim: int, plot_name: Optional[str] = None) -> None:
    r"""
    Plot the average number of poca point per voxel along a certain direction.

    Args:
        - n_poca_per_vox (Tensor): The number of poca point per voxel,
        with size (nx, ny, nz) with ni the number of voxels along the i direction.
        - dim (int): Integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]

    fig, ax = plt.subplots()
    fig.suptitle(
        "Average {}{} distribution of POCA points".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )
    im = ax.imshow(
        n_poca_per_vox.sum(dim=dim).T / n_poca_per_vox.size()[dim],
        interpolation="nearest",
        origin="lower",
    )

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")

    cbar_ax = fig.add_axes(rect=(1.0, 0.15, 0.05, 0.7))
    fig.colorbar(im, cax=cbar_ax, label=" Average # poca per voxel")
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def plot_poca_points_per_pixel(
    poca_points: Tensor,
    dim: int,
    plot_name: Optional[str] = None,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> None:
    r"""
    Plot the number of poca point per pixel (screen pixel).

    Args:
        - poca_points (Tensor): Tensor poca points location.
        - dim (int): Integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """

    from matplotlib.colors import LinearSegmentedColormap

    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list(
        "white_viridis",
        [
            (0, "#ffffff"),
            (1e-20, "#440053"),
            (0.2, "#404388"),
            (0.4, "#2a788e"),
            (0.6, "#21a784"),
            (0.8, "#78d151"),
            (1, "#fde624"),
        ],
        N=256,
    )

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]
    dimx = [1, 0, 0]
    dimy = [2, 2, 1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

    density = ax.scatter_density(  # type: ignore
        poca_points[:, dimx[dim]].numpy(),
        poca_points[:, dimy[dim]].numpy(),
        cmap=white_viridis,
    )

    fig.suptitle(
        "Average {}{} distribution of POCA points".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")
    ax.set_aspect("equal")
    cbar_ax = fig.add_axes(rect=(1.0, 0.15, 0.05, 0.7))
    fig.colorbar(density, cax=cbar_ax, label="# points per pixel")
    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def plot_poca_points(
    poca_points: Tensor,
    dim: int,
    plot_name: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 8),
) -> None:
    r"""
    Plot the poca points location.

    Args:
        - poca_points (Tensor): poca points location.
        - dim (int): Integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """
    import mpl_scatter_density  # noqa: F401

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]
    dimx = [1, 0, 0]
    dimy = [2, 2, 1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

    ax.scatter(poca_points[:, dimx[dim]].numpy(), poca_points[:, dimy[dim]].numpy(), alpha=0.01)

    fig.suptitle(
        "Average {}{} distribution of POCA points".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")
    ax.set_aspect("equal")

    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def plot_poca_points_hist2d(poca_points: Tensor, voi: Volume, dim: int = 2) -> None:
    r"""
    Plot the poca points location as 2D histogram.

    Args:
        - poca_points: Tensor poca points location.
        - dim: integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """
    poca_points_np = poca_points.detach().cpu().numpy()

    dims = [_ for _ in [0, 1, 2] if _ != dim]
    labels = ["x", "y", "z"]
    fig, ax = plt.subplots()
    fig.suptitle("POCA points location", fontsize=15, fontweight="bold")

    im = ax.hist2d(
        poca_points_np[:, dims[0]],
        poca_points_np[:, dims[1]],
        bins=(voi.n_vox_xyz[dims[0]], voi.n_vox_xyz[dims[1]]),
    )

    ax.set_xlabel(labels[dims[0]] + " [{}]".format(d_unit))
    ax.set_ylabel(labels[dims[1]] + " [{}]".format(d_unit))

    cbar_ax = fig.add_axes(rect=(1.01, 0.15, 0.05, 0.7))
    fig.colorbar(im[3], cax=cbar_ax, label="# POCA points per pixel")
    ax.set_aspect("equal")

    plt.show()


def plot_voxel_pred(pred: Tensor, dim: int, plot_name: Optional[str] = None) -> None:
    r"""
    Plot the voxel-wise predictions averaged along a certain direction.

    Args:
        - pred (Tensor): The scattering denisyt predictions,
        with size (nx, ny, nz) with ni the number of voxels along the i direction.
        - dim (int): Integer defining the projection: dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]

    fig, ax = plt.subplots()
    fig.suptitle(
        "Average {}{} density predictions".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )
    im = ax.imshow(pred.sum(dim=dim).T / pred.size()[dim], interpolation="nearest", origin="lower")

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")

    cbar_ax = fig.add_axes(rect=(1.0, 0.15, 0.05, 0.7))
    fig.colorbar(im, cax=cbar_ax, label=" Average voxel density [a.u]")
    if plot_name is not None:
        plt.savefig(plot_name)
        plt.tight_layout()
    plt.show()


def plot_2d_vector(
    ax: matplotlib.axes._axes.Axes,
    vector: np.ndarray,
    origin: np.ndarray,
    scale: float = 1.0,
) -> None:
    """
    Plots a 2D vector starting from the origin.

    Args:
        - ax (matplotlib.axes._axes.Axes) The axes to plot on.
        - vector (np.ndarray): A 2D vector [x, y].
        - origin (np.ndarray): The starting point of the vector.
    """
    ax.quiver(
        *origin.tolist(),
        vector[0],
        vector[1],
        angles="xy",
        scale_units="xy",
        alpha=0.5,
        scale=scale,
        color="green",
    )

    ax.set_aspect("equal")


def get_n_bins_xy_from_xy_span(dx: float, dy: float, n_bins: int = n_bins_2D) -> Tuple[int, int, int]:
    r"""
    Calculate the number of bins along the x and y dimensions based on the
    aspect ratio of the plot and the total desired number of bins.

    The function adjusts the bin counts proportionally according to the span
    of the plot along the x (`dx`) and y (`dy`) dimensions, ensuring that the
    total number of bins along both dimensions is close to the specified
    `n_bins`, while maintaining the plot's aspect ratio.

    Args:
        dx (float): The span of the plot along the x-axis.
        dy (float): The span of the plot along the y-axis.
        n_bins (int): The total desired number of bins (along the longer dimension).

    Returns:
        Tuple[int, int, float]: The number of bins along the x-axis (`nx`) and y-axis (`ny`), and the pixel size.
                         The larger span will be allocated `n_bins`, while the other
                         dimension will have a proportional number of bins based on
                         the aspect ratio of `dx` and `dy`.
    """

    if dx == dy:
        # If spans are equal, assign equal bins to both x and y.
        nx, ny = n_bins, n_bins
        pixel_size = round(dx / n_bins)
    if dx > dy:
        # If x-span is larger, allocate n_bins to x and scale y accordingly.
        nx = n_bins
        ny = round(n_bins * (dy / dx))
        pixel_size = round(dx / n_bins)
    else:
        # If y-span is larger, allocate n_bins to y and scale x accordingly.
        ny = n_bins
        nx = round(n_bins * (dx / dy))
        pixel_size = round(dy / n_bins)

    return nx, ny, pixel_size


def plot_hist(
    data_1D: Union[Tensor, np.ndarray],
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    logy: bool = False,
    logx: bool = False,
    figname: Optional[str] = None,
    title: Optional[str] = None,
    n_bins: int = n_bins,
    range: Optional[Tuple] = None,
) -> None:
    set_plot_style()

    fig, ax = plt.subplots()

    ax.hist(
        data_1D.detach().cpu().numpy() if isinstance(data_1D, Tensor) else data_1D,
        bins=n_bins,
        range=range,
        log=logy,
        color="blue",
        alpha=0.4,
        edgecolor="black",
    )

    xlabel = "x" if xlabel is None else xlabel
    ylabel = "frequency [a.u]" if ylabel is None else ylabel

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")

    if title is not None:
        ax.set_title(title)

    if figname is not None:
        plt.savefig(figname, bbox_inches="tight")

    plt.show()

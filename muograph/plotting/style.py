import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional, List, Tuple

colors: List[str] = list(mpl.colors.TABLEAU_COLORS.values())  # type: ignore
subplots_4_figsize: Tuple[float, float] = (9, 6)
subplots_2_figsize: Tuple[float, float] = (8, 3.5)
n_bins: int = 50
alpha: float = 0.5


def set_plot_style() -> None:
    """
    Configure Matplotlib + Seaborn + LaTeX for consistent publication-quality plots.
    """
    # Use Seaborn base style (muted, minimal grid)
    sns.set_theme(context="paper", style="whitegrid", palette="deep")

    mpl.rcParams.update(mpl.rcParamsDefault)

    # Matplotlib rcParams overrides
    mpl.rcParams.update(
        {
            # === Font ===
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "font.size": 14,
            # === Axes ===
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.grid": True,
            # === Ticks ===
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            # === Grid ===
            "grid.color": "gray",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.5,
            # === Legend ===
            "legend.fontsize": 12,
            "legend.frameon": False,
            # === Lines ===
            "lines.linewidth": 2,
            "lines.markersize": 6,
            # === Figure ===
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.figsize": [6, 4],
            "figure.titlesize": "large",
            # === Colors ===
            "axes.prop_cycle": plt.cycler(color=sns.color_palette("deep")),
            # === LaTeX ===
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )


def add_colorbar_right(
    ax: mpl.axes.Axes,
    mappable: mpl.cm.ScalarMappable,
    label: Optional[str] = None,
    size: str = "5%",
    pad: float = 0.2,
) -> mpl.colorbar.Colorbar:
    """
    Adds a colorbar to the right of an axis with consistent spacing.

    Parameters:
    - ax: Matplotlib Axes to attach the colorbar to
    - mappable: The object returned by `imshow`, `scatter`, etc.
    - label: Optional label for the colorbar
    - size: Width of the colorbar (as % of the axes width)
    - pad: Padding between axes and colorbar (in figure fraction)
    - aspect: Aspect ratio of the colorbar (height / width)
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = ax.figure.colorbar(mappable, cax=cax)
    if label:
        cbar.set_label(label)
    return cbar

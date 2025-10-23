from typing import Tuple, Dict, Union, List
import seaborn as sns
import matplotlib

# Number of bins for histograms
n_bins: int = 50

# Histogram transparency
alpha: float = 0.6
alpha_sns: float = 0.4

# Distance unit
d_unit: str = "mm"

# Matplotlib default font
# fontsize: int = 18
# labelsize: int = 19
# titlesize: int = 20

fontsize: int = 12
labelsize: int = 12
titlesize: int = 22


fontweigh: str = "normal"
font: Dict[str, Union[str, int]] = {"weight": "normal", "size": fontsize, "family": "sans-serif"}

# Matplotlib colormaps
cmap: str = "jet"

# Plot colors
colors: List[str] = list(matplotlib.colors.TABLEAU_COLORS.values())  # type: ignore

# Histogram XY ratio
xy_golden_ratio: float = 1.5
hist_scale = 6
hist_figsize: Tuple[float, float] = (xy_golden_ratio * hist_scale, 1 * hist_scale)
hist2_figsize: Tuple[float, float] = (2 * xy_golden_ratio * hist_scale, 1 * hist_scale)

# tracking plot
tracking_figsize: Tuple[float, float] = (10, 5)

# Scale of matplotlib figures with subplots
scale: int = 3

# 2D histogram
n_bins_2D = 50


# Set seabron theme
def configure_plot_theme(font: Dict[str, Union[str, int]]) -> None:
    """Configure the plotting theme and default font."""
    sns.set_theme(
        style="darkgrid",
        rc={
            "font.family": font["family"],
            "font.size": font["size"],
            "axes.labelsize": font["size"],
            "axes.titlesize": font["size"],
            "xtick.labelsize": font["size"],
            "ytick.labelsize": font["size"],
        },
    )
    matplotlib.rc("font", **font)

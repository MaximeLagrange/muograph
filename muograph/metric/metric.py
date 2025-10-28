import torch
import math
import numpy as np
from copy import deepcopy
from torch import Tensor
import pandas as pd
from typing import Optional, Dict, Tuple, Any
from muograph.plotting.plotting import plot_hist
from muograph.plotting.voxel import VoxelPlotting
from muograph.reconstruction.poca import POCA

r"""
Provides classes for computing regression error metrics.
"""


class SSIM:
    _mssim: Optional[float] = None
    _ssim: Optional[float] = None

    r"""
    Compute the mean structural similarity index between (MSSIM) two 3D images.
    Instead of computing the SSMI over the whole 3D image, it is computed within a window
    with size (window_size x window_size x window size), which moves voxel-by-voxel
    over the entire image. At each step, the local statistics and SSIM
    index are calculated within the local window. The mean SSIM (MSSIM) is the average
    of all the local SSIMs.

    References
    ----------
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       :DOI:`10.1109/TIP.2003.819861`

    .. [2] Avanaki, A. N. (2009). Exact global histogram specification
       optimized for structural similarity. Optical Review, 16, 613-621.
       :arxiv:`0901.0065`
       :DOI:`10.1007/s10043-009-0119-z`

    """

    def __init__(
        self,
        image_x: Tensor,
        image_y: Tensor,
        data_range: float = 1,
        k1: float = 0.01,
        k2: float = 0.03,
        alpha: float = 1,
        beta: float = 1,
        gamma: float = 1,
        win_size: int = 11,
        sigma: Optional[float] = 1.5,
    ) -> None:
        """
        Initializes the SSIM class.
        Args:
            ground_truth (torch.Tensor): The ground truth tensor of shape (Nx, Ny, Nz).
            predictions (torch.Tensor): The predicted tensor of the same shape as ground_truth.
        """
        assert image_x.shape == image_y.shape, "The two images must have the same shape."
        assert win_size % 2 == 1, "The window size must be odd."
        assert image_x.dim() == 3, "The images must be 3D."

        self.image_x = image_x
        self.image_y = image_y

        self.data_range = data_range
        self.k1 = k1
        self.k2 = k2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.win_size = win_size

    @staticmethod
    def get_unbiased_std(image_3d: Tensor) -> float:
        return (1 / (image_3d.numel() - 1) * (image_3d - image_3d.mean()) ** 2).sum().sqrt().item()

    @staticmethod
    def get_mean_intensity(image_3d: Tensor) -> float:
        return image_3d.mean(dim=0).mean(dim=0).mean(dim=0).detach().cpu().item()

    @staticmethod
    def get_contrast_comparision(
        image_x: Tensor,
        image_y: Tensor,
        data_range: float,
        k2: float,
    ) -> float:
        r"""
        Compute the contrast comparison between two images.

        Args:
            image_x (Tensor): The image to compare to the reference.
            image_y (Tensor): The reference image.
            l: The dynamic range of the pixel values (255 for 8-bit grayscale images).
            k2: The constant for the contrast comparison, a small constant << 1.
        """

        sigma_x, sigma_y = SSIM.get_unbiased_std(image_x), SSIM.get_unbiased_std(image_y)
        c2 = (data_range * k2) ** 2

        return (c2 + (2 * sigma_x * sigma_y)) / (sigma_x**2 + sigma_y**2 + c2)

    @staticmethod
    def get_luminance_comparision(
        image_x: Tensor,
        image_y: Tensor,
        data_range: float,
        k1: float,
    ) -> float:
        r"""
        Compute the luminance comparison between two images.

        Args:
            image_x (Tensor): The image to compare to the reference.
            image_y (Tensor): The reference image.
            weights (Tensor): The weights for the luminance comparison.
            l: The dynamic range of the pixel values (255 for 8-bit grayscale images).
            k1: The constant for the luminance comparison, a small constant << 1.
        """

        mu_x, mu_y = SSIM.get_mean_intensity(image_x), SSIM.get_mean_intensity(image_y)
        c1 = (data_range * k1) ** 2

        return (c1 + (2 * mu_x * mu_y)) / (mu_x**2 + mu_y**2 + c1)

    @staticmethod
    def get_structure_comparision(image_x: Tensor, image_y: Tensor, data_range: float, k3: float) -> float:
        r"""
        Compute the structure comparison between two images.

        Args:
            image_x (Tensor): The image to compare to the reference.
            image_y (Tensor): The reference image.
            l: The dynamic range of the pixel values (255 for 8-bit grayscale images).
            k1: The constant for the luminance comparison, a small constant << 1.
        """

        sigma_xy = (image_x.numel()) ** (-1) * ((image_x - SSIM.get_mean_intensity(image_x)) * (image_y - SSIM.get_mean_intensity(image_y))).sum().sum().sum()
        sigma_x = SSIM.get_unbiased_std(image_x)
        sigma_y = SSIM.get_unbiased_std(image_y)
        c3 = (data_range * k3) ** 2
        return ((sigma_xy + c3) / ((sigma_x * sigma_y) + c3)).detach().cpu().item()

    @staticmethod
    def gaussian_kernel_2d(window_size: int, sigma: float) -> Tensor:
        r"""
        Create a 3D Gaussian kernel with the specified window size and standard deviation.

        Args:
            window_size (int): The size of the window.
            sigma (float): The standard deviation of the Gaussian kernel.
            ndim (int): The number of dimensions of the kernel (either 1, 2, or 3).
        """

        if window_size % 2 == 0:
            print("The window size must be odd." "The window size is increased by 1.")
            window_size += 1

        # Create coordinate grid
        x = torch.arange(window_size) - window_size // 2
        y = torch.arange(window_size) - window_size // 2

        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Compute Gaussian function
        kernel = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))

        return kernel

    @staticmethod
    def get_ssim(
        image_x: Tensor,
        image_y: Tensor,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        weights: Optional[Tensor] = None,
    ) -> float:
        r"""
        Compute the structural similarity index between two images.

        Args:
            image_x (Tensor): The image to compare to the reference.
            image_y (Tensor): The reference image.
            data_range: The dynamic range of the pixel values (255 for 8-bit grayscale images).
            k1: The constant for the luminance comparison, a small constant << 1.
            k2: The constant for the contrast comparison, a small constant << 1.
            alpha: The exponent for the luminance comparison.
            beta: The exponent for the contrast comparison.
            gamma: The exponent for the structure comparison.
            weights: The weights to apply to the images before computing the SSIM.
        """

        if weights is not None:
            image_x = image_x * weights
            image_y = image_y * weights

        luminance = SSIM.get_luminance_comparision(image_x, image_y, data_range, k1)
        contrast = SSIM.get_contrast_comparision(image_x, image_y, data_range, k2)
        structure = SSIM.get_structure_comparision(image_x, image_y, data_range, k2 / 2)

        return (luminance) ** alpha * (contrast) ** beta * (structure) ** gamma

    @staticmethod
    def get_mssim(
        image_x: Tensor,
        image_y: Tensor,
        data_range: float = 1.0,
        k1: float = 0.01,
        k2: float = 0.03,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        window_size: int = 7,
        sigma: Optional[float] = None,
    ) -> float:
        r"""
        Compute the mean structural similarity index between (MSSIM) two 3D images.
        Instead of computing the SSMI over the whole 3D image, it is computed within a window
        with size (window_size x window_size x window size), which moves voxel-by-voxel
        over the entire image. At each step, the local statistics and SSIM
        index are calculated within the local window. The mean SSIM (MSSIM) is the average
        of all the local SSIMs.

        References
        ----------
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
        (2004). Image quality assessment: From error visibility to
        structural similarity. IEEE Transactions on Image Processing,
        13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        :DOI:`10.1109/TIP.2003.819861`

        .. [2] Avanaki, A. N. (2009). Exact global histogram specification
        optimized for structural similarity. Optical Review, 16, 613-621.
        :arxiv:`0901.0065`
        :DOI:`10.1007/s10043-009-0119-z`
        """

        image_x, image_y = image_x[1:-1, 1:-1, 1:-1], image_y[1:-1, 1:-1, 1:-1]

        d, h, w = image_x.shape
        pad = window_size // 2

        if sigma is not None:
            kernel = SSIM.gaussian_kernel_2d(window_size, sigma)
        else:
            kernel = torch.ones(window_size, window_size)

        full_mssim = []
        for k in range(0, w):
            layer_mssim = []
            for i in range(pad, d - pad):
                for j in range(pad, h - pad):
                    sub_image_x = image_x[i - pad : i + pad + 1, j - pad : j + pad + 1, k] * kernel
                    sub_image_y = image_y[i - pad : i + pad + 1, j - pad : j + pad + 1, k] * kernel

                    ssim_value = SSIM.get_ssim(
                        image_x=sub_image_x, image_y=sub_image_y, data_range=data_range, k1=k1, k2=k2, alpha=alpha, beta=beta, gamma=gamma
                    )

                    layer_mssim.append(ssim_value)
            full_mssim.append(torch.tensor(layer_mssim).mean().item())

        return torch.tensor(full_mssim).mean().detach().cpu().item()

    @property
    def mssim(self) -> float:
        if self._mssim is None:
            self._mssim = self.get_mssim(
                self.image_x, self.image_y, self.data_range, self.k1, self.k2, self.alpha, self.beta, self.gamma, window_size=self.win_size, sigma=self.sigma
            )
        return self._mssim


class RegressionErrorMetrics:
    r"""
    Class for computing regression error metrics from voxelized gronud truth and predictions.
    ground truth and predictions must have the same shape.
    It is advised to normalize the data before computing metrics.
    """

    def __init__(self, ground_truth: Tensor, predictions: Tensor, mask: Optional[Tensor] = None) -> None:
        """
        Initializes the RegressionErrorMetrics class.
        Args:
            ground_truth (torch.Tensor): The ground truth tensor of shape (Nx, Ny, Nz).
            predictions (torch.Tensor): The predicted tensor of the same shape as ground_truth.
        """
        assert (
            ground_truth.shape == predictions.shape
        ), f"Ground truth and predictions must have the same shape, but have {ground_truth.shape} and {predictions.shape}."

        self.ground_truth = ground_truth[mask]
        self.predictions = predictions[mask]

        self.errors = self.ground_truth - self.predictions

    def normalize(self, data: Tensor) -> Tensor:
        """
        Normalizes the data to have a range of [0, 1].
        Args:
            data (torch.Tensor): Input tensor to normalize.
        Returns:
            torch.Tensor: Normalized tensor with range [0, 1].
        """
        data_min = torch.min(data)
        data_max = torch.max(data)
        return (data - data_min) / (data_max - data_min + 1e-8)  # Adding epsilon to avoid division by zero

    def normalize_data(self) -> None:
        """
        Normalizes both ground truth and predictions to have a range of [0, 1].
        Updates self.ground_truth and self.predictions.
        """
        self.ground_truth = self.normalize(self.ground_truth)
        self.predictions = self.normalize(self.predictions)
        self.errors = self.ground_truth - self.predictions

    @property
    def mae(self) -> float:
        """Computes the Mean Absolute Error (MAE)."""
        mae = torch.mean(torch.abs(self.errors))
        return mae.detach().cpu().item()

    @property
    def mse(self) -> float:
        """Computes the Mean Squared Error (MSE)."""
        mse = torch.mean(self.errors**2)
        return mse.detach().cpu().item()

    @property
    def rmse(self) -> float:
        """Computes the Root Mean Squared Error (RMSE)."""
        mse = self.mse
        rmse = torch.sqrt(torch.tensor(mse))
        return rmse.detach().cpu().item()

    @property
    def r_squared(self) -> float:
        """Computes the R-squared (coefficient of determination) metric."""
        total_variance = torch.var(self.ground_truth)
        unexplained_variance = torch.var(self.errors)
        r2 = 1 - (unexplained_variance / total_variance)
        return r2.detach().cpu().item()

    def summary(self, normalize: bool = False) -> Dict[str, float]:
        """
        Generates a summary of all metrics.
        Args:
            normalize (bool): If True, normalizes the data before computing metrics.
        Returns:
            dict: A dictionary of error metrics.
        """
        if normalize:
            self.normalize_data()
        return {"MAE": self.mae, "MSE": self.mse, "RMSE": self.rmse, "R^2": self.r_squared}

    def plot_preds(self, logx: bool = False, logy: bool = False, figname: Optional[str] = None) -> None:
        plot_hist(data_1D=self.predictions.ravel(), logx=logx, logy=logy, xlabel="Predictions", figname=figname)

    def plot_gt(self, logx: bool = False, logy: bool = False, figname: Optional[str] = None) -> None:
        plot_hist(data_1D=self.ground_truth.ravel(), logx=logx, logy=logy, xlabel="Ground Truth", figname=figname)

    def plot_preds_gt_1D(self, dim: int = 0, title: Optional[str] = None) -> None:
        VoxelPlotting.plot_3D_to_1D([self.predictions, self.ground_truth], data_labels=["Predictions", "Ground Truth"], dim=dim, title=title)


class PocaErrorMetrics:
    """Class for comparing POCA objects and computing error metrics based on POCA points."""

    _distances: Optional[Tensor] = None  # Cached distances tensor (N,)
    _masks: Optional[Tuple[Tensor, Tensor]] = None  # Cached masks tuple (N,)

    def __init__(self, poca_ref: "POCA", poca: "POCA", output_dir: Optional[str] = None, label: Optional[str] = None):
        """
        Initializes the PocaErrorMetrics class.

        Args:
            poca_ref (POCA): Reference POCA instance.
            poca (POCA): POCA instance to compare against the reference.
            output_dir (Optional[str]): Directory to save outputs.
            label (Optional[str]): Label for saving outputs.
        """
        if not isinstance(poca_ref, POCA) or not isinstance(poca, POCA):
            raise TypeError("Both poca_ref and poca must be instances of the POCA class.")

        self.poca_ref = poca_ref
        self.poca = poca
        self.output_dir = output_dir
        self.label = label

        self.n_mu_ref = deepcopy(poca_ref.n_mu)
        self.n_mu = deepcopy(poca.n_mu)
        self.n_mu_lost = self.n_mu - self.n_mu_ref

        self._filter_events()

    @staticmethod
    def get_compatible_event_masks(poca1: "POCA", poca2: "POCA") -> Tuple[Tensor, Tensor]:
        """
        Finds the compatible events between two POCA objects.

        Args:
            poca1 (POCA): First POCA instance.
            poca2 (POCA): Second POCA instance.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of masks indicating compatible events.
        """
        cross_mask = poca1.full_mask & poca2.full_mask

        def compute_mask(poca: "POCA") -> Tensor:
            valid_indices = torch.where(poca.parallel_mask)[0][poca.mask_in_voi]
            cross_indices = torch.where(cross_mask)[0]
            return torch.isin(valid_indices, cross_indices)

        return compute_mask(poca1), compute_mask(poca2)

    @staticmethod
    def compute_distances(points_1: Tensor, points_2: Tensor) -> Tensor:
        """
        Computes distances between corresponding 3D points from two tensors.

        Args:
            points_1 (Tensor): Tensor of points (N, 3).
            points_2 (Tensor): Tensor of points (N, 3).

        Returns:
            Tensor: Tensor of distances (N,).
        """
        if points_1.shape != points_2.shape or points_1.shape[1] != 3:
            raise ValueError("Input tensors must have shape (N, 3) and be of the same size.")
        return torch.norm(points_1 - points_2, dim=1)

    def _filter_events(self) -> None:
        """Filters events in both POCA objects to retain only compatible events."""
        self.poca_ref._filter_pocas(self.masks[0])
        self.poca_ref.tracks._filter_muons(self.masks[0])
        self.poca._filter_pocas(self.masks[1])
        self.poca.tracks._filter_muons(self.masks[1])

    def plot_distance(self, mask: Optional[Tensor] = None, figname: Optional[str] = None, title: Optional[str] = None) -> None:
        """
        Plots the distribution of distances between the POCA points of two POCA objects.

        Args:
            mask (Optional[Tensor]): Mask to filter distances for plotting.
            figname (Optional[str]): Filename to save the plot.
            title (Optional[str]): Plot title.
        """
        mask = mask if mask is not None else torch.ones_like(self.distances, dtype=torch.bool)
        figname = figname or (self.output_dir and f"{self.output_dir}/distance_distribution")
        title = title or f"Distance between POCA points - {mask.sum().item():,d} events"

        plot_hist(data_1D=self.distances[mask], xlabel="Distance [mm]", figname=figname, title=title, logy=True)

    def save(self) -> None:
        """Saves the summary as a CSV file in the specified output directory."""
        filename = (self.output_dir or "") + (self.label or "poca_metric_summary")
        pd.DataFrame(self.summary).to_csv(filename)
        print(f"Poca metric summary saved at {filename}")

    @property
    def distances(self) -> Tensor:
        """Returns the distances between POCA points."""
        if self._distances is None:
            self._distances = self.compute_distances(self.poca_ref.poca_points, self.poca.poca_points)
        return self._distances

    @property
    def masks(self) -> Tuple[Tensor, Tensor]:
        """Returns masks of compatible events."""
        if self._masks is None:
            self._masks = self.get_compatible_event_masks(self.poca_ref, self.poca)
        return self._masks

    @property
    def summary(self) -> Dict[str, Any]:
        """Returns a summary of metrics for the POCA comparison."""

        # Convert distances and dtheta to NumPy arrays
        d_np = self.distances.detach().cpu().numpy()
        dtheta_np = self.poca_ref.tracks.dtheta.detach().cpu().numpy()

        # Precompute masks in NumPy
        mask_1deg_np = dtheta_np > 1 * math.pi / 180
        mask_3deg_np = dtheta_np > 3 * math.pi / 180

        # Helper function to compute metrics
        def compute_metrics(data: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
            if mask is not None:
                data = data[mask]
            return {
                "mean": data.mean(),
                "q25": np.quantile(data, q=0.25),
                "q50": np.quantile(data, q=0.5),
                "q75": np.quantile(data, q=0.75),
                "std": data.std(),
            }

        # Base metrics
        base_metrics = compute_metrics(d_np)
        metrics_1deg = compute_metrics(d_np, mask_1deg_np)
        metrics_3deg = compute_metrics(d_np, mask_3deg_np)

        # Combine results
        return {
            "angular_res": self.poca.tracks.angular_res_in,
            "d_mean": base_metrics["mean"],
            "d_q25": base_metrics["q25"],
            "d_q50": base_metrics["q50"],
            "d_q75": base_metrics["q75"],
            "d_mean_1deg": metrics_1deg["mean"],
            "d_q25_1deg": metrics_1deg["q25"],
            "d_q50_1deg": metrics_1deg["q50"],
            "d_q75_1deg": metrics_1deg["q75"],
            "d_mean_3deg": metrics_3deg["mean"],
            "d_q25_3deg": metrics_3deg["q25"],
            "d_q50_3deg": metrics_3deg["q50"],
            "d_q75_3deg": metrics_3deg["q75"],
            "d_std": base_metrics["std"],
            "d_std_1deg": metrics_1deg["std"],
            "d_std_3deg": metrics_3deg["std"],
            "n_mu_ref": self.n_mu_ref,
            "n_mu": self.n_mu,
            "n_mu_lost": self.n_mu_lost,
            "n_mu_shared": self.poca_ref.n_mu,
        }

from typing import Optional, Tuple, List, Dict, Union
import torch
from torch import Tensor
from copy import deepcopy
from functools import partial
import math
from pathlib import Path

from muograph.utils.device import DEVICE
from muograph.tracking.tracking import TrackingMST
from muograph.reconstruction.poca import POCA
from muograph.volume.volume import Volume
from muograph.reconstruction.voxel_inferer import AbsVoxelInferer

value_type = Union[float, partial, Tuple[float, float], bool, int]
bca_params_type = Dict[str, value_type]
P_MEAN = 4.0  # GeV

r"""
Provides class for computing voxelized scattering density predictions
based on the Binned clustered algorithm (A binned clustering algorithm
to detect high-Z material using cosmic muons, 2013 JINST 8 P10013,
(http://iopscience.iop.org/1748-0221/8/10/P10013)).
"""


class BCA(POCA, AbsVoxelInferer):
    _bca_params: bca_params_type = {
        "n_max_per_vox": 50,
        "n_min_per_vox": 3,
        "score_method": partial(torch.quantile, q=0.5),
        "metric_method": partial(torch.log),  # type: ignore
        "p_range": (0.0, 10000000),  # MeV
        "dtheta_range": (0.0, math.pi / 3),
        "use_p": False,
    }

    _vars_to_save = ["xyz_voxel_pred", "n_poca_per_vox"]

    def __init__(
        self,
        voi: Volume,
        tracking: TrackingMST,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        r"""
        Initializes the BCA object with an instance of the TrackingMST class.

        Args:
            - voi (Volume) Instance of the Volume class. The BCA algo. relying on voxelized
            volume, `voi` has to be provided.
            - tracking (Optional[TrackingMST]) Instance of the TrackingMST class.
            - output_dir (Optional[str]) Path to a directory where to save BCA `pred`
            and `hit_per_voxel` attributes in a hdf5 file.
        """
        AbsVoxelInferer.__init__(self, voi=voi, tracking=tracking)
        POCA.__init__(
            self,
            tracking=tracking,
            voi=voi,
            output_dir=output_dir,
            filename=filename,
        )

        self.bca_indices: Tensor = deepcopy(self.poca_indices)
        self.bca_poca_points: Tensor = deepcopy(self.poca_points)
        self.bca_tracks: TrackingMST = deepcopy(self.tracks)

    @staticmethod
    def compute_distance_2_points(points: Tensor) -> Tensor:
        r"""
        Compute the distances between each pair of 3D points.

        Args:
            - points (Tensor): A tensor of shape (n, 3) representing n 3D points.

        Returns:
            - distances (Tensor): A symmetric matrix of shape (n, n) with distances between points.
        """

        points = points.reshape((points.size()[0], 3, 1))
        points_transposed = points.permute(2, 1, 0)
        distances = torch.sqrt(torch.sum(torch.square(points - points_transposed), dim=1))
        return distances

    @staticmethod
    def compute_scattering_momentum_weight(dtheta: Tensor, p: Tensor) -> Tensor:
        r"""
        Compute weights based on the muon scattering angle and momentum (if availabe).

        Args:
            - dtheta (Tensor) the muons scattering angle with size (mu).
            - p (Tensor) the muons momentum with size (mu).

        Returns:
            - weights (Tensor) a symmetric matrix with size (mu, mu), if momentum is provided,
            computed as `(p * dtheta) * (p * dtheta).T`. If not, computed as `dtheta * dtheta.T`.
        """

        dtheta = dtheta.unsqueeze(1)
        p = p.unsqueeze(0)

        # Normalize the momentum
        if not (p == 1).all():
            p_norm = p / P_MEAN
        else:
            p_norm = p

        if p is not None:
            weights = dtheta * p_norm * (dtheta * p_norm).T
        else:
            weights = dtheta * dtheta.T

        return weights

    @staticmethod
    def compute_low_theta_events_voxel_wise_mask(
        n_max_per_voxel: int,
        bca_indices: Tensor,
        dtheta: Tensor,
        voi: Volume,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Only keep the n-th highest scattering event among poca points within a given voxel.
        Other events will be rejected.

        Args:
            - n_max_per_voxel (int) Number of highest scattering angle to keep
            among poca points located within a given voxel.
            - bca_indices (Tensor) Indices of voxels the poca points are located in.
            - dtheta (Tensor): Muon scattering angle.
            - voi (Volume) Instance of the voi class.

        Returns:
            - mask (Tensor) a mask specifying true if the muon scattering angle is ranked
            in the n-th highest scatterings in its voxel.
            - nhit (Tensor) nhit[i,j,k] is the number of poca points within voxel i,j,k,
            BEFORE removing events, with size (Nx,Ny,Nz,1) with Ni the number of voxels along a given dimension.
            - nhit (Tensor) nhit[i,j,k] is the number of poca points within voxel i,j,k,
            AFTER removing events,with size (Nx,Ny,Nz,1) with Ni the number of voxels along a given dimension.
        """

        # Initialize hit counters
        nhit = torch.zeros(voi.n_vox_xyz, dtype=torch.int64, device=DEVICE)
        nhit_cut = torch.zeros(voi.n_vox_xyz, dtype=torch.int64, device=DEVICE)

        flat_vox_indices = bca_indices[:, 0] * (voi.n_vox_xyz[1] * voi.n_vox_xyz[2]) + bca_indices[:, 1] * voi.n_vox_xyz[2] + bca_indices[:, 2]

        # Use unique to find all distinct voxels and counts
        unique_voxels, counts = torch.unique(flat_vox_indices, return_counts=True)
        unique_voxels = unique_voxels.type(torch.int64)

        # Scatter to get the number of hits per voxel
        nhit.view(-1).scatter_(0, unique_voxels, counts)

        # Mask to identify which events should be rejected
        rejected_events = torch.tensor([], dtype=torch.long, device=DEVICE)

        for voxel_idx in unique_voxels:
            # Get mask for the current voxel
            voxel_mask = flat_vox_indices == voxel_idx

            # Sort the scattering angles within the voxel
            sorted_dtheta, order = torch.sort(dtheta[voxel_mask], descending=True)

            # Get the number of hits in the current voxel
            num_hits = sorted_dtheta.size(0)

            # Update nhit_cut for this voxel
            if num_hits > n_max_per_voxel:
                nhit_cut.view(-1)[voxel_idx] = n_max_per_voxel

                # Get the indices of the events to reject
                to_reject = torch.nonzero(voxel_mask)[order[n_max_per_voxel:]].squeeze()

                # Ensure to_reject is 2D before concatenation
                if to_reject.dim() == 0:
                    to_reject = to_reject.unsqueeze(0)

                # Append rejected events
                rejected_events = torch.cat((rejected_events, to_reject))
            else:
                nhit_cut.view(-1)[voxel_idx] = num_hits

        # Sort rejected events and create mask
        rejected_events, _ = torch.sort(rejected_events)
        mask = torch.ones_like(dtheta, dtype=torch.bool, device=DEVICE)
        mask[rejected_events] = False

        return mask, nhit, nhit_cut

    def compute_vox_wise_metric(
        self,
        vox_id: Tensor,
        use_p: bool,
        bca_indices: Tensor,
        poca_points: Tensor,
        dtheta: Tensor,
        momentum: Tensor,
        metric_method: Optional[partial] = None,
    ) -> Tensor:
        r"""
        Computes a voxel-wise scattering density metric.
        CREDITS: A binned clustering algorithm to detect high-Z material using cosmic muons,
                 2013 JINST 8 P10013, (http://iopscience.iop.org/1748-0221/8/10/P10013)

        Args:
            - vox_id (Tensor) Voxel indices.
            - use_p (bool) If True, the muon momentum is used in the `scattering_weights`
            computation.
            - bca_indices (Tensor) Indinces of voxels the poca points are located in.
            - poca_points (Tensor) Coordinates of poca points.
            - dtheta (Tensor) The muons scattering angle.
            - momentum (Tensor) Muons momentum if available.
            - metric_method (Optional[partial]): Function to compute the voxel-wise metric.


        Returns:
            - full_metric (Tensor) The voxel-wise scattering density metric with size (n, n),
            where n is the number of poca points within the voxel with indices vox_id and satisfies the condition on p and dtheta.
            Zero elements are removed.
        """

        # Mask events outside the voxel
        poca_in_vox_mask = (bca_indices == vox_id).sum(dim=-1) == 3

        # POCA points within the voxel
        poca_in_vox = poca_points[poca_in_vox_mask]

        # Distance metric
        distance_metric = self.compute_distance_2_points(points=poca_in_vox)

        # Scattering metric
        if use_p is False:
            momentum = torch.ones_like(dtheta, device=DEVICE, dtype=torch.float32)
        scattering_weights = self.compute_scattering_momentum_weight(
            dtheta=dtheta[poca_in_vox_mask],
            p=momentum[poca_in_vox_mask],
        )

        # Keep only lower triangle element within symmetric matrix
        full_metric = torch.tril(
            torch.where(
                (distance_metric != 0) & (scattering_weights != 0),
                # distance_metric / scattering_weights,
                scattering_weights / distance_metric,
                0.0,
            )
        )
        if metric_method is not None:
            return metric_method(full_metric[full_metric != 0.0])
        else:
            return full_metric[full_metric != 0.0]

    def compute_voxels_distribution(
        self,
        use_p: bool,
        n_min_per_vox: int,
        voi: Volume,
        bca_indices: Tensor,
        poca_points: Tensor,
        momentum: Tensor,
        dtheta: Tensor,
        metric_method: partial,
    ) -> List:
        r"""
        Compute voxel-wise weight distribution, according to the `compute_vox_wise_metric()` method.

        Args:
            - use_p (bool): Whether to include momentum in the computation.
            - n_min_per_vox (int): Minimum number of POCA points required per voxel.
            - voi (Volume): Instance of the `Volume` class.
            - bca_indices (Tensor): Voxel indices of POCA points.
            - poca_points (Tensor): Coordinates of POCA points.
            - momentum (Tensor): Momentum of POCA points.
            - dtheta (Tensor): Scattering angles of POCA points.
            - metric_method (partial): Function to compute voxel-wise metrics.

        Returns:
             - score_list (List) List containing lists of scores for each voxel.
        """

        # Initialize score tensor with large negative values
        score_list = torch.zeros(voi.n_vox_xyz, dtype=torch.int16, device=DEVICE).tolist()

        # Compute voxel indices and POCA point counts
        flat_vox_indices = bca_indices[:, 0] * (voi.n_vox_xyz[1] * voi.n_vox_xyz[2]) + bca_indices[:, 1] * voi.n_vox_xyz[2] + bca_indices[:, 2]

        unique_voxels, counts = torch.unique(flat_vox_indices, return_counts=True)

        # Filter voxels with at least n_min_per_vox points
        valid_voxels = unique_voxels[counts >= n_min_per_vox]

        # Convert flat indices back to 3D coordinates
        voxel_coords = torch.stack(
            [
                valid_voxels // (voi.n_vox_xyz[1] * voi.n_vox_xyz[2]),
                (valid_voxels % (voi.n_vox_xyz[1] * voi.n_vox_xyz[2])) // voi.n_vox_xyz[2],
                valid_voxels % voi.n_vox_xyz[2],
            ],
            dim=-1,
        )

        # Compute metrics for valid voxels
        for coord in voxel_coords:
            i, j, k = coord.tolist()
            score_list[i][j][k] = self.compute_vox_wise_metric(
                metric_method=metric_method,
                vox_id=torch.tensor([i, j, k], device=bca_indices.device),
                use_p=use_p,
                bca_indices=bca_indices,
                poca_points=poca_points,
                dtheta=dtheta,
                momentum=momentum,
            )

        return score_list

    def compute_final_scores(self, score_list: List, score_method: partial) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Compute voxel-wise scores from a voxel-wise list of scores.

        Args:
             - score_list (List) A list containing a list of scores for each voxel.
             - score_method (partial) The function used to convert the list of scores into a float.

        Returns:
             - final_voxel_scores (Tensor) containing the final voxel score with size (Nx, Ny, Nz),
            where Ni is the number of voxels along a certain axis.
             - hit_per_voxel (Tensor) containing the number of POCA points within each voxel,
            with size (Nx, Ny, Nz).
        """

        Nx, Ny, Nz = self.voi.n_vox_xyz
        final_voxel_scores = torch.zeros((Nx, Ny, Nz), device=DEVICE, dtype=torch.float32)
        hit_per_voxel = torch.zeros((Nx, Ny, Nz), device=DEVICE, dtype=torch.int16)

        # Iterate over each voxel in the score_list
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    voxel_scores = score_list[i][j][k]

                    if isinstance(voxel_scores, Tensor):
                        # Compute the final score using the provided score_method
                        final_voxel_scores[i, j, k] = score_method(voxel_scores)
                        # Count the number of POCA points (hits) in this voxel
                        hit_per_voxel[i, j, k] = len(voxel_scores)

        # replace zero values by minimum value
        min_value = torch.min(final_voxel_scores[final_voxel_scores != 0.0])
        final_voxel_scores[final_voxel_scores == 0.0] = min_value

        return final_voxel_scores, hit_per_voxel

    def _filter_events(self, mask: Tensor) -> None:
        r"""
        Remove events specified as False in `mask`.

        Args:
            - mask (Tensor) events with False elements will be removed.
        """
        self.bca_indices = self.bca_indices[mask]
        self.bca_poca_points = self.bca_poca_points[mask]
        self.bca_tracks._filter_muons(mask=mask)

    def get_dir_name(self) -> Path:
        """Returns the name of the BCA algo given its parameters.

        Returns:
            Path: Path to the BCA directory.
        """
        return Path(str(self.output_dir) + "/" + self.bca_name + "/")

    def get_xyz_voxel_pred(self) -> Tensor:
        """
        Run the BCA algorithm, as implemented in:
        A binned clustering algorithm to detect high-Z material using cosmic muons,
        2013 JINST 8 P10013,
        (http://iopscience.iop.org/1748-0221/8/10/P10013).

        Compute voxel-wise scattering density predictions.

        Uses parameters stored in `_bca_params`. The algorithm calculates:
            - Scattering density predictions (`pred`).
            - Number of POCA points used for each voxel's prediction (`hit_per_voxel`).

        Returns:
            - pred (Tensor): Tensor of voxel-wise scattering density predictions.
        """

        # Copy relevant features before event selection
        self.bca_tracks = deepcopy(self.tracks)
        self.bca_indices = deepcopy(self.poca_indices)
        self.bca_poca_points = deepcopy(self.poca_points)

        # Keep only the n poca points with highest scattering angle within a voxel
        (
            self.mask,
            self.nhit,
            self.nhit_rejected,
        ) = self.compute_low_theta_events_voxel_wise_mask(
            n_max_per_voxel=int(self.bca_params["n_max_per_vox"]),  # type: ignore
            voi=self.voi,
            bca_indices=self.bca_indices,
            dtheta=self.bca_tracks.dtheta,
        )
        self._filter_events(self.mask)

        # momentum cut
        if self.bca_params["use_p"]:
            p_mask = (self.bca_tracks.E > self.bca_params["p_range"][0]) & (  # type: ignore
                self.bca_tracks.E < self.bca_params["p_range"][1]  # type: ignore
            )
        else:
            p_mask = torch.ones_like(self.bca_tracks.dtheta, dtype=torch.bool, device=DEVICE)

        # scattering angle cut
        dtheta_mask = (self.bca_tracks.dtheta > self.bca_params["dtheta_range"][0]) & (  # type: ignore
            self.bca_tracks.dtheta < self.bca_params["dtheta_range"][1]  # type: ignore
        )

        # apply dtheta, p cuts
        self._filter_events(mask=p_mask & dtheta_mask)

        # compute voxels distribution
        self.score_list = self.compute_voxels_distribution(
            metric_method=self.bca_params["metric_method"],  # type: ignore
            use_p=self.bca_params["use_p"],  # type: ignore
            n_min_per_vox=self.bca_params["n_min_per_vox"],  # type: ignore
            voi=self.voi,
            momentum=self.bca_tracks.E,
            bca_indices=self.bca_indices,
            poca_points=self.bca_poca_points,
            dtheta=self.bca_tracks.dtheta,
        )

        # compute fina scores
        pred, self._hit_per_voxel = self.compute_final_scores(score_list=self.score_list, score_method=self.bca_params["score_method"])  # type: ignore

        self._recompute_preds = False

        return pred

    def get_bca_name(
        self,
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

        method = "method_{}_".format(get_partial_name_args(self.bca_params["score_method"]))  # type: ignore
        metric = "metric_{}_".format(get_partial_name_args(self.bca_params["metric_method"]))  # type: ignore
        dtheta = "{:.1f}_{:.1f}_mrad_".format(self.bca_params["dtheta_range"][0] * 1000, self.bca_params["dtheta_range"][1] * 1000)  # type: ignore
        dp = "{:.0f}_{:.0f}_MeV_".format(self.bca_params["p_range"][0], self.bca_params["p_range"][1])  # type: ignore
        n_min_max = "n_min_max_{}_{}_".format(self.bca_params["n_min_per_vox"], self.bca_params["n_max_per_vox"])
        use_p = "use_p_{}".format(self.bca_params["use_p"])

        bca_name = method + metric + dtheta + dp + n_min_max + use_p
        bca_name = bca_name.replace(".", "p")
        return bca_name

    @property
    def bca_name(self) -> str:
        r"""
        The name of the bca algorithm given its parameters
        """
        return self.get_bca_name()

    @property
    def bca_params(self) -> Dict[str, value_type]:
        r"""
        The parameters of the bca algorithm.
        """
        return self._bca_params

    @bca_params.setter
    def bca_params(self, value: Dict[str, value_type]) -> None:
        r"""
        Sets the parameters of the bca algorithm.
        Args:
            - Dict containing the parameters name and value. Only parameters with
            valid name and non `None` values wil be updated.
        """
        for key in value.keys():
            if key in self._bca_params.keys():
                if value[key] is not None:
                    self._bca_params[key] = value[key]

        self._recompute_preds = True

    @property
    def hit_per_voxel(self) -> Tensor:
        return self._hit_per_voxel

    @property
    def dir_name(self) -> Path:
        """The path to the directory corresponding to the current set of parameters."""
        return self.get_dir_name()

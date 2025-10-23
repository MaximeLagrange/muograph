from typing import Optional
import torch
from torch import Tensor
from abc import ABC, abstractmethod

from muograph.volume.volume import Volume
from muograph.tracking.tracking import TrackingMST
from muograph.plotting.voxel import VoxelPlotting


r"""
Provides class for voxel-wise scattering density predictions.
"""


class AbsVoxelInferer(VoxelPlotting, ABC):
    r"""
    Class used for handling the computation and plotting of voxel-wise scattering density predictions
    """

    _xyz_voxel_pred: Optional[Tensor] = None  # (Nx, Ny, Nz)
    _recompute_preds = True

    def __init__(self, voi: Volume, tracking: TrackingMST) -> None:
        r"""
        Initializes the AbsVoxelInferer object with an instance of the TrackingMST class and Volume class.

        Args:
            - voi (Volume) Instance of the Volume class. The BCA algo. relying on voxelized
            volume, `voi` has to be provided.
            - tracking (Optional[TrackingMST]) Instance of the TrackingMST class.
        """

        super().__init__(voi=voi)
        self.tracks = tracking

    @abstractmethod
    def get_xyz_voxel_pred(self) -> Tensor:
        """
        Abstract method to compute voxel-wise predictions.

        Returns:
            Tensor: A tensor of shape (nx, ny, nz) with voxel-wise predictions.
        """
        pass

    @property
    def xyz_voxel_pred(self) -> Tensor:
        r"""
        The scattering density predictions.
        """
        if self._xyz_voxel_pred is None or self._recompute_preds:
            self._xyz_voxel_pred = self.get_xyz_voxel_pred()
            assert self._xyz_voxel_pred is not None
        return self._xyz_voxel_pred

    @property
    def xyz_voxel_pred_norm(self) -> Tensor:
        r"""
        The normalized scattering density predictions.
        """
        return (self.xyz_voxel_pred.float() - torch.min(self.xyz_voxel_pred)) / (torch.max(self.xyz_voxel_pred.float()) - torch.min(self.xyz_voxel_pred))

import torch
from torch import Tensor
from typing import Tuple
from muograph.utils.device import DEVICE

r"""
Provides container classes for a batch of many muons.
"""


class Volume:
    r"""
    Class for handling volume of interest and its voxelization.
    """

    _n_vox_xyz = None
    _voxel_centers = None  # (nx, ny, nz, 3)
    _voxel_edges = None  # (nx, ny, nz, 2, 3)

    def __init__(
        self,
        position: Tuple[float, float, float],
        dimension: Tuple[float, float, float],
        voxel_width: Tuple[float, float, float] = (10.0, 10.0, 10.0),
    ) -> None:
        """
        Initialize the Volume object.

        ASSUMPTION: X, Y are the horizontal coordinates and Z the vertical coordinate.

        Args:
            position (`Tuple[float, float, float]`): Position of the center of the volume in xyz.
            dimension (`Tuple[float, float, float]`): The xyz span of the volume.
            voxel_width (`float`): The size of the voxel. The user must ensure that
            the dimension of the volume can be divided by the voxel width.
        """
        self.xyz = torch.tensor(position, dtype=torch.float32, device=DEVICE)
        self.dxyz = torch.tensor(dimension, dtype=torch.float32, device=DEVICE)
        self.xyz_min = self.xyz - self.dxyz / 2
        self.xyz_max = self.xyz + self.dxyz / 2
        self.vox_width = torch.tensor(voxel_width, device=DEVICE)

    def __repr__(self) -> str:
        return (
            f"Volume of interest at x,y,z = "
            f"{self.xyz[0]:.2f},{self.xyz[1]:.2f},{self.xyz[2]:.2f}, "
            f"voxel size = {self.vox_width[0]:.2f},{self.vox_width[1]:.2f},{self.vox_width[2]:.2f} mm"
        )

    @staticmethod
    def compute_n_voxel(vox_width: Tensor, dxyz: Tensor) -> Tuple[int, int, int]:
        r"""
        Calculate the number of voxels along each axis.

        Args:
            vox_width (`float`): The size of the voxel.
            dxyz (`Tensor`): The dimensions of the volume.

        Returns:
            x (`Tuple[int, int, int]`): The number of voxels along the x, y, z dimensions.

        Raises:
            ValueError: If the dimensions are not divisible by the voxel width.
        """

        n_vox = dxyz / vox_width
        if not torch.all(n_vox % 1 == 0):
            raise ValueError("Voxel size does not match VOI dimensions. " "Ensure that dimension / voxel_width = integer.")

        nx, ny, nz = n_vox.int().tolist()

        return (nx, ny, nz)

    @staticmethod
    def generate_voxels(
        xyz_min: Tensor,
        xyz_max: Tensor,
        vox_width: Tensor,
        n_vox_xyz: Tuple[int, int, int],
    ) -> Tuple[Tensor, Tensor]:
        """
        Generate the voxel centers and edges for the 3D grid.

        Args:
            xyz_min (`Tensor`): The minimum xyz coordinates of the volume.
            xyz_max (`Tensor`): The maximum xyz coordinates of the volume.
            vox_width (`float`): The size of the voxel.
            n_vox_xyz (`Tuple[int, int, int]`): The number of voxels along each dimension.

        Returns:
            x (`Tuple[Tensor, Tensor]`): Voxel centers and voxel edges as tensors.
        """
        # Compute voxel centers for each axis
        xs = torch.linspace(
            xyz_min[0] + vox_width[0] / 2,
            xyz_max[0] - vox_width[0] / 2,
            n_vox_xyz[0],
            device=DEVICE,
        )
        ys = torch.linspace(
            xyz_min[1] + vox_width[1] / 2,
            xyz_max[1] - vox_width[1] / 2,
            n_vox_xyz[1],
            device=DEVICE,
        )
        zs = torch.linspace(
            xyz_min[2] + vox_width[2] / 2,
            xyz_max[2] - vox_width[2] / 2,
            n_vox_xyz[2],
            device=DEVICE,
        )

        # Create a meshgrid for the voxel centers
        xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")

        # Stack the grid coordinates along the last axis to get the voxel centers
        voxel_centers = torch.stack([xx, yy, zz], dim=-1)

        # Compute voxel edges by adding/subtracting half of the voxel width
        half_width = vox_width / 2
        voxel_edges = torch.stack([voxel_centers - half_width, voxel_centers + half_width], dim=-2)

        return voxel_centers, voxel_edges

    @property
    def n_vox_xyz(self) -> Tuple[int, int, int]:
        """
        Get the number of voxels along the x, y, z dimensions.

        Returns:
            x (`Tuple[int, int, int]`): The number of voxels along the x, y, z dimensions.
        """
        if self._n_vox_xyz is None:
            self._n_vox_xyz = self.compute_n_voxel(vox_width=self.vox_width, dxyz=self.dxyz)
        return self._n_vox_xyz

    @property
    def voxel_centers(self) -> Tensor:
        """
        Get the xyz position of the center of each voxel.

        Returns:
            voxel_centers (Tensor): Voxel centers with size (nx, ny, nz, 3).
        """
        if self._voxel_centers is None:
            self._voxel_centers, self._voxel_edges = self.generate_voxels(
                xyz_max=self.xyz_max,
                xyz_min=self.xyz_min,
                vox_width=self.vox_width,
                n_vox_xyz=self.n_vox_xyz,
            )
        return self._voxel_centers

    @property
    def voxel_edges(self) -> Tensor:
        """
        Get the xyz position of the front-bottom-left and back-upper-right corner of each voxel.

        Returns:
            voxel_edges (`Tensor`): Voxel edges with size (nx, ny, nz, 2, 3).
        """
        if self._voxel_edges is None:
            self._voxel_centers, self._voxel_edges = self.generate_voxels(
                xyz_max=self.xyz_max,
                xyz_min=self.xyz_min,
                vox_width=self.vox_width,
                n_vox_xyz=self.n_vox_xyz,
            )
        return self._voxel_edges

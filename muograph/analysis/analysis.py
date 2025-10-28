from __future__ import annotations
from typing import Union, Tuple, List, Optional, Any
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, fields

from muograph.volume.volume import Volume
from muograph.hits.hits import Hits
from muograph.tracking.tracking import Tracking, TrackingMST
from muograph.reconstruction.asr import ASR
from muograph.reconstruction.poca import POCA
from muograph.reconstruction.binned_clustered import BCA

ALGO_REGISTRY = {"ASR": ASR, "POCA": POCA, "BCA": BCA}


@dataclass
class Algorithm:
    class_ref: type
    params: Any
    preds: Optional[np.ndarray] = None
    name: Optional[str] = None


class Scan:
    def __init__(
        self,
        input_data: Union[str, Path, pd.DataFrame],
        voi: Volume,
        algorithms: List[Algorithm],
        plane_labels_in_out: Tuple[Tuple[int, ...], Tuple[int, ...]],
        energy_range: Optional[Tuple[float, float]] = None,
        spatial_res: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.input_data = input_data
        self.voi = voi
        self.algorithms = algorithms
        self.plane_label_in, self.plane_label_out = plane_labels_in_out
        self.energy_range = energy_range
        self.spatial_res = spatial_res

        self._hits_in = self._make_hits(self.plane_label_in)
        self._hits_out = self._make_hits(self.plane_label_out)
        self._tracking = TrackingMST((Tracking(hits=self._hits_in, label="above"), Tracking(hits=self._hits_out, label="below")))

    def _make_hits(self, plane_labels: Tuple[int, ...]) -> Hits:
        return Hits(
            data=self.input_data,
            plane_labels=plane_labels,
            spatial_res=self.spatial_res,
            energy_range=self.energy_range,
        )

    def get_preds(self, algo: Algorithm) -> np.ndarray:
        reconstructor = algo.class_ref(tracking=self._tracking, voi=self.voi)

        # Set params if applicable
        for field in fields(algo.params):
            name = field.name
            if hasattr(reconstructor, name):
                setattr(reconstructor, name, getattr(algo.params, name))
        algo.name = reconstructor.name
        preds = reconstructor.xyz_voxel_pred.detach().cpu().numpy()
        return preds.copy()

    def scan_all_algos(self, save_dir: Optional[Path] = None) -> None:
        for algo in self.algorithms:
            preds = self.get_preds(algo)
            if save_dir:
                np.save(save_dir / f"{algo.name}_preds.npy", preds)

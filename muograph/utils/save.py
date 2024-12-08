from pathlib import Path
from typing import List, Optional
from torch import  Tensor
import torch
import numpy as np
import h5py
import os

from muograph.utils.device import DEVICE

muograph_path = str(Path(__file__).parent.parent.parent).split("/muograph/")[0] + "/muograph/"
default_output_dir: str = muograph_path + "/output/"


class AbsSave:
    """
    A base class for managing directory creation and handling class attributes
    saving/loading.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Initializes the AbsSave object and ensures the output directory exists.

        Args:
            output_dir (str): The path to the directory where output files will be saved.
        """

        if output_dir is not None:
            self.output_dir = Path(output_dir)
            try:
                self.create_directory(self.output_dir)
            except FileNotFoundError:
                print(f"Directory not found: {self.output_dir}")
            except PermissionError:
                print(f"Permission denied: Could not create {self.output_dir}")
            except OSError as e:
                # General fallback for other OS-related errors.
                print(f"OS error occurred while creating {self.output_dir}: {e}")
        else:
            self.output_dir = None  # type: ignore

    @staticmethod
    def create_directory(directory: Path) -> None:
        r"""
        Creates a directory at the specified path if it does not already exist.

        Args:
            directory (Path): The path to the directory to be created.

        Notes:
            If the directory already exists, this method will not raise an exception.
            It will simply indicate that the directory already exists.
        """

        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"{directory} directory created")
        except PermissionError:
            print(f"Permission denied: Cannot create {directory}")
        except OSError as e:
            print(f"OS error: {e}")

    def save_attr(self, attributes: List[str], directory: Path, filename: str) -> None:
        r"""
        Saves class attributes to hdf5 file.

        Args:
            attributes (List[str]): The list of the class attributes to save.
            directory (Path): The path to the directory to be created.
            filename (str): the name of the file where to save the attributes.
        """

        filename += ".hdf5"
        with h5py.File(directory / filename, "w") as f:
            for attr in attributes:
                value = getattr(self, attr)
                if isinstance(value, Tensor):
                    f.create_dataset(attr, data=value.detach().cpu().numpy())
                elif isinstance(value, (np.ndarray, float)):
                    f.create_dataset(attr, data=value)
                elif isinstance(value, str):
                    f.create_dataset(attr, data=np.string_(value))

        f.close()
        print("Class attributes saved at {}".format(directory / filename))

    def load_attr(self, attributes: List[str], filename: str, tag: Optional[str] = None) -> None:
        r"""
        Loads class attributes from hdf5 file.

        Args:
            attributes (List[str]): the list of the class attributes to save.
            directory (Path): The path to the directory to be created.
            filename (str): the name of the file where to save the attributes.
            tag (str): tag to add to the attributes to that it matches
            the parent class attribute names. e.g: TrackingMST differentiate
            incoming and outgoing track attributes.
        """

        with h5py.File(filename, "r") as f:
            for attr in attributes:
                data = f[attr]
                if tag is not None:
                    if attr != "E":  # Do not differenciate incoming energy from outgoing energy
                        attr += tag
                if data.ndim == 0:
                    if isinstance(data[()], bytes):
                        setattr(self, attr, data[()].decode("utf-8"))
                    else:
                        setattr(self, attr, data[()])
                elif type(data[:]) is np.ndarray:
                  
                    setattr(self, attr, torch.tensor(data[:], device=DEVICE))
                    
                elif isinstance(data[()], bytes):  # Strings are usually stored as bytes
                    setattr(self, attr, data[()].decode("utf-8"))

        f.close()
        print("\nTracking attributes loaded from {}".format(filename))

    @staticmethod
    def files_in_dir(dir: str, files: List[str]) -> bool:
        r"""Returns `True` if the the directory `dir` contains the files listed in `files`.

        Args:
            dir (str): Path to the directory.
            files (List[str]): List of file names.

        Returns:
            bool
        """

        # Get file names from the input directory
        files = [f for f in os.listdir(dir) if os.path.isfile(dir + f)]

        # Make sure the directory contains the required files
        all_exist = all(any(file == name for file in files) for name in files)

        return all_exist

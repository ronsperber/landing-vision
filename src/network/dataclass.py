"""
module for dataset used for
Lunar Lander videos
"""

from typing import cast

import h5py
import torch
from torch.utils.data import Dataset


class LunarLanderDataset(Dataset):
    """
    Dataset class for Luna Lander
    video
    """

    def __init__(self, h5_path: str):
        """
        Parameters
        ----------
        h5_path : str
            path where the h5 data is stored
        """
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.n_samples = int(cast(h5py.Dataset, f["rewards"]).shape[0])

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            number of samples
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        idx : int
            index to get data
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            frames, length ,reward at idx
        """
        with h5py.File(self.h5_path, "r") as f:
            frames = torch.tensor(cast(h5py.Dataset, f["frames"])[idx]).float() / 255.0
            length = torch.tensor(int(cast(h5py.Dataset, f["lengths"])[idx]))
            reward = torch.tensor(float(cast(h5py.Dataset, f["rewards"])[idx]))
        return frames, length, reward

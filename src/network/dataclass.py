from typing import cast

import h5py
import torch
from torch.utils.data import Dataset


class LunarLanderDataset(Dataset):
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.n_samples = int(cast(h5py.Dataset, f["rewards"]).shape[0])

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_path, "r") as f:
            frames = torch.tensor(cast(h5py.Dataset, f["frames"])[idx]).float() / 255.0
            length = torch.tensor(int(cast(h5py.Dataset, f["lengths"])[idx]))
            reward = torch.tensor(float(cast(h5py.Dataset, f["rewards"])[idx]))
        return frames, length, reward

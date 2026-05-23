import numpy as np
import pytest
import torch
import h5py

from network.dataclass import LunarLanderDataset


def make_h5(path, n_samples: int, n_frames: int = 10):
    with h5py.File(path, "w") as f:
        f.create_dataset("rewards", data=np.random.rand(n_samples).astype(np.float32))
        f.create_dataset(
            "frames",
            data=np.zeros((n_samples, n_frames, 3, 84, 84), dtype=np.uint8),
        )
        f.create_dataset(
            "lengths", data=np.full(n_samples, n_frames, dtype=np.int64)
        )
    return path


def test_len(tmp_path):
    h5_path = make_h5(tmp_path / "data.h5", n_samples=5)
    dataset = LunarLanderDataset(str(h5_path))
    assert len(dataset) == 5


def test_getitem_types(tmp_path):
    h5_path = make_h5(tmp_path / "data.h5", n_samples=3)
    dataset = LunarLanderDataset(str(h5_path))
    frames, length, reward = dataset[0]
    assert isinstance(frames, torch.Tensor)
    assert isinstance(length, torch.Tensor)
    assert isinstance(reward, torch.Tensor)


def test_getitem_frames_normalized(tmp_path):
    with h5py.File(tmp_path / "data.h5", "w") as f:
        f.create_dataset(
            "frames", data=np.full((1, 5, 3, 84, 84), 255, dtype=np.uint8)
        )
        f.create_dataset("rewards", data=np.array([1.0], dtype=np.float32))
        f.create_dataset("lengths", data=np.array([5], dtype=np.int64))

    dataset = LunarLanderDataset(str(tmp_path / "data.h5"))
    frames, _, _ = dataset[0]
    assert frames.dtype == torch.float32
    assert torch.allclose(frames, torch.ones_like(frames))


def test_getitem_frames_shape(tmp_path):
    n_frames = 10
    h5_path = make_h5(tmp_path / "data.h5", n_samples=2, n_frames=n_frames)
    dataset = LunarLanderDataset(str(h5_path))
    frames, length, reward = dataset[1]
    assert frames.shape == (n_frames, 3, 84, 84)
    assert length.shape == ()
    assert reward.shape == ()


def test_getitem_all_indices(tmp_path):
    h5_path = make_h5(tmp_path / "data.h5", n_samples=4)
    dataset = LunarLanderDataset(str(h5_path))
    for i in range(len(dataset)):
        frames, length, reward = dataset[i]
        assert frames is not None

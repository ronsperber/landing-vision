from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from preprocessing.video_process import (
    get_filename_df,
    get_joined_df,
    get_video_path,
    get_video_tensor,
    make_transform,
    pad_video,
)


# --- pad_video ---

def test_pad_video_short():
    t = torch.zeros(10, 3, 84, 84)
    out = pad_video(t, max_len=20)
    assert out.shape == (20, 3, 84, 84)
    assert torch.all(out[10:] == 0)


def test_pad_video_truncate():
    t = torch.ones(30, 3, 84, 84)
    out = pad_video(t, max_len=20)
    assert out.shape == (20, 3, 84, 84)


def test_pad_video_exact_length():
    t = torch.ones(20, 3, 84, 84)
    out = pad_video(t, max_len=20)
    assert out.shape == (20, 3, 84, 84)
    assert torch.all(out == 1)


# --- make_transform ---

def test_make_transform_output_shape():
    transform = make_transform(84, 84)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    result = transform(frame)
    assert result.shape == (3, 84, 84)


def test_make_transform_custom_size():
    transform = make_transform(32, 64)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = transform(frame)
    assert result.shape == (3, 32, 64)


# --- get_filename_df ---

def test_get_filename_df(tmp_path):
    for ep in [1, 5, 10]:
        (tmp_path / f"episode-{ep}.mp4").touch()
    df = get_filename_df(tmp_path, suffix="mp4")
    assert set(df["Episode"]) == {1, 5, 10}
    assert "filename" in df.columns
    assert len(df) == 3


def test_get_filename_df_empty_dir(tmp_path):
    df = get_filename_df(tmp_path, suffix="mp4")
    assert len(df) == 0


# --- get_joined_df ---

def test_get_joined_df_basic():
    rewards_df = pd.DataFrame({"Episode": [1, 2, 3], "reward": [10.0, 20.0, 30.0]})
    file_df = pd.DataFrame({"Episode": [1, 2], "filename": ["ep-1.mp4", "ep-2.mp4"]})
    with patch("preprocessing.video_process.get_filename_df", return_value=file_df):
        result = get_joined_df(rewards_df, Path("/fake"), "Episode")
    assert len(result) == 2
    assert "filename" in result.columns


def test_get_joined_df_strips_leading_spaces():
    rewards_df = pd.DataFrame({" Episode": [1, 2], " reward": [1.0, 2.0]})
    file_df = pd.DataFrame({"Episode": [1, 2], "filename": ["a.mp4", "b.mp4"]})
    with patch("preprocessing.video_process.get_filename_df", return_value=file_df):
        result = get_joined_df(rewards_df, Path("/fake"), " Episode")
    assert len(result) == 2


def test_get_joined_df_renames_episode_col():
    rewards_df = pd.DataFrame({"Ep": [1, 2], "reward": [1.0, 2.0]})
    file_df = pd.DataFrame({"Episode": [1, 2], "filename": ["a.mp4", "b.mp4"]})
    with patch("preprocessing.video_process.get_filename_df", return_value=file_df):
        result = get_joined_df(rewards_df, Path("/fake"), "Ep")
    assert len(result) == 2
    assert "Episode" in result.columns


def test_get_joined_df_missing_column_raises():
    rewards_df = pd.DataFrame({"Episode": [1], "reward": [1.0]})
    with patch("preprocessing.video_process.get_filename_df"):
        with pytest.raises(ValueError, match="not in rewards_df"):
            get_joined_df(rewards_df, Path("/fake"), "NonExistent")


# --- get_video_path ---

def test_get_video_path():
    df = pd.DataFrame({"filename": ["ep-1.mp4", "ep-2.mp4"]})
    path = get_video_path(Path("/videos"), df, 0)
    assert path == Path("/videos/ep-1.mp4")


def test_get_video_path_second_row():
    df = pd.DataFrame({"filename": ["ep-1.mp4", "ep-2.mp4"]})
    path = get_video_path(Path("/videos"), df, 1)
    assert path == Path("/videos/ep-2.mp4")


# --- get_video_tensor ---

def test_get_video_tensor_shape():
    fake_frame = np.zeros((84, 84, 3), dtype=np.uint8)
    transform = make_transform(84, 84)

    mock_cap = MagicMock()
    mock_cap.read.side_effect = [
        (True, fake_frame),
        (True, fake_frame),
        (False, None),
    ]

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with patch("cv2.cvtColor", side_effect=lambda f, _: f):
            result = get_video_tensor(Path("/fake/video.mp4"), transform)

    assert result.shape == (2, 3, 84, 84)
    assert result.dtype == torch.float32


def test_get_video_tensor_empty_video():
    transform = make_transform(84, 84)
    mock_cap = MagicMock()
    mock_cap.read.return_value = (False, None)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with patch("cv2.cvtColor", side_effect=lambda f, _: f):
            with pytest.raises(Exception):
                get_video_tensor(Path("/fake/video.mp4"), transform)

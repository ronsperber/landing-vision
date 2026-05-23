from pathlib import Path

import cv2
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import Compose


def get_filename_df(dir: Path, suffix: str = "mp4") -> pd.DataFrame:
    video_paths = list(dir.glob(f"*.{suffix}"))
    video_files = [Path(x).name for x in video_paths]
    video_episodes = [int(f.split("-")[-1].replace(".mp4", "")) for f in video_files]
    return pd.DataFrame({"Episode": video_episodes, "filename": video_files})


def pad_video(tensor: torch.Tensor, max_len=500) -> torch.Tensor:
    T = tensor.shape[0]
    if T >= max_len:
        return tensor[:max_len]  # truncate
    pad = torch.zeros(max_len - T, *tensor.shape[1:])
    return torch.cat([tensor, pad], dim=0)


def make_transform(h: int = 84, w: int = 84) -> Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),  # handles both the permute and normalization
            transforms.Resize((h, w), antialias=True),
        ]
    )


def get_joined_df(
    rewards_df: pd.DataFrame,
    video_dir: Path,
    episode_col: str,
    suffix: str = "mp4",
) -> pd.DataFrame:
    filename_df = get_filename_df(video_dir, suffix)
    # strip leading spaces from rewards_df column names:
    rewards_df = rewards_df.rename(columns={c: c.lstrip() for c in rewards_df.columns})
    # rename the episode column "Episodes" to match the output of get_filename_df
    if episode_col.lstrip() not in rewards_df.columns:
        raise ValueError(f"Column {episode_col} not in rewards_df")
    if episode_col.lstrip() != "Episode":
        rewards_df = rewards_df.rename(columns={episode_col.lstrip(): "Episode"})
    return rewards_df.merge(filename_df, on="Episode", how="inner")


def get_video_path(video_dir: Path, df: pd.DataFrame, idx: int) -> Path:
    return video_dir / str(df.iloc[idx]["filename"])


def get_video_tensor(video_path: Path, transform: Compose) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    # change to RGB
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    torch_frames = [transform(frame) for frame in frames]
    return torch.stack(torch_frames)

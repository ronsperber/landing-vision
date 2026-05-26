import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import video_process

# load configs and create basic baths to be used
with open("config.json") as f:
    config = json.load(f) or {}
video_root = config.get("video_root", "videos")
game = config.get("game", "")
timestamp = config.get("timestamp", "")
hidden_dims = config.get("hidden_dims", "")
video_dir = Path(video_root) / game / timestamp / hidden_dims
# read in the rewards
rewards_df = pd.read_csv(video_dir / config.get("reward_file", ""))

DATA_PATH = Path(".") / "data"
DATA_PATH.mkdir(parents=True, exist_ok=True)
# make the transform to be used later
transform = video_process.make_transform()
# get joint dataframe with episode, reward, filename
joint_df = video_process.get_joined_df(
    rewards_df=rewards_df,
    video_dir=video_dir,
    episode_col="Episode",
)
n_videos = len(joint_df)
max_len = 500

with h5py.File(DATA_PATH / "preprocessed.h5", "w") as f:
    # create datasets for the frames, lengths, and rewards
    frames_ds = f.create_dataset(
        "frames",
        shape=(n_videos, max_len, 3, 84, 84),
        dtype="uint8",
    )
    lengths_ds = f.create_dataset("lengths", shape=(n_videos,), dtype="int32")
    rewards_ds = f.create_dataset("rewards", shape=(n_videos,), dtype="float32")

    for i, row in enumerate(tqdm(joint_df.itertuples(), total=n_videos)):
        # get the video, pad, and save everything needed to the h5
        # file
        video_path = video_process.get_video_path(
            video_dir=video_dir, df=joint_df, idx=i
        )
        tensor = video_process.get_video_tensor(video_path, transform)
        padded = video_process.pad_video(tensor, max_len=500)
        frames_ds[i] = (padded.numpy() * 255).astype(np.uint8)
        lengths_ds[i] = min(tensor.shape[0], 500)
        rewards_ds[i] = row.Episode_Reward

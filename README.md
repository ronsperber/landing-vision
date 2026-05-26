# landing-vision

Predicting the outcome of lunar lander episodes from raw video footage using a CNN + LSTM model.

## Overview

Each episode of the [Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment is recorded as an mp4 video. This project processes those videos into a fixed-length sequence of frames and trains a neural network to predict the episode reward — either as a regression task (predict the raw reward) or a binary classification task (did the episode exceed a reward threshold?).

**Architecture**: per-frame features are extracted by two Conv2D layers (with optional max pooling and dropout), projected to an embedding, fed into a 2-layer LSTM, and a linear head produces the final scalar prediction.

## Project structure

```
scripts/
  process_videos.py       # preprocess mp4s → data/preprocessed.h5
  train.py                # regression training
  train_classification.py # classification training
src/
  preprocessing/
    video_process.py      # video loading, transforms, dataframe helpers
  network/
    model.py              # LunarLanderConv (CNN + LSTM)
    dataclass.py          # LunarLanderDataset (HDF5-backed)
    training.py           # train() loop with early stopping
tests/                    # pytest test suite
local_notebooks/          # exploratory notebooks
```

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Usage

### 1. Preprocess videos

Videos are expected to live in a directory and be named `*-{episode}.mp4`. A CSV with episode rewards is also required. Configure paths in `config.json`:

```json
{
  "video_root": "videos",
  "game": "LunarLander-v3",
  "timestamp": "20240101_120000",
  "hidden_dims": "256",
  "reward_file": "rewards.csv"
}
```

Then run:

```bash
uv run python scripts/process_videos.py
```

This writes `data/preprocessed.h5` containing frames (resized to 84x84, stored as uint8), sequence lengths, and rewards.

### 2. Train — regression

Predicts the raw episode reward. Rewards are standardized using a `StandardScaler` fit on the training split.

```bash
uv run python scripts/train.py
uv run python scripts/train.py --epochs 200 --dropout 0.3
```

| Flag | Default | Description |
|------|---------|-------------|
| `-e` / `--epochs` | 500 | Maximum training epochs |
| `-d` / `--dropout` | 0.0 | Dropout rate (applied after embed layer and between LSTM layers) |

### 3. Train — classification

Converts rewards to binary labels (`reward >= threshold`) and trains with `BCEWithLogitsLoss`.

```bash
uv run python scripts/train_classification.py
uv run python scripts/train_classification.py --threshold 150 --dropout 0.2
```

| Flag | Default | Description |
|------|---------|-------------|
| `-e` / `--epochs` | 500 | Maximum training epochs |
| `-d` / `--dropout` | 0.0 | Dropout rate |
| `-t` / `--threshold` | 200.0 | Reward threshold for positive class |

### Outputs

Both training scripts write to `output/{run_name}/{timestamp}/`:

| File | Description |
|------|-------------|
| `best_model.pt` | Checkpoint with lowest validation loss |
| `model.pt` | Final model state at end of training |
| `history.json` | Per-epoch train and validation loss |
| `test_indices.pt` | Indices of the held-out test split |
| `reward_scaler.pkl` | Fitted `StandardScaler` (regression only) |

Training uses a 70 / 15 / 15 train/val/test split and stops early if validation loss does not improve for 20 consecutive epochs.

## Running tests

```bash
uv run pytest
```

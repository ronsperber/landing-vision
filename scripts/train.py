import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, random_split

from network.dataclass import LunarLanderDataset
from network.model import LunarLanderConv
from network.training import train

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=500)
args = parser.parse_args()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_PATH = "data/preprocessed.h5"
OUTPUT_PATH = Path("output") / timestamp
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = LunarLanderDataset(DATA_PATH)
# train/val/test split
n = len(dataset)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
# get train rewards
print("Getting training rewards for scaling.")
train_rewards = np.array([dataset[i][2].item() for i in train_ds.indices])
# create scaler for the rewards and save for inference
scaler = StandardScaler()
scaler.fit(train_rewards.reshape(-1, 1))
joblib.dump(scaler, OUTPUT_PATH / "reward_scaler.pkl")
print("Scaler fit and saved.")
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
model = LunarLanderConv()
hist, val_hist = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.MSELoss(),
    epochs=args.epochs,
    reward_scaler=scaler,
    device=device,
    output_path=OUTPUT_PATH,
)


history_dict = {
    "train_history": hist,
    "val_history": val_hist,
}
history_file = OUTPUT_PATH / "history.json"
with open(history_file, "w") as f:
    json.dump(history_dict, f)

torch.save(model.state_dict(), OUTPUT_PATH / "model.pt")
torch.save(test_ds.indices, OUTPUT_PATH / "test_indices.pt")

import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from network.dataclass import LunarLanderDataset
from network.model import LunarLanderConv
from network.training import train

parser = ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=500)
parser.add_argument("-t", "--threshold", type=float, default=200.0)
args = parser.parse_args()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_PATH = "data/preprocessed.h5"
OUTPUT_PATH = Path("output") / f"threshold_{args.threshold}" / timestamp
OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = LunarLanderDataset(DATA_PATH)
# train/val/test split
n = len(dataset)
n_train = int(0.7 * n)
n_val = int(0.15 * n)
n_test = n - n_train - n_val
train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
model = LunarLanderConv()
hist, val_hist = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.BCEWithLogitsLoss(),
    epochs=args.epochs,
    classification_threshold=args.threshold,
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

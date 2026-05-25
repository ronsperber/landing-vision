from typing import Callable, cast

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from network.model import LunarLanderConv


def train(
    model: LunarLanderConv,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type = torch.optim.Adam,
    lr: float = 1e-4,
    epochs: int = 500,
    device: str = "cpu",
    reward_scaler : StandardScaler | None,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    model = model.to(device)
    optimizer: torch.optim.Optimizer = optimizer_cls(params=model.parameters(), lr=lr)
    losses = []
    val_losses = []
    tqdm_bar = tqdm(range(epochs))
    for epoch in tqdm_bar:
        epoch_losses = []
        for batch in tqdm(train_loader, desc="train", leave=False):
            optimizer.zero_grad()
            frames_batch: torch.Tensor
            length_batch: torch.Tensor
            rewards_batch: torch.Tensor
            frames_batch, length_batch, rewards_batch = batch
            frames_batch = frames_batch.to(device)
            length_batch = length_batch.to(device)
            rewards_batch = rewards_batch.to(device)
            if reward_scaler is not None:
                rewards_batch = torch.tensor(
                    reward_scaler.transform(rewards_batch.cpu().numpy().reshape(-1, 1)),
                    dtype=torch.float32
                    ).squeeze().to(device)
            out = cast(torch.Tensor, model(frames_batch, length_batch))
            loss = criterion(out.squeeze(), rewards_batch)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append((epoch + 1, epoch_loss))
        model.eval()
        with torch.no_grad():
            epoch_val_losses = []
            for batch in tqdm(val_loader, desc="val", leave=False):
                frames: torch.Tensor
                lengths: torch.Tensor
                rewards: torch.Tensor
                frames, lengths, rewards = batch
                frames = frames.to(device)
                lengths = lengths.to(device)
                rewards = rewards.to(device)
                out = cast(torch.Tensor, model(frames, lengths))
                loss = criterion(out.squeeze(), rewards)
                epoch_val_losses.append(loss.item())
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        model.train()
        val_losses.append((epoch + 1, avg_val_loss))
        tqdm_bar.set_postfix(
            {
                "train_loss": f"{epoch_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
            }
        )
    return losses, val_losses

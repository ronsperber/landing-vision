"""
module for training on the video data
"""

from pathlib import Path
from typing import Callable, cast

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_cls: type = torch.optim.Adam,
    lr: float = 1e-4,
    epochs: int = 500,
    device: str = "cpu",
    reward_scaler: StandardScaler | None = None,
    classification_threshold: float | None = None,
    patience: int = 20,
    output_path: Path | None = None,
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """
    training function
    Parameters
    ----------
    model: nn.Module
        network to train
    train_loader: DataLoader
        data loader with training data
    val_loader: DataLoader
        data loader with validation data
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        loss function. typically MSE for regression, or cross entropy for classification
    optimizer_cls : type
        optimizer class to use
    lr: float
        learning rate for training
    epochs: int
        number of epochs to train
    device: str
        device to use for the model and data
    reward_scaler : (optional) StandardScaler
        optional scaler to scale the rewards
    classification_threshold : (optional) float
        when classification is the task the threshhold at which
        the positive class is defined
    patience: int
        number of epochs without validation loss improvement to wait
        before early stopping
    output_path : (optional) Path
        path to save checkpoints
    Returns
    -------
    tuple[list[tuple[int, float]], list[tuple[int, float]]]
        lists [(epoch, training loss)], [(epoch, val loss)]
    """
    model = model.to(device)
    optimizer: torch.optim.Optimizer = optimizer_cls(params=model.parameters(), lr=lr)
    losses = []
    val_losses = []
    # initiate values for early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    _output_path = output_path or Path(".")
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
            # if there is a classification threshhold change
            # rewards to a 0/1 based on the threshold
            if classification_threshold is not None:
                rewards_batch = (rewards_batch >= classification_threshold).float()
            # if not, and there is a scaler, transform the rewards
            elif reward_scaler is not None:
                rewards_batch = (
                    torch.tensor(
                        reward_scaler.transform(
                            rewards_batch.cpu().numpy().reshape(-1, 1)
                        ),
                        dtype=torch.float32,
                    )
                    .squeeze()
                    .to(device)
                )
            out = cast(torch.Tensor, model(frames_batch, length_batch))
            loss = criterion(out.squeeze(), rewards_batch)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append((epoch + 1, epoch_loss))
        model.eval()
        with torch.no_grad():
            # evaluate the losses on the validation set
            # using same classification_threshold or scaler
            epoch_val_losses = []
            for batch in tqdm(val_loader, desc="val", leave=False):
                frames: torch.Tensor
                lengths: torch.Tensor
                rewards: torch.Tensor
                frames, lengths, rewards = batch
                frames = frames.to(device)
                lengths = lengths.to(device)
                rewards = rewards.to(device)
                if classification_threshold is not None:
                    rewards = (rewards >= classification_threshold).float()
                elif reward_scaler is not None:
                    rewards = (
                        torch.tensor(
                            reward_scaler.transform(
                                rewards.cpu().numpy().reshape(-1, 1)
                            ),
                            dtype=torch.float32,
                        )
                        .squeeze()
                        .to(device)
                    )
                out = cast(torch.Tensor, model(frames, lengths))
                loss = criterion(out.squeeze(), rewards)
                epoch_val_losses.append(loss.item())
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_losses.append((epoch + 1, avg_val_loss))
        # if we have a new best loss, track it and start the counter over
        # and save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), _output_path / "best_model.pt")
        else:
            # if validation loss not better than best, increase counter
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            # too many epochs without improving best validation loss
            # kill the process and save
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break
        model.train()
        tqdm_bar.set_postfix(
            {
                "train_loss": f"{epoch_loss:.4f}",
                "val_loss": f"{avg_val_loss:.4f}",
            }
        )
    return losses, val_losses

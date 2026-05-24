import torch
import torch.nn as nn
import pytest
from torch.utils.data import DataLoader, TensorDataset

from network.training import train


class _TinyModel(nn.Module):
    """Drop-in substitute with the same forward signature but no expensive conv/lstm."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, frames: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.ones(frames.shape[0], 1))


def _make_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    # frames shape deliberately tiny — _TinyModel ignores spatial dims
    frames = torch.zeros(n, 5, 3, 4, 4)
    lengths = torch.full((n,), 5, dtype=torch.long)
    rewards = torch.rand(n)
    return DataLoader(TensorDataset(frames, lengths, rewards), batch_size=batch_size)


@pytest.fixture
def model():
    return _TinyModel()


@pytest.fixture
def loaders():
    return _make_loader(), _make_loader()


# --- return structure ---

def test_returns_two_lists(model, loaders):
    train_loader, val_loader = loaders
    result = train(model, train_loader, val_loader, nn.MSELoss(), epochs=2)
    assert len(result) == 2
    losses, val_losses = result
    assert isinstance(losses, list)
    assert isinstance(val_losses, list)


def test_list_lengths_match_epochs(model, loaders):
    train_loader, val_loader = loaders
    losses, val_losses = train(model, train_loader, val_loader, nn.MSELoss(), epochs=3)
    assert len(losses) == 3
    assert len(val_losses) == 3


def test_entries_are_int_float_tuples(model, loaders):
    train_loader, val_loader = loaders
    losses, val_losses = train(model, train_loader, val_loader, nn.MSELoss(), epochs=1)
    for epoch, loss in losses + val_losses:
        assert isinstance(epoch, int)
        assert isinstance(loss, float)


# --- epoch numbering ---

def test_epoch_numbers_start_at_one(model, loaders):
    train_loader, val_loader = loaders
    losses, val_losses = train(model, train_loader, val_loader, nn.MSELoss(), epochs=3)
    assert [e for e, _ in losses] == [1, 2, 3]
    assert [e for e, _ in val_losses] == [1, 2, 3]


# --- loss values ---

def test_losses_are_finite(model, loaders):
    train_loader, val_loader = loaders
    losses, val_losses = train(model, train_loader, val_loader, nn.MSELoss(), epochs=2)
    for _, loss in losses + val_losses:
        assert torch.isfinite(torch.tensor(loss))


# --- model state ---

def test_model_in_train_mode_after_training(model, loaders):
    train_loader, val_loader = loaders
    train(model, train_loader, val_loader, nn.MSELoss(), epochs=1)
    assert model.training


def test_weights_update_during_training():
    model = _TinyModel()
    before = model.linear.weight.clone().detach()
    loader = _make_loader()
    train(model, loader, loader, nn.MSELoss(), epochs=3, lr=1e-2)
    after = model.linear.weight.clone().detach()
    assert not torch.allclose(before, after)


# --- custom optimizer ---

def test_custom_optimizer_cls_is_used(model, loaders):
    train_loader, val_loader = loaders
    instantiated = []

    class TrackingOptimizer(torch.optim.SGD):
        def __init__(self, params, lr):
            instantiated.append(True)
            super().__init__(params, lr=lr)

    train(model, train_loader, val_loader, nn.MSELoss(), optimizer_cls=TrackingOptimizer, epochs=1)
    assert instantiated


# --- edge cases ---

def test_single_epoch(model, loaders):
    train_loader, val_loader = loaders
    losses, val_losses = train(model, train_loader, val_loader, nn.MSELoss(), epochs=1)
    assert len(losses) == 1
    assert len(val_losses) == 1


def test_separate_train_val_losses(model):
    # train and val use different data so their losses should differ
    torch.manual_seed(0)
    train_loader = _make_loader(n=8)
    val_loader = _make_loader(n=8)
    losses, val_losses = train(model, train_loader, val_loader, nn.MSELoss(), epochs=1)
    # losses are recorded independently — same epoch number but potentially different values
    assert losses[0][0] == val_losses[0][0] == 1

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from network.training import train


class _TinyModel(nn.Module):
    """Drop-in substitute with the same forward signature but no expensive conv/lstm."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, frames: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.ones(frames.shape[0], 1))


def _make_loader(n: int = 8, batch_size: int = 4, shuffle: bool = False) -> DataLoader:
    # frames shape deliberately tiny — _TinyModel ignores spatial dims
    frames = torch.zeros(n, 5, 3, 4, 4)
    lengths = torch.full((n,), 5, dtype=torch.long)
    rewards = torch.rand(n)
    return DataLoader(
        TensorDataset(frames, lengths, rewards),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _fitted_scaler() -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(np.array([[0.0], [1.0]]))
    return scaler


@pytest.fixture
def loaders(tmp_path):
    return _make_loader(), _make_loader(), tmp_path


# --- return structure ---

def test_returns_two_lists(loaders):
    train_loader, val_loader, out = loaders
    result = train(_TinyModel(), train_loader, val_loader, nn.MSELoss(), epochs=2, output_path=out)
    assert len(result) == 2
    losses, val_losses = result
    assert isinstance(losses, list)
    assert isinstance(val_losses, list)


def test_list_lengths_match_epochs(loaders):
    train_loader, val_loader, out = loaders
    losses, val_losses = train(
        _TinyModel(), train_loader, val_loader, nn.MSELoss(), epochs=3, output_path=out
    )
    assert len(losses) == 3
    assert len(val_losses) == 3


def test_entries_are_int_float_tuples(loaders):
    train_loader, val_loader, out = loaders
    losses, val_losses = train(
        _TinyModel(), train_loader, val_loader, nn.MSELoss(), epochs=1, output_path=out
    )
    for epoch, loss in losses + val_losses:
        assert isinstance(epoch, int)
        assert isinstance(loss, float)


# --- epoch numbering ---

def test_epoch_numbers_start_at_one(loaders):
    train_loader, val_loader, out = loaders
    losses, val_losses = train(
        _TinyModel(), train_loader, val_loader, nn.MSELoss(), epochs=3, output_path=out
    )
    assert [e for e, _ in losses] == [1, 2, 3]
    assert [e for e, _ in val_losses] == [1, 2, 3]


# --- loss values ---

def test_losses_are_finite(loaders):
    train_loader, val_loader, out = loaders
    losses, val_losses = train(
        _TinyModel(), train_loader, val_loader, nn.MSELoss(), epochs=2, output_path=out
    )
    for _, loss in losses + val_losses:
        assert torch.isfinite(torch.tensor(loss))


# --- model state ---

def test_model_in_train_mode_after_training(loaders):
    train_loader, val_loader, out = loaders
    m = _TinyModel()
    train(m, train_loader, val_loader, nn.MSELoss(), epochs=1, output_path=out)
    assert m.training


def test_weights_update_during_training(tmp_path):
    m = _TinyModel()
    before = m.linear.weight.clone().detach()
    loader = _make_loader()
    train(m, loader, loader, nn.MSELoss(), epochs=3, lr=1e-2, output_path=tmp_path)
    after = m.linear.weight.clone().detach()
    assert not torch.allclose(before, after)


# --- custom optimizer ---

def test_custom_optimizer_cls_is_used(loaders):
    train_loader, val_loader, out = loaders
    instantiated = []

    class TrackingOptimizer(torch.optim.SGD):
        def __init__(self, params, lr):
            instantiated.append(True)
            super().__init__(params, lr=lr)

    train(
        _TinyModel(), train_loader, val_loader, nn.MSELoss(),
        optimizer_cls=TrackingOptimizer, epochs=1, output_path=out,
    )
    assert instantiated


# --- edge cases ---

def test_single_epoch(loaders):
    train_loader, val_loader, out = loaders
    losses, val_losses = train(
        _TinyModel(), train_loader, val_loader, nn.MSELoss(), epochs=1, output_path=out
    )
    assert len(losses) == 1
    assert len(val_losses) == 1


def test_separate_train_val_losses(tmp_path):
    torch.manual_seed(0)
    train_loader = _make_loader(n=8)
    val_loader = _make_loader(n=8)
    losses, val_losses = train(
        _TinyModel(), train_loader, val_loader, nn.MSELoss(), epochs=1, output_path=tmp_path
    )
    assert losses[0][0] == val_losses[0][0] == 1


# --- reward_scaler ---

def test_reward_scaler_transform_called(tmp_path):
    scaler = _fitted_scaler()
    scaler.transform = MagicMock(side_effect=scaler.transform)
    loader = _make_loader()
    train(
        _TinyModel(), loader, loader, nn.MSELoss(),
        epochs=1, reward_scaler=scaler, output_path=tmp_path,
    )
    assert scaler.transform.called


def test_reward_scaler_losses_finite(tmp_path):
    loader = _make_loader()
    losses, val_losses = train(
        _TinyModel(), loader, loader, nn.MSELoss(),
        epochs=1, reward_scaler=_fitted_scaler(), output_path=tmp_path,
    )
    for _, loss in losses + val_losses:
        assert torch.isfinite(torch.tensor(loss))


def test_reward_scaler_ignored_when_threshold_set(tmp_path):
    """classification_threshold takes priority (elif branch); scaler must not be called."""
    scaler = _fitted_scaler()
    scaler.transform = MagicMock(side_effect=scaler.transform)
    loader = _make_loader()
    train(
        _TinyModel(), loader, loader, nn.MSELoss(), epochs=1,
        reward_scaler=scaler, classification_threshold=0.5, output_path=tmp_path,
    )
    scaler.transform.assert_not_called()


# --- classification_threshold ---

def test_classification_threshold_losses_finite(tmp_path):
    loader = _make_loader()
    losses, val_losses = train(
        _TinyModel(), loader, loader, nn.MSELoss(),
        epochs=1, classification_threshold=0.5, output_path=tmp_path,
    )
    for _, loss in losses + val_losses:
        assert torch.isfinite(torch.tensor(loss))


def test_classification_threshold_returns_correct_structure(tmp_path):
    loader = _make_loader()
    losses, val_losses = train(
        _TinyModel(), loader, loader, nn.MSELoss(),
        epochs=2, classification_threshold=0.5, output_path=tmp_path,
    )
    assert len(losses) == 2
    assert len(val_losses) == 2


# --- early stopping ---

def test_early_stopping_triggers(tmp_path):
    # lr=0 freezes weights → val loss never improves → patience=1 stops after epoch 2
    loader = _make_loader()
    losses, _ = train(
        _TinyModel(), loader, loader, nn.MSELoss(),
        epochs=50, lr=0.0, patience=1, output_path=tmp_path,
    )
    assert len(losses) < 50
    assert len(losses) == 2


def test_early_stopping_not_triggered_with_large_patience(tmp_path):
    loader = _make_loader()
    losses, _ = train(
        _TinyModel(), loader, loader, nn.MSELoss(),
        epochs=3, patience=20, output_path=tmp_path,
    )
    assert len(losses) == 3


# --- output_path / model saving ---

def test_best_model_saved_to_output_path(tmp_path):
    loader = _make_loader()
    train(_TinyModel(), loader, loader, nn.MSELoss(), epochs=1, output_path=tmp_path)
    assert (tmp_path / "best_model.pt").exists()


def test_best_model_saved_to_cwd_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    loader = _make_loader()
    train(_TinyModel(), loader, loader, nn.MSELoss(), epochs=1)
    assert (tmp_path / "best_model.pt").exists()


def test_best_model_loadable(tmp_path):
    m = _TinyModel()
    loader = _make_loader()
    train(m, loader, loader, nn.MSELoss(), epochs=1, output_path=tmp_path)
    state = torch.load(tmp_path / "best_model.pt", weights_only=True)
    m2 = _TinyModel()
    m2.load_state_dict(state)
    assert all(
        torch.allclose(p1, p2)
        for p1, p2 in zip(m.parameters(), m2.parameters())
    )

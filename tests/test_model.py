import torch
import pytest

from network.model import LunarLanderConv


@pytest.fixture
def model():
    return LunarLanderConv()


def _make_input(B: int, N: int):
    x = torch.zeros(B, N, 3, 84, 84)
    lengths = torch.tensor([N] * B)
    return x, lengths


def test_forward_output_shape(model):
    x, lengths = _make_input(B=2, N=10)
    out = model(x, lengths)
    assert out.shape == (2, 1)


def test_forward_single_sample(model):
    x, lengths = _make_input(B=1, N=5)
    out = model(x, lengths)
    assert out.shape == (1, 1)


def test_forward_variable_lengths(model):
    B, N = 3, 8
    x = torch.zeros(B, N, 3, 84, 84)
    lengths = torch.tensor([8, 5, 2])
    out = model(x, lengths)
    assert out.shape == (B, 1)


def test_forward_no_pooling():
    # use_pooling=False skips MaxPool, so linear layer is much larger — use tiny batch
    model = LunarLanderConv(use_pooling=False)
    x, lengths = _make_input(B=1, N=1)
    out = model(x, lengths)
    assert out.shape == (1, 1)


def test_forward_custom_channels():
    model = LunarLanderConv(conv_channels=(16, 32), embed_dim=64, hidden_size=64)
    x, lengths = _make_input(B=2, N=6)
    out = model(x, lengths)
    assert out.shape == (2, 1)


def test_output_is_scalar_per_sample(model):
    x, lengths = _make_input(B=4, N=10)
    out = model(x, lengths)
    assert out.shape == (4, 1)
    assert out.dtype == torch.float32


def test_gradients_flow(model):
    x, lengths = _make_input(B=2, N=5)
    x.requires_grad_(True)
    out = model(x, lengths)
    out.sum().backward()
    assert x.grad is not None

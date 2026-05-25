import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class LunarLanderConv(nn.Module):
    def __init__(
        self,
        conv_channels: tuple[int, int] = (32, 64),
        embed_dim: int = 128,
        hidden_size: int = 128,
        use_pooling: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.use_pooling = use_pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=conv_channels[0], kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=3,
            padding=1,
        )
        linear_dim = conv_channels[1] * 84 * 84
        if use_pooling:
            linear_dim //= 4 * 4
        self.linear_embed = nn.Linear(in_features=linear_dim, out_features=embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=2,  # stacked LSTMs
            batch_first=True,
            dropout=dropout_rate,
        )
        self.output_layer = nn.Linear(hidden_size, 1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C, H, W = x.shape
        x = x.reshape(B * N, C, H, W)
        x = self.conv1(x)
        x = F.relu(x)
        if self.use_pooling:
            x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        if self.use_pooling:
            x = self.pool(x)
        x = x.reshape(B * N, -1)  # flatten
        x = F.relu(self.linear_embed(x))
        x = self.dropout(x)
        x = x.reshape(B, N, -1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: (num_layers, batch, hidden_size)
        # take the last layer's hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.output_layer(last_hidden)  # (batch, 1)

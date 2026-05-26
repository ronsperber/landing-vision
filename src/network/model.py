"""
module with network used to make
predictions on lunar lander videos
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class LunarLanderConv(nn.Module):
    """
    class with network used to train on videos
    """

    def __init__(
        self,
        conv_channels: tuple[int, int] = (32, 64),
        embed_dim: int = 128,
        hidden_size: int = 128,
        use_pooling: bool = True,
        dropout_rate: float = 0.0,
    ):
        """
        Parameters
        ----------
        conv_channels: tuple[int, int]
            number of channels to use in the pair of conv layers
        embed_dim: int
            dimension of linear layer where conv output is embedded
        hidden_size : int
            hidden_size for LSTM layer
        use_pooling : bool
            whether or not to use max pooling between conv layers
        dropout_rate: float
            rate to use in dropout layer and dropout between LSTM layers
        """
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
        """
        Forward pass for the network.
        Parameters
        ----------
        x : torch.Tensor
            batch of videos with shape (B, N, C, H, W)
        lengths : torch.Tensor
            actual length of each video in the batch, used for packed padding
        Returns
        -------
        torch.Tensor
            predicted scalar per sample, shape (B, 1)
        """
        # extract batch, number of frames, channels, height, width
        B, N, C, H, W = x.shape
        # reshape to have an B*N sized batch of images
        x = x.reshape(B * N, C, H, W)
        # pass images through convolution and optional pooling
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
        # reshape to batch, length, embedding
        x = x.reshape(B, N, -1)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n shape: (num_layers, batch, hidden_size)
        # take the last layer's hidden state
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.output_layer(last_hidden)  # (batch, 1)

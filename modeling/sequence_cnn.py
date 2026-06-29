"""sequence_cnn.py

Comments : 1D CNN branch for the siRNA sequence + chemistry channels of the
           CMsiRNA model. Updated Conv1dBlock with LayerNorm instead of
           batch normalization.
"""

import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):
    """Conv -> LayerNorm -> activation -> Dropout."""

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        # LayerNorm takes the number of channels
        self.norm = nn.LayerNorm(out_channels)
        self.act = activation()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = x.permute(0, 2, 1)

        return self.drop(self.act(x))


class SiRNASequenceCNN(nn.Module):
    """1D CNN that turns the siRNA seq+chem tensor into a multi-scale feature vector.

    Skip connections from all three conv layers are concatenated (torch.cat) into the output.
    in_channels is data-dependent (2 * D, where the sequence one-hot width D varies with which nonstandard bases
    occur in the data), so it is passed in and inferred from the assembled tensor.
    """

    def __init__(self, in_channels, channels=(32, 64, 128),
                 kernel_sizes=(3, 5, 7), dropout=0.3, activation=nn.ReLU):
        super().__init__()

        self.input_norm = nn.BatchNorm1d(in_channels)

        # in_channels of each layer is the out_channels of the previous one;
        # only the first layer's in_channels is fixed by the data (= 2 * D)
        self.conv1 = Conv1dBlock(in_channels, channels[0], kernel_sizes[0], dropout, activation)
        self.conv2 = Conv1dBlock(channels[0], channels[1], kernel_sizes[1], dropout, activation)
        self.conv3 = Conv1dBlock(channels[1], channels[2], kernel_sizes[2], dropout, activation)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # concat of the three pooled layer outputs -> the head sizes itself to this
        self.out_dim = sum(channels)

    def forward(self, x):
        # x: (N, in_channels = 2 * D, seq_len)
        x = self.input_norm(x)

        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)

        # skip connections from all three layers to the final MLP (or an attempt at least)
        return torch.cat(
            [
                self.pool(h1).flatten(1),
                self.pool(h2).flatten(1),
                self.pool(h3).flatten(1),
            ],
            dim=1,
        )  # (N, sum(channels))

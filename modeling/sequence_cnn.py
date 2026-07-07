"""sequence_cnn.py

Comments : 1D CNN branch for the siRNA sequence + chemistry channels of the
           CMsiRNA model. Guide (antisense) and passenger (sense) strands are
           concatenated along the encoding/channel axis, so the input is a
           tensor of (N, 2 * D, seq_len) where D = sequence + acid + sugar +
           linker one-hot widths per strand. Three conv layers (kernel sizes
           3 -> 5 -> 7) are run in sequence; the output of each layer is global
           average pooled and the three pooled vectors are concatenated
           (torch.cat) to form skip connections from all three layers into the
           final MLP (see crew_model.py).
"""

import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):
    """Conv -> LayerNorm -> activation -> Dropout.
    Padding is kernel_size // 2 so the length is preserved.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.3, activation=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(out_channels)
        self.act = activation()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        return self.drop(self.act(x))


class SiRNASequenceCNN(nn.Module):
    """1D CNN that turns the siRNA seq+chem tensor into a multi-scale feature vector.

    Skip connections from all three conv layers are concatenated (torch.cat) into the output.
    """

    def __init__(self, in_channels, channels=(32, 64, 128),
                 kernel_sizes=(3, 5, 7), dropout=0.3, activation=nn.ReLU):
        super().__init__()

        self.input_norm = nn.LayerNorm(in_channels)

        self.conv1 = Conv1dBlock(in_channels, channels[0], kernel_sizes[0], dropout, activation)
        self.conv2 = Conv1dBlock(channels[0], channels[1], kernel_sizes[1], dropout, activation)
        self.conv3 = Conv1dBlock(channels[1], channels[2], kernel_sizes[2], dropout, activation)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_dim = sum(channels)

    def forward(self, x):
        # x: (N, in_channels, seq_len)
        x = x.permute(0, 2, 1)   # (N, C, L) -> (N, L, C)
        x = self.input_norm(x)
        x = x.permute(0, 2, 1)   # (N, L, C) -> (N, C, L)

        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)

        return torch.cat(
            [
                self.pool(h1).flatten(1),
                self.pool(h2).flatten(1),
                self.pool(h3).flatten(1),
            ],
            dim=1,
        )  # (N, sum(channels))
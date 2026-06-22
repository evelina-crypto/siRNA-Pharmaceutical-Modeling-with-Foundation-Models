"""experimental_mlp.py

Comments : MLP branch for the experimental conditions of the CMsiRNA model.

"""

import torch.nn as nn


class ExperimentalMLP(nn.Module):
    """Projects experimental condition features (concentration, time, cell type)
    into a fixed-size embedding for fusion with the sequence branch."""

    def __init__(self, input_dim, embedding_dim=64, dropout=0.3, activation=nn.ReLU, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            activation(),
            nn.Dropout(dropout),

            nn.Linear(16, 64),
            activation(),
            nn.Dropout(dropout),

            nn.Linear(64, embedding_dim),
            activation(),
        )

    def forward(self, x):
        return self.net(x)
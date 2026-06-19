"""experimental_mlp.py

Comments : MLP branch for the experimental conditions of the CMsiRNA model.

"""

import torch.nn as nn


class ExperimentalMLP(nn.Module):
    """Placeholder: single linear projection to the embedding size."""

    def __init__(self, input_dim, embedding_dim=64, **_):
        super().__init__()
        self.net = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.net(x)
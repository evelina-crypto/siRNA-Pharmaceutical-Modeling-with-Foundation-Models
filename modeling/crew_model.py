"""crew_model.py

Comments : Branches run in parallel and each emit a feature
           vector: the sequence CNN (sequence + chemistry channels, with skip
           connections from all three conv layers), the experimental-conditions
           MLP, and an mRNA foundation model (both yet to be implemented). The active branch outputs are
           concatenated and passed into a 2-layer (64-dim hidden) fusion MLP
           head that predicts the single inhibition value.

           use_experimental toggles the experimental branch, so the model can
           run sequence-only or sequence+experimental.
           The mRNA branch is disabled by default (mrna_embedding_dim=0).
"""

import torch
import torch.nn as nn

from modeling.experimental_mlp import ExperimentalMLP
from modeling.sequence_cnn import SiRNASequenceCNN


class MRNAFMEncoder(nn.Module):
    """Placeholder for the mRNA foundation model."""

    def __init__(self, input_dim, embedding_dim=64, **_):
        super().__init__()
        self.net = nn.Linear(input_dim, embedding_dim)

    def forward(self, x):
        return self.net(x)


class CrewSiRNAModel(nn.Module):
    """Multi-branch siRNA inhibition model with a 2-layer fusion MLP head.

    one hidden layer of ``fusion_hidden`` units feeding a single linear output. The output
    is linear (regression on inhibition); With use_experimental=False the model is sequence-only and forward accepts a
    single positional ``x_seq``, so it is compatible with the single-input
    train_model / evaluate_model helpers in modeling.training_utils.
    """

    def __init__(self, seq_in_channels, exp_input_dim=None, use_experimental=True,
                 emb_dim=64, fusion_hidden=64, mrna_input_dim=None,
                 mrna_embedding_dim=0, dropout=0.3, activation=nn.ReLU):
        super().__init__()

        self.seq_cnn = SiRNASequenceCNN(
            in_channels=seq_in_channels, dropout=dropout, activation=activation,
        )
        fused_dim = self.seq_cnn.out_dim

        # experimental-conditions branch placeholder
        self.use_experimental = use_experimental
        if use_experimental:
            assert exp_input_dim is not None, "exp_input_dim required when use_experimental=True"
            self.exp_mlp = ExperimentalMLP(
                input_dim=exp_input_dim, embedding_dim=emb_dim, dropout=dropout,
            )
            fused_dim += emb_dim

        # mRNA foundation-model branch placeholder
        if mrna_embedding_dim > 0 and mrna_input_dim is not None:
            self.mrna_encoder = MRNAFMEncoder(
                input_dim=mrna_input_dim, embedding_dim=mrna_embedding_dim,
                dropout=dropout, activation=activation,
            )
            fused_dim += mrna_embedding_dim
        else:
            self.mrna_encoder = None

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fusion_hidden),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1),
        )

    def forward(self, x_seq, x_exp=None, x_mrna=None):
        # x_seq: (N, 2 * D, seq_len); x_exp: (N, exp_input_dim)
        parts = [self.seq_cnn(x_seq)]

        if self.use_experimental and x_exp is not None:
            parts.append(self.exp_mlp(x_exp))

        if self.mrna_encoder is not None and x_mrna is not None:
            parts.append(self.mrna_encoder(x_mrna))

        fused = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return self.head(fused)  # (N, 1)
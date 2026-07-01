"""model_attribution.py

Comments : Integrated Gradients (captum) attribution for the multi-input
           CrewSiRNAModel (sequence tensor + experimental vector). IG is run
           with a tuple of inputs and returns one attribution map per branch.
"""

import os

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients


class ModelExplainer:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(model)

    def create_explainability_matrix(self, dataloader, seq_channel_names=None,
                                     exp_feature_names=None, n_steps=50):
        """Compute IG attributions for every sample in the dataloader.

        Expects IndexedMultiTensorDataset batches: (X_seq, X_exp, y, sample_ids).

        Returns dict with sample_ids, seq_raw (N, C, L), exp (N, exp_dim).
        """
        self.model.eval()
        seq_chunks, exp_chunks, all_ids = [], [], []

        print("Computing Integrated Gradients attributions...")
        for batch_idx, (X_seq, X_exp, _, sample_ids) in enumerate(dataloader):
            X_seq = X_seq.to(self.device).float().requires_grad_(True)
            X_exp = X_exp.to(self.device).float().requires_grad_(True)

            attr_seq, attr_exp = self.ig.attribute(
                (X_seq, X_exp),
                baselines=(torch.zeros_like(X_seq), torch.zeros_like(X_exp)),
                target=0, n_steps=n_steps,
            )
            seq_chunks.append(attr_seq.detach().cpu().numpy())
            exp_chunks.append(attr_exp.detach().cpu().numpy())
            all_ids.extend(list(sample_ids))
            print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

        return {
            "sample_ids": all_ids,
            "seq_raw": np.vstack(seq_chunks),
            "exp": np.vstack(exp_chunks),
            "seq_channel_names": seq_channel_names,
            "exp_feature_names": exp_feature_names,
        }

    @staticmethod
    def save_attributions(result, prefix):
        """Save raw sequence tensor and experimental attributions."""
        os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".", exist_ok=True)

        seq_raw = result["seq_raw"]
        np.save(f"{prefix}_seq.npy", seq_raw)
        print(f"Saved sequence attributions -> {prefix}_seq.npy (shape {seq_raw.shape})")

        exp_names = result.get("exp_feature_names")
        n_features = result["exp"].shape[1]
        if exp_names is None or len(exp_names) != n_features:
            exp_names = [f"exp{i}" for i in range(n_features)]
        exp_df = pd.DataFrame(result["exp"], index=result["sample_ids"], columns=exp_names)
        exp_df.to_csv(f"{prefix}_exp.csv")
        print(f"Saved experimental attributions -> {prefix}_exp.csv (shape {exp_df.shape})")

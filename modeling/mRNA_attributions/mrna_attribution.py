"""mrna_attribution.py

integrated-gradients attribution of the mRNA branch of the three-branch CrewSiRNAModel(siRNA, experimental, mRNA branches). 
it attributes all three inputs at once, so we can ask how much of the attribution the mRNA
branch carries next to the other two and, within it, which region the model leans on
(binding site, 5' start, 3'UTR)
"""

import numpy as np
import pandas as pd


def integrated_gradients(model, X_seq, X_exp, X_mrna, mrna_mask, device="cpu",
                         n_steps=50, batch_size=64):
    """how much each input (siRNA, experimental, mRNA) contributed to the predicted
    knockdown, via integrated gradients. IG compares the real input to an all-zeros
    baseline, which for the mRNA branch is exactly how the model represents a missing
    region. a masked (missing) region gets no attribution.

    returns attr_seq (N, C, L), attr_exp (N, exp_dim), attr_mrna (N, n_regions, dim).
    """
    import torch
    from captum.attr import IntegratedGradients

    model.eval().to(device)
    ig = IntegratedGradients(model)

    attr_seq, attr_exp, attr_mrna = [], [], []
    for lo in range(0, len(X_seq), batch_size):
        hi = lo + batch_size
        x_seq = torch.tensor(X_seq[lo:hi], dtype=torch.float32, device=device)
        x_exp = torch.tensor(X_exp[lo:hi], dtype=torch.float32, device=device)
        x_mrna = torch.tensor(X_mrna[lo:hi], dtype=torch.float32, device=device)
        mask = torch.tensor(mrna_mask[lo:hi], dtype=torch.float32, device=device)

        a_seq, a_exp, a_mrna = ig.attribute(
            inputs=(x_seq, x_exp, x_mrna),
            baselines=(torch.zeros_like(x_seq), torch.zeros_like(x_exp),
                       torch.zeros_like(x_mrna)),
            additional_forward_args=(mask,),
            target=0, n_steps=n_steps,
        )
        attr_seq.append(a_seq.detach().cpu().numpy())
        attr_exp.append(a_exp.detach().cpu().numpy())
        attr_mrna.append(a_mrna.detach().cpu().numpy())

    return (np.concatenate(attr_seq), np.concatenate(attr_exp),
            np.concatenate(attr_mrna))


def branch_share(attr_seq, attr_exp, attr_mrna):
    """what fraction of the total attribution each branch (siRNA, experimental, mRNA) gets.
    """
    seq = np.abs(attr_seq).reshape(len(attr_seq), -1).sum(axis=1)
    exp = np.abs(attr_exp).reshape(len(attr_exp), -1).sum(axis=1)
    mrna = np.abs(attr_mrna).reshape(len(attr_mrna), -1).sum(axis=1)

    totals = np.stack([seq, exp, mrna], axis=1)
    share = (totals / totals.sum(axis=1, keepdims=True)).mean(axis=0)
    return pd.DataFrame({"branch": ["siRNA", "experimental", "mRNA"], "share": share})


def region_share(attr_mrna, mrna_mask,
                 region_names=("binding", "five_prime", "three_prime")):
    """which region the model leans on.
    """
    per_region = np.abs(attr_mrna).sum(axis=2)  #(N, n_regions)
    present = mrna_mask.astype(bool)

    means = []
    for j in range(per_region.shape[1]):
        col = per_region[present[:, j], j]
        means.append(col.mean() if len(col) else np.nan)
    means = np.array(means)

    return pd.DataFrame({"region": list(region_names), "mean_abs_attr": means,
                         "share": means / np.nansum(means)})


def region_attr_per_row(attr_mrna):
    """how much the model used each region, one number per row, so we can correlate it with that region's GC and MFE.
"""
    
    return np.abs(attr_mrna).sum(axis=2)  # (N, n_regions)

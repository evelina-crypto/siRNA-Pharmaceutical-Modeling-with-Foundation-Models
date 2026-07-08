"""run_mrna_attribution.py

mRNA-branch attribution.
Loads the arrays, per-row slices and fold weights that run_crew_mrna wrote,
attributes each fold on its own held-out rows (so pooling gives one out-of-fold attribution
per sample), and reports the branch and region shares plus how each region's attribution
correlates with its GC and MFE features. Everything is loaded from --save-dir, so the rows
line up with training without re-embedding.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch

from modeling.crew_model import CrewSiRNAModel
from modeling.mRNA_attributions.mrna_attribution import (
    integrated_gradients, branch_share, region_share, region_attr_per_row,
)
from utils.mrna_features.gc_content_mfe import (
    add_binding_features, add_utr_gc, spearman_vs_inhibition,
)


def run_mrna_attribution(save_dir, out_dir=None, n_steps=50, with_mfe=True, device=None):
    out_dir = out_dir or save_dir
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    arrays = np.load(os.path.join(save_dir, "arrays.npz"), allow_pickle=True)
    X_seq, X_exp = arrays["X_seq"], arrays["X_exp"]
    X_mrna, mrna_mask = arrays["X_mrna"], arrays["mrna_mask"]
    rows = pd.read_csv(os.path.join(save_dir, "rows.csv"))

    fold_paths = sorted(glob.glob(os.path.join(save_dir, "fold*.pt")))
    if not fold_paths:
        raise FileNotFoundError(f"no fold*.pt checkpoints in {save_dir}")

    # scatter each fold's attribution back into full-length arrays, so every row gets
    # exactly one attribution from the fold that held it out
    attr_seq = np.zeros_like(X_seq)
    attr_exp = np.zeros_like(X_exp)
    attr_mrna = np.zeros_like(X_mrna)
    for path in fold_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        test_idx = ckpt["test_idx"]

        model = CrewSiRNAModel(**ckpt["model_kwargs"]).to(device)
        model.load_state_dict(ckpt["state_dict"])

        a_seq, a_exp, a_mrna = integrated_gradients(
            model, X_seq[test_idx], X_exp[test_idx], X_mrna[test_idx],
            mrna_mask[test_idx], device=device, n_steps=n_steps,
        )
        attr_seq[test_idx], attr_exp[test_idx], attr_mrna[test_idx] = a_seq, a_exp, a_mrna
        print(f"{os.path.basename(path)}: attributed {len(test_idx)} held-out rows")

    branches = branch_share(attr_seq, attr_exp, attr_mrna)
    regions = region_share(attr_mrna, mrna_mask)
    print("\nBranch share:")
    print(branches.to_string(index=False))
    print("\nmRNA region share:")
    print(regions.to_string(index=False))
    branches.to_csv(os.path.join(out_dir, "branch_share.csv"), index=False)
    regions.to_csv(os.path.join(out_dir, "region_share.csv"), index=False)

    # how much the model leaned on each region per row, correlated against that region's
    # local features. binding features have within-gene spread; the UTR features are
    # gene-constant, so only their overall (between-gene) correlation is meaningful
    region_attr = region_attr_per_row(attr_mrna)
    rows["binding_attr"] = region_attr[:, 0]
    rows["five_attr"] = region_attr[:, 1]
    rows["three_attr"] = region_attr[:, 2]
    rows = add_binding_features(rows, with_mfe=with_mfe)
    rows = add_utr_gc(rows)

    binding_cols = ["binding_gc", "binding_gc_start", "binding_gc_mid", "binding_gc_end"]
    if with_mfe:
        binding_cols.append("binding_mfe")
    print("\nBinding-region attribution vs its GC/MFE features:")
    binding_corr = spearman_vs_inhibition(rows, binding_cols, target_column="binding_attr")
    print(binding_corr.to_string(index=False))
    binding_corr.to_csv(os.path.join(out_dir, "binding_attr_vs_features.csv"), index=False)

    np.savez(os.path.join(out_dir, "mrna_attr.npz"), attr_mrna=attr_mrna, region_attr=region_attr)
    print(f"\nSaved shares, feature correlation and mrna_attr.npz to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="mRNA-branch attribution on the run_crew_mrna --save-dir folds",
    )
    parser.add_argument("--save-dir", required=True,
                        help="run_crew_mrna --save-dir with arrays.npz, rows.csv, fold*.pt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--no-mfe", action="store_true", help="skip the Vienna MFE feature")
    args = parser.parse_args()

    run_mrna_attribution(save_dir=args.save_dir, out_dir=args.out_dir,
                         n_steps=args.n_steps, with_mfe=not args.no_mfe)


if __name__ == "__main__":
    main()

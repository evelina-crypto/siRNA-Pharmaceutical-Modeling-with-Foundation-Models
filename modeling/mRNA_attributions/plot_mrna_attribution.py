"""plot_mrna_attribution.py

makes a figure for the mRNA-branch attribution. how much of the model's attribution each of the three branches gets (siRNA,
experimental, mRNA). and the region share: (within the mRNA branch), how the attribution splits
across the three regions (binding site, 5' start, 3'UTR), with error bars showing how much
it varies across the cross-validation folds.

it reuses what run_mrna_attribution already saved (mrna_attr.npz, branch_share.csv) instead
of recomputing the attribution. it only re-splits the saved values fold by fold to get the error bars.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from modeling.mRNA_attributions.mrna_attribution import region_share

region_labels = {"binding": "binding\n(100 bp)", "five_prime": "5' start", "three_prime": "3'UTR"}


def per_fold_region_share(save_dir):
    """the region share (how the mRNA attribution splits across the binding, 5' and 3'UTR
    regions) computed separately for each cross-validation fold, so we can see how stable
    the split is
    """
    attr_mrna = np.load(os.path.join(save_dir, "mrna_attr.npz"))["attr_mrna"]
    mrna_mask = np.load(os.path.join(save_dir, "arrays.npz"), allow_pickle=True)["mrna_mask"]

    shares = []
    for path in sorted(glob.glob(os.path.join(save_dir, "fold*.pt"))):
        test_idx = torch.load(path, map_location="cpu", weights_only=False)["test_idx"]
        shares.append(region_share(attr_mrna[test_idx], mrna_mask[test_idx])["share"].to_numpy())
    shares = np.stack(shares)  # (n_folds, n_regions)

    regions = region_share(attr_mrna, mrna_mask)["region"].tolist()
    return regions, shares.mean(axis=0), shares.std(axis=0)


def plot(save_dir, out=None):
    out = out or os.path.join(save_dir, "mrna_attribution_shares.png")
    branch = pd.read_csv(os.path.join(save_dir, "branch_share.csv"))
    regions, region_mean, region_std = per_fold_region_share(save_dir)
    labels = [region_labels.get(r, r) for r in regions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(branch["branch"], branch["share"], color="#4c72b0")
    ax1.set_ylabel("attribution share")
    ax1.set_title("Branch share")
    ax1.set_ylim(0, 0.8)
    for i, v in enumerate(branch["share"]):
        ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)

    ax2.bar(labels, region_mean, yerr=region_std, capsize=4, color="#55a868")
    ax2.set_ylabel("attribution share")
    ax2.set_title("mRNA region share (mean +/- SD over folds)")
    ax2.set_ylim(0, 0.8)
    for i, v in enumerate(region_mean):
        ax2.text(i, v + region_std[i] + 0.01, f"{v:.2f}", ha="center", fontsize=10)

    fig.suptitle("Integrated-gradients attribution of the three-branch model")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("saved", out)


def main():
    parser = argparse.ArgumentParser(description="report figure for the mRNA-branch attribution")
    parser.add_argument("--save-dir", required=True,
                        help="run_mrna_attribution --save-dir with mrna_attr.npz, branch_share.csv, fold*.pt")
    parser.add_argument("--out", default=None, help="output png (default: save-dir/mrna_attribution_shares.png)")
    args = parser.parse_args()
    plot(args.save_dir, args.out)


if __name__ == "__main__":
    main()

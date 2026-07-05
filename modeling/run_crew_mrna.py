"""run_crew_mrna.py
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from modeling.crew_model import CrewSiRNAModel
from modeling.training_utils import create_validation_loader, set_global_seed
from modeling.multi_input_training_utils import (
    IndexedMRNADataset, train_model_mrna, evaluate_model_mrna,
)
from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline
from utils.splitter import GroupKFoldLeakPerGroup
from utils.fm_utils import add_slice_columns, build_slice_embeddings, load_cache


def run_mrna_cv(cmsirna_path, historic_path, mrna_cache, n_splits=3, leak_n=0, val_split=0.15,
                batch_size=64, lr=5e-4, epochs=20, patience=5, seed=42, mrna_embedding_dim=64,
                device=None, save_dir=None):
    """Cross-validated run of the full sequence + experimental + mRNA model.

    Split is by gene (leak_n=0 = strict gene split). Returns the per-fold metrics.

    With save_dir set, the trained fold weights and everything the attribution
    step needs (the arrays, the per-row slice strings, the fold split and the
    y-scaler) are written out, so run_attributions can rebuild each fold's model.
    """
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # enrich with add_mrna=True so the mRNA/UTR columns and the alignment site exist
    raw_df = load_merged_dataset(cmsirna_path, historic_path)
    pipeline = SiRNADataPipeline(target_len=25)
    enriched = pipeline.enrich_dataset_with_encodings(raw_df, strict_cleaning=False, add_mrna=True)
    enriched = enriched.reset_index(drop=True)

    # sequence + experimental tensors (same rows, same order as enriched)
    X_seq, X_exp, groups, y = pipeline.prepare_for_deep_learning(enriched, target_column="Inhibition")

    # mRNA region embeddings from the cache, built on the same enriched rows so the
    # arrays line up positionally with X_seq / X_exp. three_prime_width=None is the
    # full 3'UTR, which is what the cache was built with, so the slice strings match
    enriched = add_slice_columns(enriched, three_prime_width=None)
    cache = load_cache(mrna_cache)
    X_mrna, mrna_mask, _ = build_slice_embeddings(enriched, model=None, cache=cache)

    # drop rows with a missing target, consistently across every array
    valid = np.isnan(y) == False
    X_seq, X_exp, groups, y = X_seq[valid], X_exp[valid], groups[valid], y[valid]
    X_mrna, mrna_mask = X_mrna[valid], mrna_mask[valid]
    print(f"Usable samples: {len(y)}, genes: {len(np.unique(groups))}, "
          f"seq channels: {X_seq.shape[1]}, exp dim: {X_exp.shape[1]}, "
          f"mrna: {X_mrna.shape[1]} regions x {X_mrna.shape[2]} dim")

    # the arrays and the per-row slice strings, saved once and shared by every
    # fold so the attribution step reads the same rows the model was trained on
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, "arrays.npz"), X_seq=X_seq, X_exp=X_exp,
                 X_mrna=X_mrna, mrna_mask=mrna_mask, y=y, groups=groups)
        slice_cols = ["mRNA_binding_slice", "mRNA_five_slice", "mRNA_three_slice"]
        rows = enriched.loc[valid, ["gene_target_symbol_name", "Inhibition"] + slice_cols]
        rows.reset_index(drop=True).to_csv(os.path.join(save_dir, "rows.csv"), index=False)

    cv = GroupKFoldLeakPerGroup(n_splits=n_splits, leak_n=leak_n, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_seq, y, groups)):
        print(f"\n=== Fold {fold + 1}/{n_splits} "
              f"(train {len(train_idx)}, test {len(test_idx)}) ===")
        generator = torch.Generator().manual_seed(seed + fold)

        # target scaling fit on the training fold only (leakage-free)
        scaler_y = StandardScaler().fit(y[train_idx].reshape(-1, 1))
        y_train = scaler_y.transform(y[train_idx].reshape(-1, 1)).astype(np.float32)
        y_test = scaler_y.transform(y[test_idx].reshape(-1, 1)).astype(np.float32)

        # the mask is float so it can multiply the projections in the model
        train_ds = IndexedMRNADataset(
            torch.tensor(X_seq[train_idx]), torch.tensor(X_exp[train_idx]),
            torch.tensor(X_mrna[train_idx]), torch.tensor(mrna_mask[train_idx].astype(np.float32)),
            torch.tensor(y_train), [str(g) for g in groups[train_idx]],
        )
        test_ds = IndexedMRNADataset(
            torch.tensor(X_seq[test_idx]), torch.tensor(X_exp[test_idx]),
            torch.tensor(X_mrna[test_idx]), torch.tensor(mrna_mask[test_idx].astype(np.float32)),
            torch.tensor(y_test), [str(g) for g in groups[test_idx]],
        )

        train_loader, val_loader = create_validation_loader(
            train_ds, val_split=val_split, batch_size=batch_size, generator=generator,
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = CrewSiRNAModel(
            seq_in_channels=X_seq.shape[1],
            exp_input_dim=X_exp.shape[1],
            use_experimental=True,
            mrna_input_dim=X_mrna.shape[2],
            mrna_embedding_dim=mrna_embedding_dim,
            mrna_n_regions=X_mrna.shape[1],
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model, _ = train_model_mrna(model, train_loader, val_loader, criterion, optimizer,
                                    epochs=epochs, device=device, patience=patience)

        metrics, _, _, _ = evaluate_model_mrna(scaler_y, model, test_loader, device)
        print("Fold metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        fold_metrics.append(metrics)

        # save the trained fold so run_attributions can rebuild it. model_kwargs
        # is everything CrewSiRNAModel needs; the y-scaler and the split come
        # along so attribution runs on exactly this fold's held-out rows.
        if save_dir:
            model_kwargs = dict(seq_in_channels=X_seq.shape[1], exp_input_dim=X_exp.shape[1],
                                use_experimental=True, mrna_input_dim=X_mrna.shape[2],
                                mrna_embedding_dim=mrna_embedding_dim, mrna_n_regions=X_mrna.shape[1])
            torch.save({"state_dict": model.state_dict(), "model_kwargs": model_kwargs,
                        "train_idx": train_idx, "test_idx": test_idx,
                        "scaler_mean": scaler_y.mean_, "scaler_scale": scaler_y.scale_},
                       os.path.join(save_dir, f"fold{fold}.pt"))

    print("\n=== Cross-validation summary ===")
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        print(f"{key}: mean={np.nanmean(values):.4f} +/- {np.nanstd(values):.4f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Full run of CrewSiRNAModel with the mRNA Orthrus branch")
    parser.add_argument("--cmsirna-path", required=True)
    parser.add_argument("--historic-path", required=True)
    parser.add_argument("--mrna-cache", required=True,
                        help="npz of Orthrus slice embeddings saved by fm_utils.save_cache")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--leak-n", type=int, default=0, help="0 = strict split by gene")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mrna-embedding-dim", type=int, default=64)
    parser.add_argument("--save-dir", default=None,
                        help="directory to write per-fold weights + arrays for the attribution step")
    args = parser.parse_args()

    run_mrna_cv(cmsirna_path=args.cmsirna_path, historic_path=args.historic_path,
                mrna_cache=args.mrna_cache, n_splits=args.n_splits, leak_n=args.leak_n,
                val_split=args.val_split, batch_size=args.batch_size, lr=args.lr,
                epochs=args.epochs, patience=args.patience, seed=args.seed,
                mrna_embedding_dim=args.mrna_embedding_dim, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

"""run_crew.py

Comments : Full training run for the siRNA sequence CNN + final MLP head.
           Loads the merged dataset with the usual (non-strict) cleaner, shapes
           it with prepare_for_deep_learning, splits by gene (GroupKFold), scales
           the target per training fold, and trains/evaluates the
           CrewSiRNAModel (sequence + experimental conditions) with Adam + MSE.
           Uses the multi-input train/eval helpers in
           modeling.multi_input_training_utils since use_experimental=True.
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
from modeling.multi_input_training_utils import IndexedMultiTensorDataset, train_model_multi, evaluate_model_multi
from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline
from utils.splitter import GroupKFoldLeakPerGroup

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CMSIRNA_PATH = os.path.join(REPO_ROOT, "dataset", "primary_dataset", "CMsiRNA_data_update.tsv")
DEFAULT_HISTORIC_PATH = os.path.join(REPO_ROOT, "dataset", "Historic_Takayuki_hueskan_ichihara.csv")


def run_sequence_cnn_cv(cmsirna_path=DEFAULT_CMSIRNA_PATH, historic_path=DEFAULT_HISTORIC_PATH, n_splits=3, leak_n=0,
                        val_split=0.15, batch_size=64, lr=5e-4, epochs=20, patience=5, seed=42, max_rows=None,
                        device=None):
    """Cross-validated full run of the sequence + experimental CrewSiRNAModel.

    Split is by gene (leak_n=0 = strict gene split). Returns the list of per-fold
    metric dicts.
    """

    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and enrich with the usual (non-strict) cleaner, no mRNA branch
    raw_df = load_merged_dataset(cmsirna_path, historic_path)
    if max_rows is not None:
        raw_df = raw_df.sample(n=min(max_rows, len(raw_df)), random_state=seed).reset_index(drop=True)

    pipeline = SiRNADataPipeline(target_len=25)
    enriched = pipeline.enrich_dataset_with_encodings(raw_df, strict_cleaning=False, add_mrna=False, )
    X_seq, X_exp, groups, y = pipeline.prepare_for_deep_learning(enriched, target_column="Inhibition")

    # Drop rows with a missing target
    valid = ~np.isnan(y)
    X_seq, X_exp, groups, y = X_seq[valid], X_exp[valid], groups[valid], y[valid]
    print(f"Usable samples: {len(y)}, genes: {len(np.unique(groups))}, "
          f"seq channels: {X_seq.shape[1]}, exp dim: {X_exp.shape[1]}")

    seq_in_channels = X_seq.shape[1]
    exp_input_dim = X_exp.shape[1]
    cv = GroupKFoldLeakPerGroup(n_splits=n_splits, leak_n=leak_n, random_state=seed)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_seq, y, groups)):
        print(f"\n=== Fold {fold + 1}/{n_splits} "
              f"(train {len(train_idx)}, test {len(test_idx)}) ===")
        generator = torch.Generator().manual_seed(seed + fold)

        # Target scaling fit on the training fold only (leakage-free)
        scaler_y = StandardScaler().fit(y[train_idx].reshape(-1, 1))
        y_train = scaler_y.transform(y[train_idx].reshape(-1, 1)).astype(np.float32)
        y_test = scaler_y.transform(y[test_idx].reshape(-1, 1)).astype(np.float32)

        train_ds = IndexedMultiTensorDataset(
            torch.tensor(X_seq[train_idx]), torch.tensor(X_exp[train_idx]), torch.tensor(y_train),
            [str(g) for g in groups[train_idx]],
        )
        test_ds = IndexedMultiTensorDataset(
            torch.tensor(X_seq[test_idx]), torch.tensor(X_exp[test_idx]), torch.tensor(y_test),
            [str(g) for g in groups[test_idx]],
        )

        train_loader, val_loader = create_validation_loader(train_ds, val_split=val_split, batch_size=batch_size,
            generator=generator, )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Sequence + experimental model (multi-input)
        model = CrewSiRNAModel(
            seq_in_channels=seq_in_channels,
            exp_input_dim=exp_input_dim,
            use_experimental=True,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model, history = train_model_multi(model, train_loader, val_loader, criterion, optimizer, epochs=epochs,
            device=device, patience=patience, )

        metrics, _, _, _ = evaluate_model_multi(scaler_y, model, test_loader, device)
        print("Fold metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        fold_metrics.append(metrics)

    # 5. Aggregate
    print("\n=== Cross-validation summary ===")
    for key in fold_metrics[0]:
        values = np.array([m[key] for m in fold_metrics], dtype=float)
        print(f"{key}: mean={np.nanmean(values):.4f} +/- {np.nanstd(values):.4f}")

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(description="Full run of the siRNA sequence CNN + experimental MLP + fusion head")
    parser.add_argument("--cmsirna-path", default=DEFAULT_CMSIRNA_PATH)
    parser.add_argument("--historic-path", default=DEFAULT_HISTORIC_PATH)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--leak-n", type=int, default=0, help="0 = strict split by gene")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="cap rows for a quick check")
    args = parser.parse_args()

    run_sequence_cnn_cv(cmsirna_path=args.cmsirna_path, historic_path=args.historic_path, n_splits=args.n_splits,
        leak_n=args.leak_n, val_split=args.val_split, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        patience=args.patience, seed=args.seed, max_rows=args.max_rows, )


if __name__ == "__main__":
    main()
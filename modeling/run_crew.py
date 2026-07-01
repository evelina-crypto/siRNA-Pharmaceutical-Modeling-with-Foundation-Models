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
import pandas as pd
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
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DEFAULT_WEIGHTS_DIR = os.path.join(DEFAULT_RESULTS_DIR, "weights")
DEFAULT_ATTRIBUTIONS_DIR = os.path.join(DEFAULT_RESULTS_DIR, "attributions")
DEFAULT_ATTRIBUTION_PLOTS_DIR = os.path.join(DEFAULT_ATTRIBUTIONS_DIR, "attribution_plots")


def run_sequence_cnn_cv(cmsirna_path=DEFAULT_CMSIRNA_PATH, historic_path=DEFAULT_HISTORIC_PATH, n_splits=3, leak_n=0,
                        val_split=0.15, batch_size=64, lr=5e-4, epochs=20, patience=5, seed=42, max_rows=None,
                        device=None, save_weights=True, weights_dir=DEFAULT_WEIGHTS_DIR,
                        attribution=False, attributions_dir=DEFAULT_ATTRIBUTIONS_DIR):
    """Cross-validated full run of the sequence + experimental CrewSiRNAModel.

    Split is by gene (leak_n=0 = strict gene split). Returns the list of per-fold
    metric dicts.

    For each CV split the trained model (best-epoch, early-stopping-restored) is
    saved with the metadata needed to reload the exact test split. When
    attribution=True, Integrated Gradients maps are also computed on the fold's
    test set and saved per branch (sequence + experimental).
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

    if save_weights:
        os.makedirs(weights_dir, exist_ok=True)

    # readable names (in tensor order) for the attribution maps
    seq_channel_names = exp_feature_names = None
    fold_attr_seq, fold_attr_exp, fold_X_seq, fold_X_exp, fold_sample_ids = [], [], [], [], []
    if attribution:
        os.makedirs(attributions_dir, exist_ok=True)
        seq_channel_names = pipeline.build_sequence_channel_names(enriched)
        exp_feature_names = pipeline.build_experimental_feature_names(enriched)

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
        # best epoch val loss (early stopping restores these weights) and last epoch val loss
        metrics["best_loss"] = float(np.min(history["val_loss"])) if history["val_loss"] else float("nan")
        metrics["final_val_loss"] = float(history["val_loss"][-1]) if history["val_loss"] else float("nan")
        print("Fold metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        fold_metrics.append(metrics)

        # Save the (best-epoch, early-stopping-restored) model plus the metadata
        # needed to reproduce this fold's test split for attribution.
        if save_weights:
            ckpt_path = os.path.join(weights_dir, f"crew_seed{seed}_fold{fold}.pt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "fold": fold,
                    "seq_in_channels": seq_in_channels,
                    "exp_input_dim": exp_input_dim,
                    "use_experimental": True,
                    "test_idx": np.asarray(test_idx),
                    "scaler_y_mean": scaler_y.mean_,
                    "scaler_y_scale": scaler_y.scale_,
                },
                ckpt_path,
            )
            print(f"Saved fold weights -> {ckpt_path}")

        # Integrated Gradients attribution on the fold's in-memory test set.
        if attribution:
            from modeling.model_attribution import ModelExplainer
            from modeling.attribution_plots import save_all_attribution_plots

            explainer = ModelExplainer(model, device=device)
            attr = explainer.create_explainability_matrix(
                test_loader, seq_channel_names=seq_channel_names, exp_feature_names=exp_feature_names,
            )
            prefix = os.path.join(attributions_dir, f"seed{seed}_fold{fold}")
            explainer.save_attributions(attr, prefix)

            fold_attr_seq.append(attr["seq_raw"])
            fold_attr_exp.append(attr["exp"])
            fold_X_seq.append(X_seq[test_idx])
            fold_X_exp.append(X_exp[test_idx])
            fold_sample_ids.extend(attr["sample_ids"])

            plot_dir = os.path.join(
                DEFAULT_ATTRIBUTION_PLOTS_DIR, "per_fold", f"seed{seed}_fold{fold}",
            )
            save_all_attribution_plots(
                attr["seq_raw"], attr["exp"], X_seq[test_idx], X_exp[test_idx],
                attr["sample_ids"], seq_channel_names, exp_feature_names, plot_dir,
            )

    # Pool fold test attributions into one dataset-wide map (each sample once).
    if attribution and fold_attr_seq:
        pooled_seq = np.concatenate(fold_attr_seq, axis=0)
        pooled_exp = np.concatenate(fold_attr_exp, axis=0)
        pooled_X_seq = np.concatenate(fold_X_seq, axis=0)
        pooled_X_exp = np.concatenate(fold_X_exp, axis=0)

        np.save(os.path.join(attributions_dir, "pooled_seq.npy"), pooled_seq)
        pd.DataFrame(pooled_exp, index=fold_sample_ids, columns=exp_feature_names).to_csv(
            os.path.join(attributions_dir, "pooled_exp.csv"),
        )
        print(f"Saved pooled attributions -> {attributions_dir}/pooled_seq.npy, pooled_exp.csv")

        save_all_attribution_plots(
            pooled_seq, pooled_exp, pooled_X_seq, pooled_X_exp, fold_sample_ids,
            seq_channel_names, exp_feature_names,
            os.path.join(DEFAULT_ATTRIBUTION_PLOTS_DIR, "pooled"),
        )

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
    parser.add_argument("--no-save-weights", action="store_true", help="do not save per-fold model weights")
    parser.add_argument("--attribution", action="store_true",
                        help="compute and save Integrated Gradients maps per fold")
    args = parser.parse_args()

    run_sequence_cnn_cv(cmsirna_path=args.cmsirna_path, historic_path=args.historic_path, n_splits=args.n_splits,
        leak_n=args.leak_n, val_split=args.val_split, batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
        patience=args.patience, seed=args.seed, max_rows=args.max_rows, save_weights=not args.no_save_weights,
        attribution=args.attribution)


if __name__ == "__main__":
    main()
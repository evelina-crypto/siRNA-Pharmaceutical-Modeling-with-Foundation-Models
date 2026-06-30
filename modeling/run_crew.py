"""Grouped-CV training for the multi-branch Crew siRNA model."""

import argparse
import json
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from modeling.crew_model import CrewSiRNAModel
from modeling.training_utils import create_validation_loader, set_global_seed
from modeling.multi_input_training_utils import (
    IndexedMultiTensorDataset,
    collate_runtime_slices,
    evaluate_model_multi,
    train_model_multi,
)
from utils.fm_utils import (
    add_slice_columns,
    build_slice_embeddings,
    load_cache,
    parse_three_prime_width,
)
from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline
from utils.splitter import GroupKFoldLeakPerGroup

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CMSIRNA_PATH = os.path.join(REPO_ROOT, "dataset", "primary_dataset", "CMsiRNA_data_update.tsv")
DEFAULT_HISTORIC_PATH = os.path.join(REPO_ROOT, "dataset", "Historic_Takayuki_hueskan_ichihara.csv")
DEFAULT_ORTHRUS_MODEL_DIR = os.path.join(REPO_ROOT, "models", "orthrus_v1_4_track")
DEFAULT_ORTHRUS_CHECKPOINT = "epoch=6-step=20000.ckpt"
SLICE_COLUMNS = ["mRNA_binding_slice", "mRNA_five_slice", "mRNA_three_slice"]


def run_sequence_cnn_cv(
        cmsirna_path=DEFAULT_CMSIRNA_PATH,
        historic_path=DEFAULT_HISTORIC_PATH,
        n_splits=3,
        leak_n=0,
        val_split=0.15,
        batch_size=64,
        lr=5e-4,
        epochs=20,
        patience=5,
        seed=42,
        max_rows=None,
        device=None,
        orthrus_cache=None,
        matched_mrna_baseline=False,
        three_prime_width=100,
        orthrus_runtime=False,
        orthrus_model_dir=DEFAULT_ORTHRUS_MODEL_DIR,
        orthrus_checkpoint=DEFAULT_ORTHRUS_CHECKPOINT,
        orthrus_unfreeze_last_n=1,
        orthrus_lr=1e-6,
        gradient_accumulation=1,
        mixed_precision=False,
        max_grad_norm=None,
        results_dir=None,
):
    """Run grouped CV with baseline, cached Orthrus, or runtime Orthrus."""
    if orthrus_cache and orthrus_runtime:
        raise ValueError("--orthrus-cache and --orthrus-runtime are mutually exclusive")
    if orthrus_runtime and three_prime_width != 100:
        raise ValueError("Runtime fine-tuning currently supports --three-prime-width 100 only")
    if orthrus_runtime and batch_size > 8:
        print(f"Runtime Orthrus: reducing batch size from {batch_size} to 4 for GPU safety")
        batch_size = 4
    if orthrus_runtime and results_dir is None:
        run_name = (
            f"orthrus_runtime_3utr100_last{orthrus_unfreeze_last_n}_"
            f"lr{orthrus_lr:g}_seed{seed}"
        )
        results_dir = os.path.join(REPO_ROOT, "results", run_name)

    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    raw_df = load_merged_dataset(cmsirna_path, historic_path)
    if max_rows is not None:
        raw_df = raw_df.sample(
            n=min(max_rows, len(raw_df)), random_state=seed,
        ).reset_index(drop=True)

    needs_mrna = bool(orthrus_cache or orthrus_runtime or matched_mrna_baseline)
    pipeline = SiRNADataPipeline(target_len=25, fetch_missing_mrna=True)
    # A matched baseline performs the same mRNA resolution/alignment filtering
    # as Orthrus without adding Orthrus features, keeping row comparisons fair.
    enriched = pipeline.enrich_dataset_with_encodings(
        raw_df, strict_cleaning=True, add_mrna=needs_mrna,
    )
    X_seq, X_exp, groups, y = pipeline.prepare_for_deep_learning(
        enriched, target_column="Inhibition",
    )

    X_mrna = None
    if orthrus_cache or orthrus_runtime:
        enriched = add_slice_columns(
            enriched, three_prime_width=three_prime_width,
        )
        if orthrus_runtime:
            # Raw strings are encoded by the DataLoader collator every batch;
            # embeddings cannot be cached because Orthrus weights are changing.
            X_mrna = enriched[SLICE_COLUMNS].to_numpy(dtype=object)
            print(f"Runtime Orthrus slices X_mrna shape: {X_mrna.shape}")
        else:
            cached = load_cache(orthrus_cache)
            X_slices, slice_mask, _ = build_slice_embeddings(enriched, cache=cached)
            X_mrna = np.concatenate(
                [X_slices.reshape(len(X_slices), -1), slice_mask.astype(np.float32)],
                axis=1,
            ).astype(np.float32)
            print(f"Cached Orthrus matrix X_mrna shape: {X_mrna.shape}")

    valid = ~np.isnan(y)
    X_seq, X_exp, groups, y = X_seq[valid], X_exp[valid], groups[valid], y[valid]
    if X_mrna is not None:
        X_mrna = X_mrna[valid]
    print(
        f"Usable samples: {len(y)}, genes: {len(np.unique(groups))}, "
        f"seq channels: {X_seq.shape[1]}, exp dim: {X_exp.shape[1]}"
    )

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    seq_in_channels = X_seq.shape[1]
    exp_input_dim = X_exp.shape[1]
    cv = GroupKFoldLeakPerGroup(
        n_splits=n_splits, leak_n=leak_n, random_state=seed,
    )
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_seq, y, groups)):
        print(
            f"\n=== Fold {fold + 1}/{n_splits} "
            f"(train {len(train_idx)}, test {len(test_idx)}) ==="
        )
        generator = torch.Generator().manual_seed(seed + fold)

        scaler_y = StandardScaler().fit(y[train_idx].reshape(-1, 1))
        y_train = scaler_y.transform(y[train_idx].reshape(-1, 1)).astype(np.float32)
        y_test = scaler_y.transform(y[test_idx].reshape(-1, 1)).astype(np.float32)

        def mrna_subset(indices):
            if X_mrna is None:
                return None
            if orthrus_runtime:
                return X_mrna[indices]
            return torch.tensor(X_mrna[indices])

        train_ds = IndexedMultiTensorDataset(
            torch.tensor(X_seq[train_idx]),
            torch.tensor(X_exp[train_idx]),
            torch.tensor(y_train),
            [str(g) for g in groups[train_idx]],
            X_mrna_tensor=mrna_subset(train_idx),
        )
        test_ds = IndexedMultiTensorDataset(
            torch.tensor(X_seq[test_idx]),
            torch.tensor(X_exp[test_idx]),
            torch.tensor(y_test),
            [str(g) for g in groups[test_idx]],
            X_mrna_tensor=mrna_subset(test_idx),
        )

        collate_fn = partial(collate_runtime_slices, width=100) if orthrus_runtime else None
        train_loader, val_loader = create_validation_loader(
            train_ds,
            val_split=val_split,
            batch_size=batch_size,
            generator=generator,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )

        # A new model reloads the original Orthrus checkpoint in every fold,
        # preventing fine-tuned information from crossing fold boundaries.
        model = CrewSiRNAModel(
            seq_in_channels=seq_in_channels,
            exp_input_dim=exp_input_dim,
            use_experimental=True,
            mrna_input_dim=(
                X_mrna.shape[1]
                if X_mrna is not None and not orthrus_runtime else None
            ),
            mrna_embedding_dim=64 if X_mrna is not None else 0,
            orthrus_model_dir=orthrus_model_dir if orthrus_runtime else None,
            orthrus_checkpoint=orthrus_checkpoint if orthrus_runtime else None,
            orthrus_unfreeze_last_n=orthrus_unfreeze_last_n,
        ).to(device)
        criterion = nn.MSELoss()

        if orthrus_runtime:
            backbone_parameters = [
                parameter for parameter in model.mrna_encoder.backbone.parameters()
                if parameter.requires_grad
            ]
            backbone_ids = {id(parameter) for parameter in backbone_parameters}
            crew_parameters = [
                parameter for parameter in model.parameters()
                if parameter.requires_grad and id(parameter) not in backbone_ids
            ]
            parameter_groups = [{
                "params": crew_parameters, "lr": lr, "weight_decay": 0.0,
            }]
            if backbone_parameters:
                parameter_groups.append({
                    "params": backbone_parameters,
                    "lr": orthrus_lr,
                    "weight_decay": 1e-5,
                })
            optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
            print(
                f"Trainable Orthrus parameters: "
                f"{sum(p.numel() for p in backbone_parameters):,}; "
                f"Crew parameters: {sum(p.numel() for p in crew_parameters):,}"
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model, history = train_model_multi(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            epochs=epochs,
            device=device,
            patience=patience,
            gradient_accumulation=gradient_accumulation,
            mixed_precision=mixed_precision,
            max_grad_norm=max_grad_norm,
        )

        metrics, predictions, actuals, sample_names = evaluate_model_multi(
            scaler_y, model, test_loader, device, mixed_precision=mixed_precision,
        )
        print("Fold metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        fold_metrics.append(metrics)

        if results_dir:
            np.savez_compressed(
                os.path.join(results_dir, f"fold_{fold + 1}_predictions.npz"),
                predictions=predictions,
                actuals=actuals,
                sample_ids=np.asarray(sample_names, dtype=object),
            )

        del model, optimizer
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n=== Cross-validation summary ===")
    summary = {}
    for key in fold_metrics[0]:
        values = np.array([metrics[key] for metrics in fold_metrics], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values)),
        }
        print(f"{key}: mean={summary[key]['mean']:.4f} +/- {summary[key]['std']:.4f}")

    if results_dir:
        with open(os.path.join(results_dir, "metrics.json"), "w") as handle:
            json.dump({"folds": fold_metrics, "summary": summary}, handle, indent=2)

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train the Crew siRNA model with optional Orthrus features",
    )
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
    parser.add_argument("--max-rows", type=int, default=None)
    orthus_group = parser.add_mutually_exclusive_group()
    orthus_group.add_argument(
        "--orthrus-cache", default=None,
        help="static slice embedding cache from modeling.build_orthrus_cache",
    )
    orthus_group.add_argument(
        "--orthrus-runtime", action="store_true",
        help="run Orthrus inside training and fine-tune selected final layers",
    )
    parser.add_argument(
        "--three-prime-width", type=parse_three_prime_width, default=100,
        metavar="N|full",
    )
    parser.add_argument("--matched-mrna-baseline", action="store_true")
    parser.add_argument("--orthrus-model-dir", default=DEFAULT_ORTHRUS_MODEL_DIR)
    parser.add_argument("--orthrus-checkpoint", default=DEFAULT_ORTHRUS_CHECKPOINT)
    parser.add_argument("--orthrus-unfreeze-last-n", type=int, default=1)
    parser.add_argument("--orthrus-lr", type=float, default=1e-6)
    parser.add_argument(
        "--gradient-accumulation", type=int, default=None,
        help="default: 16 in runtime mode, otherwise 1",
    )
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    run_sequence_cnn_cv(
        cmsirna_path=args.cmsirna_path,
        historic_path=args.historic_path,
        n_splits=args.n_splits,
        leak_n=args.leak_n,
        val_split=args.val_split,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        max_rows=args.max_rows,
        orthrus_cache=args.orthrus_cache,
        matched_mrna_baseline=args.matched_mrna_baseline,
        three_prime_width=args.three_prime_width,
        orthrus_runtime=args.orthrus_runtime,
        orthrus_model_dir=args.orthrus_model_dir,
        orthrus_checkpoint=args.orthrus_checkpoint,
        orthrus_unfreeze_last_n=args.orthrus_unfreeze_last_n,
        orthrus_lr=args.orthrus_lr,
        gradient_accumulation=(
            args.gradient_accumulation
            if args.gradient_accumulation is not None
            else (16 if args.orthrus_runtime else 1)
        ),
        mixed_precision=args.mixed_precision,
        max_grad_norm=(
            args.max_grad_norm
            if args.max_grad_norm is not None
            else (1.0 if args.orthrus_runtime else None)
        ),
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()

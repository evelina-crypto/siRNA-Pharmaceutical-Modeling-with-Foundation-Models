"""Runtime Orthrus fine-tuning runner for the Crew siRNA model.

This file is intentionally separate from ``run_crew.py`` and
``run_crew_mrna.py``:

* ``run_crew.py`` keeps the clean sequence + experimental baseline.
* ``run_crew_mrna.py`` runs the frozen/static Orthrus cache experiment.
* this runner loads Orthrus inside every CV fold and fine-tunes selected layers.

The model is rebuilt inside each fold, so fine-tuned Orthrus weights never leak
from one held-out gene fold into another.
"""

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
from modeling.multi_input_training_utils import (
    IndexedMultiTensorDataset,
    collate_runtime_slices,
    evaluate_model_multi,
    train_model_multi,
)
from modeling.training_utils import set_global_seed
from utils.fm_utils import add_slice_columns, parse_three_prime_width
from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline
from utils.splitter import GroupKFoldLeakPerGroup


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CMSIRNA_PATH = os.path.join(
    REPO_ROOT, "dataset", "primary_dataset", "CMsiRNA_data_update.tsv",
)
DEFAULT_HISTORIC_PATH = os.path.join(
    REPO_ROOT, "dataset", "Historic_Takayuki_hueskan_ichihara.csv",
)
DEFAULT_ORTHRUS_MODEL_DIR = os.path.join(REPO_ROOT, "models", "orthrus_v1_4_track")
DEFAULT_ORTHRUS_CHECKPOINT = "epoch=6-step=20000.ckpt"
SLICE_COLUMNS = ["mRNA_binding_slice", "mRNA_five_slice", "mRNA_three_slice"]


def grouped_train_val_split(train_idx, groups, val_split=0.15, seed=42):
    """Split the outer training fold into train/validation by gene.

    Runtime fine-tuning can overfit gene-specific mRNA representations. A random
    row-level validation split can share the same gene between train and val,
    making validation look better than the held-out-gene test fold. This helper
    keeps validation genes separate from inner-training genes.
    """
    train_idx = np.asarray(train_idx)
    train_groups = np.asarray(groups)[train_idx]
    unique_groups = np.unique(train_groups)
    if len(unique_groups) < 2:
        raise ValueError("Need at least two genes in the outer train fold for grouped validation")

    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n_val_groups = int(round(len(shuffled) * val_split))
    n_val_groups = max(1, min(len(shuffled) - 1, n_val_groups))
    val_groups = set(shuffled[:n_val_groups])

    is_val = np.array([group in val_groups for group in train_groups])
    inner_train_idx = train_idx[~is_val]
    val_idx = train_idx[is_val]
    if len(inner_train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Grouped validation split produced an empty train or validation fold")
    return inner_train_idx, val_idx


def run_runtime_orthrus_cv(
        cmsirna_path=DEFAULT_CMSIRNA_PATH,
        historic_path=DEFAULT_HISTORIC_PATH,
        n_splits=3,
        leak_n=0,
        val_split=0.15,
        batch_size=4,
        lr=1e-4,
        epochs=20,
        patience=5,
        seed=42,
        max_rows=None,
        device=None,
        three_prime_width=100,
        orthrus_model_dir=DEFAULT_ORTHRUS_MODEL_DIR,
        orthrus_checkpoint=DEFAULT_ORTHRUS_CHECKPOINT,
        orthrus_unfreeze_last_n=1,
        orthrus_lr=3e-7,
        gradient_accumulation=16,
        mixed_precision=False,
        max_grad_norm=1.0,
        results_dir=None,
):
    """Run grouped CV with Orthrus inside the training loop."""
    if results_dir is None:
        three_prime_label = "full" if three_prime_width is None else str(three_prime_width)
        run_name = (
            f"orthrus_runtime_3utr{three_prime_label}_last{orthrus_unfreeze_last_n}_"
            f"olr{orthrus_lr:g}_clr{lr:g}_seed{seed}"
        )
        results_dir = os.path.join(REPO_ROOT, "results", run_name)

    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "Runtime Orthrus fine-tuning requires a CUDA GPU. "
            "torch.cuda.is_available() is False in this session, so PyTorch "
            "selected CPU, but Orthrus/Mamba uses CUDA-only kernels. "
            "Switch the Lightning Studio hardware to a GPU runtime and rerun."
        )
    print(f"Using device: {device}")

    raw_df = load_merged_dataset(cmsirna_path, historic_path)
    if max_rows is not None:
        raw_df = raw_df.sample(
            n=min(max_rows, len(raw_df)), random_state=seed,
        ).reset_index(drop=True)

    pipeline = SiRNADataPipeline(target_len=25, fetch_missing_mrna=True)
    enriched = pipeline.enrich_dataset_with_encodings(
        raw_df, strict_cleaning=True, add_mrna=True,
    )
    X_seq, X_exp, groups, y = pipeline.prepare_for_deep_learning(
        enriched, target_column="Inhibition",
    )
    enriched = add_slice_columns(enriched, three_prime_width=three_prime_width)
    X_mrna = enriched[SLICE_COLUMNS].to_numpy(dtype=object)

    valid = ~np.isnan(y)
    X_seq, X_exp, groups, y = X_seq[valid], X_exp[valid], groups[valid], y[valid]
    X_mrna = X_mrna[valid]
    print(
        f"Usable samples: {len(y)}, genes: {len(np.unique(groups))}, "
        f"seq channels: {X_seq.shape[1]}, exp dim: {X_exp.shape[1]}, "
        f"runtime slices: {X_mrna.shape}"
    )

    os.makedirs(results_dir, exist_ok=True)
    seq_in_channels = X_seq.shape[1]
    exp_input_dim = X_exp.shape[1]
    cv = GroupKFoldLeakPerGroup(
        n_splits=n_splits, leak_n=leak_n, random_state=seed,
    )
    fold_metrics = []

    for fold, (outer_train_idx, test_idx) in enumerate(cv.split(X_seq, y, groups)):
        print(
            f"\n=== Fold {fold + 1}/{n_splits} "
            f"(outer train {len(outer_train_idx)}, test {len(test_idx)}) ==="
        )
        train_idx, val_idx = grouped_train_val_split(
            outer_train_idx, groups, val_split=val_split, seed=seed + fold,
        )
        print(
            f"Inner grouped validation: train {len(train_idx)} rows/"
            f"{len(np.unique(groups[train_idx]))} genes, val {len(val_idx)} rows/"
            f"{len(np.unique(groups[val_idx]))} genes"
        )

        scaler_y = StandardScaler().fit(y[train_idx].reshape(-1, 1))
        y_train = scaler_y.transform(y[train_idx].reshape(-1, 1)).astype(np.float32)
        y_val = scaler_y.transform(y[val_idx].reshape(-1, 1)).astype(np.float32)
        y_test = scaler_y.transform(y[test_idx].reshape(-1, 1)).astype(np.float32)

        def make_dataset(indices, targets):
            return IndexedMultiTensorDataset(
                torch.tensor(X_seq[indices]),
                torch.tensor(X_exp[indices]),
                torch.tensor(targets),
                [str(g) for g in groups[indices]],
                X_mrna_tensor=X_mrna[indices],
            )

        train_ds = make_dataset(train_idx, y_train)
        val_ds = make_dataset(val_idx, y_val)
        test_ds = make_dataset(test_idx, y_test)

        collate_fn = partial(collate_runtime_slices, width=three_prime_width)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
            generator=torch.Generator().manual_seed(seed + fold),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        )

        # Critical CV safety: a fresh Crew model reloads the original Orthrus
        # checkpoint inside every fold, so fine-tuned weights from fold k cannot
        # leak into fold k+1.
        model = CrewSiRNAModel(
            seq_in_channels=seq_in_channels,
            exp_input_dim=exp_input_dim,
            use_experimental=True,
            mrna_embedding_dim=64,
            orthrus_model_dir=orthrus_model_dir,
            orthrus_checkpoint=orthrus_checkpoint,
            orthrus_unfreeze_last_n=orthrus_unfreeze_last_n,
        ).to(device)
        criterion = nn.MSELoss()

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
        metrics["best_loss"] = float(np.min(history["val_loss"])) if history["val_loss"] else float("nan")
        metrics["final_val_loss"] = float(history["val_loss"][-1]) if history["val_loss"] else float("nan")
        print("Fold metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        fold_metrics.append(metrics)

        np.savez_compressed(
            os.path.join(results_dir, f"fold_{fold + 1}_predictions.npz"),
            predictions=predictions,
            actuals=actuals,
            sample_ids=np.asarray(sample_names, dtype=object),
            train_idx=np.asarray(train_idx),
            val_idx=np.asarray(val_idx),
            test_idx=np.asarray(test_idx),
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

    with open(os.path.join(results_dir, "metrics.json"), "w") as handle:
        json.dump({"folds": fold_metrics, "summary": summary}, handle, indent=2)

    return fold_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Runtime Orthrus fine-tuning for the Crew siRNA model",
    )
    parser.add_argument("--cmsirna-path", default=DEFAULT_CMSIRNA_PATH)
    parser.add_argument("--historic-path", default=DEFAULT_HISTORIC_PATH)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--leak-n", type=int, default=0, help="0 = strict split by gene")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Crew/head learning rate")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--three-prime-width", type=parse_three_prime_width, default=100,
        metavar="N|full",
    )
    parser.add_argument("--orthrus-model-dir", default=DEFAULT_ORTHRUS_MODEL_DIR)
    parser.add_argument("--orthrus-checkpoint", default=DEFAULT_ORTHRUS_CHECKPOINT)
    parser.add_argument("--orthrus-unfreeze-last-n", type=int, default=1)
    parser.add_argument("--orthrus-lr", type=float, default=3e-7)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    run_runtime_orthrus_cv(
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
        three_prime_width=args.three_prime_width,
        orthrus_model_dir=args.orthrus_model_dir,
        orthrus_checkpoint=args.orthrus_checkpoint,
        orthrus_unfreeze_last_n=args.orthrus_unfreeze_last_n,
        orthrus_lr=args.orthrus_lr,
        gradient_accumulation=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
        max_grad_norm=args.max_grad_norm,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()

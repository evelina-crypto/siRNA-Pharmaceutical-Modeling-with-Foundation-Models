"""run_attribution.py

Comments : Integrated Gradients attribution for the CrewSiRNAModel from saved
           per-fold checkpoints (results/weights/crew_seed{seed}_fold{fold}.pt).
           Reloads the dataset with the same preprocessing as training, indexes
           each fold's test set via checkpoint test_idx, and writes per-fold and
           pooled attribution arrays + plots.

           Must use the same --cmsirna-path, --historic-path, --seed, and
           --max-rows as the training run that produced the checkpoints.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from modeling.attribution_plots import save_all_attribution_plots
from modeling.crew_model import CrewSiRNAModel
from modeling.model_attribution import ModelExplainer
from modeling.multi_input_training_utils import IndexedMultiTensorDataset
from modeling.training_utils import set_global_seed
from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CMSIRNA_PATH = os.path.join(REPO_ROOT, "dataset", "primary_dataset", "CMsiRNA_data_update.tsv")
DEFAULT_HISTORIC_PATH = os.path.join(REPO_ROOT, "dataset", "Historic_Takayuki_hueskan_ichihara.csv")
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DEFAULT_WEIGHTS_DIR = os.path.join(DEFAULT_RESULTS_DIR, "weights")
DEFAULT_ATTRIBUTIONS_DIR = os.path.join(DEFAULT_RESULTS_DIR, "attributions")
DEFAULT_ATTRIBUTION_PLOTS_DIR = os.path.join(DEFAULT_ATTRIBUTIONS_DIR, "attribution_plots")


def _load_crew_data(cmsirna_path, historic_path, seed=42, max_rows=None):
    """Load merged data, enrich, and prepare tensors for attribution."""
    raw_df = load_merged_dataset(cmsirna_path, historic_path)
    if max_rows is not None:
        raw_df = raw_df.sample(n=min(max_rows, len(raw_df)), random_state=seed).reset_index(drop=True)

    pipeline = SiRNADataPipeline(target_len=25)
    enriched = pipeline.enrich_dataset_with_encodings(raw_df, strict_cleaning=False, add_mrna=False)
    X_seq, X_exp, groups, y = pipeline.prepare_for_deep_learning(enriched, target_column="Inhibition")

    valid = ~np.isnan(y)
    X_seq, X_exp, groups, y = X_seq[valid], X_exp[valid], groups[valid], y[valid]
    seq_channel_names = pipeline.build_sequence_channel_names(enriched)
    exp_feature_names = pipeline.build_experimental_feature_names(enriched)

    return X_seq, X_exp, groups, y, seq_channel_names, exp_feature_names


def _attribution_for_fold(model, test_loader, X_seq_test, X_exp_test, seq_channel_names,
                          exp_feature_names, seed, fold, device, attributions_dir):
    """Compute IG attributions_arch, save arrays, and write per-fold plots."""
    explainer = ModelExplainer(model, device=device)
    attr = explainer.create_explainability_matrix(
        test_loader, seq_channel_names=seq_channel_names, exp_feature_names=exp_feature_names,
    )
    prefix = os.path.join(attributions_dir, f"seed{seed}_fold{fold}")
    explainer.save_attributions(attr, prefix)

    plot_dir = os.path.join(DEFAULT_ATTRIBUTION_PLOTS_DIR, "per_fold", f"seed{seed}_fold{fold}")
    save_all_attribution_plots(
        attr["seq_raw"], attr["exp"], X_seq_test, X_exp_test,
        attr["sample_ids"], seq_channel_names, exp_feature_names, plot_dir,
    )
    return attr


def _pool_and_save_attributions(fold_attrs, fold_X_seq, fold_X_exp, fold_sample_ids,
                                seq_channel_names, exp_feature_names, attributions_dir):
    """Concatenate per-fold test attributions_arch and write pooled arrays + plots."""
    pooled_seq = np.concatenate([a["seq_raw"] for a in fold_attrs], axis=0)
    pooled_exp = np.concatenate([a["exp"] for a in fold_attrs], axis=0)
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


def _load_checkpoint(ckpt_path):
    """Load a fold checkpoint; prefer weights_only when supported."""
    try:
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location="cpu")


def run_attribution_from_weights(cmsirna_path=DEFAULT_CMSIRNA_PATH, historic_path=DEFAULT_HISTORIC_PATH,
                                 n_splits=3, batch_size=64, seed=42, max_rows=None, device=None,
                                 weights_dir=DEFAULT_WEIGHTS_DIR, attributions_dir=DEFAULT_ATTRIBUTIONS_DIR):
    """Load saved fold weights and compute IG attribution maps + plots."""
    set_global_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading weights from", weights_dir)

    X_seq, X_exp, groups, y, seq_channel_names, exp_feature_names = _load_crew_data(
        cmsirna_path, historic_path, seed=seed, max_rows=max_rows,
    )
    print(f"Usable samples: {len(y)}, genes: {len(np.unique(groups))}, "
          f"seq channels: {X_seq.shape[1]}, exp dim: {X_exp.shape[1]}")

    os.makedirs(attributions_dir, exist_ok=True)
    fold_attrs, fold_X_seq, fold_X_exp, fold_sample_ids = [], [], [], []

    for fold in range(n_splits):
        ckpt_path = os.path.join(weights_dir, f"crew_seed{seed}_fold{fold}.pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"Missing checkpoint for fold {fold}: {ckpt_path}. "
                f"Run training first or check --seed / --n-splits."
            )

        print(f"\n=== Attribution fold {fold + 1}/{n_splits} ===")
        ckpt = _load_checkpoint(ckpt_path)
        test_idx = np.asarray(ckpt["test_idx"])

        model = CrewSiRNAModel(
            seq_in_channels=int(ckpt["seq_in_channels"]),
            exp_input_dim=int(ckpt["exp_input_dim"]),
            use_experimental=bool(ckpt.get("use_experimental", True)),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        dummy_y = np.zeros((len(test_idx), 1), dtype=np.float32)
        test_ds = IndexedMultiTensorDataset(
            torch.tensor(X_seq[test_idx]), torch.tensor(X_exp[test_idx]), torch.tensor(dummy_y),
            [str(g) for g in groups[test_idx]],
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        attr = _attribution_for_fold(
            model, test_loader, X_seq[test_idx], X_exp[test_idx],
            seq_channel_names, exp_feature_names, seed, fold, device, attributions_dir,
        )
        fold_attrs.append(attr)
        fold_X_seq.append(X_seq[test_idx])
        fold_X_exp.append(X_exp[test_idx])
        fold_sample_ids.extend(attr["sample_ids"])

    if fold_attrs:
        _pool_and_save_attributions(
            fold_attrs, fold_X_seq, fold_X_exp, fold_sample_ids,
            seq_channel_names, exp_feature_names, attributions_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Integrated Gradients attribution from saved CrewSiRNAModel fold weights",
    )
    parser.add_argument("--cmsirna-path", default=DEFAULT_CMSIRNA_PATH)
    parser.add_argument("--historic-path", default=DEFAULT_HISTORIC_PATH)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="must match training if used")
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--attributions-dir", default=DEFAULT_ATTRIBUTIONS_DIR)
    args = parser.parse_args()

    run_attribution_from_weights(
        cmsirna_path=args.cmsirna_path,
        historic_path=args.historic_path,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        seed=args.seed,
        max_rows=args.max_rows,
        weights_dir=args.weights_dir,
        attributions_dir=args.attributions_dir,
    )


if __name__ == "__main__":
    main()

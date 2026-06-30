"""Reusable audit/export utilities for exact duplicate same-condition conflicts."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline
from utils.splitter import GroupKFoldLeakPerGroup


FROZEN_PARAMS = {
    "n_estimators": 800,
    "max_depth": 4,
    "learning_rate": 0.15881823130907038,
    "subsample": 0.8812898741586134,
    "colsample_bytree": 0.7824379872752019,
    "min_child_weight": 4,
    "reg_lambda": 0.8342807691178866,
    "reg_alpha": 1.4296995092035882,
    "gamma": 0.07531958697602548,
}


def normalize_seq(value):
    if pd.isna(value):
        return None
    return "".join(str(value).upper().split())


@dataclass(slots=True)
class SameConditionConflictArtifacts:
    predictions_df: pd.DataFrame
    duplicate_audit: pd.DataFrame
    same_condition_conflicts: pd.DataFrame
    candidate_removals: pd.DataFrame
    candidate_rows: pd.DataFrame
    summary: pd.DataFrame


class SameConditionConflictExporter:
    """Builds and exports exact duplicate same-condition conflict tables."""

    def __init__(
        self,
        target_len: int = 25,
        fetch_missing_mrna: bool = True,
        strict_cleaning: bool = True,
        add_mrna: bool = True,
        use_normalized_conditions: bool = False,
        n_splits: int = 3,
        leak_n: int = 0,
        random_state: int = 42,
        frozen_params: dict[str, object] | None = None,
    ):
        self.target_len = target_len
        self.fetch_missing_mrna = fetch_missing_mrna
        self.strict_cleaning = strict_cleaning
        self.add_mrna = add_mrna
        self.use_normalized_conditions = use_normalized_conditions
        self.n_splits = n_splits
        self.leak_n = leak_n
        self.random_state = random_state
        self.frozen_params = dict(FROZEN_PARAMS if frozen_params is None else frozen_params)

    def build_predictions_frame(
        self,
        cmsirna_path: str | None = None,
        historic_path: str | None = None,
    ) -> pd.DataFrame:
        cmsirna_path = cmsirna_path or os.environ.get("CMSIRNA_RAW_DATA_PATH")
        historic_path = historic_path or os.environ.get("CMSIRNA_RAW_HISTORIC_DATA_PATH")

        if not cmsirna_path:
            raise RuntimeError("CMSIRNA_RAW_DATA_PATH is not set")
        if not historic_path:
            raise RuntimeError("CMSIRNA_RAW_HISTORIC_DATA_PATH is not set")

        raw_df = load_merged_dataset(cmsirna_path, historic_path)
        pipeline = SiRNADataPipeline(
            target_len=self.target_len,
            fetch_missing_mrna=self.fetch_missing_mrna,
        )
        enriched_df = pipeline.enrich_dataset_with_encodings(
            raw_df,
            strict_cleaning=self.strict_cleaning,
            add_mrna=self.add_mrna,
        )
        X, groups, y = pipeline.prepare_for_classical_ml(
            enriched_df,
            target_column="Inhibition",
            use_normalized_conditions=self.use_normalized_conditions,
        )

        mask = ~np.isnan(y)
        X = X[mask]
        groups = groups[mask]
        y = y[mask]
        analysis_df = enriched_df.loc[mask].reset_index(drop=True).copy()
        analysis_df["patent_group"] = analysis_df.get(
            "patent_ID", pd.Series(index=analysis_df.index, dtype=object)
        ).fillna("HISTORIC_OR_UNKNOWN")

        splitter = GroupKFoldLeakPerGroup(
            n_splits=self.n_splits,
            leak_n=self.leak_n,
            random_state=self.random_state,
        )
        oof_frames: list[pd.DataFrame] = []
        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model = XGBRegressor(
                tree_method="hist",
                n_jobs=-1,
                random_state=self.random_state,
                **self.frozen_params,
            )
            model.fit(X[train_idx], y[train_idx])
            fold_pred = model.predict(X[test_idx])

            fold_frame = analysis_df.iloc[test_idx].reset_index().rename(columns={"index": "source_index"}).copy()
            fold_frame["row_index"] = test_idx
            fold_frame["fold_id"] = fold_id
            fold_frame["group"] = groups[test_idx]
            fold_frame["y_true"] = y[test_idx]
            fold_frame["y_pred"] = fold_pred
            fold_frame["abs_error"] = (fold_frame["y_true"] - fold_frame["y_pred"]).abs()
            oof_frames.append(fold_frame)

        predictions_df = pd.concat(oof_frames, ignore_index=True)
        predictions_df["sense_seq_norm"] = predictions_df["Sense_seqence"].map(normalize_seq)
        predictions_df["antisense_seq_norm"] = predictions_df["Antisense_seqence"].map(normalize_seq)
        predictions_df["duplex_key"] = (
            predictions_df["sense_seq_norm"].fillna("MISSING")
            + "||"
            + predictions_df["antisense_seq_norm"].fillna("MISSING")
        )
        condition_cols = ["group", "Cell_Type", "Concentration_nM", "Time_of_administration_h"]
        predictions_df["condition_key"] = predictions_df[condition_cols].astype(str).agg(" || ".join, axis=1)
        return predictions_df

    def build_duplicate_audit(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for duplex_key, dup_df in predictions_df.groupby("duplex_key"):
            if len(dup_df) < 2:
                continue
            unique_conditions = dup_df["condition_key"].nunique()
            rows.append(
                {
                    "duplex_key": duplex_key,
                    "n_rows": len(dup_df),
                    "n_condition_keys": unique_conditions,
                    "same_condition_only": unique_conditions == 1,
                    "group": dup_df["group"].mode().iloc[0],
                    "cell_types": "|".join(sorted(dup_df["Cell_Type"].astype(str).unique().tolist())),
                    "concentrations": "|".join(map(str, sorted(dup_df["Concentration_nM"].astype(float).unique().tolist()))),
                    "times": "|".join(map(str, sorted(dup_df["Time_of_administration_h"].astype(float).unique().tolist()))),
                    "patents": "|".join(sorted(dup_df["patent_group"].astype(str).unique().tolist())),
                    "y_true_min": float(dup_df["y_true"].min()),
                    "y_true_max": float(dup_df["y_true"].max()),
                    "y_true_range": float(dup_df["y_true"].max() - dup_df["y_true"].min()),
                    "mean_y_true": float(dup_df["y_true"].mean()),
                    "row_indices": "|".join(map(str, dup_df["row_index"].tolist())),
                    "ids": "|".join(dup_df.get("ID", pd.Series(index=dup_df.index, dtype=object)).astype(str).tolist()),
                }
            )
        return pd.DataFrame(rows)

    def build_artifacts(
        self,
        range_threshold: float = 40.0,
        cmsirna_path: str | None = None,
        historic_path: str | None = None,
    ) -> SameConditionConflictArtifacts:
        predictions_df = self.build_predictions_frame(cmsirna_path=cmsirna_path, historic_path=historic_path)
        duplicate_audit = self.build_duplicate_audit(predictions_df)
        same_condition_conflicts = duplicate_audit.loc[duplicate_audit["same_condition_only"]].copy()
        same_condition_conflicts = same_condition_conflicts.sort_values(
            ["y_true_range", "n_rows"], ascending=[False, False]
        ).reset_index(drop=True)
        candidate_removals = same_condition_conflicts.loc[
            same_condition_conflicts["y_true_range"] >= range_threshold
        ].copy()

        if candidate_removals.empty:
            candidate_rows = pd.DataFrame(columns=predictions_df.columns)
        else:
            keys = set(candidate_removals["duplex_key"])
            candidate_rows = predictions_df.loc[predictions_df["duplex_key"].isin(keys)].copy()

        summary = pd.DataFrame(
            [
                {
                    "range_threshold": range_threshold,
                    "n_same_condition_duplicate_groups": int(len(same_condition_conflicts)),
                    "n_candidate_removal_groups": int(len(candidate_removals)),
                    "n_candidate_rows": int(len(candidate_rows)),
                }
            ]
        )

        return SameConditionConflictArtifacts(
            predictions_df=predictions_df,
            duplicate_audit=duplicate_audit,
            same_condition_conflicts=same_condition_conflicts,
            candidate_removals=candidate_removals,
            candidate_rows=candidate_rows,
            summary=summary,
        )

    def export(
        self,
        output_dir: str | Path,
        range_threshold: float = 40.0,
        cmsirna_path: str | None = None,
        historic_path: str | None = None,
    ) -> SameConditionConflictArtifacts:
        artifacts = self.build_artifacts(
            range_threshold=range_threshold,
            cmsirna_path=cmsirna_path,
            historic_path=historic_path,
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts.same_condition_conflicts.to_csv(output_dir / "same_condition_conflicts_all.csv", index=False)
        artifacts.candidate_removals.to_csv(output_dir / "same_condition_conflicts_thresholded.csv", index=False)
        artifacts.candidate_rows.to_csv(output_dir / "same_condition_conflict_rows_thresholded.csv", index=False)
        artifacts.summary.to_csv(output_dir / "same_condition_conflict_export_summary.csv", index=False)
        return artifacts


def main() -> None:
    range_threshold = float(os.environ.get("CMSIRNA_DUPLICATE_RANGE_THRESHOLD", "40"))
    output_dir = Path(
        os.environ.get(
            "CMSIRNA_PROCESSED_DIR",
            PROJECT_ROOT / "classical_ml" / "xgboost" / "outputs",
        )
    )
    exporter = SameConditionConflictExporter()
    artifacts = exporter.export(output_dir=output_dir, range_threshold=range_threshold)

    print("Saved:")
    print(output_dir / "same_condition_conflicts_all.csv")
    print(output_dir / "same_condition_conflicts_thresholded.csv")
    print(output_dir / "same_condition_conflict_rows_thresholded.csv")
    print(output_dir / "same_condition_conflict_export_summary.csv")
    print()
    print(artifacts.summary.to_string(index=False))


if __name__ == "__main__":
    main()

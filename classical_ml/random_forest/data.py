"""Dataset preparation helpers for Random Forest experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from classical_ml.random_forest.config import RandomForestExperimentConfig
from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline


@dataclass(slots=True)
class PreparedRandomForestData:
    """Container for the enriched dataframe and the flat ML arrays."""

    enriched_df: pd.DataFrame
    X: object
    groups: object
    y: object


class RandomForestDataBuilder:
    """Runs the shared preprocessing pipeline and returns RF-ready arrays."""

    def __init__(self, config: RandomForestExperimentConfig):
        self.config = config
        self.pipeline = SiRNADataPipeline(
            target_len=config.target_len,
            fetch_missing_mrna=config.fetch_missing_mrna,
        )

    def build_from_dataframe(self, df: pd.DataFrame) -> PreparedRandomForestData:
        enriched_df = self.pipeline.enrich_dataset_with_encodings(
            df,
            strict_cleaning=self.config.strict_cleaning,
            add_mrna=self.config.add_mrna,
        )
        X, groups, y = self.pipeline.prepare_for_classical_ml(
            enriched_df,
            target_column=self.config.target_column,
            use_normalized_conditions=self.config.use_normalized_conditions,
        )
        return PreparedRandomForestData(
            enriched_df=enriched_df,
            X=X,
            groups=groups,
            y=y,
        )

    def build_from_cmsirna_csv(self, path: str | Path, sep: str = "\t") -> PreparedRandomForestData:
        raw_df = pd.read_csv(path, sep=sep, low_memory=False)
        if "Target_Gene" in raw_df.columns and "gene_target_symbol_name" not in raw_df.columns:
            raw_df = raw_df.rename(columns={"Target_Gene": "gene_target_symbol_name"})
        return self.build_from_dataframe(raw_df)

    def build_from_merged_sources(
            self,
            cmsirna_path: str | Path,
            historic_path: str | Path,
    ) -> PreparedRandomForestData:
        merged_df = load_merged_dataset(str(cmsirna_path), str(historic_path))
        return self.build_from_dataframe(merged_df)

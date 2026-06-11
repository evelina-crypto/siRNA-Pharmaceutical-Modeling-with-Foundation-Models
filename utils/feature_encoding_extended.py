"""feature_encoding.py

Comments : Class for encoding the experimental conditions of the CMsiRNAdb dataset. The data should be cleaned and the values parsed before using this class. The encoding logic is based on the notebook but extended to handle more cases and ensure robustness.
Date     : 2026/06/11
"""

import re
import numpy as np
import pandas as pd

class FeatureEncoderExtended:

    def __init__(self, file: pd.DataFrame):
        # Work on a copy so the caller keeps the cleaned input unchanged.
        self.file = file.copy()
        self.cell_type_columns = []

    def _zscore(self, series: pd.Series) -> pd.Series:
        """Standardizes a numeric series to zero mean and unit variance."""
        std = series.std()
        if pd.isna(std) or std == 0:
            return pd.Series(np.nan, index=series.index, dtype=float)
        return (series - series.mean()) / std

    def encode(self) -> pd.DataFrame:
        """Encodes the cleaned experimental features using the notebook logic."""
        self.file["Concentration_nM"] = pd.to_numeric(self.file["Concentration"], errors="coerce")
        self.file["Concentration_log10_nM"] = np.log10(
            self.file["Concentration_nM"].where(self.file["Concentration_nM"] > 0)
        )
        self.file["Concentration_zscore"] = self._zscore(self.file["Concentration_log10_nM"])

        self.file["Time_hours"] = pd.to_numeric(self.file["Time_of_administration"], errors="coerce")
        self.file["Time_zscore"] = self._zscore(self.file["Time_hours"])
        self.file["Time_bin"] = pd.cut(
            self.file["Time_hours"],
            bins=[0, 24, 48, np.inf],
            labels=["<=24h", "24-48h", ">48h"],
            include_lowest=True,
        )

        cell_type_ohe = pd.get_dummies(self.file["Cell_Type"], prefix="cell", dtype=int)
        self.cell_type_columns = cell_type_ohe.columns.tolist()
        self.file = pd.concat([self.file, cell_type_ohe], axis=1)

        if "SD" in self.file.columns:
            self.file["has_sd"] = self.file["SD"].notna().astype(int)
        else:
            self.file["has_sd"] = 0

        return self.file

    def experimental_feature_table(self) -> pd.DataFrame:
        """Returns the notebook-style experimental feature subset."""
        if not self.cell_type_columns:
            self.encode()

        base_feature_columns = [
            "ID",
            "Target_Gene",
            "Inhibition",
            "Cell_Type",
            "Concentration",
            "Concentration_nM",
            "Concentration_log10_nM",
            "Concentration_zscore",
            "Time_of_administration",
            "Time_hours",
            "Time_zscore",
            "Time_bin",
            "has_sd",
        ]
        available_base_columns = [column for column in base_feature_columns if column in self.file.columns]
        return self.file[available_base_columns + self.cell_type_columns].copy()

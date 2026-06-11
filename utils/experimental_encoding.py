"""experimental_encoding.py

Comments : Class for encoding the experimental conditions of the CMsiRNAdb dataset
Date     : 2026/06/06
"""

import numpy as np
import pandas as pd


class ExperimentalEncoder:

    def __init__(self, file: pd.DataFrame):
        self.file = file

        # fixed cell-line order so every row's one-hot vector lines up
        self.cell_types = sorted(self.file["Cell_Type"].dropna().unique())

    def encode_cell_type(self, value):
        """One-hot column over the known cell lines"""
        vector = np.zeros(len(self.cell_types))
        if value in self.cell_types:
            vector[self.cell_types.index(value)] = 1.0
        return vector

    def encode(self):
        """Adds the one-hot cell-type column"""
        self.file["Cell_Type_One_Hot"] = [self.encode_cell_type(v) for v in self.file["Cell_Type"]]
        return self.file


class FeatureNormalizer:

    def __init__(self, file: pd.DataFrame):
        self.file = file

        # concentration is scaled on a log10, then maped to [0, 1]
        log_conc = np.log10(self.file["Concentration_nM"].where(self.file["Concentration_nM"] > 0))
        self.conc_log_min = log_conc.min()
        self.conc_log_max = log_conc.max()

        # time is just min-max normalized to [0, 1]
        self.time_min = self.file["Time_of_administration_h"].min()
        self.time_max = self.file["Time_of_administration_h"].max()

    def scale_concentration(self, value):
        """Concentration on a log10 scale, min-max normalized to [0, 1]."""
        if pd.isna(value) or value <= 0:
            return np.nan
        span = self.conc_log_max - self.conc_log_min
        if span == 0:  # avoiding dividing by zero
            return 0.0
        return (np.log10(value) - self.conc_log_min) / span

    def scale_time(self, value):
        """Time of administration min-max normalized to [0, 1]."""
        if pd.isna(value):
            return np.nan
        span = self.time_max - self.time_min
        if span == 0:  
            return 0.0
        return (value - self.time_min) / span

    def normalize(self):
        """Adds the [0, 1] concentration and time columns (raw hours kept as-is)."""
        self.file["Concentration_norm"] = self.file["Concentration_nM"].map(self.scale_concentration)
        self.file["Time_norm"] = self.file["Time_of_administration_h"].map(self.scale_time)
        return self.file

"""experimental_encoding.py

Comments : Class for encoding the experimental conditions of the CMsiRNAdb dataset
Date     : 2026/06/06
"""

import re
import numpy as np
import pandas as pd


class ExperimentalEncoder:

    def __init__(self, file: pd.DataFrame):
        self.file = file

        # molar units scaled to nM, the data is cleaned first, so only these remain
        self.molar_to_nm = {"pm": 1e-3, "nm": 1.0, "um": 1e3}

        # fixed cell-line order so every row's one-hot vector lines up
        self.cell_types = sorted(self.file["Cell_Type"].dropna().unique())

    def concentration_unit(self, value):
        """Reads the molar unit out of one Concentration cell, e.g. 'nm', 'um', 'pm"""
        if pd.isna(value):
            return None
        text = str(value).lower().replace("µ", "u").replace("μ", "u")
        for unit in ["um", "nm", "pm"]:
            if unit in text:
                return unit
        return None

    def parse_concentration(self, value):
        """Concentration in nM, or NaN when it cannot be read."""
        unit = self.concentration_unit(value)
        if unit is None:
            return np.nan
        number = re.search(r"[\d.]+", str(value))
        return float(number.group()) * self.molar_to_nm[unit] if number else np.nan

    def parse_time(self, value):
        """Time of administration in hours"""
        if pd.isna(value):
            return np.nan
        text = str(value).lower()
        numbers = [float(n) for n in re.findall(r"[\d.]+", text)]
        if not numbers:
            return np.nan
        hours = np.mean(numbers)
        return hours * 24 if "day" in text else hours

    def encode_cell_type(self, value):
        """One-hot column over the known cell lines"""
        vector = np.zeros(len(self.cell_types))
        if value in self.cell_types:
            vector[self.cell_types.index(value)] = 1.0
        return vector

    def encode(self):
        """Adds the parsed concentration, time and one-hot cell-type columns."""
        self.file["Concentration_nM"] = self.file["Concentration"].map(self.parse_concentration)
        self.file["Time_of_administration_h"] = self.file["Time_of_administration"].map(self.parse_time)
        self.file["Cell_Type_One_Hot"] = [self.encode_cell_type(v) for v in self.file["Cell_Type"]]
        return self.file

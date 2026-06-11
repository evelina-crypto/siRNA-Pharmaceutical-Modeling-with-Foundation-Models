"""data_cleaning.py

Comments : Class for QC of the CMsiRNAdb dataset
Date     : 2026/06/06
"""

import re

import numpy as np
import pandas as pd


class MoreStrictDataCleaner:

    def __init__(self, file: pd.DataFrame):
        self.file = file

        # words that mark a whole animal (in vivo), primary cultures contain these
        # words too, but are real in-vitro cells, so they are excluded separately
        self.animal_terms = ["mice", "mouse", "musculus", "rat", "macaca",
                             "macaque", "monkey", "patient", "muscle", "serum"]
        self.molar_to_nm = {"pm": 1e-3, "nm": 1.0, "um": 1e3}

    def parse_time_to_hours(self, value):
        """Parses time-of-administration text into hours."""
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower().replace(" ", "")
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?h", text):
            return float(text[:-1])
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?hr", text):
            return float(text[:-2])
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?hrs", text):
            return float(text[:-3])
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?hours?", text):
            return float(re.findall(r"[0-9]+(?:\.[0-9]+)?", text)[0])
        if re.fullmatch(r"[0-9]+(\.[0-9]+)?days?", text):
            return float(re.findall(r"[0-9]+(?:\.[0-9]+)?", text)[0]) * 24
        return np.nan

    def parsed_time_series(self):
        """Returns time of administration parsed to hours without mutating the dataframe."""
        return self.file["Time_of_administration"].apply(self.parse_time_to_hours)

    def replace_time_with_parsed_value(self):
        """Replaces parsable time values with their numeric hour value."""
        parsed = self.parsed_time_series()
        self.file["Time_of_administration"] = parsed.where(parsed.notna(), self.file["Time_of_administration"])

    def fill_missing_time_of_administration(self, fallback: float = 24.0):
        """Fills missing time using the modal time for matching Cell_Type and Concentration."""
        time_hours = self.parsed_time_series()
        observed_time_options = (
            self.file.assign(Time_hours=time_hours)
            .dropna(subset=["Concentration", "Time_hours"])
            .groupby(["Cell_Type", "Concentration"])["Time_hours"]
            .agg(
                modal_time_hours=lambda series: series.value_counts().index[0]
                if not series.value_counts().empty else np.nan
            )
            .reset_index()
        )

        fill_lookup = observed_time_options.set_index(["Cell_Type", "Concentration"])["modal_time_hours"]
        missing_mask = self.file["Time_of_administration"].isna()

        def infer_time(row):
            modal_hours = fill_lookup.get((row["Cell_Type"], row["Concentration"]), np.nan)
            if pd.isna(modal_hours):
                return fallback
            return float(modal_hours)

        filled = self.file.loc[missing_mask].apply(infer_time, axis=1)
        self.file.loc[missing_mask, "Time_of_administration"] = filled
        print(f"filled {int(missing_mask.sum())} rows for missing time of administration")

    def normalize_concentration_text(self, value):
        """Normalizes concentration text before parsing."""
        if pd.isna(value):
            return value
        text = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
        # observed typo: "lnm" appears to mean "1nm", e.g. "0.041lnM" -> "0.0411nm"
        return text.replace("lnm", "1nm")

    def parse_concentration_value(self, value):
        """Reads the numeric part out of one Concentration cell."""
        if pd.isna(value):
            return np.nan
        number = re.search(r"[0-9]*\.?[0-9]+", self.normalize_concentration_text(value))
        return float(number.group()) if number else np.nan

    def concentration_unit(self, value):
        """Reads the molar unit out of one Concentration cell."""
        if pd.isna(value):
            return None
        text = self.normalize_concentration_text(value)
        match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-z/]+)", text)
        if not match:
            return None
        unit = match.group(2)
        return unit

    def parse_concentration_nm(self, value):
        """Concentration converted to nM, or NaN when it cannot be read."""
        unit = self.concentration_unit(value)
        if unit is None or unit in {"mm", "mg/kg"}:
            return np.nan
        number = self.parse_concentration_value(value)
        return number * self.molar_to_nm[unit] if not pd.isna(number) else np.nan

    def parsed_concentration_series(self):
        """Returns concentration parsed to nM without mutating the dataframe."""
        return self.file["Concentration"].map(self.parse_concentration_nm)

    def replace_concentration_with_parsed_value(self):
        """Replaces parsable concentration values with their nM numeric value."""
        parsed = self.parsed_concentration_series()
        self.file["Concentration"] = parsed.where(parsed.notna(), self.file["Concentration"])

    def _drop_mask(self, mask: pd.Series, reason: str):
        """Drops rows where mask is true and prints the number removed."""
        removed = int(mask.sum())
        self.file = self.file.loc[~mask].copy()
        print(f"dropped {removed} rows for {reason}")

    def drop_in_vivo_readings(self):
        """Drops mg/kg rows and animal-model rows."""
        conc = self.file["Concentration"].fillna("").str.lower()
        cells = self.file["Cell_Type"].fillna("").str.strip().str.lower()
        animal = cells.str.contains("|".join(self.animal_terms)) & ~cells.str.contains("primary")
        in_vivo = conc.str.contains("mg") | animal
        self._drop_mask(in_vivo, "in-vivo readings")

    def drop_mM_readings(self):
        """Drops rows reported in mM."""
        conc = self.file["Concentration"].fillna("").str.lower()
        wrong_unit = conc.str.contains("mm")
        self._drop_mask(wrong_unit, "mM readings")

    def drop_unknown_cell_lines(self):
        """Drops rows with unknown or out-of-scope cell lines."""
        cells = self.file["Cell_Type"].fillna("").str.strip().str.lower()
        unwanted = cells.isin(["unknown cell line", "rga cell"])
        self._drop_mask(unwanted, "unknown or unwanted cell lines")

    def limit_inhibition(self, start: float = -100, to: float =100):
        """Drops rows where inhibition is outside [-100, 100]."""
        inhibition = pd.to_numeric(self.file["Inhibition"], errors="coerce")
        out_of_range = ~inhibition.between(start, to)
        self._drop_mask(out_of_range, "out-of-range inhibition")

    def drop_missing_concentration(self):
        """Drops rows with missing, blank, or unknown concentration values."""
        conc = self.file["Concentration"].fillna("").str.lower()
        missing_or_unknown = self.file["Concentration"].isna() | conc.str.strip().eq("") | conc.str.contains("unknown")
        self._drop_mask(missing_or_unknown, "missing or unknown concentration")

    # def drop_invalid_concentration(self):
    #     """Drops rows whose concentration cannot be parsed to supported nM units."""
    #     invalid = self.file["Concentration"].notna() & self.parsed_concentration_series().isna()
    #     self._drop_mask(invalid, "invalid concentration")

    def limit_concentration(self, to: float = 200):
        """Drops rows whose parsed concentration exceeds the given nM limit."""
        over_limit = self.parsed_concentration_series() > to
        self._drop_mask(over_limit, f"concentration > {to} nM")

    def drop_columns(self):
        """Drop unnecessary columns - columns with information that is implicitly contained in other columns"""

        cols_to_drop = [
            # encoding related columns

            # safe to drop 
            "Modification_locations_Sense_strand",  # e.g Base, Sugar, Phosphate (what chemical structure modification effects)
            "Modification_locations_Antisense_strand",# --- (same as above, but for the other strand)
            "Modifications_sense_strand",   # the machine unreadable encoding of the modifications in the form "VPudGcadAgdTgagg"
            "Modifications_Antisense_strand", # --- (same as above, but for the other strand)

            # presumably useless columns
            "Modifications_AntiSense_strand_3_5",
            "position_Antisense_strand",    # list of positions of the modifications, safe to drop when matching with positions in the `Modification_Types_Antisense_strand` successfull 
            "position_Sense_strand",    # --- (same as above, but for the other strand)
            
            # experimental conditions related columns
            # to be filled
        ]
        present = [c for c in cols_to_drop if c in self.file.columns]
        self.file = self.file.drop(columns=present)
        print(f"dropped {len(present)} columns: {present}")

    def drop_rows(self):
        """Drops out-of-scope rows for modeling."""
        self.drop_in_vivo_readings()
        self.drop_mM_readings()
        self.drop_unknown_cell_lines()
        self.limit_inhibition(start= -100, to=100)
        self.drop_missing_concentration()
        # self.drop_invalid_concentration()
        self.limit_concentration(to=200)
        self.replace_concentration_with_parsed_value()
        self.fill_missing_time_of_administration()
        self.replace_time_with_parsed_value()
        self.drop_invalid_sequences()


    def drop_invalid_sequences(self, max_len: int = 25):
        """Drops rows where the sense or antisense strand is missing/empty or longer
        than 25 nt. Length is computed from the sequence (the length_* column
        is unreliable)"""
        # valid strands are 1-25 nt
        sense_len = self.file["Sense_seqence"].fillna("").str.replace(r"\s", "", regex=True).str.len()
        anti_len = self.file["Antisense_seqence"].fillna("").str.replace(r"\s", "", regex=True).str.len()
        invalid = ~sense_len.between(1, max_len) | ~anti_len.between(1, max_len)
        print(f"dropped {invalid.sum()} rows with a missing or >{max_len} nt strand")
        self.file = self.file[~invalid].copy()

    def clean(self):
        """Runs the quality control and returns the cleaned table."""
        self.drop_rows()
        self.drop_columns()
        return self.file

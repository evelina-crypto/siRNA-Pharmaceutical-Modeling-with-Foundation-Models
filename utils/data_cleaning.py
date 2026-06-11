"""data_cleaning.py

Comments : Class for QC of the CMsiRNAdb dataset
Date     : 2026/06/06
"""

import re
import numpy as np
import pandas as pd


class DataCleaner:

    def __init__(self, file: pd.DataFrame):
        self.file = file

        # words that mark a whole animal (in vivo), primary cultures contain these
        # words too, but are real in-vitro cells, so they are excluded separately
        self.animal_terms = ["mice", "mouse", "musculus", "rat", "macaca",
                             "macaque", "monkey", "patient", "muscle", "serum"]

        # molar units scaled to nM; anything else (mM, mg/kg, blank) cannot be read
        self.molar_to_nm = {"pm": 1e-3, "nm": 1.0, "um": 1e3}

    def concentration_unit(self, value):
        """Reads the molar unit out of one Concentration cell, e.g. 'nm', 'um', 'pm'."""
        if pd.isna(value):
            return None
        text = str(value).lower().replace("µ", "u").replace("μ", "u")
        for unit in ["um", "nm", "pm"]:
            if unit in text:
                return unit
        return None

    def parse_concentration(self, value):
        """Concentration in nM, or NaN when the unit cannot be read (mM, mg/kg, missing)."""
        unit = self.concentration_unit(value)
        if unit is None:
            return np.nan
        number = re.search(r"[\d.]+", str(value))
        return float(number.group()) * self.molar_to_nm[unit] if number else np.nan

    def parse_time(self, value):
        """Time of administration in hours."""
        if pd.isna(value):
            return np.nan
        text = str(value).lower()
        numbers = [float(n) for n in re.findall(r"[\d.]+", text)]
        if not numbers:
            return np.nan
        hours = np.mean(numbers)
        return hours * 24 if "day" in text else hours

    def parse_conditions(self):
        """Parses the experimental conditions into numbers: concentration in nM, time in hours."""
        self.file["Concentration_nM"] = self.file["Concentration"].map(self.parse_concentration)
        self.file["Time_of_administration_h"] = self.file["Time_of_administration"].map(self.parse_time)

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
            "Modifications_AntiSense_strand_3_5"
            "position_Antisense_strand",    # list of positions of the modifications, safe to drop when matching with positions in the `Modification_Types_Antisense_strand` successfull 
            "position_Sense_strand",    # --- (same as above, but for the other strand)
            
            # experimental conditions related columns
            # to be filled
        ]
        present = [c for c in cols_to_drop if c in self.file.columns]
        self.file.drop(columns=present)
        print(f"dropped {len(present)} columns: {present}")

    def drop_rows(self):
        """Drops in-vivo, mM, above-200-nM, unknown/RGA cell line, and out-of-range inhibition rows.
        A missing or blank dose stays as NaN, we keep it."""
        conc = self.file["Concentration"].fillna("").str.lower()
        cells = self.file["Cell_Type"].fillna("").str.strip().str.lower()
        inhibition = pd.to_numeric(self.file["Inhibition"], errors="coerce")

        animal = cells.str.contains("|".join(self.animal_terms)) & ~cells.str.contains("primary")
        in_vivo = conc.str.contains("mg") | animal     # mg/kg dose or a live animal
        mM_dose = conc.str.contains("mm")              # mM is not a real siRNA concentration
        over_200nM = self.file["Concentration_nM"] > 200  
        unwanted = cells.isin(["unknown cell line", "rga cell"])
        out_of_range = ~inhibition.between(-100, 100)   # inhibition filter

        keep = ~(in_vivo | mM_dose | over_200nM | unwanted | out_of_range)
        print(f"dropped {len(self.file) - keep.sum()} rows "
              f"(in-vivo {in_vivo.sum()}, mM {mM_dose.sum()}, conc>200 {over_200nM.sum()}, "
              f"cell {unwanted.sum()}, inhibition {out_of_range.sum()})")
        self.file = self.file[keep].copy()

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
        self.parse_conditions()
        self.drop_rows()
        self.drop_invalid_sequences()
        self.drop_columns()
        return self.file

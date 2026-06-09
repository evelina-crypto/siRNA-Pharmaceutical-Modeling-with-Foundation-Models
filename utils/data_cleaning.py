"""data_cleaning.py

Comments : Class for QC of the CMsiRNAdb dataset
Date     : 2026/06/06
"""

import pandas as pd


class DataCleaner:

    def __init__(self, file: pd.DataFrame):
        self.file = file

        # words that mark a whole animal (in vivo), primary cultures contain these
        # words too, but are real in-vitro cells, so they are excluded separately
        self.animal_terms = ["mice", "mouse", "musculus", "rat", "macaca",
                             "macaque", "monkey", "patient", "muscle", "serum"]

    def drop_rows(self):
        """Drops in-vivo, mM, uknown cell lines, RGA cell lines, and inhibition rows < -100"""
        conc = self.file["Concentration"].fillna("").str.lower()
        cells = self.file["Cell_Type"].fillna("").str.strip().str.lower()
        inhibition = pd.to_numeric(self.file["Inhibition"], errors="coerce")

        animal = cells.str.contains("|".join(self.animal_terms)) & ~cells.str.contains("primary")
        in_vivo = conc.str.contains("mg") | animal     # mg/kg dose or a live animal
        wrong_unit = conc.str.contains("mm")           # mM is not a real siRNA concentration
        unwanted = cells.isin(["unknown cell line", "rga cell"])
        out_of_range = ~inhibition.between(-100, 100)   # inhibition filter

        keep = ~(in_vivo | wrong_unit | unwanted | out_of_range)
        print(f"dropped {len(self.file) - keep.sum()} rows "
              f"(in-vivo {in_vivo.sum()}, mM {wrong_unit.sum()}, "
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
        self.drop_rows()
        self.drop_invalid_sequences()
        return self.file

"""chemistry_encoding.py

Comments : Class for one-hot encoding of chemical modifications columns for CMsiRNAdb dataset
Date     : 2026/06/05
"""


import numpy as np
import pandas as pd


class ChemistryEncoder:

    def __init__(self, file: pd.DataFrame, target_len=25):
        self.file = file
        self.target_len = target_len  # matrix size for the siRNA sequence length

        # split to different categories
        self.acid_map = {"RNA": 0, "DNA": 1, "GNA": 2}
        self.sugar_map = {"Unmodified": 0, "2'-OMe": 1, "2'-F": 2}
        self.linker_map = {"Normal": 0, "PS": 1, "VP": 2, "Both_VP_PS": 3}

    def extract_chemistry(self, token_text):
        """
        Parses chemical modifications (columns: 'Modification_Types_Sense_strand' and
        'Modification_Types_Antisense_strand') into acid, sugar, and linker pieces
        """

        # acid extraction
        if "Deoxy" in token_text:
            acid = "DNA"
        elif "Glycol" in token_text:
            acid = "GNA"
        else:
            acid = "RNA"

        # sugar extraction
        if "2'-O-Methyl" in token_text or "2'-OMe" in token_text:
            sugar = "2'-OMe"
        elif "Fluoro" in token_text or "2'-F" in token_text:
            sugar = "2'-F"
        else:
            sugar = "Unmodified"

        # linker extraction
        if ("Phosphorothioate" in token_text) and ("Vinyl" in token_text):
            linker = "Both_VP_PS"
        elif "Phosphorothioate" in token_text:
            linker = "PS"
        elif "Vinyl" in token_text:
            linker = "VP"
        else:
            linker = "Normal"

        return acid, sugar, linker

    def encode_strand_chemistry(self, cell_value):
        """Turns a single row's raw chemical string into numerical one-hot matrices."""
        # empty matrices first
        acid_matrix = np.zeros((self.target_len, len(self.acid_map)))
        sugar_matrix = np.zeros((self.target_len, len(self.sugar_map)))
        linker_matrix = np.zeros((self.target_len, len(self.linker_map)))

        # return the matrices as all zeros if there are NA values
        if pd.isna(cell_value):
            return acid_matrix, sugar_matrix, linker_matrix

        tokens = cell_value.split(" || ")

        for token in tokens:
            if not "*" in token:
                continue

            parts = token.split("*")

            # align indices
            pos_idx = int(parts[0]) - 1
            if pos_idx < 0 or pos_idx >= self.target_len:
                continue

            chemistry_text = parts[1]

            acid_type, sugar_mod, linker_type = self.extract_chemistry(chemistry_text)

            acid_col = self.acid_map[acid_type]
            sugar_col = self.sugar_map[sugar_mod]
            linker_col = self.linker_map[linker_type]

            acid_matrix[pos_idx, acid_col] = 1.0
            sugar_matrix[pos_idx, sugar_col] = 1.0
            linker_matrix[pos_idx, linker_col] = 1.0

        return acid_matrix, sugar_matrix, linker_matrix

    def update_dataframe(self):
        """Loops through the dataframe and generates one-hot columns for sense and antisense strands"""

        sense_acids, sense_sugars, sense_linkers = [], [], []
        anti_acids, anti_sugars, anti_linkers = [], [], []

        for cell_value in self.file["Modification_Types_Sense_strand"]:
            a_mat, s_mat, l_mat = self.encode_strand_chemistry(cell_value)
            sense_acids.append(a_mat)
            sense_sugars.append(s_mat)
            sense_linkers.append(l_mat)

        for cell_value in self.file["Modification_Types_Antisense_strand"]:
            a_mat, s_mat, l_mat = self.encode_strand_chemistry(cell_value)
            anti_acids.append(a_mat)
            anti_sugars.append(s_mat)
            anti_linkers.append(l_mat)

        self.file["Sense_Acid_One_Hot"] = sense_acids # one df row will hold 25x3 matrix
        self.file["Sense_Sugar_One_Hot"] = sense_sugars # one df row will hold 25x3 matrix
        self.file["Sense_Linker_One_Hot"] = sense_linkers # one df row will hold 25x4 matrix

        self.file["Antisense_Acid_One_Hot"] = anti_acids
        self.file["Antisense_Sugar_One_Hot"] = anti_sugars
        self.file["Antisense_Linker_One_Hot"] = anti_linkers

        return "Dataframe was updated"
"""chemistry_encoding.py

Comments : Class for one-hot encoding of chemical modifications columns for CMsiRNAdb dataset
Date     : 2026/06/05
"""


import numpy as np
import pandas as pd
import logging

# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='modifications to update.log', encoding='utf-8', level=logging.DEBUG)

class ChemistryEncoder:

    def __init__(self, file: pd.DataFrame, target_len=25):
        self.file = file
        self.target_len = target_len  # matrix size for the siRNA sequence length

        # split to different categories
        self.acid_map = {"RNA": 0, "DNA": 1, "GNA": 2, "UNA": 3, "LNA": 4, "Unknown": 5}
        self.sugar_map = {
            "Unmodified": 0,
            "2'-OMe": 1,
            "2'-F": 2,
            "2'-M": 3,
            "2'-OHe": 4,
            "2'-P": 5,
            "Abasic": 6,
            "2'-F-4'-Thio": 7,
            "Unknown": 8
        }
        self.linker_map = {
            "Normal": 0,
            "PS": 1,
            "VP": 2,
            "Both_VP_PS": 3,
            "Phosphonate": 4,
            "Unknown": 5
        }

    def extract_chemistry(self, token_text):
        """
        Parses chemical modifications (columns: 'Modification_Types_Sense_strand' and
        'Modification_Types_Antisense_strand') into acid, sugar, and linker pieces
        """
        # log for debugging
        # known_vocabulary = [
        #     "Deoxy", "Glycol", "RNA", "DNA", "GNA",  # acids
        #     "2'-O-Methyl", "2'-OMe", "Fluoro", "2'-F",  # sugars
        #     "Phosphorothioate", "Vinyl"  # linkers
        # ]
        #
        # is_unknown = not any(word in token_text for word in known_vocabulary)
        #
        # if is_unknown:
        #     logger.warning(f"New mods found: '{token_text}'")

        # acid extraction
        if "Deoxy" in token_text:
            if "thio" in token_text:
                acid = "RNA"
            else:
                acid = "DNA"
        elif "Glycol" in token_text:
            acid = "GNA"
        elif "Unlocked" in token_text or "UNA" in token_text:
            acid = "UNA"
        elif "Locked" in token_text or "LNA" in token_text:
            acid = "LNA"
        else:
            acid = "RNA"

        # sugar extraction
        if "thio" in token_text:
            sugar = "2'-F-4'-Thio"
        elif "Abasic" in token_text:
            sugar = "Abasic"
        elif "2'-O-Methyl" in token_text:
            sugar = "2'-OMe"
        elif "Fluoro" in token_text:
            sugar = "2'-F"
        elif "Methoxy" in token_text:
            sugar = "2'-M"
        elif "2'-O-hexadecyl" in token_text:
            sugar = "2'-OHe"
        elif "2'-Phosphate" in token_text:
            sugar = "2'-P"
        else:
            sugar = "Unmodified"

        # linker extraction
        if ("Phosphorothioate" in token_text) and ("Vinyl" in token_text):
            linker = "Both_VP_PS"
        elif "Phosphorothioate" in token_text:
            linker = "PS"
        elif "Vinyl" in token_text:
            linker = "VP"
        elif "phosphonate" in token_text:
            linker = "Phosphonate"
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
        if pd.isna(cell_value) or not isinstance(cell_value, str):
            acid_matrix[:, self.acid_map["Unknown"]] = 1.0
            sugar_matrix[:, self.sugar_map["Unknown"]] = 1.0
            linker_matrix[:, self.linker_map["Unknown"]] = 1.0
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

        self.file["Sense_Acid_One_Hot"] = sense_acids # one df row will hold 25x6 matrix
        self.file["Sense_Sugar_One_Hot"] = sense_sugars # one df row will hold 25x9 matrix
        self.file["Sense_Linker_One_Hot"] = sense_linkers # one df row will hold 25x6 matrix

        self.file["Antisense_Acid_One_Hot"] = anti_acids
        self.file["Antisense_Sugar_One_Hot"] = anti_sugars
        self.file["Antisense_Linker_One_Hot"] = anti_linkers

        return "Dataframe was updated"
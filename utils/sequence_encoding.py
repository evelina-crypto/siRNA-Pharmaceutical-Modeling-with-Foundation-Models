"""sequence_encoding.py

Comments : Class for one-hot encoding of siRNA sequence columns for CMsiRNAdb dataset
"""


import numpy as np
import pandas as pd


class SequenceEncoder:

    def __init__(self, file: pd.DataFrame, target_len=25):
        self.file = file
        self.target_len = target_len  # matrix size for the siRNA sequence length

        # nucleotide alphabet (T kept for DNA overhangs in chemically modified siRNAs)
        self.base_map = {"A": 0, "U": 1, "G": 2, "C": 3, "T": 4}

    def extract_base(self, char):
        """
        Maps a single character from columns 'Sense_seqence' or 'Antisense_seqence'
        to a base label in self.base_map
        """
        char = char.upper()
        if char in self.base_map:
            return char
        return None

    def encode_strand_sequence(self, cell_value):
        """Turns a single row's raw sequence string into a numerical one-hot matrix."""
        # empty matrix first
        seq_matrix = np.zeros((self.target_len, len(self.base_map)))

        # return the matrix as all zeros if there are NA values
        if pd.isna(cell_value) or not isinstance(cell_value, str):
            return seq_matrix

        sequence = cell_value.upper().replace(" ", "")

        for pos_idx, char in enumerate(sequence[: self.target_len]):
            base = self.extract_base(char)
            if base is None:
                continue

            base_col = self.base_map[base]
            seq_matrix[pos_idx, base_col] = 1.0

        return seq_matrix

    def update_dataframe(self):
        """Loops through the dataframe and generates one-hot columns for sense and antisense strands"""

        sense_seqs, anti_seqs = [], []

        for cell_value in self.file["Sense_seqence"]:
            seq_mat = self.encode_strand_sequence(cell_value)
            sense_seqs.append(seq_mat)

        for cell_value in self.file["Antisense_seqence"]:
            seq_mat = self.encode_strand_sequence(cell_value)
            anti_seqs.append(seq_mat)

        self.file["Sense_Sequence_One_Hot"] = sense_seqs  # one df row will hold 25x5 matrix
        self.file["Antisense_Sequence_One_Hot"] = anti_seqs  # one df row will hold 25x5 matrix

        return "Dataframe was updated"

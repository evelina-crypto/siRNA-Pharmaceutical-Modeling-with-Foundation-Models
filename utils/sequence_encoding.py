"""sequence_encoding.py

Comments : Class for one-hot encoding of siRNA sequence columns for CMsiRNAdb dataset
"""


import numpy as np
import pandas as pd


class SequenceEncoder:

    def __init__(self, file: pd.DataFrame, target_len=25):
        self.file = file
        self.target_len = target_len  # matrix size for the siRNA sequence length

        # standard nucleotide alphabet 
        self.base_map = {"A": 0, "U": 1, "G": 2, "C": 3, "T": 4}

        # nonstandard single-character symbols that can remain in a sequence after
        # normalize_sequence(), with their meaning. Only the symbols that actually
        # occur in the data are given a one-hot column (see build_encoding_map()).
        self.nonstandard_char_map = {
            "N": "2'-O-Methyl-2'-aminoadenosine (from (A2N))",
            "V": "inverted abasic (from (invAb))",
            "Y": "abasic (from Ab)",
            "I": "inosine",
            "X": "unknown / unspecified base / Y1",
        }

        # one-hot column index map: standard bases plus the nonstandard symbols
        # present in the data. Populated by build_encoding_map().
        self.encoding_map = dict(self.base_map)

    def extract_base(self, char):
        """
        Maps a single character from columns 'Sense_seqence' or 'Antisense_seqence'
        to a column label in self.encoding_map (standard or nonstandard base).
        """
        char = char.upper()
        if char in self.encoding_map:
            return char
        return None

    def build_encoding_map(self):
        """Build the one-hot column index map for the current dataframe.

        Standard bases always get columns 0..4. A nonstandard symbol from
        self.nonstandard_char_map only gets a column if it actually appears in the
        normalized sequences, so absent classes (e.g. 'Y' when no abasic exists)
        do not add empty matrix columns.
        """
        present = set()
        for column in self.mod_type_columns:  # the two sequence columns
            if column not in self.file.columns:
                continue
            for cell_value in self.file[column].dropna():
                normalized = self.normalize_sequence(cell_value)
                if isinstance(normalized, str):
                    present.update(normalized)

        encoding_map = dict(self.base_map)
        next_col = len(encoding_map)
        for symbol in self.nonstandard_char_map:
            if symbol in present:
                encoding_map[symbol] = next_col
                next_col += 1

        self.encoding_map = encoding_map
        return encoding_map

    def encode_strand_sequence(self, cell_value):
        """Turns a single row's raw sequence string into a numerical one-hot matrix.

        The sequence is normalized first and each position is encoded against
        self.encoding_map (standard bases plus the nonstandard classes present in
        the data). Symbols without a column are skipped.
        """
        # empty matrix first
        seq_matrix = np.zeros((self.target_len, len(self.encoding_map)))

        # return the matrix as all zeros if there are NA values
        if pd.isna(cell_value) or not isinstance(cell_value, str):
            return seq_matrix

        sequence = self.normalize_sequence(cell_value)

        for pos_idx, char in enumerate(sequence[: self.target_len]):
            base = self.extract_base(char)
            if base is None:
                continue

            seq_matrix[pos_idx, self.encoding_map[base]] = 1.0

        return seq_matrix

    def update_dataframe(self):
        """Loops through the dataframe and generates one-hot columns for sense and antisense strands"""

        # determine which (non)standard base classes are present before encoding
        self.build_encoding_map()
        n_cols = len(self.encoding_map)

        sense_seqs, anti_seqs = [], []

        for cell_value in self.file["Sense_seqence"]:
            seq_mat = self.encode_strand_sequence(cell_value)
            sense_seqs.append(seq_mat)

        for cell_value in self.file["Antisense_seqence"]:
            seq_mat = self.encode_strand_sequence(cell_value)
            anti_seqs.append(seq_mat)

        # one df row holds a (target_len x n_cols) matrix, n_cols depends on the data
        self.file["Sense_Sequence_One_Hot"] = sense_seqs
        self.file["Antisense_Sequence_One_Hot"] = anti_seqs

        return f"Dataframe was updated (one-hot width = {n_cols}: {list(self.encoding_map)})"

    # sequence column -> column holding its per-strand modification types
    mod_type_columns = {
        "Antisense_seqence": "Modification_Types_Antisense_strand",
        "Sense_seqence": "Modification_Types_Sense_strand",
    }

    def check_nonstandard_bases(self, column_name: str) -> set[str]:
        """Report bases in a sequence column that are not in self.base_map.

        For every sequence containing a nonstandard base, the matching
        modification-types cell is printed underneath the sequence
        (Modification_Types_Antisense_strand for the antisense column and
        Modification_Types_Sense_strand for the sense column).
        """
        standard_bases = set(self.base_map)
        mod_column = self.mod_type_columns.get(column_name)

        nonstandard_bases = set()
        for idx, cell_value in self.file[column_name].dropna().items():
            sequence = str(cell_value).replace(" ", "")
            found = {base for base in sequence if base not in standard_bases}
            if found:
                nonstandard_bases.update(found)
                print(sequence)
                if mod_column is not None and mod_column in self.file.columns:
                    print(f"{self.file.at[idx, mod_column]}")

        if nonstandard_bases:
            print(f"[{column_name}] nonstandard bases: {sorted(nonstandard_bases)}")
        else:
            print(f"[{column_name}] all bases are standard ({sorted(standard_bases)})")

        return nonstandard_bases

    # known modification subwords -> single-character replacement
    normalization_map = {
        "(A2N)": "N", # 2'-O-Methyl-2'-aminoadenosine 
        "(invAb)": "V", # inverted abasic
        "Ab": "Y", # abasic
        "g": "G",
        "c": "C",
        "u": "U",
    }

    def normalize_sequence(self, cell_value):
        """Replace known modification subwords with single-character symbols.

        drops the lowercase deoxy prefix 'd' (e.g. 'dT' -> 'T', since the
        deoxy/DNA is captured in the chemistry encoding); strips any
        whitespace/newlines left in the raw cells.
        """
        if pd.isna(cell_value) or not isinstance(cell_value, str):
            return cell_value

        sequence = cell_value
        for subword, replacement in self.normalization_map.items():
            sequence = sequence.replace(subword, replacement)
        sequence = sequence.replace("d", "")  # drop deoxy prefix (dT -> T)
        sequence = "".join(sequence.split())
        return sequence

    def normalize_sequences(self, columns=("Sense_seqence", "Antisense_seqence")):
        """Normalize modification subwords in the given sequence columns of self.file."""
        for column in columns:
            if column in self.file.columns:
                self.file[column] = self.file[column].apply(self.normalize_sequence)
        return self.file

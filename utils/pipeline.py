"""pipeline.py

Comments : Layer which adds every encoding together and adds mRNA column, preparing CMsiRNA
           dataset for classical machine learning models.
Date     : 2026/06/13
"""

import numpy as np
import pandas as pd
from utils.data_cleaning_more_strict import MoreStrictDataCleaner
from utils.data_cleaning import DataCleaner
from utils.preprocess_invitro import add_mrna_column
from utils.experimental_encoding import ExperimentalEncoder, FeatureNormalizer
from utils.sequence_encoding import SequenceEncoder
from utils.chemistry_encoding import ChemistryEncoder
from utils.mrna_alignment import add_alignment_columns


class SiRNADataPipeline:
    def __init__(self, target_len=25, fetch_missing_mrna=False):
        self.target_len = target_len
        self.fetch_missing_mrna = fetch_missing_mrna

    def enrich_dataset_with_encodings(
            self,
            df: pd.DataFrame,
            strict_cleaning: bool = True,
            add_mrna: bool = False
    ) -> pd.DataFrame:
        """
        Combines all cleaning, tokenization, normalizations, and encodings,
        returning an updated df containing all newly engineered features
        """
        working_df = df.copy()

        # 1. Clean data and parse strings (like '100nM' and '48h') into raw numbers
        print("Running qc and data cleaning")
        if strict_cleaning:
            working_df = MoreStrictDataCleaner(working_df).clean()
            working_df["Concentration_nM"] = pd.to_numeric(working_df["Concentration"], errors="coerce")
            working_df["Time_of_administration_h"] = pd.to_numeric(
                working_df["Time_of_administration"],
                errors="coerce")
        else:
            working_df = DataCleaner(working_df).clean()

        # 2. mRNA sequence additions if requested
        if add_mrna:
            print("Mapping mRNA structural profiles")
            working_df = add_mrna_column(working_df, fetch_missing_from_ncbi=self.fetch_missing_mrna)
            working_df = add_alignment_columns(working_df)

        # 3. Apply chemistry and sequence encoding
        seq_encoder = SequenceEncoder(working_df, target_len=self.target_len)
        seq_encoder.update_dataframe()
        working_df = seq_encoder.file

        chem_encoder = ChemistryEncoder(working_df, target_len=self.target_len)
        chem_encoder.update_dataframe()
        working_df = chem_encoder.file

        # 4. Apply cell type and experimental conditions encoding
        working_df = ExperimentalEncoder(working_df).encode()
        working_df = FeatureNormalizer(working_df).normalize()

        print("Dataset successfully enriched")
        return working_df

    def prepare_for_classical_ml(
            self,
            enriched_df: pd.DataFrame,
            target_column: str = "Inhibition",
            use_normalized_conditions: bool = True,
    ):
        """
        Flattening and extraction for classical ml.
        Takes the fully updated df and flattens matrices for training.
        """
        df_ml = enriched_df.copy()

        # 1. Extract target label (y) and gene groups
        y = pd.to_numeric(df_ml[target_column], errors="coerce").values
        groups = df_ml["gene_target_symbol_name"].values

        # 2. Flatten sequence configurations
        s_seq = np.stack(df_ml["Sense_Sequence_One_Hot"].values).reshape(len(df_ml), -1)
        as_seq = np.stack(df_ml["Antisense_Sequence_One_Hot"].values).reshape(len(df_ml), -1)

        s_acid = np.stack(df_ml["Sense_Acid_One_Hot"].values).reshape(len(df_ml), -1)
        s_sugar = np.stack(df_ml["Sense_Sugar_One_Hot"].values).reshape(len(df_ml), -1)
        s_linker = np.stack(df_ml["Sense_Linker_One_Hot"].values).reshape(len(df_ml), -1)

        as_acid = np.stack(df_ml["Antisense_Acid_One_Hot"].values).reshape(len(df_ml), -1)
        as_sugar = np.stack(df_ml["Antisense_Sugar_One_Hot"].values).reshape(len(df_ml), -1)
        as_linker = np.stack(df_ml["Antisense_Linker_One_Hot"].values).reshape(len(df_ml), -1)

        # 3. Gather experimental conditions

        if use_normalized_conditions:
            concentration = df_ml["Concentration_norm"].values.reshape(-1, 1)
            time = df_ml["Time_norm"].values.reshape(-1, 1)
        else:
            concentration = pd.to_numeric(df_ml["Concentration_log10_nM"], errors="coerce").values.reshape(-1, 1)
            time = pd.to_numeric(df_ml["Time_of_administration_h"], errors="coerce").values.reshape(-1, 1)

        cell_type_oh = np.stack(df_ml["Cell_Type_One_Hot"].values)

        # 4. mRNA alignment features (only present when add_mrna=True)
        mrna_cols = []
        if "edit_distance" in df_ml.columns and "target_site_pct" in df_ml.columns:
            mrna_features = df_ml[["edit_distance", "target_site_pct"]].apply(
                pd.to_numeric, errors="coerce"
            ).fillna(0).values
            mrna_cols.append(mrna_features)

        # 5. Combine everything into a single flat 2D matrix
        X_flat = np.hstack([
            concentration,
            time,
            cell_type_oh,
            s_seq, as_seq,
            s_acid, s_sugar, s_linker,
            as_acid, as_sugar, as_linker,
            *mrna_cols,
        ])

        print(f"Feature matrix X shape: {X_flat.shape}, target y shape: {y.shape}")
        return X_flat, groups, y

    def prepare_for_deep_learning(self, enriched_df: pd.DataFrame, target_column: str = "Inhibition"):
        """Shape the enriched df for the multi-branch siRNA model.

        Keeps the sequence/chemistry channels as a 3D tensor for
        the 1D CNN, returns the experimental conditions separately for the
        experimental MLP.

        Returns
            X_seq   : (N, 2 * D, target_len) float32. Guide (antisense) and
                      passenger (sense) strands concatenated along the channel
                      axis, where D = sequence + acid + sugar + linker one-hot
                      widths per strand. Layout matches Conv1d (N, channels, L).
            X_exp   : (N, 2 + n_cell_types) float32. Concentration_norm,
                      Time_norm and the cell-type one-hot.
            groups  : gene-target group labels (for grouped CV).
            y       : target (Inhibition), raw numeric.
        """
        df_ml = enriched_df.copy()

        # 1. Target and gene groups
        y = pd.to_numeric(df_ml[target_column], errors="coerce").values
        groups = df_ml["gene_target_symbol_name"].values

        # 2. Sequence + chemistry blocks, each stored per row as (target_len, width)
        def stack_block(column):
            return np.stack(df_ml[column].values)  # (N, target_len, width)

        sense_blocks = [
            stack_block("Sense_Sequence_One_Hot"),
            stack_block("Sense_Acid_One_Hot"),
            stack_block("Sense_Sugar_One_Hot"),
            stack_block("Sense_Linker_One_Hot"),
        ]
        antisense_blocks = [
            stack_block("Antisense_Sequence_One_Hot"),
            stack_block("Antisense_Acid_One_Hot"),
            stack_block("Antisense_Sugar_One_Hot"),
            stack_block("Antisense_Linker_One_Hot"),
        ]

        # concat all blocks along the feature axis -> (N, target_len, 2 * D),
        # then move features to the channel axis -> (N, 2 * D, target_len)
        X_seq = np.concatenate(sense_blocks + antisense_blocks, axis=2)
        X_seq = np.transpose(X_seq, (0, 2, 1)).astype(np.float32)

        # 3. Experimental conditions for the experimental MLP
        conc_norm = df_ml["Concentration_norm"].values.reshape(-1, 1)
        time_norm = df_ml["Time_norm"].values.reshape(-1, 1)
        cell_type_oh = np.stack(df_ml["Cell_Type_One_Hot"].values)
        X_exp = np.hstack([conc_norm, time_norm, cell_type_oh]).astype(np.float32)

        print(f"Sequence tensor X_seq shape: {X_seq.shape}, "
              f"experimental matrix X_exp shape: {X_exp.shape}, target y shape: {y.shape}")
        return X_seq, X_exp, groups, y

    def build_feature_names(self, enriched_df: pd.DataFrame) -> list[str]:
        """readable column names in the same order as prepare_for_classical_ml's hstack.
        """
        df = enriched_df

        def clean(label):
            return str(label).replace("[", "(").replace("]", ")").replace("<", "lt")

        names = ["Concentration_norm", "Time_norm"]

        cell_types = ExperimentalEncoder(df).cell_types
        names += [f"Cell_{clean(c)}" for c in cell_types]

        seq_encoder = SequenceEncoder(df, target_len=self.target_len)
        seq_encoder.build_encoding_map()
        seq_bases = list(seq_encoder.encoding_map)
        for strand in ("Sense", "Antisense"):
            for pos in range(1, self.target_len + 1):
                for base in seq_bases:
                    names.append(f"{strand}_seq_pos{pos}_{clean(base)}")

        chem = ChemistryEncoder(df, target_len=self.target_len)
        chem_blocks = (("acid", chem.acid_map), ("sugar", chem.sugar_map), ("linker", chem.linker_map))
        for strand in ("Sense", "Antisense"):
            for block_name, block_map in chem_blocks:
                for pos in range(1, self.target_len + 1):
                    for label in block_map:
                        names.append(f"{strand}_{block_name}_pos{pos}_{clean(label)}")

        if "edit_distance" in df.columns and "target_site_pct" in df.columns:
            names += ["edit_distance", "target_site_pct"]

        return names

    def build_experimental_feature_names(self, enriched_df: pd.DataFrame) -> list[str]:
        """Names for X_exp from prepare_for_deep_learning, in column order.

        Matches X_exp = [Concentration_norm, Time_norm, Cell_Type_One_Hot].
        """
        def clean(label):
            return str(label).replace("[", "(").replace("]", ")").replace("<", "lt")

        cell_types = ExperimentalEncoder(enriched_df).cell_types
        return ["Concentration_norm", "Time_norm"] + [f"Cell_{clean(c)}" for c in cell_types]

    def build_sequence_channel_names(self, enriched_df: pd.DataFrame) -> list[str]:
        """Channel names for X_seq from prepare_for_deep_learning, in channel order.

        X_seq has shape (N, 2 * D, target_len); the channel axis follows the
        block concatenation order sense_blocks + antisense_blocks, where each
        block is (seq, acid, sugar, linker) and contributes one channel per
        one-hot category. The length axis (positions 1..target_len) is separate
        and is NOT part of these names.
        """
        df = enriched_df

        def clean(label):
            return str(label).replace("[", "(").replace("]", ")").replace("<", "lt")

        seq_encoder = SequenceEncoder(df, target_len=self.target_len)
        seq_encoder.build_encoding_map()
        seq_bases = list(seq_encoder.encoding_map)

        chem = ChemistryEncoder(df, target_len=self.target_len)
        blocks = (
            ("seq", seq_bases),
            ("acid", list(chem.acid_map)),
            ("sugar", list(chem.sugar_map)),
            ("linker", list(chem.linker_map)),
        )

        names = []
        for strand in ("Sense", "Antisense"):
            for block_name, categories in blocks:
                for category in categories:
                    names.append(f"{strand}_{block_name}_{clean(category)}")
        return names

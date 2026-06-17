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

    def prepare_for_classical_ml(self, enriched_df: pd.DataFrame, target_column: str = "Inhibition"):
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
        conc_norm = df_ml["Concentration_norm"].values.reshape(-1, 1)
        #time_norm = df_ml["Time_norm"].values.reshape(-1, 1)
        time_raw = df_ml["Time_of_administration_h"].values.reshape(-1, 1)
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
            conc_norm,
            time_raw,
            cell_type_oh,
            s_seq, as_seq,
            s_acid, s_sugar, s_linker,
            as_acid, as_sugar, as_linker,
            *mrna_cols,
        ])

        print(f"Feature matrix X shape: {X_flat.shape}, target y shape: {y.shape}")
        return X_flat, groups, y
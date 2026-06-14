"""mrna_alignment.py

Comments : QC alignment of siRNA antisense strand reverse complement against target mRNA.
           Uses edlib alignment after normalizing U -> T on both sequences.
Date     : 2026/06/14
"""

import edlib
import pandas as pd
import numpy as np


def reverse_complement(seq: str) -> str:
    complement = {"A": "T", "T": "A", "U": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(complement.get(b, "N") for b in reversed(seq.upper()))


def align_guide_to_mrna(antisense_seq, mrna_seq):
    """Aligns reverse complement of antisense strand against mRNA.
    Returns edit_distance (int) and target_site_pct (float 0-1, position in mRNA).
    0.0 - near the 5' end / start codon area; 1.0 - the siRNA targets the very end (near the 3' UTR).
    """
    if not isinstance(antisense_seq, str) or not isinstance(mrna_seq, str):
        return pd.Series({"edit_distance": np.nan, "target_site_pct": np.nan})

    query = reverse_complement(antisense_seq).replace("U", "T")
    target = mrna_seq.upper().replace("U", "T")

    result = edlib.align(query, target, mode="HW", task="path")

    edit_distance = result["editDistance"]
    locations = result["locations"]

    if not locations:
        return pd.Series({"edit_distance": edit_distance, "target_site_pct": np.nan})

    start, _ = locations[0]
    target_site_pct = start / len(target)

    return pd.Series({"edit_distance": edit_distance, "target_site_pct": target_site_pct})


def add_alignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds edit_distance and target_site_pct columns for use as ml model features
    and QC inspection.
    """
    df = df.copy()

    alignment = df.apply(
        lambda row: align_guide_to_mrna(row["Antisense_seqence"], row["mRNA"]),
        axis=1
    )
    df = pd.concat([df, alignment], axis=1)

    print(f"edit_distance distribution:\n{df['edit_distance'].value_counts().sort_index()}")
    return df
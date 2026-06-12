#!/usr/bin/env python

# Author Alexander William Larsen
# Email: alarsen525@gmail.com

# Authored 2026-03-10
"""
Preprocess an invitro training CSV and optionally add:
- siRNA_list: tokenized guide sequence
- siRNA_passenger_list: tokenized and reversed passenger sequence
- siRNA: unmodified guide sequence
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_INPUT = "/home/larsena8/software/fennec/src/fennec/support_files/invitro_curated_train.csv"
DEFAULT_OUTPUT = "/home/larsena8/software/fennec/src/fennec/support_files/invitro_curated_train_preprocessed.csv"
REQUIRED_COLUMNS = ["siRNA_guide", "siRNA_passenger"]


def process_passenger(pass_seq):
    """Tokenize passenger sequence, reverse tokens, and join with spaces."""
    from fennec.featurizer import tokenize_seq_to_list

    if pd.isna(pass_seq):
        return None
    pass_seq_list = tokenize_seq_to_list(str(pass_seq))
    return " ".join(pass_seq_list[::-1])


def process_guide(guide_seq):
    """Tokenize guide sequence and join with spaces."""
    from fennec.featurizer import tokenize_seq_to_list

    if pd.isna(guide_seq):
        return None
    return " ".join(tokenize_seq_to_list(str(guide_seq)))


def preprocess_dataframe(df, skip_sirna_conversion=False):
    """Add siRNA-derived columns unless conversion is explicitly skipped."""

    df = df.copy()

    if skip_sirna_conversion:
        print("Skipping siRNA_guide/siRNA_passenger conversion; leaving existing siRNA columns unchanged.")
        return df

    from fennec.featurizer import seq_list_to_seq

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        missing_str = ", ".join(missing_cols)
        raise ValueError(f"Missing required columns: {missing_str}")

    print("Converting siRNA_guide to siRNA_list...")
    df["siRNA_list"] = df["siRNA_guide"].apply(process_guide)

    print("Converting siRNA_passenger to siRNA_passenger_list (tokenized and reversed)...")
    df["siRNA_passenger_list"] = df["siRNA_passenger"].apply(process_passenger)

    print("Converting siRNA_list to unmodified siRNA sequence...")
    df["siRNA"] = df["siRNA_list"].apply(lambda x: seq_list_to_seq(str(x)) if pd.notna(x) else None)

    return df


def add_mrna_column(df, fetch_missing_from_ncbi=False):
    """Add mRNA, mRNA_five_prime, and mRNA_three_prime columns using helpers from add_mrna_sequences.py."""
    from utils.add_mrna_sequences import (
        load_local_cache,
        get_mrna_from_local_cache,
        get_mrna_from_reference,
        fetch_mrna_from_ncbi,
        upsert_gene_in_cache,
        load_reference_sequences,
        find_utr_regions,
    )

    if "gene_target_symbol_name" not in df.columns:
        raise ValueError("Missing required column for mRNA mapping: gene_target_symbol_name")

    df = df.copy()

    # Keep behavior consistent with add_mrna_sequences.py
    alect2_count = (df["gene_target_symbol_name"] == "ALECT2").sum()
    if alect2_count > 0:
        print(f"Fixing ALECT2 -> LECT2 ({alect2_count} rows)")
        df.loc[df["gene_target_symbol_name"] == "ALECT2", "gene_target_symbol_name"] = "LECT2"

    local_cache = load_local_cache()
    reference_sequences = load_reference_sequences()

    print(f"Loaded {len(local_cache)} gene sequences from local cache")
    print(f"Loaded {len(reference_sequences)} gene sequences from reference CSV")

    unique_genes = df["gene_target_symbol_name"].dropna().unique()
    print(f"Building gene -> mRNA mapping for {len(unique_genes)} genes...")

    gene_mrna_map = {}
    gene_cds_map = {}

    for i, gene in enumerate(unique_genes, 1):
        print(f"[{i}/{len(unique_genes)}] Processing {gene}...")

        result = get_mrna_from_local_cache(gene, local_cache)
        if result is not None:
            mrna, cds_start, cds_end = result
            print(f"  Found in local cache ({len(mrna)} bp)")
            gene_mrna_map[gene] = mrna
            if cds_start is not None:
                gene_cds_map[gene] = (cds_start, cds_end)
            continue

        result = get_mrna_from_reference(gene, reference_sequences)
        if result is not None:
            mrna, cds_start, cds_end = result
            print(f"  Found in reference ({len(mrna)} bp)")
            gene_mrna_map[gene] = mrna
            continue

        if fetch_missing_from_ncbi:
            result = fetch_mrna_from_ncbi(gene)
            if result is not None:
                mrna, cds_start, cds_end, accession = result
                gene_mrna_map[gene] = mrna
                if cds_start is not None:
                    gene_cds_map[gene] = (cds_start, cds_end)
                gene_key = str(gene).strip().upper()
                if gene_key not in {str(k).strip().upper() for k in local_cache}:
                    upsert_gene_in_cache(gene, mrna, cds_start=cds_start, cds_end=cds_end, accession=accession)
                    local_cache[gene] = {"sequence": mrna, "cds_start": cds_start,
                                         "cds_end": cds_end, "accession": accession}
                continue

        print(f"  WARNING: Could not resolve mRNA for {gene}")
        gene_mrna_map[gene] = None

    df["mRNA"] = df["gene_target_symbol_name"].map(gene_mrna_map)
    rows_with_mrna = df["mRNA"].notna().sum()
    print(f"Rows with mRNA: {rows_with_mrna}/{len(df)} ({100 * rows_with_mrna / len(df):.1f}%)")

    # Derive UTR columns per gene using annotated CDS when available.
    gene_utr_map = {}
    for gene, mrna in gene_mrna_map.items():
        if not isinstance(mrna, str):
            gene_utr_map[gene] = (None, None)
            continue
        cds = gene_cds_map.get(gene)
        cds_start, cds_end = cds if cds else (None, None)
        gene_utr_map[gene] = find_utr_regions(mrna, cds_start=cds_start, cds_end=cds_end)

    df["mRNA_five_prime"] = df["gene_target_symbol_name"].map(
        lambda g: gene_utr_map.get(g, (None, None))[0]
    )
    df["mRNA_three_prime"] = df["gene_target_symbol_name"].map(
        lambda g: gene_utr_map.get(g, (None, None))[1]
    )

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess invitro CSV and optionally add tokenized guide/passenger plus unmodified guide columns."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=3,
        help="Number of sample rows to print after preprocessing (default: 3)",
    )
    parser.add_argument(
        "--add-mrna",
        action="store_true",
        help="Add mRNA column using helper functions from add_mrna_sequences.py",
    )
    parser.add_argument(
        "--fetch-missing-mrna",
        action="store_true",
        help="When used with --add-mrna, fetch unresolved genes from NCBI Entrez",
    )
    parser.add_argument(
        "--skip-sirna-conversion",
        action="store_true",
        help="Leave siRNA_guide/siRNA_passenger and any existing siRNA-derived columns unchanged.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")

    df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} rows from CSV")
    print(f"Columns: {df.columns.tolist()}")

    processed = preprocess_dataframe(df, skip_sirna_conversion=args.skip_sirna_conversion)

    if args.add_mrna:
        print("\nAdding mRNA sequences...")
        processed = add_mrna_column(processed, fetch_missing_from_ncbi=args.fetch_missing_mrna)

    sample_cols = [
        "siRNA_guide",
        "siRNA_list",
        "siRNA",
        "siRNA_passenger",
        "siRNA_passenger_list",
        "gene_target_symbol_name",
        "mRNA",
        "mRNA_five_prime",
        "mRNA_three_prime",
    ]
    existing_sample_cols = [col for col in sample_cols if col in processed.columns]

    print("\n=== Sample of processed data ===")
    print(processed[existing_sample_cols].head(args.sample_rows).to_string())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    print(f"\nSaved preprocessed CSV to: {output_path}")
    if args.skip_sirna_conversion:
        print("siRNA conversion skipped; existing siRNA-related columns were preserved.")
    else:
        print("New columns added: siRNA_list, siRNA_passenger_list, siRNA")
    if args.add_mrna:
        print("mRNA, mRNA_five_prime, and mRNA_three_prime columns added using shared helper functions")


if __name__ == "__main__":
    main()

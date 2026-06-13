#!/usr/bin/env python

# Author Alexander William Larsen
# Email: alarsen525@gmail.com

# Authored 2026-03-05
"""
Add mRNA transcript sequences to invitro_curated_train_preprocessed.csv

This script:
1. Extracts mRNA sequences from reference dataset for overlapping genes
2. Fetches missing gene transcripts from NCBI Entrez
3. Maps mRNA sequences to all rows based on gene_target_symbol_name
"""

import json
import pandas as pd
from Bio import Entrez, SeqIO
from io import StringIO
import time
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# Configure Entrez
Entrez.email = "alarsen525@gmail.com"  # Required by NCBI
Entrez.tool = "FennecPreprocessing"
Entrez.api_key = "f8119655d5c8bd87ceb28b81ba2f4df82208"

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_FASTA_PATH = SCRIPT_DIR / 'gene_mrna_sequences.fasta'   # legacy – kept for migration
LOCAL_CACHE_PATH = SCRIPT_DIR / 'gene_mrna_sequences.json'
REFERENCE_CSV_PATH = Path('/home/larsena8/software/fennec/src/fennec/support_files/train_data_v1.1.0_N=27742.csv')
PREPROCESSED_CSV_PATH = Path('/home/larsena8/software/fennec/src/fennec/support_files/invitro_curated_train_preprocessed.csv')
MIN_VALID_MRNA_LENGTH = 120


def normalize_rna_sequence(sequence):
    """Normalize DNA/RNA sequence text to uppercase RNA."""
    if not isinstance(sequence, str):
        return None

    normalized = ''.join(sequence.split()).upper().replace('T', 'U')
    return normalized or None


def is_valid_mrna(sequence, min_length=MIN_VALID_MRNA_LENGTH):
    """Return True when sequence looks like a transcript and not a short target-site snippet."""
    if not isinstance(sequence, str):
        return False
    return len(sequence) >= min_length


def find_utr_regions(sequence, cds_start=None, cds_end=None):
    """Return (five_prime_utr, three_prime_utr) for an mRNA sequence.

    When cds_start/cds_end are provided (e.g. from GenBank annotation) they are
    used directly, after verifying the indicated position begins with AUG.  If
    the annotation is inconsistent with the sequence, or no annotation is given,
    the function falls back to finding the longest complete ORF (AUG … stop
    codon) across all positions in the sequence.  This is more robust than
    first-AUG alone for transcripts that have upstream AUGs in the 5' UTR.

    Returns (None, None) when no AUG can be found at all.
    """
    if not isinstance(sequence, str):
        return None, None

    seq = sequence.upper().replace('T', 'U')
    n = len(seq)

    # --- Use annotated CDS when it is consistent with this exact sequence ---
    if cds_start is not None and cds_end is not None:
        if 0 <= cds_start < n - 2 and seq[cds_start:cds_start + 3] == 'AUG':
            five_prime = seq[:cds_start] if cds_start > 0 else None
            three_prime = seq[cds_end:] if cds_end < n else None
            return five_prime, three_prime
        # Annotation doesn't match this sequence (different transcript) —
        # fall through to heuristic below.

    # --- Longest-ORF heuristic ---
    STOP_CODONS = {'UAA', 'UAG', 'UGA'}
    best_start = None
    best_stop = None
    best_len = 0
    search_from = 0

    while True:
        pos = seq.find('AUG', search_from)
        if pos == -1:
            break
        for stop_pos in range(pos + 3, n - 2, 3):
            if seq[stop_pos:stop_pos + 3] in STOP_CODONS:
                orf_len = stop_pos + 3 - pos
                if orf_len > best_len:
                    best_len = orf_len
                    best_start = pos
                    best_stop = stop_pos + 3
                break
        search_from = pos + 1

    if best_start is None:
        # No complete ORF found; return region before first AUG as 5' UTR if present.
        first_aug = seq.find('AUG')
        if first_aug == -1:
            return None, None
        return seq[:first_aug] or None, None

    five_prime = seq[:best_start] if best_start > 0 else None
    three_prime = seq[best_stop:] if best_stop < n else None
    return five_prime, three_prime


def load_local_cache(cache_path=LOCAL_CACHE_PATH):
    """Load the JSON gene cache.

    Returns a dict keyed by gene symbol.  Each value is a dict with keys:
        sequence  (str)
        cds_start (int | None)
        cds_end   (int | None)
        accession (str | None)

    On first run the JSON may not exist yet; in that case we migrate any
    existing FASTA entries automatically.
    """
    if not cache_path.exists():
        return _migrate_fasta_to_json(cache_path)

    with open(cache_path) as fh:
        return json.load(fh)


def _migrate_fasta_to_json(cache_path=LOCAL_CACHE_PATH):
    """One-time migration: read the legacy FASTA and write a JSON cache."""
    cache = {}

    if LOCAL_FASTA_PATH.exists():
        print(f"Migrating {LOCAL_FASTA_PATH} -> {cache_path} ...")
        gene_data_map = _load_fasta_for_migration(LOCAL_FASTA_PATH)
        for gene, (seq, cds_start, cds_end) in gene_data_map.items():
            cache[gene] = {
                "sequence": seq,
                "cds_start": cds_start,
                "cds_end": cds_end,
                "accession": None,
            }
        _save_cache(cache, cache_path)
        print(f"  Migrated {len(cache)} entries.")
    else:
        print(f"No existing FASTA found; starting with an empty cache at {cache_path}")
        _save_cache(cache, cache_path)

    return cache


def _load_fasta_for_migration(fasta_path):
    """Parse the legacy FASTA and return {gene: (seq, cds_start, cds_end)}."""
    gene_data_map = {}
    current_gene = None
    current_cds = None
    sequence_chunks = []

    def _flush(gene, chunks, cds):
        seq = normalize_rna_sequence(''.join(chunks))
        if seq:
            gene_data_map[gene] = (seq, cds[0] if cds else None, cds[1] if cds else None)

    with open(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_gene and sequence_chunks:
                    _flush(current_gene, sequence_chunks, current_cds)
                parts = line[1:].split()
                current_gene = parts[0]
                current_cds = None
                for part in parts[1:]:
                    if part.startswith('cds='):
                        try:
                            s, e = part[4:].split(':')
                            current_cds = (int(s), int(e))
                        except (ValueError, AttributeError):
                            pass
                sequence_chunks = []
                continue
            sequence_chunks.append(line)

    if current_gene and sequence_chunks:
        _flush(current_gene, sequence_chunks, current_cds)

    return gene_data_map


def _save_cache(cache, cache_path=LOCAL_CACHE_PATH):
    """Write the gene cache dict to disk atomically."""
    tmp = cache_path.with_suffix('.json.tmp')
    with open(tmp, 'w') as fh:
        json.dump(cache, fh, indent=2)
    tmp.replace(cache_path)


def upsert_gene_in_cache(gene_symbol, sequence, cds_start=None, cds_end=None,
                         accession=None, cache_path=LOCAL_CACHE_PATH):
    """Add or update a single gene entry in the JSON cache."""
    if not gene_symbol or not sequence:
        return

    seq = normalize_rna_sequence(sequence)
    if not seq:
        return

    cache = load_local_cache(cache_path)
    cache[str(gene_symbol).strip()] = {
        "sequence": seq,
        "cds_start": cds_start,
        "cds_end": cds_end,
        "accession": accession,
    }
    _save_cache(cache, cache_path)
    print(f"  Cached {gene_symbol} in {cache_path}")


def get_mrna_from_local_cache(gene_symbol, local_cache):
    """Return (sequence, cds_start, cds_end) from the JSON cache, or None."""
    entry = local_cache.get(gene_symbol)
    if entry is None:
        return None
    return entry["sequence"], entry.get("cds_start"), entry.get("cds_end")



def load_reference_sequences(csv_path=REFERENCE_CSV_PATH):
    """Load gene -> mRNA sequences from the reference training CSV."""
    try:
        ref = pd.read_csv(csv_path, low_memory=False)
    except Exception as error:
        print(f"Error reading reference dataset: {error}")
        return {}

    gene_mrna_map = {}
    for gene_symbol, gene_data in ref.groupby('gene_target_symbol_name'):
        mrna_values = gene_data['mRNA'].dropna().map(normalize_rna_sequence)
        mrna_values = [seq for seq in mrna_values if is_valid_mrna(seq)]
        if len(mrna_values) == 0:
            continue

        sequence = mrna_values[0]
        if sequence:
            gene_mrna_map[gene_symbol] = sequence

    return gene_mrna_map

def get_mrna_from_reference(gene_symbol, reference_sequences):
    """Return (sequence, None, None) from the reference training data, or None."""
    seq = reference_sequences.get(gene_symbol)
    return (seq, None, None) if seq else None


def fetch_genome_from_ncbi(gene_symbol):
    """Fetch viral genome from NCBI."""
    try:
        print(f"  Fetching {gene_symbol} genome from NCBI...")
        
        # Map common names to search terms
        virus_map = {
            'HBV': 'Hepatitis B virus[Organism] AND complete genome',
            'SARS-CoV-2': 'Severe acute respiratory syndrome coronavirus 2[Organism] AND complete genome'
        }
        
        search_term = virus_map.get(gene_symbol)
        if not search_term:
            return None
        
        handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=5)
        record = Entrez.read(handle)
        handle.close()
        
        if not record["IdList"]:
            print(f"    No genome found for {gene_symbol}")
            return None
        
        # Get the first complete genome (usually reference)
        gene_id = record["IdList"][0]
        
        # Fetch sequence summary to get info
        summary_handle = Entrez.esummary(db="nucleotide", id=gene_id)
        summary = Entrez.read(summary_handle)
        summary_handle.close()
        
        accession = summary[0].get('AccessionVersion', '')
        length = summary[0].get('Length', 0)
        
        print(f"    Fetching {accession} ({length} bp)...")
        fetch_handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="fasta", retmode="text")
        fasta_data = fetch_handle.read()
        fetch_handle.close()
        
        # Parse FASTA
        lines = fasta_data.strip().split('\n')
        sequence = ''.join(lines[1:])
        
        print(f"    ✓ Successfully fetched viral genome")
        time.sleep(0.35)
        return sequence.upper().replace('T', 'U'), None, None, accession
        
    except Exception as e:
        print(f"    Error fetching {gene_symbol} genome: {e}")
        time.sleep(1)
        return None

def fetch_mrna_from_ncbi(gene_symbol, species="Homo sapiens", max_length=50000):
    """Fetch mRNA transcript from NCBI Entrez.
    
    Args:
        gene_symbol: Gene symbol (e.g., 'VEGFA')
        species: Species name
        max_length: Maximum allowed sequence length (bp) to avoid genomic sequences
    """
    # Check if it's a viral genome
    if gene_symbol in ['HBV', 'SARS-CoV-2']:
        return fetch_genome_from_ncbi(gene_symbol)
    
    gene_symbol = gene_symbol.strip(' ')
    
    try:
        print(f"  Fetching {gene_symbol} from NCBI...")
        
        # Search specifically for mRNA with refseq filter
        search_term = f"{gene_symbol}[Gene Name] AND {species}[Organism] AND biomol_mrna[PROP] AND refseq[filter]"
        handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=20)
        record = Entrez.read(handle)
        handle.close()
        
        if not record["IdList"]:
            # Search fallback
            search_term = f"{gene_symbol}[Gene Name] AND {species}[Organism] AND mRNA"
            handle = Entrez.esearch(db="nucleotide", term=search_term, retmax=20)
            record = Entrez.read(handle)
            handle.close()
            if not record["IdList"]:
                print(f"    No mRNA results found for {gene_symbol}")
                return None
        
        # Fetch summaries to find appropriate mRNA
        summary_handle = Entrez.esummary(db="nucleotide", id=",".join(record["IdList"]))
        summaries = Entrez.read(summary_handle)
        summary_handle.close()
        
        # Sort by preference: NM_ > XM_, shorter transcripts first, avoid huge sequences
        candidates = []
        for summary in summaries:
            accession = summary.get('AccessionVersion', '')
            length = summary.get('Length', 0)
            title = summary.get('Title', '')
            
            # Skip if too large (likely genomic sequence)
            if length > max_length:
                print(f"    Skipping {accession} (too large: {length} bp)")
                continue
            
            # Prioritize NM_ (curated) over XM_ (predicted)
            priority = 0 if accession.startswith('NM_') else 1
            candidates.append((priority, length, summary['Id'], accession, title))
        
        if not candidates:
            print(f"    No suitable mRNA transcripts found (all too large or filtered)")
            return None
        
        # Sort by priority (NM_ first), then by length (shorter first for typical transcripts)
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        # Fetch the best candidate
        priority, length, gene_id, accession, title = candidates[0]
        
        print(f"    Fetching {accession} ({length} bp)...")
        fetch_handle = Entrez.efetch(db="nucleotide", id=gene_id, rettype="gb", retmode="text")
        gb_data = fetch_handle.read()
        fetch_handle.close()

        # Parse GenBank record to get sequence and annotated CDS boundaries.
        record = SeqIO.read(StringIO(gb_data), "genbank")
        sequence = str(record.seq).upper().replace('T', 'U')

        cds_start = None
        cds_end = None
        for feature in record.features:
            if feature.type == 'CDS':
                cds_start = int(feature.location.start)
                cds_end = int(feature.location.end)
                break

        cds_info = f" (CDS: {cds_start}–{cds_end})" if cds_start is not None else ""
        print(f"    ✓ Successfully fetched {accession}{cds_info}")
        time.sleep(0.35)  # Rate limiting for NCBI API (3 requests/sec)
        return sequence, cds_start, cds_end, accession
        
    except Exception as e:
        print(f"    Error fetching {gene_symbol}: {e}")
        time.sleep(1)  # Wait longer after error
        return None

def main():
    # Load the preprocessed CSV
    csv_path = PREPROCESSED_CSV_PATH
    df = pd.read_csv(csv_path, low_memory=False)
    local_cache = load_local_cache()
    reference_sequences = load_reference_sequences()

    print(f"Loaded {len(df)} rows from CSV")
    print(f"Loaded {len(local_cache)} gene sequences from local cache")
    print(f"Loaded {len(reference_sequences)} gene sequences from reference CSV")

    # Fix ALECT2 -> LECT2 (typo in original data)
    alect2_count = (df['gene_target_symbol_name'] == 'ALECT2').sum()
    if alect2_count > 0:
        print(f"Fixing ALECT2 -> LECT2 ({alect2_count} rows)")
        df.loc[df['gene_target_symbol_name'] == 'ALECT2', 'gene_target_symbol_name'] = 'LECT2'

    # Get unique genes
    unique_genes = df['gene_target_symbol_name'].dropna().unique()
    print(f"Found {len(unique_genes)} unique genes")

    # Build gene -> mRNA mapping
    gene_mrna_map = {}
    gene_cds_map = {}   # gene -> (cds_start, cds_end) when annotation is available

    print("\nBuilding gene -> mRNA mapping...")
    for i, gene in enumerate(unique_genes, 1):
        print(f"[{i}/{len(unique_genes)}] Processing {gene}...")

        # First try the local JSON cache
        result = get_mrna_from_local_cache(gene, local_cache)
        if result is not None:
            mrna, cds_start, cds_end = result
            print(f"  Found in local cache ({len(mrna)} bp)")
            gene_mrna_map[gene] = mrna
            if cds_start is not None:
                gene_cds_map[gene] = (cds_start, cds_end)
            continue

        # Then try reference dataset
        result = get_mrna_from_reference(gene, reference_sequences)
        if result is not None:
            mrna, cds_start, cds_end = result
            print(f"  Found in reference ({len(mrna)} bp)")
            gene_mrna_map[gene] = mrna
            continue

        # If not in reference, fetch from NCBI
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
        else:
            print(f"  WARNING: Could not fetch mRNA for {gene}")
            gene_mrna_map[gene] = None
    
    print(f"\n=== Summary ===")
    print(f"Successfully fetched: {sum(1 for v in gene_mrna_map.values() if v is not None)}/{len(unique_genes)} genes")
    print(f"Failed to fetch: {sum(1 for v in gene_mrna_map.values() if v is None)} genes")
    
    # Map mRNA to dataframe while preserving existing valid transcript sequences.
    print("\nMapping mRNA sequences to dataframe...")
    existing_mrna = df['mRNA'].map(normalize_rna_sequence)
    existing_valid_mask = existing_mrna.map(is_valid_mrna)
    mapped_mrna = df['gene_target_symbol_name'].map(gene_mrna_map).map(normalize_rna_sequence)

    df['mRNA_source'] = 'mapped_gene_reference'
    df.loc[existing_valid_mask, 'mRNA_source'] = 'existing_in_input'
    df.loc[~existing_valid_mask & mapped_mrna.isna(), 'mRNA_source'] = 'unresolved'

    df['mRNA'] = existing_mrna.where(existing_valid_mask, mapped_mrna)

    # Derive 5' and 3' UTR columns.
    # Use annotated CDS (from GenBank) per gene when available; fall back to
    # the longest-ORF heuristic for sequences from the local FASTA / reference CSV.
    gene_utr_map = {}
    for gene, mrna in gene_mrna_map.items():
        if not isinstance(mrna, str):
            gene_utr_map[gene] = (None, None)
            continue
        cds = gene_cds_map.get(gene)
        cds_start, cds_end = cds if cds else (None, None)
        gene_utr_map[gene] = find_utr_regions(mrna, cds_start=cds_start, cds_end=cds_end)

    df['mRNA_five_prime'] = df['gene_target_symbol_name'].map(
        lambda g: gene_utr_map.get(g, (None, None))[0]
    )
    df['mRNA_three_prime'] = df['gene_target_symbol_name'].map(
        lambda g: gene_utr_map.get(g, (None, None))[1]
    )

    # Check how many rows have mRNA
    rows_with_mrna = df['mRNA'].notna().sum()
    print(f"Rows with mRNA: {rows_with_mrna}/{len(df)} ({100*rows_with_mrna/len(df):.1f}%)")
    print("mRNA source breakdown:")
    print(df['mRNA_source'].value_counts(dropna=False).to_string())
    
    # Save updated CSV
    output_path = csv_path  # Overwrite the preprocessed file
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved updated CSV to: {output_path}")
    
    # Display sample
    print("\n=== Sample rows ===")
    sample_cols = ['gene_target_symbol_name', 'siRNA', 'mRNA']
    for _, row in df[sample_cols].head(3).iterrows():
        print(f"\nGene: {row['gene_target_symbol_name']}")
        print(f"siRNA: {row['siRNA']}")
        mrna = row['mRNA']
        if pd.notna(mrna) and isinstance(mrna, str):
            print(f"mRNA: {mrna[:80]}... ({len(mrna)} bp)")
        else:
            print(f"mRNA: None")

if __name__ == "__main__":
    main()

"""mrna_features.py

GC content of the 100 nt binding slice (whole, and split into start/middle/end
thirds) and the Vienna minimum free energy of that slice. works off the slice columns from
fm_utils.add_slice_columns.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def gc_content(seq):
    """fraction of G and C. None for a missing slice."""
    if not isinstance(seq, str) or len(seq) == 0:
        return None
    return (seq.count("G") + seq.count("C")) / len(seq)


def gc_by_third(seq):
    """GC of the start, middle and end thirds, to see if GC leans toward one end."""
    if not isinstance(seq, str) or len(seq) < 3:
        return (None, None, None)
    n = len(seq) // 3
    return (gc_content(seq[:n]), gc_content(seq[n:2 * n]), gc_content(seq[2 * n:]))


def fold_mfe(seq):
    """Vienna minimum free energy of one slice in kcal/mol. None for a missing slice.
    """
    if not isinstance(seq, str) or len(seq) == 0:
        return None
    import RNA
    _, mfe = RNA.fold(seq)
    return mfe


def mfe_unique(seqs):
    """fold only the distinct slices, since the same target site repeats across many
    rows. returns a mapping from each slice to its mfe so folding runs once per sequence.
    """
    cache = {}
    todo = sorted({s for s in seqs if isinstance(s, str) and s})
    for i, s in enumerate(todo, 1):
        if i % 500 == 0:
            print(f"  folded {i}/{len(todo)} unique slices")
        cache[s] = fold_mfe(s)
    return cache


def add_binding_features(df, with_mfe=True):
    """add GC (and optionally MFE) features of the binding slice.

    expects the mRNA_binding_slice column from fm_utils.add_slice_columns.
    """
    df = df.copy()
    slices = df["mRNA_binding_slice"]

    df["binding_gc"] = slices.map(gc_content)
    # split the 100 nt window into start, middle and end thirds to see where in the
    # guide footprint the GC sits
    thirds = slices.map(gc_by_third)
    df["binding_gc_start"] = [t[0] for t in thirds]
    df["binding_gc_mid"] = [t[1] for t in thirds]
    df["binding_gc_end"] = [t[2] for t in thirds]

    if with_mfe:
        mfe = mfe_unique(slices.tolist())
        df["binding_mfe"] = slices.map(mfe)

    return df


def add_utr_gc(df):
    """add GC of the two transcript-end regions: the 5' slice (first 100 nt, the
    short 5'UTR plus the start of the CDS) and the whole 3'UTR. these are the
    region-level features, separate from the local binding window above.

    expects mRNA_five_slice and mRNA_three_slice from fm_utils.add_slice_columns.
    """
    df = df.copy()
    df["utr5_gc"] = df["mRNA_five_slice"].map(gc_content)
    df["utr3_gc"] = df["mRNA_three_slice"].map(gc_content)
    return df


def spearman_vs_inhibition(df, feature_cols, target_column="Inhibition",
                           gene_column="gene_target_symbol_name"):
    """Spearman of each feature against the target, computed two ways.

    spearman_overall pools every siRNA together. spearman_by_gene ranks within each
    gene on its own and then averages, which is the fair number, since the models
    are scored on unseen genes (gene-grouped split) and only within-gene ranking
    transfers to a new gene. reading the two together is the point: a large overall
    next to a near-zero per-gene means the feature only tracks which gene it is, not
    which siRNA within the gene. a feature that is constant within a gene (a UTR
    feature, say) has no within-gene spread, so its per-gene value is NaN and only
    the overall number exists.
    """
    y = df[target_column]
    rows = []
    for col in feature_cols:
        x = df[col]
        keep = x.notna() & y.notna()
        overall = spearmanr(x[keep], y[keep]).correlation

        per_gene = []
        for _, g in df[keep].groupby(gene_column):
            # need spread in both to rank within a gene
            if g[col].nunique() > 1 and g[target_column].nunique() > 1:
                per_gene.append(spearmanr(g[col], g[target_column]).correlation)
        by_gene = np.nanmean(per_gene) if per_gene else np.nan

        rows.append({"feature": col, "spearman_overall": overall,
                     "spearman_by_gene": by_gene, "n": int(keep.sum())})
    return pd.DataFrame(rows)

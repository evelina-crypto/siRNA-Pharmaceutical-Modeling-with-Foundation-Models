"""fm_utils.py

turns each row's mRNA into three local regions and runs them through the Orthrus
RNA foundation model. the regions are the binding site (100 nt around where the
guide lands), the 5' start (first 100 nt, which spans the short 5'UTR plus the
start of the CDS) and the entire 3'UTR 

"""

import numpy as np
import pandas as pd


# orthrus 4-track one-hot uses the DNA alphabet (T, not U), like its README
def normalize_rna(seq):
    return seq.upper().replace("U", "T")


### Slicing

def slice_siRNA_100_bp(mrna, target_site_pct, guide_len, width=100):
    """100 nt window centred on where the guide reverse complement aligns.

    target_site_pct is the alignment start over mRNA length (see
    mrna_alignment.align_guide_to_mrna), so the start nucleotide is
    target_site_pct * len(mrna) and the footprint is start..start+guide_len.
    """
    if not isinstance(mrna, str) or pd.isna(target_site_pct):
        return None

    n = len(mrna)
    if n <= width:
        return normalize_rna(mrna)

    start = int(round(target_site_pct * n))
    centre = start + guide_len // 2
    lo = centre - width // 2
    # keep the window inside the transcript
    lo = max(0, min(lo, n - width))
    return normalize_rna(mrna[lo:lo + width])


def slice_5_prime_utr(seq, width=100):
    """the 5'UTR plus the start of the CDS."""
    if not isinstance(seq, str) or len(seq) == 0:
        return None
    return normalize_rna(seq[:width])


def slice_3_prime_utr(seq):
    """the entire 3'UTR."""
    if not isinstance(seq, str) or len(seq) == 0:
        return None
    return normalize_rna(seq)


def add_slice_columns(df, width=100):
    """add the three slice columns. a slice is None when its source is missing
    (binding needs mRNA + target_site_pct, 5' needs mRNA, 3' needs the annotated
    3'UTR). build_slice_embeddings turns those Nones into zero vectors plus a
    mask, so no separate present-flag is needed here.
    """
    df = df.copy()

    guide_len = df["Antisense_seqence"].str.len()

    df["mRNA_binding_slice"] = [
        slice_siRNA_100_bp(mrna, pct, glen, width)
        for mrna, pct, glen in zip(df["mRNA"], df["target_site_pct"], guide_len)
    ]
    df["mRNA_five_slice"] = [slice_5_prime_utr(s, width) for s in df["mRNA"]]
    df["mRNA_three_slice"] = [slice_3_prime_utr(s) for s in df["mRNA_three_prime"]]

    for col in ("mRNA_binding_slice", "mRNA_five_slice", "mRNA_three_slice"):
        print(f"{col}: {df[col].notna().sum()}/{len(df)} rows have a slice")

    return df


### Orthrus

# README defaults for the 4-track base model
orthrus_base_dir = "./models/orthrus_base_4_track"
orthrus_base_ckpt = "epoch=18-step=20000.ckpt"


def load_orthrus(model_dir=orthrus_base_dir, checkpoint_name=orthrus_base_ckpt, device="cpu", freeze=True):
    """load Orthrus, freeze=True for the static track (preprocess mRNA before putting through model)."""
    from orthrus.model_loader import load_model
    import torch

    model = load_model(model_dir, checkpoint_name=checkpoint_name)
    model = model.to(torch.device(device))

    if freeze:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    return model


def seq_to_one_hot(seq):
    """one sequence to the (4, L) one-hot Orthrus expects. shared by both tracks(static and 'at runtime')."""
    from orthrus.encoding import seq_to_oh
    return seq_to_oh(normalize_rna(seq)).T  #(4, L)


def embed_sequence(seq, model, device="cpu"):
    """frozen-Orthrus embedding of a single slice. returns a (256,) vector."""
    import torch

    one_hot = torch.tensor(seq_to_one_hot(seq), dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([one_hot.shape[2]]).to(device)
    with torch.no_grad():
        emb = model.representation(one_hot, lengths)
    return emb.squeeze(0).cpu().numpy()


def embed_unique(seqs, model, device="cpu", cache=None):
    """embed only the distinct slices, since 5' and 3'UTR slices repeat per gene and binding slices repeat per target
    site. cache maps each slice to its embedding.
    """
    if cache is None:
        cache = {}

    todo = {s for s in seqs if isinstance(s, str) and s not in cache}
    for i, s in enumerate(sorted(todo), 1):
        if i % 200 == 0:
            print(f"  embedded {i}/{len(todo)} unique slices")
        cache[s] = embed_sequence(s, model, device)

    return cache


def build_slice_embeddings(df, model, device="cpu", cache=None):
    """per-row Orthrus embeddings for the three slices.

    returns X_mrna of shape (N, 3, dim) ordered binding, five, three, and a
    (N, 3) present mask. a missing slice gets a zero vector, the mask records it.
    expects the slice columns from add_slice_columns.
    """
    slice_cols = ["mRNA_binding_slice", "mRNA_five_slice", "mRNA_three_slice"]

    all_slices = pd.concat([df[c] for c in slice_cols])
    cache = embed_unique(all_slices.tolist(), model, device, cache=cache)

    # dim is whatever Orthrus returned (256 for the 4-track base model)
    dim = len(next(iter(cache.values())))
    zero = np.zeros(dim, dtype=np.float32)

    n = len(df)
    x_mrna = np.zeros((n, 3, dim), dtype=np.float32)
    mask = np.zeros((n, 3), dtype=bool)
    for j, col in enumerate(slice_cols):
        for i, s in enumerate(df[col].tolist()):
            if isinstance(s, str):
                x_mrna[i, j] = cache.get(s, zero)
                mask[i, j] = True

    return x_mrna, mask, cache


### Cache persistence

def save_cache(cache, path):
    """save the slice -> vector cache so embeddings are computed once."""
    seqs = np.array(list(cache.keys()), dtype=object)
    vecs = np.stack(list(cache.values())).astype(np.float32)
    np.savez(path, seqs=seqs, vecs=vecs)


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    return {s: v for s, v in zip(data["seqs"], data["vecs"])}


# quick check of the slicing alone, runs without orthrus installed
if __name__ == "__main__":
    mrna = "ATG" + "ACGT" * 60  #test transcript (243 nt)
    three = "TTT" * 50
    print("binding:", slice_siRNA_100_bp(mrna, 0.5, guide_len=21, width=100))
    print("five:   ", slice_5_prime_utr(mrna, 100))
    print("three:  ", slice_3_prime_utr(three), "len", len(three))
    print("missing:", slice_siRNA_100_bp(None, 0.5, 21), slice_3_prime_utr(None))

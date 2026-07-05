"""fm_utils.py

turns each row's mRNA into three local regions and runs them through the Orthrus
RNA foundation model. the regions are the binding site (100 nt around where the
guide lands), the 5' start (first 100 nt, which spans the short 5'UTR plus the
start of the CDS) and either the first 100 nt (default) or the entire 3'UTR.

"""

from pathlib import Path

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


def parse_three_prime_width(value):
    """Parse a positive integer width or ``full`` (represented as ``None``)."""
    if value is None or str(value).lower() == "full":
        return None
    width = int(value)
    if width <= 0:
        raise ValueError("3' UTR width must be a positive integer or 'full'")
    return width


def slice_3_prime_utr(seq, width=100):
    """Return the first ``width`` 3' UTR bases, or the full UTR if width is None."""
    if not isinstance(seq, str) or len(seq) == 0:
        return None
    return normalize_rna(seq if width is None else seq[:width])


def add_slice_columns(df, width=100, three_prime_width=100):
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
    df["mRNA_three_slice"] = [
        slice_3_prime_utr(s, three_prime_width)
        for s in df["mRNA_three_prime"]
    ]

    for col in ("mRNA_binding_slice", "mRNA_five_slice", "mRNA_three_slice"):
        print(f"{col}: {df[col].notna().sum()}/{len(df)} rows have a slice")

    return df


### Orthrus

# Defaults for the downloaded Orthrus v1 4-track checkpoint
orthrus_base_dir = "./models/orthrus_v1_4_track"
orthrus_base_ckpt = "epoch=6-step=20000.ckpt"


def load_orthrus(model_dir=orthrus_base_dir, checkpoint_name=orthrus_base_ckpt, device="cpu", freeze=True):
    """load Orthrus, freeze=True for the static track (preprocess mRNA before putting through model)."""
    from orthrus.eval_utils import load_model
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
    from orthrus.gk_utils import seq_to_oh
    return seq_to_oh(normalize_rna(seq)).T  #(4, L)


def embed_sequence(seq, model, device="cpu"):
    """Frozen-Orthrus embedding of one slice; dimension depends on the checkpoint."""
    import torch

    one_hot = torch.tensor(seq_to_one_hot(seq), dtype=torch.float32).unsqueeze(0).to(device)
    lengths = torch.tensor([one_hot.shape[2]]).to(device)
    with torch.no_grad():
        emb = model.representation(one_hot, lengths)
    return emb.squeeze(0).cpu().numpy()


def embed_unique(seqs, model, device="cpu", cache=None, batch_size=32,
                 save_every=0, cache_path=None):
    """embed only the distinct slices, since 5' and 3'UTR slices repeat per gene and binding slices repeat per target
    site. cache maps each slice to its embedding.
    """
    import torch

    if cache is None:
        cache = {}

    # Deduplicate repeated slices and skip sequences already cached. Frozen
    # Orthrus is deterministic in eval mode, so recomputing an identical slice
    # would waste GPU time without adding information.
    todo = sorted(
        {s for s in seqs if isinstance(s, str) and s and s not in cache},
        key=len,
    )
    if todo and model is None:
        raise ValueError(
            f"Embedding cache is missing {len(todo)} slices, but no Orthrus model was supplied"
        )

    # Embed several similarly sized sequences per GPU call. Sorting by length
    # above limits padding, improving throughput and reducing wasted GPU memory.
    for start in range(0, len(todo), batch_size):
        batch_seqs = todo[start:start + batch_size]
        encoded = [seq_to_one_hot(s) for s in batch_seqs]
        lengths = torch.tensor([x.shape[1] for x in encoded], device=device)
        max_len = int(lengths.max().item())
        batch = torch.zeros(
            (len(encoded), 4, max_len), dtype=torch.float32, device=device,
        )
        for i, one_hot in enumerate(encoded):
            batch[i, :, :one_hot.shape[1]] = torch.from_numpy(one_hot).to(device)

        with torch.inference_mode():
            embeddings = model.representation(batch, lengths).float().cpu().numpy()
        cache.update(zip(batch_seqs, embeddings))

        done = min(start + batch_size, len(todo))
        if done == len(todo) or done % 200 < batch_size:
            print(f"  embedded {done}/{len(todo)} unique slices")
        # Save partial progress periodically so an interrupted job can resume
        # instead of recomputing every completed embedding.
        if cache_path and save_every and (
            done == len(todo) or done % save_every < batch_size
        ):
            save_cache(cache, cache_path)

    return cache


def build_slice_embeddings(df, model=None, device="cpu", cache=None,
                           batch_size=32, save_every=0, cache_path=None):
    """per-row Orthrus embeddings for the three slices.

    returns X_mrna of shape (N, 3, dim) ordered binding, five, three, and a
    (N, 3) present mask. a missing slice gets a zero vector, the mask records it.
    expects the slice columns from add_slice_columns.
    """
    slice_cols = ["mRNA_binding_slice", "mRNA_five_slice", "mRNA_three_slice"]

    all_slices = pd.concat([df[c] for c in slice_cols])
    # with a model, embed any slice not yet cached. with model=None assemble from
    # the cache only (the training env has no Orthrus installed), and a
    # slice missing from the cache stays absent via the mask below
    if model is not None:
        cache = embed_unique(
            all_slices.tolist(), model, device, cache=cache, batch_size=batch_size,
            save_every=save_every, cache_path=cache_path,
        )

    # Dimension is inferred from the checkpoint (512 for Orthrus v1 4-track).
    if not cache:
        raise ValueError("No valid mRNA slices were available to embed")
    dim = len(next(iter(cache.values())))

    n = len(df)
    x_mrna = np.zeros((n, 3, dim), dtype=np.float32)
    # Missing slices receive a zero vector. This separate presence mask lets
    # the downstream model distinguish "missing" from a genuine embedding
    # whose values happen to be near zero.
    mask = np.zeros((n, 3), dtype=bool)
    for j, col in enumerate(slice_cols):
        for i, s in enumerate(df[col].tolist()):
            # present only if actually embedded
            if isinstance(s, str) and s in cache:
                x_mrna[i, j] = cache[s]
                mask[i, j] = True

    return x_mrna, mask, cache


### Cache persistence

def save_cache(cache, path):
    """save the slice -> vector cache so embeddings are computed once."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seqs = np.array(list(cache.keys()), dtype=object)
    vecs = np.stack(list(cache.values())).astype(np.float32)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, seqs=seqs, vecs=vecs)
    tmp.replace(path)


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    return {s: v for s, v in zip(data["seqs"], data["vecs"])}


# quick check of the slicing alone, runs without orthrus installed
if __name__ == "__main__":
    mrna = "ATG" + "ACGT" * 60  #test transcript (243 nt)
    three = "TTT" * 50
    print("binding:", slice_siRNA_100_bp(mrna, 0.5, guide_len=21, width=100))
    print("five:   ", slice_5_prime_utr(mrna, 100))
    print("three:  ", slice_3_prime_utr(three), "len", len(slice_3_prime_utr(three)))
    print("missing:", slice_siRNA_100_bp(None, 0.5, 21), slice_3_prime_utr(None))

"""Generate a resumable frozen-Orthrus cache for all unique mRNA slices."""

import argparse
import os

import torch

from modeling.run_crew import DEFAULT_CMSIRNA_PATH, DEFAULT_HISTORIC_PATH
from utils.fm_utils import (
    add_slice_columns,
    build_slice_embeddings,
    load_cache,
    load_orthrus,
    parse_three_prime_width,
    save_cache,
)
from utils.merge_historic_data import load_merged_dataset
from utils.pipeline import SiRNADataPipeline


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "models", "orthrus_v1_4_track")


def main():
    parser = argparse.ArgumentParser(
        description="Embed unique binding/5-prime/3-prime slices with frozen Orthrus",
    )
    parser.add_argument("--cmsirna-path", default=DEFAULT_CMSIRNA_PATH)
    parser.add_argument("--historic-path", default=DEFAULT_HISTORIC_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", default="epoch=6-step=20000.ckpt")
    parser.add_argument(
        "--cache", default=None,
        help="output cache path; defaults to a strategy-specific filename",
    )
    parser.add_argument(
        "--three-prime-width", type=parse_three_prime_width, default=100,
        metavar="N|full",
        help="3' UTR slice length (default: 100); use 'full' for the entire UTR",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    strategy = "full3utr" if args.three_prime_width is None else f"3utr0_{args.three_prime_width}"
    cache_path = args.cache or os.path.join(
        REPO_ROOT, "cache", f"orthrus_v1_4track_strict_fetch_{strategy}.npz",
    )
    raw_df = load_merged_dataset(args.cmsirna_path, args.historic_path)
    if args.max_rows is not None:
        raw_df = raw_df.head(args.max_rows).copy()

    pipeline = SiRNADataPipeline(target_len=25, fetch_missing_mrna=True)
    enriched = pipeline.enrich_dataset_with_encodings(
        raw_df, strict_cleaning=True, add_mrna=True,
    )
    enriched = add_slice_columns(
        enriched, three_prime_width=args.three_prime_width,
    )

    # Reuse a partial cache after interruption. fm_utils only embeds sequences
    # that are absent from this mapping and checkpoints new progress regularly.
    cache = load_cache(cache_path) if os.path.exists(cache_path) else {}
    print(f"Loaded {len(cache)} cached slices; using {device}")
    model = load_orthrus(
        args.model_dir, checkpoint_name=args.checkpoint, device=device, freeze=True,
    )
    _, _, cache = build_slice_embeddings(
        enriched,
        model=model,
        device=device,
        cache=cache,
        batch_size=args.batch_size,
        save_every=args.save_every,
        cache_path=cache_path,
    )
    save_cache(cache, cache_path)
    print(f"Saved {len(cache)} unique slice embeddings to {cache_path}")


if __name__ == "__main__":
    main()

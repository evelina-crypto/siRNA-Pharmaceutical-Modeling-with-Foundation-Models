import argparse

import numpy as np

from utils.mrna_alignment import reverse_complement, align_guide_to_mrna
from utils.merge_historic_data import load_merged_dataset


def clean_rna(seq):
    if not isinstance(seq, str):
        return None
    seq = seq.upper().replace("U", "T")
    return seq if set(seq) <= set("ACGT") and len(seq) >= 15 else None


def head_match(a, b):
    #match from the 5' end so the 2 nt 3' overhang does not count against it
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else 0.0


def check(df):
    guide = df["Antisense_seqence"].map(clean_rna)
    passenger = df["Sense_seqence"].map(clean_rna)
    keep = guide.notna() & passenger.notna()

    standard = np.array([head_match(g, reverse_complement(p)) for g, p in zip(guide[keep], passenger[keep])])
    print(f"guide == revcomp(passenger): {100 * np.mean(standard > 0.8):.1f}% of {len(standard)} rows")

    # revcomp(guide) should hit the mRNA (antisense), the guide should not
    mrna = df["mRNA"]
    keep = guide.notna() & mrna.notna()
    correct = np.array([align_guide_to_mrna(g, m)["edit_distance"] for g, m in zip(guide[keep], mrna[keep])], float)
    print(f"revcomp(guide) aligns to mRNA (edit<=3): {100 * np.mean(correct <= 3):.1f}% of {len(correct)} rows")

    ok = np.mean(standard > 0.8) > 0.8
    print("Ok, no flip" if ok else " orientation is off")


def main():
    data_dir = ".."
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmsirna-path", default=f"{data_dir}/CMsiRNA_data_update.tsv")
    parser.add_argument("--historic-path", default=f"{data_dir}/Historic_Takayuki_hueskan_ichihara.csv")
    args = parser.parse_args()

    check(load_merged_dataset(args.cmsirna_path, args.historic_path))


if __name__ == "__main__":
    main()

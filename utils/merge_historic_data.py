"""merge_historic_data.py

Comments : Reshapes the historic siRNA data set into the CMsiRNAdb column
           layout so the same cleaning and encoding pipeline can take it in.
Date     : 2026/06/13
"""

import pandas as pd


def align_historic_to_cmsirna(path):
    """Reads the historic dataset and renames its columns to the CMsiRNAdb ones.  
    These siRNAs are unmodified, so the modification columns are left out and the chemistry
    encoder reads the absent values as all 0 modification matrices."""
    historic = pd.read_csv(path, low_memory=False)

    cmsirna = pd.DataFrame({
        "gene_target_symbol_name": historic["gene_target_symbol_name"],
        "Antisense_seqence": historic["siRNA_guide"], 
        "Sense_seqence": historic["siRNA_passenger"], 
        "Cell_Type": historic["cell_line_donor"],
        "Concentration": historic["siRNA_concentration"].astype(str) + historic["concentration_unit"],
        "Time_of_administration": historic["Duration_after_transfection_h"].astype(str) + "h",
        "Inhibition": 100 - historic["mRNA_remaining_pct"], #remaining mRNA is the inverse of knockdown
        "mRNA": historic["mRNA"], #the historic set already has mRNA
    })

    print(f"loaded {len(cmsirna)} historic rows")
    return cmsirna


def load_merged_dataset(cmsirna_path, historic_path):
    """Loads the primary dataset, reshapes the historic set to match, and stacks them into one
    frame. Target_Gene is renamed so both sources share gene_target_symbol_name."""
    cmsirna = pd.read_csv(cmsirna_path, sep="\t", low_memory=False)
    cmsirna = cmsirna.rename(columns={"Target_Gene": "gene_target_symbol_name"})
    historic = align_historic_to_cmsirna(historic_path)

    merged = pd.concat([cmsirna, historic], ignore_index=True)
    print(f"merged {len(cmsirna)} CMsiRNA and {len(historic)} historic rows into {len(merged)}")
    return merged

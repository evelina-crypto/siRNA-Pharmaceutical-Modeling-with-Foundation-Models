# siRNA Pharmaceutical Modeling with Foundation Models

This project predicts how well a chemically modified siRNA knocks down its target gene, from the siRNA itself, the experimental conditions it was tested in, and the target mRNA. It is built on CMsiRNA database, a large set of patent-curated therapeutic siRNAs, and it plugs an RNA foundation model (Orthrus) in as an extra source of information about the mRNA.

The model is a small multi-branch neural net: one branch reads the siRNA sequence and its chemistry, one reads the experimental conditions, and one reads Orthrus embeddings of the mRNA. On top of the models we run integrated-gradients attribution to see what each part of the model actually uses, and tie that back to biology.

## What is in here

- `utils/` cleaning, QC, sequence/chemistry encoding, mRNA alignment, normalization, and the Orthrus slicing helpers (`fm_utils.py`).
- `modeling/` the neural nets and training. `crew_model.py` is the multi-branch model,
  `run_crew.py` trains the sequence + experimental version, `run_crew_mrna.py` adds the
  static Orthrus branch, `build_orthrus_cache.py` precomputes the Orthrus embeddings.
- `modeling/` attribution: `run_attribution.py` for the sequence/experimental branches,
  `modeling/mRNA_attributions/` for the mRNA branch (which region the model leans on, and
  how that relates to GC content and folding energy).
- `classical_ml/` random forest/XGBoost baselines.
- `notebooks/`, `eda/` data exploration and the feature analyses.

## Setup

You need Python 3.11 or newer. A minimal environment:

```
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install torch captum
pip install edlib biopython
pip install logomaker ViennaRNA
```

The mRNA branch uses embeddings from Orthrus, an RNA foundation model. You only run it once though, when you build the embedding cache with
`build_orthrus_cache`. It needs a GPU and mamba-ssm, so it lives in its own
environment (see the notes in `fm_utils.py`). After the cache exists, training and attribution just read it and do not need Orthrus installed.

ToDo: explain paths to datasets

## Running it

Train the sequence + experimental model with grouped cross-validation by gene:

```
python -m modeling.run_crew \
  --cmsirna-path data/CMsiRNA_data_update.tsv \
  --historic-path data/Historic_Takayuki_hueskan_ichihara.csv
```

Add the Orthrus mRNA branch. This is two steps. First build the cache of Orthrus mRNA embeddings once. This is the only part that needs Orthrus and a GPU, and it expects the Orthrus checkpoint under `models/` (see `fm_utils.py`). Use `--three-prime-width full`:

```
python -m modeling.build_orthrus_cache \
  --cmsirna-path data/CMsiRNA_data_update.tsv \
  --historic-path data/Historic_Takayuki_hueskan_ichihara.csv \
  --three-prime-width full --cache slices.npz
```

Then train with that cache. The training itself only reads the saved embeddings:

```
python -m modeling.run_crew_mrna \
  --cmsirna-path data/CMsiRNA_data_update.tsv \
  --historic-path data/Historic_Takayuki_hueskan_ichihara.csv \
  --mrna-cache slices.npz --save-dir results/orthrus_static
```

The `--save-dir` keeps the per-fold weights and arrays, which the attribution step reads.

Attribution of the sequence and experimental branches (integrated gradients, saved as logos and importance plots). This reads the per-fold weights that `run_crew` saved, and reloads the data to line up the folds, so pass the same paths:

```
python -m modeling.run_attribution \
  --cmsirna-path data/CMsiRNA_data_update.tsv \
  --historic-path data/Historic_Takayuki_hueskan_ichihara.csv
```

Attribution of the mRNA branch (how much each branch and each mRNA region the model uses, plus the GC and folding-energy correlations), then the figure:

```
python -m modeling.mRNA_attributions.run_mrna_attribution --save-dir results/orthrus_static
python -m modeling.mRNA_attributions.plot_mrna_attribution --save-dir results/orthrus_static
```

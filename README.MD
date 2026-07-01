### Evolutionary Optimization for the Identification of Biologically Relevant CpG Sites in DNA Methylation Arrays

----------------------------------------------------------------------------

#### 1) Download and install every dependency through uv (you can check uv's setup guide [here](https://docs.astral.sh/uv/).

```bash
uv sync 
```


#### 2) Create a data directory where you will store your bval.csv as well as the relevant metadata.csv



#### 3) Run data preprocessor to narrow down CpG sites:

```bash
uv run preprocess_methylation.py --betas ./data/bvals.csv --targets ./data/metadata.csv --output ./data/bvals_processed.csv
```


#### 4) Run feature ranking on narrowed down CpG list (few thousands):

```bash
uv run rank_features.py --input ./data/bvals_processed.csv --target-col condition --output-kw ./data/kw_ranking.csv --output-rf ./data/rf_ranking.csv
```


#### 5) Genetic algorithm execution (terminal)

```bash
uv run ga_select_features.py --betas ./data/bvals_processed.csv --kw ./data/kw_ranking.csv --rf ./data/rf_ranking.csv --gene-map ./data/cpg_annotation.csv
```



#### 6) Run classification comparision (5000 vs 209 CpG sites)

```bash
uv run classify_compare.py --betas ./data/bvals_processed.csv --selected ./data/selected_cpgs.csv --n-folds 10 --random-state 42
```


#### 7) Run feature selection algorithm comparison

```bash
uv run compare_fs_methods.py --betas ./data/bvals_processed.csv --ga-selected ./data/selected_cpgs.csv --gene-map ./data/cpg_annotation.csv --results-dir ./results --random-state 42
```


### Data availability:

The beta value matrix used for the baseline experiment as well as relevant metadata are available upon request.
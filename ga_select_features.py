"""
Genetic Algorithm Feature Selection for DNA Methylation
========================================================

Narrows down the preprocessed CpG set (a few thousand) to a few hundred
using a genetic algorithm with a multi-criteria fitness function.

Each GA individual is a binary mask over CpGs (1 = selected, 0 = not).
Each individual's selected subset is scored on 4 criteria:

    1. Mean Kruskal-Wallis H statistic of selected CpGs   -- maximize
    2. Mean Random Forest importance of selected CpGs     -- maximize
    3. Gene compression: 1 - (unique_genes / n_selected)  -- maximize
       (rewards subsets where many CpGs map to few genes;
        biologically: redundant CpG signals on the same gene reinforce
        that gene's role rather than scattering signal across many genes)
    4. Mean contrast score across selected CpGs           -- maximize
       (computed on the per-class mean beta matrix:
        for each CpG, find the class whose mean is most "set apart"
        from the other classes; high contrast = condition-specific marker)

Combination: each criterion is min-max normalized to [0, 1] up front
using the global per-CpG distribution. The fitness of a subset is then
the (weighted) MEAN of the four normalized criterion values, also in
[0, 1]. Default weights are equal (simple arithmetic mean). With this
formulation, fitness has a clean interpretation: 1.0 = best possible on
every criterion; 0.5 = average on every criterion; values are directly
comparable both within and across generations.

Subset size: bounded in [--min-size, --max-size] (default 100-300).
Individuals violating the bounds are repaired (random bits flipped to
satisfy the constraint).

Outputs:
  1. Mean beta-value matrix (4 conditions x CpGs) -- the table used to
     compute contrast scores; useful in its own right.
  2. Final selected CpG list with all 4 per-CpG scores.
  3. Per-generation GA progress log.

Usage:
    python ga_select_features.py \\
        --betas preprocessed.csv \\
        --kw ranking_kruskal_wallis.csv \\
        --rf ranking_random_forest.csv \\
        --gene-map cpg_to_gene.csv \\
        --output-mean-beta mean_beta.csv \\
        --output-selected selected_cpgs.csv \\
        --min-size 100 --max-size 300 \\
        --population 100 --generations 100 \\
        --random-state 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading and alignment
# ---------------------------------------------------------------------------
def load_inputs(
    betas_path: Path,
    kw_path: Path,
    rf_path: Path,
    gene_map_path: Path,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Load all inputs and align them on a common CpG axis.

    Returns:
        X            : beta matrix (samples x CpGs), aligned to common CpGs
        y            : target labels (samples,)
        kw_score     : KW H statistic per CpG (CpGs,)
        rf_score     : RF importance per CpG (CpGs,)
        gene_map     : CpG -> gene symbol (CpGs,), missing -> NaN
    """
    print(f"[load] reading beta matrix from {betas_path}")
    df = pd.read_csv(betas_path, index_col=0)
    if target_col not in df.columns:
        sys.exit(f"[error] target column '{target_col}' not found in {betas_path}")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    print(f"[load] beta matrix: {X.shape[0]} samples x {X.shape[1]:,} CpGs")
    print(f"[load] classes: {sorted(y.unique())}")

    print(f"[load] reading KW ranking from {kw_path}")
    kw_df = pd.read_csv(kw_path)
    kw_score = pd.Series(
        kw_df["h_statistic"].values, index=kw_df["feature"].values, name="kw"
    )

    print(f"[load] reading RF ranking from {rf_path}")
    rf_df = pd.read_csv(rf_path)
    # RF ranking can have either 'importance' (MDI) or 'importance_mean' (perm)
    rf_col = "importance" if "importance" in rf_df.columns else "importance_mean"
    rf_score = pd.Series(
        rf_df[rf_col].values, index=rf_df["feature"].values, name="rf"
    )

    print(f"[load] reading gene map from {gene_map_path}")
    gene_df = pd.read_csv(gene_map_path)
    if gene_df.shape[1] < 2:
        sys.exit(f"[error] gene map needs >=2 columns (CpG, gene)")
    # First column = CpG ID, second = gene symbol
    gene_map = pd.Series(
        gene_df.iloc[:, 1].values, index=gene_df.iloc[:, 0].values, name="gene"
    )

    # Align everything to the CpGs that appear in the beta matrix
    common = X.columns
    kw_score = kw_score.reindex(common)
    rf_score = rf_score.reindex(common)
    gene_map = gene_map.reindex(common)

    n_missing_kw = int(kw_score.isna().sum())
    n_missing_rf = int(rf_score.isna().sum())
    n_missing_gene = int(gene_map.isna().sum())
    if n_missing_kw or n_missing_rf:
        print(
            f"[load] warning: {n_missing_kw} CpGs missing KW score, "
            f"{n_missing_rf} missing RF score; filling with 0 (worst)"
        )
        kw_score = kw_score.fillna(0.0)
        rf_score = rf_score.fillna(0.0)
    print(f"[load] {n_missing_gene:,} CpGs have no gene annotation "
          f"(treated as 'unmapped' -- penalize selection)")

    return X, y, kw_score, rf_score, gene_map


# ---------------------------------------------------------------------------
# Mean beta matrix and contrast scores (computed once, reused every gen)
# ---------------------------------------------------------------------------
def compute_mean_beta_matrix(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Group-by-class mean of beta values: returns (n_classes x n_CpGs)."""
    print(f"[mean] computing per-class mean beta matrix")
    mean_beta = X.groupby(y).mean()
    print(f"[mean] shape: {mean_beta.shape} (classes x CpGs)")
    return mean_beta


def compute_contrast_scores(mean_beta: pd.DataFrame) -> pd.Series:
    """Per-CpG contrast score on the mean-beta matrix.

    For each CpG (column), the 4 condition means are col = [v0, v1, v2, v3].
    For each condition i, contrast_i = sum_j |col[i] - col[j]| (j != i).
    Returns max over i. High score = at least one condition stands apart.
    """
    print(f"[contrast] computing contrast scores for {mean_beta.shape[1]:,} CpGs")
    arr = mean_beta.values  # (n_classes, n_cpgs)
    n_classes = arr.shape[0]
    if n_classes < 2:
        sys.exit("[error] contrast score requires at least 2 classes")

    # Vectorized: for each CpG, contrast_i = sum_j |arr[i,c] - arr[j,c]|
    # Compute pairwise abs diffs across the class axis, then for each row i
    # sum the absolute diffs to all other rows.
    # Shape trick: |arr[i] - arr[j]| via broadcasting -> (n_classes, n_classes, n_cpgs)
    diffs = np.abs(arr[:, None, :] - arr[None, :, :])  # (C, C, n_cpgs)
    # contrast_i[c] = sum_j diffs[i, j, c]   (diagonal j=i contributes 0)
    contrasts = diffs.sum(axis=1)  # (C, n_cpgs)
    max_contrast = contrasts.max(axis=0)  # (n_cpgs,)

    # NaN propagation: if any class mean is NaN for that CpG, score is NaN
    nan_cols = np.isnan(arr).any(axis=0)
    max_contrast[nan_cols] = np.nan

    result = pd.Series(max_contrast, index=mean_beta.columns, name="contrast")
    n_nan = int(result.isna().sum())
    if n_nan:
        print(f"[contrast] {n_nan} CpGs have NaN contrast (NaN class mean); "
              f"filling with 0 (worst)")
        result = result.fillna(0.0)
    print(f"[contrast] range: {result.min():.4f} -- {result.max():.4f}")
    return result


# ---------------------------------------------------------------------------
# Fitness function
# ---------------------------------------------------------------------------
class FitnessEvaluator:
    """Holds precomputed per-CpG scores and evaluates subsets quickly.

    The four criteria (KW, RF, gene compression, contrast) are on totally
    different natural scales (KW in hundreds, RF in thousandths, the
    other two in [0, 1]). To average them meaningfully, we min-max
    normalize each criterion to [0, 1] up front using the global range
    of values it can take. After normalization, fitness is simply the
    mean of the four normalized criterion values: a number in [0, 1]
    where 1 = "best possible on every criterion".

    Per-CpG normalization (criteria 1, 2, 4):
        For criteria computed as a mean over selected CpGs (KW, RF,
        contrast), we min-max normalize the per-CpG scores themselves.
        The mean of normalized per-CpG scores is then automatically
        in [0, 1] -- it equals the average normalized score of the
        selected CpGs.

    Gene compression (criterion 3):
        Already in [0, 1] by construction (1 - unique_genes / n_selected),
        so no normalization needed.
    """

    def __init__(
        self,
        kw_score: np.ndarray,
        rf_score: np.ndarray,
        contrast_score: np.ndarray,
        gene_codes: np.ndarray,    # int code per CpG; -1 = unmapped
        n_genes_total: int,
        weights: tuple[float, float, float, float],
    ):
        # Min-max normalize each per-CpG score array to [0, 1].
        # If max == min (degenerate), all values become 0.
        self.kw = self._minmax(kw_score)
        self.rf = self._minmax(rf_score)
        self.contrast = self._minmax(contrast_score)
        self.gene_codes = gene_codes
        self.n_genes_total = n_genes_total
        # Weights are normalized so they sum to 1 -> fitness stays in [0, 1]
        w = np.asarray(weights, dtype=float)
        self.weights = w / w.sum() if w.sum() > 0 else np.full(4, 0.25)

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        """Min-max scale to [0, 1]; flat arrays become all zeros."""
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - lo) / (hi - lo)

    def raw_scores(self, mask: np.ndarray) -> np.ndarray:
        """Return the 4 normalized criterion values for one individual."""
        sel = mask.astype(bool)
        n_sel = int(sel.sum())
        if n_sel == 0:
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Criteria 1, 2, 4: mean of normalized per-CpG scores -> in [0, 1]
        kw_mean = self.kw[sel].mean()
        rf_mean = self.rf[sel].mean()
        contrast_mean = self.contrast[sel].mean()

        # Criterion 3: gene compression, already in [0, 1]
        codes = self.gene_codes[sel]
        mapped = codes[codes >= 0]
        if mapped.size == 0:
            gene_score = 0.0
        else:
            n_unique_genes = np.unique(mapped).size
            gene_score = 1.0 - (n_unique_genes / n_sel)

        return np.array([kw_mean, rf_mean, gene_score, contrast_mean])

    def population_raw_scores(self, population: np.ndarray) -> np.ndarray:
        """Return raw scores for every individual: (pop_size, 4)."""
        return np.array([self.raw_scores(ind) for ind in population])

    def fitness(self, raw_scores: np.ndarray) -> np.ndarray:
        """Combine the 4 normalized criterion values into a single fitness
        by weighted mean. With default equal weights this is the simple
        arithmetic mean of (KW_norm, RF_norm, gene, contrast).

        Fitness is in [0, 1]. 1.0 = best possible on every criterion.
        Comparable both within and across generations.
        """
        return raw_scores @ self.weights


# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------
def initialize_population(
    pop_size: int, n_features: int, min_size: int, max_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Random binary masks with selected count uniformly in [min_size, max_size]."""
    pop = np.zeros((pop_size, n_features), dtype=np.int8)
    for i in range(pop_size):
        k = rng.integers(min_size, max_size + 1)
        idx = rng.choice(n_features, size=k, replace=False)
        pop[i, idx] = 1
    return pop


def repair(mask: np.ndarray, min_size: int, max_size: int, rng: np.random.Generator) -> np.ndarray:
    """Force a mask to satisfy [min_size, max_size] by adding/removing bits."""
    n_sel = int(mask.sum())
    if min_size <= n_sel <= max_size:
        return mask
    if n_sel < min_size:
        zero_idx = np.where(mask == 0)[0]
        n_add = min_size - n_sel
        chosen = rng.choice(zero_idx, size=n_add, replace=False)
        mask[chosen] = 1
    else:  # n_sel > max_size
        one_idx = np.where(mask == 1)[0]
        n_remove = n_sel - max_size
        chosen = rng.choice(one_idx, size=n_remove, replace=False)
        mask[chosen] = 0
    return mask


def tournament_select(
    fitness: np.ndarray, tournament_size: int, rng: np.random.Generator
) -> int:
    """Standard tournament selection: pick k random individuals, return best."""
    contenders = rng.integers(0, len(fitness), size=tournament_size)
    return int(contenders[np.argmax(fitness[contenders])])


def uniform_crossover(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Each bit independently inherited from one parent or the other."""
    mask = rng.random(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2).astype(np.int8)
    c2 = np.where(mask, p2, p1).astype(np.int8)
    return c1, c2


def mutate(mask: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    """Bit-flip mutation: each bit flipped with probability mutation_rate."""
    flips = rng.random(len(mask)) < mutation_rate
    mask = mask.copy()
    mask[flips] = 1 - mask[flips]
    return mask


def run_ga(
    evaluator: FitnessEvaluator,
    n_features: int,
    pop_size: int,
    n_generations: int,
    min_size: int,
    max_size: int,
    mutation_rate: float,
    tournament_size: int,
    elitism: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict]]:
    """Run the GA. Returns the best mask found and a per-generation log."""
    print(f"[ga] initializing population ({pop_size} individuals, "
          f"size in [{min_size}, {max_size}])")
    population = initialize_population(pop_size, n_features, min_size, max_size, rng)
    raw = evaluator.population_raw_scores(population)
    fitness = evaluator.fitness(raw)

    best_idx = int(np.argmax(fitness))
    best_mask = population[best_idx].copy()
    best_raw = raw[best_idx].copy()
    best_fit = float(fitness[best_idx])
    log = []

    print(f"[ga] running {n_generations} generations")
    print(f"[ga]   gen   best   mean | KW_n   RF_n  gene  contr_n | size")
    for gen in range(n_generations):
        new_pop = []

        # Elitism: carry over the top `elitism` individuals
        if elitism > 0:
            elite_idx = np.argsort(fitness)[-elitism:]
            for idx in elite_idx:
                new_pop.append(population[idx].copy())

        # Fill the rest via selection -> crossover -> mutation -> repair
        while len(new_pop) < pop_size:
            i1 = tournament_select(fitness, tournament_size, rng)
            i2 = tournament_select(fitness, tournament_size, rng)
            c1, c2 = uniform_crossover(population[i1], population[i2], rng)
            c1 = mutate(c1, mutation_rate, rng)
            c2 = mutate(c2, mutation_rate, rng)
            c1 = repair(c1, min_size, max_size, rng)
            c2 = repair(c2, min_size, max_size, rng)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = np.array(new_pop[:pop_size])
        raw = evaluator.population_raw_scores(population)
        fitness = evaluator.fitness(raw)

        gen_best_idx = int(np.argmax(fitness))
        if fitness[gen_best_idx] > best_fit:
            best_fit = float(fitness[gen_best_idx])
            best_mask = population[gen_best_idx].copy()
            best_raw = raw[gen_best_idx].copy()

        sizes = population.sum(axis=1)
        log.append({
            "generation": gen + 1,
            "best_fitness": float(fitness.max()),
            "mean_fitness": float(fitness.mean()),
            "best_kw_norm": float(raw[gen_best_idx, 0]),
            "best_rf_norm": float(raw[gen_best_idx, 1]),
            "best_gene": float(raw[gen_best_idx, 2]),
            "best_contrast_norm": float(raw[gen_best_idx, 3]),
            "best_size": int(sizes[gen_best_idx]),
            "mean_size": float(sizes.mean()),
        })

        if (gen + 1) % max(1, n_generations // 20) == 0 or gen == 0:
            r = log[-1]
            print(
                f"[ga]  {gen + 1:4d}  {r['best_fitness']:.3f}  {r['mean_fitness']:.3f} | "
                f"{r['best_kw_norm']:.3f}  {r['best_rf_norm']:.3f}  "
                f"{r['best_gene']:.3f}  {r['best_contrast_norm']:.3f} | {r['best_size']:4d}"
            )

    print(f"[ga] done. best fitness: {best_fit:.4f} (max possible: 1.0), "
          f"best subset size: {int(best_mask.sum())}")
    print(f"[ga] best normalized scores: "
          f"KW={best_raw[0]:.3f}, RF={best_raw[1]:.3f}, "
          f"gene_compression={best_raw[2]:.3f}, contrast={best_raw[3]:.3f}")
    return best_mask, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Genetic algorithm feature selection on methylation data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Inputs
    p.add_argument("--betas", type=Path, required=True,
                   help="Preprocessed CSV (samples x CpGs + condition column)")
    p.add_argument("--kw", type=Path, required=True,
                   help="Kruskal-Wallis ranking CSV")
    p.add_argument("--rf", type=Path, required=True,
                   help="Random Forest ranking CSV")
    p.add_argument("--gene-map", type=Path, required=True,
                   help="CSV mapping CpG IDs to gene symbols (2 columns)")
    p.add_argument("--target-col", type=str, default="condition",
                   help="Name of the target column in the betas CSV")
    # Outputs
    p.add_argument("--output-mean-beta", type=Path, default=Path("mean_beta.csv"),
                   help="Output: per-class mean beta matrix")
    p.add_argument("--output-selected", type=Path, default=Path("selected_cpgs.csv"),
                   help="Output: GA-selected CpGs with their per-CpG scores")
    p.add_argument("--output-log", type=Path, default=Path("ga_log.csv"),
                   help="Output: per-generation GA progress log")
    # Subset size constraints
    p.add_argument("--min-size", type=int, default=100,
                   help="Minimum number of CpGs in any subset")
    p.add_argument("--max-size", type=int, default=300,
                   help="Maximum number of CpGs in any subset")
    # GA hyperparameters
    p.add_argument("--population", type=int, default=100,
                   help="Population size")
    p.add_argument("--generations", type=int, default=100,
                   help="Number of generations")
    p.add_argument("--mutation-rate", type=float, default=None,
                   help="Per-bit mutation rate (default: 1/n_features)")
    p.add_argument("--tournament-size", type=int, default=3,
                   help="Tournament size for selection")
    p.add_argument("--elitism", type=int, default=2,
                   help="Number of top individuals carried over each generation")
    # Fitness weights (raw, will be applied after z-normalization)
    p.add_argument("--w-kw", type=float, default=1.0,
                   help="Weight on KW criterion in fitness")
    p.add_argument("--w-rf", type=float, default=1.0,
                   help="Weight on RF criterion in fitness")
    p.add_argument("--w-gene", type=float, default=1.0,
                   help="Weight on gene compression criterion")
    p.add_argument("--w-contrast", type=float, default=1.0,
                   help="Weight on contrast criterion")
    # Reproducibility
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_size > args.max_size:
        sys.exit(f"[error] --min-size ({args.min_size}) > --max-size ({args.max_size})")

    rng = np.random.default_rng(args.random_state)
    t_start = time.time()

    # 1) Load and align everything
    X, y, kw_score, rf_score, gene_map = load_inputs(
        args.betas, args.kw, args.rf, args.gene_map, args.target_col
    )
    if args.max_size >= X.shape[1]:
        sys.exit(f"[error] --max-size ({args.max_size}) >= n_features ({X.shape[1]})")

    # 2) Mean beta matrix + contrast scores
    mean_beta = compute_mean_beta_matrix(X, y)
    contrast_score = compute_contrast_scores(mean_beta)

    # Save the mean beta matrix as required output #1
    print(f"[out] writing mean beta matrix to {args.output_mean_beta}")
    mean_beta.to_csv(args.output_mean_beta)

    # 3) Encode gene symbols as integer codes for fast unique-counting
    # Treat NaN/missing as code -1 (unmapped); each unique gene gets a unique code
    gene_codes = pd.Categorical(gene_map.values).codes.astype(np.int64)
    n_genes_total = int((gene_codes >= 0).sum() and gene_codes.max() + 1)
    print(f"[ga] {n_genes_total:,} unique genes in annotation")

    # 4) Build evaluator
    evaluator = FitnessEvaluator(
        kw_score=kw_score.values,
        rf_score=rf_score.values,
        contrast_score=contrast_score.values,
        gene_codes=gene_codes,
        n_genes_total=n_genes_total,
        weights=(args.w_kw, args.w_rf, args.w_gene, args.w_contrast),
    )

    # 5) Run GA
    n_features = X.shape[1]
    mutation_rate = args.mutation_rate if args.mutation_rate is not None else 1.0 / n_features
    print(f"[ga] mutation rate: {mutation_rate:.6f}")

    best_mask, log = run_ga(
        evaluator=evaluator,
        n_features=n_features,
        pop_size=args.population,
        n_generations=args.generations,
        min_size=args.min_size,
        max_size=args.max_size,
        mutation_rate=mutation_rate,
        tournament_size=args.tournament_size,
        elitism=args.elitism,
        rng=rng,
    )

    # 6) Build the selected-CpGs output table with all per-CpG scores
    selected_idx = np.where(best_mask == 1)[0]
    selected_cpgs = X.columns[selected_idx]
    out = pd.DataFrame({
        "feature": selected_cpgs,
        "gene": gene_map.loc[selected_cpgs].values,
        "kw_h_statistic": kw_score.loc[selected_cpgs].values,
        "rf_importance": rf_score.loc[selected_cpgs].values,
        "contrast_score": contrast_score.loc[selected_cpgs].values,
    })
    # Sort selected CpGs by KW H statistic (a sensible default ordering)
    out = out.sort_values("kw_h_statistic", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))

    print(f"[out] writing {len(out)} selected CpGs to {args.output_selected}")
    out.to_csv(args.output_selected, index=False)

    print(f"[out] writing GA log to {args.output_log}")
    pd.DataFrame(log).to_csv(args.output_log, index=False)

    elapsed = time.time() - t_start
    print(f"[done] total runtime: {elapsed:.1f}s")
    print(f"[done] selected {len(out)} CpGs mapping to "
          f"{out['gene'].dropna().nunique()} unique genes")


if __name__ == "__main__":
    main()

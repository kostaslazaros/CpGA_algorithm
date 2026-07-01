"""
DNA Methylation Beta-Value Preprocessing Pipeline
==================================================

Preprocesses prostate cancer DNA methylation data:
  1. Reads beta-value matrix (CpGs x Samples in input) and target labels;
     transposes so downstream code sees Samples x CpGs.
  2. Handles missing values (drop CpG columns, or class-aware KNN imputation
     where neighbors are restricted to the same class as the missing sample).
  3. Selects informative CpGs by ranking on Median Absolute Deviation (MAD)
     and keeping the top N. CpGs without dynamic range across samples sink
     to the bottom of the ranking and are dropped without any threshold.

Usage:
    python preprocess_methylation.py \
        --betas betas.csv \
        --targets targets.csv \
        --output preprocessed.csv \
        --missing knn \
        --k 5 \
        --top-n 5000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_data(betas_path: Path, targets_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load beta matrix (transposes so samples are rows) and target labels.

    The input beta CSV is assumed to be CpGs (rows) x Samples (columns),
    which is the standard layout from most methylation pipelines. We
    transpose it so downstream code sees samples as rows.
    """
    print(f"[load] reading betas from {betas_path}")
    betas = pd.read_csv(betas_path, index_col=0)
    print(f"[load] raw shape: {betas.shape} (CpGs x Samples)")

    # Transpose: rows = samples, columns = CpG sites
    betas = betas.T
    betas.index.name = "sample_id"
    print(f"[load] transposed shape: {betas.shape} (Samples x CpGs)")

    print(f"[load] reading targets from {targets_path}")
    targets_df = pd.read_csv(targets_path, index_col=0)
    if targets_df.shape[1] != 1:
        # Be forgiving: take the first column if multiple
        print(
            f"[load] target file has {targets_df.shape[1]} columns; "
            f"using first column '{targets_df.columns[0]}'"
        )
    targets = targets_df.iloc[:, 0]
    targets.name = "condition"

    # Align samples between the two files
    common = betas.index.intersection(targets.index)
    if len(common) == 0:
        sys.exit(
            "[error] no overlap between sample IDs in betas and targets. "
            "Check that the index columns match."
        )
    if len(common) < len(betas):
        print(
            f"[load] dropping {len(betas) - len(common)} samples without targets"
        )
    betas = betas.loc[common]
    targets = targets.loc[common]

    print(f"[load] final aligned shape: {betas.shape}")
    print(f"[load] class distribution:\n{targets.value_counts().to_string()}")
    return betas, targets


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------
def report_missing(betas: pd.DataFrame) -> int:
    """Report missingness and return total NA count."""
    n_missing = int(betas.isna().sum().sum())
    if n_missing == 0:
        print("[na] no missing values detected")
        return 0

    n_cells = betas.size
    cpgs_with_na = int(betas.isna().any(axis=0).sum())
    samples_with_na = int(betas.isna().any(axis=1).sum())
    print(
        f"[na] {n_missing:,} missing cells "
        f"({100 * n_missing / n_cells:.3f}% of matrix)"
    )
    print(
        f"[na] {cpgs_with_na:,} CpGs and {samples_with_na:,} samples "
        f"contain at least one NA"
    )
    return n_missing


def drop_missing(betas: pd.DataFrame, targets: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop CpG columns containing any NA. We drop columns rather than rows
    because we have only 370 samples and don't want to lose any of them
    (especially given class imbalance), while we have many redundant CpGs."""
    before = betas.shape[1]
    betas = betas.dropna(axis=1, how="any")
    after = betas.shape[1]
    print(f"[na] dropped {before - after:,} CpG columns containing NAs")
    return betas, targets


def knn_impute_per_class(
    betas: pd.DataFrame, targets: pd.Series, k: int
) -> pd.DataFrame:
    """Impute NAs using KNN, restricting neighbors to the same class as the
    sample with the missing value. This preserves class-conditional
    structure, which matters for downstream supervised learning.

    Implementation: run a separate KNNImputer per class subset.
    """
    print(f"[na] running per-class KNN imputation (k={k})")
    imputed_blocks = []
    for cls, idx in targets.groupby(targets).groups.items():
        block = betas.loc[idx]
        n_class = len(idx)
        # KNNImputer needs at least k+1 samples (the sample itself + k neighbors)
        effective_k = min(k, max(1, n_class - 1))
        if effective_k < k:
            print(
                f"[na]   class '{cls}' has only {n_class} samples; "
                f"using k={effective_k}"
            )
        else:
            print(f"[na]   class '{cls}': {n_class} samples, k={effective_k}")

        imputer = KNNImputer(n_neighbors=effective_k, weights="distance")
        imputed = imputer.fit_transform(block.values)
        imputed_blocks.append(
            pd.DataFrame(imputed, index=block.index, columns=block.columns)
        )

    result = pd.concat(imputed_blocks).loc[betas.index]  # restore order
    remaining = int(result.isna().sum().sum())
    if remaining > 0:
        # Can happen if a CpG is entirely NA within a class
        print(
            f"[na] {remaining:,} cells still NA after per-class KNN "
            f"(entire-class missingness); filling with class mean of other classes"
        )
        result = result.fillna(result.mean())
    print("[na] imputation complete")
    return result


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def select_informative_cpgs(betas: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Rank CpGs by Median Absolute Deviation (MAD) and keep the top_n.

    Why MAD:
        MAD measures how much a CpG's beta values vary across samples,
        using the median of absolute deviations from the median. CpGs
        that don't vary across samples carry no information about
        conditions regardless of what those conditions are -- they
        lack biological dynamic range. Ranking by MAD pushes such sites
        (constitutively methylated or unmethylated across the cohort)
        to the bottom automatically, with no threshold needed.

    Why MAD instead of variance:
        Beta values are bounded in [0, 1] and often bimodal (clustered
        near 0 or 1). Variance is sensitive to outliers and to the
        bounded/bimodal distribution; MAD is robust to both. The factor
        1.4826 makes MAD a consistent estimator of standard deviation
        under normality, so values stay on a familiar scale.

    Why top-N instead of a MAD threshold:
        A ranking adapts to your dataset's distribution -- no arbitrary
        cutoff to defend. N itself is a budget decision (how many features
        downstream models can handle given your sample size), not a
        biological claim about what counts as "variable enough".
    """
    print(f"[fs] ranking {betas.shape[1]:,} CpGs by MAD")

    medians = betas.median(axis=0)
    mad = (betas - medians).abs().median(axis=0) * 1.4826

    if top_n >= betas.shape[1]:
        print(f"[fs] top_n ({top_n}) >= total CpGs ({betas.shape[1]}); keeping all")
        return betas

    top_cpgs = mad.sort_values(ascending=False).head(top_n).index
    selected = betas.loc[:, top_cpgs]
    print(
        f"[fs] selected top {top_n:,} CpGs by MAD "
        f"(MAD range: {mad[top_cpgs].min():.4f} -- {mad[top_cpgs].max():.4f}, "
        f"dropped tail min MAD: {mad.min():.4f})"
    )
    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess DNA methylation beta-value data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--betas", type=Path, required=True,
                   help="CSV with beta values (CpGs x Samples)")
    p.add_argument("--targets", type=Path, required=True,
                   help="CSV with target labels (sample_id, condition)")
    p.add_argument("--output", type=Path, default=Path("preprocessed.csv"),
                   help="Output CSV path (Samples x CpGs + 'condition')")
    p.add_argument("--missing", choices=["drop", "knn", "auto"], default="auto",
                   help="How to handle NAs: drop columns, class-aware KNN, "
                        "or auto (asks interactively if NAs are found)")
    p.add_argument("--k", type=int, default=5,
                   help="Number of neighbors for KNN imputation")
    p.add_argument("--top-n", type=int, default=5000,
                   help="Number of CpGs to keep after MAD ranking")
    return p.parse_args()


def resolve_missing_strategy(strategy: str) -> str:
    """If strategy is 'auto', ask the user interactively."""
    if strategy != "auto":
        return strategy
    while True:
        ans = input(
            "Missing values were found. Choose: "
            "[d]rop CpGs with NAs, [k]nn impute (class-aware), or [q]uit: "
        ).strip().lower()
        if ans in ("d", "drop"):
            return "drop"
        if ans in ("k", "knn"):
            return "knn"
        if ans in ("q", "quit"):
            sys.exit("[exit] user quit")
        print("  please enter d, k, or q")


def main() -> None:
    args = parse_args()

    betas, targets = load_data(args.betas, args.targets)

    n_missing = report_missing(betas)
    if n_missing > 0:
        strategy = resolve_missing_strategy(args.missing)
        if strategy == "drop":
            betas, targets = drop_missing(betas, targets)
        else:  # knn
            betas = pd.DataFrame(
                knn_impute_per_class(betas, targets, k=args.k),
                index=betas.index, columns=betas.columns,
            )

    betas = select_informative_cpgs(betas, top_n=args.top_n)

    # Assemble final output: samples x (CpGs + condition)
    output = betas.copy()
    output["condition"] = targets
    print(f"[out] writing {output.shape} to {args.output}")
    output.to_csv(args.output)
    print("[done]")


if __name__ == "__main__":
    main()

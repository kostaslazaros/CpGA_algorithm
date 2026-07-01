"""
DNA Methylation Feature Ranking
================================

Ranks CpG features (columns) of the preprocessed methylation dataset
from most-to-least informative for distinguishing the 4 prostate cancer
conditions, using two complementary methods:

  1. Kruskal-Wallis H test
       Non-parametric test for whether the 4 condition groups have
       different distributions of beta values at each CpG. Distribution-
       free, so it doesn't assume normality (which methylation beta
       values typically violate -- they're bounded in [0,1] and often
       bimodal).

  2. Random Forest feature importance
       Multivariate, captures non-linear effects and interactions that
       a univariate test like Kruskal-Wallis cannot. Default uses Mean
       Decrease in Impurity (Gini); --permutation switches to permutation
       importance (slower but less biased).

The two rankings are complementary: Kruskal-Wallis is univariate and
ranks each CpG independently; RF is multivariate and accounts for joint
information. CpGs that rank high in BOTH are the most robust candidates.

NOTE on methodology:
    Both rankings are computed on the full dataset. This is appropriate
    for exploratory ranking but NOT for nested feature selection inside
    a model evaluation pipeline -- if you plan to use these rankings to
    select features for a classifier you'll evaluate, do the ranking
    INSIDE cross-validation folds to avoid information leakage.

Usage:
    python rank_features.py \
        --input preprocessed.csv \
        --target-col condition \
        --output-kw ranking_kruskal_wallis.csv \
        --output-rf ranking_random_forest.csv \
        --n-estimators 500 \
        --n-jobs -1 \
        --random-state 42
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_preprocessed(path: Path, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load the preprocessed CSV (Samples x CpGs + target column).

    Splits into feature matrix X and target vector y, validates no NAs.
    """
    print(f"[load] reading preprocessed dataset from {path}")
    df = pd.read_csv(path, index_col=0)
    print(f"[load] shape: {df.shape}")

    if target_col not in df.columns:
        sys.exit(
            f"[error] target column '{target_col}' not found. "
            f"Available columns end with: {list(df.columns[-5:])}"
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if X.isna().any().any():
        sys.exit(
            "[error] feature matrix contains NaNs. The preprocessing step "
            "should have handled these. Please re-run preprocessing."
        )
    if y.isna().any():
        sys.exit("[error] target column contains NaNs.")

    print(f"[load] features: {X.shape[1]:,} CpGs, samples: {X.shape[0]}")
    print(f"[load] class distribution:\n{y.value_counts().to_string()}")
    return X, y


# ---------------------------------------------------------------------------
# Kruskal-Wallis ranking
# ---------------------------------------------------------------------------
def rank_kruskal_wallis(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Run Kruskal-Wallis H test per CpG and rank by H statistic.

    For each CpG, splits its beta values by condition and tests whether
    the per-condition distributions differ. Returns a DataFrame sorted
    by H descending, with FDR-corrected p-values (Benjamini-Hochberg).

    Why H and not p-value for ranking:
        H is the test statistic and is monotonically related to effect
        size for a given sample size. p-values get squashed against zero
        for strong effects (many CpGs may have p ~= 0 and tie), losing
        ranking resolution. H stays informative across the full range.
    """
    print(f"[kw] running Kruskal-Wallis on {X.shape[1]:,} CpGs")
    t0 = time.time()

    # Pre-split sample indices by class once (much faster than re-grouping
    # per CpG inside the loop)
    groups_idx = [np.where(y.values == cls)[0] for cls in y.unique()]
    n_classes = len(groups_idx)
    if n_classes < 2:
        sys.exit("[error] Kruskal-Wallis requires at least 2 classes")

    X_arr = X.values  # numpy is faster than pandas for column iteration
    h_stats = np.empty(X.shape[1])
    p_values = np.empty(X.shape[1])

    for j in range(X.shape[1]):
        col = X_arr[:, j]
        samples_per_group = [col[idx] for idx in groups_idx]
        try:
            h, p = stats.kruskal(*samples_per_group)
        except ValueError:
            # Happens if all values are identical within every group
            h, p = 0.0, 1.0
        h_stats[j] = h
        p_values[j] = p

        if (j + 1) % 5000 == 0:
            print(f"[kw]   processed {j + 1:,}/{X.shape[1]:,} CpGs")

    elapsed = time.time() - t0
    print(f"[kw] completed in {elapsed:.1f}s")

    # Benjamini-Hochberg FDR correction
    p_adj = _benjamini_hochberg(p_values)

    result = pd.DataFrame({
        "feature": X.columns,
        "h_statistic": h_stats,
        "p_value": p_values,
        "p_value_fdr_bh": p_adj,
    })
    # Sort by H descending (most discriminative first)
    result = result.sort_values("h_statistic", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))

    n_sig = int((result["p_value_fdr_bh"] < 0.05).sum())
    print(f"[kw] {n_sig:,} CpGs have FDR-adjusted p < 0.05")
    return result


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    # Standard BH formula: p_adj[i] = min(p[i] * n / rank, 1), enforced monotonic
    adj = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the right
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    # Restore original order
    out = np.empty_like(adj)
    out[order] = adj
    return out


# ---------------------------------------------------------------------------
# Random Forest ranking
# ---------------------------------------------------------------------------
def rank_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
    n_jobs: int,
    random_state: int,
    use_permutation: bool,
    n_permutation_repeats: int,
) -> pd.DataFrame:
    """Train a Random Forest and rank CpGs by feature importance.

    With imbalanced classes (which you have), we set
    `class_weight='balanced'` so the forest doesn't get dominated by
    the majority class. We also use stratified bootstrap implicitly via
    sklearn's defaults.

    Default importance is Mean Decrease in Impurity (Gini). Pass
    --permutation for permutation importance, which is more reliable
    but proportionally slower (n_permutation_repeats * cost of one
    forest pass).
    """
    print(
        f"[rf] training RandomForest "
        f"(n_estimators={n_estimators}, n_jobs={n_jobs}, "
        f"class_weight='balanced')"
    )
    t0 = time.time()

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight="balanced",
        # max_features='sqrt' is the default for classification and is
        # appropriate for high-dimensional methylation data
    )
    rf.fit(X.values, y.values)
    elapsed = time.time() - t0
    print(f"[rf] training completed in {elapsed:.1f}s")
    print(f"[rf] OOB-free training accuracy: {rf.score(X.values, y.values):.3f}")

    if use_permutation:
        print(
            f"[rf] computing permutation importance "
            f"(n_repeats={n_permutation_repeats}); this may take a while..."
        )
        t1 = time.time()
        perm = permutation_importance(
            rf, X.values, y.values,
            n_repeats=n_permutation_repeats,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        elapsed = time.time() - t1
        print(f"[rf] permutation importance computed in {elapsed:.1f}s")

        result = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
            "importance_method": "permutation",
        })
        sort_col = "importance_mean"
    else:
        result = pd.DataFrame({
            "feature": X.columns,
            "importance": rf.feature_importances_,
            "importance_method": "mean_decrease_impurity",
        })
        sort_col = "importance"

    result = result.sort_values(sort_col, ascending=False).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rank CpG features by Kruskal-Wallis and Random Forest importance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Preprocessed CSV (output of preprocess_methylation.py)")
    p.add_argument("--target-col", type=str, default="condition",
                   help="Name of the target column in the input CSV")
    p.add_argument("--output-kw", type=Path, default=Path("ranking_kruskal_wallis.csv"),
                   help="Output CSV for Kruskal-Wallis ranking")
    p.add_argument("--output-rf", type=Path, default=Path("ranking_random_forest.csv"),
                   help="Output CSV for Random Forest ranking")
    p.add_argument("--n-estimators", type=int, default=500,
                   help="Number of trees in the random forest")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs for RF training (-1 = all cores)")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--permutation", action="store_true",
                   help="Use permutation importance instead of MDI (slower, more reliable)")
    p.add_argument("--n-permutation-repeats", type=int, default=10,
                   help="Repeats for permutation importance (only used with --permutation)")
    p.add_argument("--skip-kw", action="store_true",
                   help="Skip Kruskal-Wallis ranking")
    p.add_argument("--skip-rf", action="store_true",
                   help="Skip Random Forest ranking")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_kw and args.skip_rf:
        sys.exit("[error] both rankings skipped; nothing to do")

    X, y = load_preprocessed(args.input, args.target_col)

    if not args.skip_kw:
        kw_ranking = rank_kruskal_wallis(X, y)
        print(f"[kw] writing ranking to {args.output_kw}")
        kw_ranking.to_csv(args.output_kw, index=False)
        print(f"[kw] top 5 CpGs by H statistic:")
        print(kw_ranking.head().to_string(index=False))

    if not args.skip_rf:
        rf_ranking = rank_random_forest(
            X, y,
            n_estimators=args.n_estimators,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            use_permutation=args.permutation,
            n_permutation_repeats=args.n_permutation_repeats,
        )
        print(f"[rf] writing ranking to {args.output_rf}")
        rf_ranking.to_csv(args.output_rf, index=False)
        print(f"[rf] top 5 CpGs by importance:")
        print(rf_ranking.head().to_string(index=False))

    print("[done]")


if __name__ == "__main__":
    main()

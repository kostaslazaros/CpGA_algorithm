"""
Feature Selection Comparison: GA vs. 14 Standard Methods
=========================================================

Compares the GA-selected CpG list against 14 popular feature selection
methods, all picking the same number of top features (matched to the
size of the GA list, read from selected_cpgs.csv). Each method's
selection is then evaluated on:

  1. Mean contrast score (per-class mean beta matrix; same formula as
     used inside the GA).
  2. Number of unique genes its CpGs map to (lower = more biologically
     coherent).

Method lineup (5 univariate filters, 2 multivariate filters, 2 wrappers,
5 embedded):

  Univariate filters:
    1. anova_f          ANOVA F-test (parametric)
    2. mutual_info      Mutual information (non-parametric)
    3. variance         Variance ranking (simplest baseline)
    4. chi2             Chi-squared test (works on beta values since
                        they are non-negative, in [0, 1])
    5. fisher_score     Fisher score (between-class vs within-class
                        variance; classic genomics filter)

  Multivariate filters:
    6. mrmr             max-Relevance min-Redundancy (in-script
                        implementation: F-stat relevance, Pearson
                        redundancy)
    7. relieff          ReliefF (captures feature interactions; via
                        skrebate package)

  Wrappers (some pre-filter to a smaller pool first; pure forward SFS
  on 5000 features is infeasible):
    8. rfe_svm          Recursive Feature Elimination, linear SVM
    9. forward_lr       Forward SFS, logistic regression (pre-filtered)

  Embedded:
   10. l1_logistic      L1-regularized logistic regression (Lasso)
   11. extratrees       ExtraTrees impurity importance
   12. xgboost          XGBoost gain importance
   13. lightgbm         LightGBM gain importance
   14. adaboost         AdaBoost feature importance (different inductive
                        bias from gradient boosting: reweights samples
                        each round rather than fitting residuals)

NOTE: Kruskal-Wallis, Random Forest importance, and MAD are deliberately
excluded -- they are used inside the GA pipeline (KW and RF as ranking
inputs to the fitness function; MAD in preprocessing), so including them
as comparison baselines would be circular.

Outputs:
  - method_summary.csv      : per-method metrics (mean/median contrast,
                              n_unique_genes, compression_ratio, runtime)
  - method_selections.csv   : long-format table of which CpGs each method
                              selected

Usage:
  python compare_fs_methods.py \\
      --betas bvals_processed.csv \\
      --ga-selected selected_cpgs.csv \\
      --gene-map cpg_annotation.csv \\
      --random-state 42
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.feature_selection import (
    RFE,
    SequentialFeatureSelector,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_inputs(
    betas_path: Path, ga_path: Path, gene_path: Path, target_col: str
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.Series]:
    """Load beta matrix, GA-selected CpGs, and gene annotation."""
    print(f"[load] reading beta matrix from {betas_path}")
    df = pd.read_csv(betas_path, index_col=0)
    if target_col not in df.columns:
        sys.exit(f"[error] target column '{target_col}' not in beta matrix")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    print(f"[load] beta matrix: {X.shape[0]} samples x {X.shape[1]:,} CpGs")
    print(f"[load] class distribution:\n{y.value_counts().to_string()}")

    print(f"[load] reading GA-selected CpGs from {ga_path}")
    ga_df = pd.read_csv(ga_path)
    if "feature" not in ga_df.columns:
        sys.exit(f"[error] '{ga_path}' must have a 'feature' column")
    ga_cpgs = ga_df["feature"].tolist()
    ga_cpgs = [c for c in ga_cpgs if c in X.columns]
    print(f"[load] {len(ga_cpgs):,} GA-selected CpGs (top-N for all methods)")

    print(f"[load] reading gene annotation from {gene_path}")
    gene_df = pd.read_csv(gene_path)
    gene_map = pd.Series(
        gene_df.iloc[:, 1].values, index=gene_df.iloc[:, 0].values, name="gene"
    )
    print(f"[load] {gene_map.notna().sum():,} CpGs have gene annotations")
    return X, y, ga_cpgs, gene_map


# ---------------------------------------------------------------------------
# Contrast score
# ---------------------------------------------------------------------------
def compute_mean_beta_matrix(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    return X.groupby(y).mean()


def compute_contrast_scores(mean_beta: pd.DataFrame) -> pd.Series:
    arr = mean_beta.values
    diffs = np.abs(arr[:, None, :] - arr[None, :, :])
    contrasts = diffs.sum(axis=1)
    max_contrast = contrasts.max(axis=0)
    nan_cols = np.isnan(arr).any(axis=0)
    max_contrast[nan_cols] = np.nan
    return pd.Series(max_contrast, index=mean_beta.columns,
                     name="contrast").fillna(0.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _timed(label: str, fn):
    """Run fn() with a timer. Returns (result, elapsed)."""
    print(f"[fs] {label}")
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"[fs]   {elapsed:.1f}s")
    return result, elapsed


# ---------------------------------------------------------------------------
# Univariate filters
# ---------------------------------------------------------------------------
def fs_anova(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[list[str], float]:
    def run():
        f_stats, _ = f_classif(X.values, y.values)
        f_stats = np.nan_to_num(f_stats, nan=0.0)
        return X.columns[np.argsort(-f_stats)[:n]].tolist()
    return _timed("ANOVA F-test", run)


def fs_mutual_info(X: pd.DataFrame, y: pd.Series, n: int,
                    rs: int) -> tuple[list[str], float]:
    def run():
        mi = mutual_info_classif(X.values, y.values,
                                 random_state=rs, n_neighbors=3)
        return X.columns[np.argsort(-mi)[:n]].tolist()
    return _timed("Mutual Information", run)


def fs_chi2(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[list[str], float]:
    """Chi-squared test of independence between each feature and the class.
    Requires non-negative inputs; beta values are in [0, 1] which qualifies."""
    def run():
        # Defensive: clip any tiny negative values from floating-point noise
        X_safe = np.clip(X.values, 0.0, None)
        chi2_stats, _ = chi2(X_safe, y.values)
        chi2_stats = np.nan_to_num(chi2_stats, nan=0.0)
        return X.columns[np.argsort(-chi2_stats)[:n]].tolist()
    return _timed("Chi-squared test", run)


def fs_variance(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[list[str], float]:
    def run():
        v = X.var(axis=0).values
        return X.columns[np.argsort(-v)[:n]].tolist()
    return _timed("Variance ranking", run)


def fs_fisher_score(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[list[str], float]:
    """Fisher score for multiclass (Duda & Hart formulation):
        score(f) = sum_c n_c * (mu_c - mu)^2  /  sum_c n_c * sigma_c^2
    Numerator: between-class variance; denominator: within-class variance.
    Higher score = feature better discriminates between classes.
    Classic statistical filter; used widely in genomics and bioinformatics.
    """
    def run():
        X_arr = X.values
        y_arr = y.values
        classes = np.unique(y_arr)
        global_mean = X_arr.mean(axis=0)

        between = np.zeros(X_arr.shape[1])
        within = np.zeros(X_arr.shape[1])
        for cls in classes:
            mask = y_arr == cls
            n_c = int(mask.sum())
            class_mean = X_arr[mask].mean(axis=0)
            class_var = X_arr[mask].var(axis=0)
            between += n_c * (class_mean - global_mean) ** 2
            within += n_c * class_var

        # Avoid div-by-zero for constant features
        within = np.where(within < 1e-12, 1e-12, within)
        score = between / within
        return X.columns[np.argsort(-score)[:n]].tolist()
    return _timed("Fisher score", run)


# ---------------------------------------------------------------------------
# Multivariate filters
# ---------------------------------------------------------------------------
def fs_mrmr(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[list[str], float]:
    """In-script mRMR (max-Relevance min-Redundancy).

    Standard FCQ formulation:
      score(f) = relevance(f, y) - mean(redundancy(f, s) for s in selected)
    Relevance: ANOVA F-statistic of f against y (multiclass-friendly).
    Redundancy: mean absolute Pearson correlation between f and already
    selected features.

    Greedy selection: starts with most relevant feature, then iteratively
    picks the feature maximizing relevance - mean_abs_corr_with_selected.

    Pre-filter to top 500 by relevance to keep correlation computation
    tractable on 5000 features (otherwise we'd compute a 5000x5000
    correlation matrix).
    """
    def run():
        pool_size = 500
        f_stats, _ = f_classif(X.values, y.values)
        f_stats = np.nan_to_num(f_stats, nan=0.0)

        # Pre-filter to top `pool_size` by relevance
        pool_idx = np.argsort(-f_stats)[:pool_size]
        X_pool = X.iloc[:, pool_idx].values
        relevance = f_stats[pool_idx]
        cols = X.columns[pool_idx].tolist()

        # Standardize for Pearson correlation
        X_std = (X_pool - X_pool.mean(axis=0)) / (X_pool.std(axis=0) + 1e-12)

        selected_idx = []
        remaining = set(range(len(cols)))

        # First pick: most relevant
        first = int(np.argmax(relevance))
        selected_idx.append(first)
        remaining.discard(first)

        # Greedy mRMR
        while len(selected_idx) < n and remaining:
            sel_arr = X_std[:, selected_idx]            # (samples, n_sel)
            rem = sorted(remaining)
            rem_arr = X_std[:, rem]                     # (samples, n_rem)
            # mean absolute correlation with selected set
            # corr_matrix shape (n_rem, n_sel)
            corr = (rem_arr.T @ sel_arr) / X_pool.shape[0]
            redundancy = np.mean(np.abs(corr), axis=1)
            score = relevance[rem] - redundancy
            best_local = int(np.argmax(score))
            best_global = rem[best_local]
            selected_idx.append(best_global)
            remaining.discard(best_global)

        return [cols[i] for i in selected_idx]
    return _timed("mRMR (pre-filtered to 500 by F-stat)", run)


def fs_relieff(X: pd.DataFrame, y: pd.Series, n: int) -> tuple[list[str], float]:
    """ReliefF via the skrebate package. Captures feature interactions
    by comparing each sample's distance to nearest hits and misses."""
    def run():
        try:
            from skrebate import ReliefF
        except ImportError:
            sys.exit("[error] skrebate not installed. Add 'skrebate>=0.62' "
                     "to pyproject.toml and run `uv sync`.")
        # n_neighbors=10 is the standard default
        rf = ReliefF(n_features_to_select=n, n_neighbors=10, n_jobs=-1)
        rf.fit(X.values, y.values)
        # rf.top_features_ is sorted by importance descending
        return X.columns[rf.top_features_[:n]].tolist()
    return _timed("ReliefF (this can take a few minutes)", run)


# ---------------------------------------------------------------------------
# Wrappers (all use a pre-filtered pool to be tractable)
# ---------------------------------------------------------------------------
def fs_rfe_svm(X: pd.DataFrame, y: pd.Series, n: int,
                rs: int) -> tuple[list[str], float]:
    """RFE with Linear SVM. Uses step=100 to keep runtime reasonable
    (~48 SVM fits for 5000->209)."""
    def run():
        X_scaled = StandardScaler().fit_transform(X.values)
        svm = LinearSVC(C=1.0, class_weight="balanced", random_state=rs,
                        max_iter=2000, dual="auto")
        rfe = RFE(estimator=svm, n_features_to_select=n, step=100)
        rfe.fit(X_scaled, y.values)
        return X.columns[rfe.support_].tolist()
    return _timed("RFE with Linear SVM", run)


def fs_forward_lr(X: pd.DataFrame, y: pd.Series, n: int, rs: int,
                   pool_size: int) -> tuple[list[str], float]:
    """Forward SFS with logistic regression. Pre-filtered to `pool_size`
    via ANOVA-F first; pure forward SFS on 5000 features is infeasible.

    Speed-tuning: cv=2 instead of cv=3 (33% fewer fits), and max_iter=200
    instead of 2000 (LR converges quickly on small samples; we only need
    a relative ranking of candidate features at each step, not full
    convergence). The default --sfs-pool-size is also reduced to 250
    (from 500) since SFS cost scales with pool x n_selected."""
    def run():
        f_stats, _ = f_classif(X.values, y.values)
        f_stats = np.nan_to_num(f_stats, nan=0.0)
        pool_idx = np.argsort(-f_stats)[:pool_size]
        X_pool = X.iloc[:, pool_idx]
        X_pool_scaled = StandardScaler().fit_transform(X_pool.values)
        lr = LogisticRegression(
            max_iter=200, class_weight="balanced", random_state=rs,
            solver="lbfgs",
        )
        sfs = SequentialFeatureSelector(
            estimator=lr, n_features_to_select=n, direction="forward",
            scoring="balanced_accuracy", cv=2, n_jobs=-1,
        )
        sfs.fit(X_pool_scaled, y.values)
        return X_pool.columns[sfs.get_support()].tolist()
    return _timed(f"Forward SFS with LR (pool={pool_size}, cv=2)", run)


# ---------------------------------------------------------------------------
# Embedded
# ---------------------------------------------------------------------------
def fs_l1_logistic(X: pd.DataFrame, y: pd.Series, n: int,
                    rs: int) -> tuple[list[str], float]:
    """L1-regularized multinomial logistic regression. Auto-tunes C
    upward if too few features survive.

    Uses saga solver (multinomial L1 support; liblinear is no longer
    allowed for multiclass in sklearn 1.7+). max_iter=300 is sufficient
    since we only need feature ordering, not full coefficient
    convergence to the global optimum.
    """
    def run():
        X_scaled = StandardScaler().fit_transform(X.values)
        coef_magnitude = None
        for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
            lr = LogisticRegression(
                penalty="l1", C=C, solver="saga",
                max_iter=300,
                class_weight="balanced", random_state=rs,
            )
            lr.fit(X_scaled, y.values)
            coef_magnitude = np.max(np.abs(lr.coef_), axis=0)
            n_nonzero = int((coef_magnitude > 1e-10).sum())
            print(f"[fs]     C={C}: {n_nonzero} non-zero")
            if n_nonzero >= n:
                break
        return X.columns[np.argsort(-coef_magnitude)[:n]].tolist()
    return _timed("L1-regularized Logistic Regression", run)


def fs_adaboost(X: pd.DataFrame, y: pd.Series, n: int,
                 rs: int) -> tuple[list[str], float]:
    """AdaBoost feature importance. Different inductive bias from
    gradient boosting (XGBoost/LightGBM): AdaBoost reweights misclassified
    samples each round rather than fitting residuals. Importance is
    derived from the weighted feature usage across base learners.

    Uses shallow decision trees (max_depth=2) as the standard base
    learner; deeper trees would dilute the per-feature importance.
    Class imbalance is handled via sample weights at fit time (AdaBoost
    has no class_weight parameter for multiclass).
    """
    def run():
        sw = compute_sample_weight("balanced", y.values)
        base = DecisionTreeClassifier(max_depth=2, random_state=rs)
        ada = AdaBoostClassifier(
            estimator=base,
            n_estimators=200,
            learning_rate=1.0,
            random_state=rs,
        )
        ada.fit(X.values, y.values, sample_weight=sw)
        return X.columns[np.argsort(-ada.feature_importances_)[:n]].tolist()
    return _timed("AdaBoost feature importance", run)


def fs_extratrees(X: pd.DataFrame, y: pd.Series, n: int,
                   rs: int) -> tuple[list[str], float]:
    """ExtraTrees (Extremely Randomized Trees) impurity importance.
    Different inductive bias from RF: random thresholds at each split."""
    def run():
        et = ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=rs, n_jobs=-1,
        )
        et.fit(X.values, y.values)
        return X.columns[np.argsort(-et.feature_importances_)[:n]].tolist()
    return _timed("ExtraTrees impurity importance", run)


def fs_xgboost(X: pd.DataFrame, y: pd.Series, n: int,
                rs: int) -> tuple[list[str], float]:
    """XGBoost gain importance."""
    def run():
        try:
            from xgboost import XGBClassifier
        except ImportError:
            sys.exit("[error] xgboost not installed. Add 'xgboost>=2.0' "
                     "to pyproject.toml and run `uv sync`.")
        classes = sorted(y.unique())
        cls_to_int = {c: i for i, c in enumerate(classes)}
        y_int = y.map(cls_to_int).values
        sw = compute_sample_weight("balanced", y_int)
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            objective="multi:softprob", num_class=len(classes),
            random_state=rs, n_jobs=-1, tree_method="hist",
            eval_metric="mlogloss",
        )
        xgb.fit(X.values, y_int, sample_weight=sw)
        return X.columns[np.argsort(-xgb.feature_importances_)[:n]].tolist()
    return _timed("XGBoost gain importance", run)


def fs_lightgbm(X: pd.DataFrame, y: pd.Series, n: int,
                 rs: int) -> tuple[list[str], float]:
    """LightGBM gain importance."""
    def run():
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            sys.exit("[error] lightgbm not installed. Add 'lightgbm>=4.0' "
                     "to pyproject.toml and run `uv sync`.")
        sw = compute_sample_weight("balanced", y.values)
        lgbm = LGBMClassifier(
            n_estimators=300, max_depth=-1, num_leaves=31,
            learning_rate=0.1, random_state=rs, n_jobs=-1,
            objective="multiclass", verbose=-1,
            importance_type="gain",
        )
        lgbm.fit(X.values, y.values, sample_weight=sw)
        return X.columns[np.argsort(-lgbm.feature_importances_)[:n]].tolist()
    return _timed("LightGBM gain importance", run)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_selection(
    cpgs: list[str], contrast: pd.Series, gene_map: pd.Series
) -> dict:
    """Compute biological-coherence metrics for one method's selection."""
    sel_contrast = contrast.reindex(cpgs)
    sel_genes = gene_map.reindex(cpgs)
    n_unmapped = int(sel_genes.isna().sum())
    unique_genes = int(sel_genes.dropna().nunique())
    n = len(cpgs)
    compress = (unique_genes / n) if n else float("nan")
    return {
        "n_cpgs": n,
        "mean_contrast": float(sel_contrast.mean()),
        "median_contrast": float(sel_contrast.median()),
        "n_unique_genes": unique_genes,
        "n_unmapped": n_unmapped,
        "compress": compress,                    # in [0,1], lower = fewer genes
        "one_minus_compress": 1.0 - compress,    # in [0,1], higher = fewer genes
        "cpgs_per_gene": (n / unique_genes) if unique_genes else float("nan"),
    }


def build_classifiers(rs: int) -> dict[str, Pipeline]:
    """Return name -> Pipeline(StandardScaler + classifier).

    Pipeline ensures the scaler is fit ONLY on training folds (no leakage).
    class_weight='balanced' is set wherever supported (LR, SVM, DT).
    KNN and Naive Bayes have no class_weight mechanism -- they are
    deliberately weight-naive baselines.
    """
    return {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced",
                        random_state=rs)),
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000,
                                        class_weight="balanced",
                                        random_state=rs, solver="lbfgs")),
        ]),
        "DecisionTree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(class_weight="balanced",
                                            random_state=rs)),
        ]),
        "NaiveBayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
    }


def classify_with_cv(
    X_sub: pd.DataFrame, y: pd.Series, n_folds: int, rs: int,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Run stratified K-fold CV with 5 classifiers on a feature subset.

    Returns:
      - summary: {classifier_name: {'f1_mean': ..., 'f1_std': ...}}
      - per_fold: list of {classifier, fold, macro_f1} dicts
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
    X_arr = X_sub.values
    y_arr = y.values

    classifiers = build_classifiers(rs)
    fold_scores = {name: [] for name in classifiers}
    per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        skf.split(X_arr, y_arr), 1
    ):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        for name, pipe in classifiers.items():
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            score = f1_score(y_te, y_pred, average="macro", zero_division=0)
            fold_scores[name].append(score)
            per_fold.append({
                "classifier": name, "fold": fold_idx, "macro_f1": score,
            })

    summary = {
        name: {
            "f1_mean": float(np.mean(scores)),
            "f1_std": float(np.std(scores)),
        }
        for name, scores in fold_scores.items()
    }
    return summary, per_fold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare GA feature selection against 14 standard methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--betas", type=Path, required=True,
                   help="Beta matrix CSV (samples x CpGs + condition column)")
    p.add_argument("--ga-selected", type=Path, required=True,
                   help="GA-selected CpGs CSV (must have 'feature' column)")
    p.add_argument("--gene-map", type=Path, required=True,
                   help="CpG-to-gene CSV (2 columns: CpG ID, gene symbol)")
    p.add_argument("--target-col", type=str, default="condition")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--sfs-pool-size", type=int, default=250,
                   help="Pre-filter pool size for forward SFS")
    p.add_argument("--n-folds", type=int, default=10,
                   help="Stratified CV folds for classification evaluation")
    p.add_argument("--results-dir", type=Path, default=Path("./results"),
                   help="Directory where all output CSVs are written")
    p.add_argument("--skip", type=str, default="",
                   help="Comma-separated method names to skip. Valid names: "
                        "anova_f, mutual_info, variance, chi2, fisher_score, "
                        "mrmr, relieff, rfe_svm, forward_lr, "
                        "l1_logistic, extratrees, xgboost, lightgbm, adaboost")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    t_start = time.time()

    X, y, ga_cpgs, gene_map = load_inputs(
        args.betas, args.ga_selected, args.gene_map, args.target_col
    )
    n = len(ga_cpgs)

    print("[mean] computing mean beta matrix and contrast scores")
    mean_beta = compute_mean_beta_matrix(X, y)
    contrast = compute_contrast_scores(mean_beta)
    print(f"[mean] contrast range: {contrast.min():.4f} -- {contrast.max():.4f}")

    rs = args.random_state
    pool = args.sfs_pool_size

    # Method registry: name -> callable returning (selected_cpgs, elapsed)
    methods = {
        # univariate filters (5)
        "anova_f":         lambda: fs_anova(X, y, n),
        "mutual_info":     lambda: fs_mutual_info(X, y, n, rs),
        "variance":        lambda: fs_variance(X, y, n),
        "chi2":            lambda: fs_chi2(X, y, n),
        "fisher_score":    lambda: fs_fisher_score(X, y, n),
        # multivariate filters (2)
        "mrmr":            lambda: fs_mrmr(X, y, n),
        "relieff":         lambda: fs_relieff(X, y, n),
        # wrappers (2)
        "rfe_svm":         lambda: fs_rfe_svm(X, y, n, rs),
        "forward_lr":      lambda: fs_forward_lr(X, y, n, rs, pool),
        # embedded (5)
        "l1_logistic":     lambda: fs_l1_logistic(X, y, n, rs),
        "extratrees":      lambda: fs_extratrees(X, y, n, rs),
        "xgboost":         lambda: fs_xgboost(X, y, n, rs),
        "lightgbm":        lambda: fs_lightgbm(X, y, n, rs),
        "adaboost":        lambda: fs_adaboost(X, y, n, rs),
    }

    selections = {"GA": (ga_cpgs, 0.0)}
    for name, fn in methods.items():
        if name in skip:
            print(f"[fs] skipping {name}")
            continue
        try:
            cpgs, elapsed = fn()
            selections[name] = (cpgs, elapsed)
        except Exception as e:
            print(f"[fs] ERROR in {name}: {e}")
            continue

    # Set up output directories
    results_dir = args.results_dir
    selections_dir = results_dir / "selections"
    results_dir.mkdir(parents=True, exist_ok=True)
    selections_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[out] results directory: {results_dir.resolve()}")

    # ---- Write per-method selection CSVs ----
    print("[out] writing per-method selection CSVs")
    for method_name, (cpgs, _) in selections.items():
        sel_df = pd.DataFrame({
            "feature": cpgs,
            "gene": [gene_map.get(c, np.nan) for c in cpgs],
            "contrast": [float(contrast.get(c, np.nan)) for c in cpgs],
        })
        sel_df.to_csv(selections_dir / f"{method_name}_selected.csv",
                      index=False)

    # ---- Compute biological metrics + run classification CV per method ----
    print(f"\n[clf] running {args.n_folds}-fold stratified CV with 5 "
          f"classifiers on each method's CpG subset")
    summary_rows = []
    all_per_fold_rows = []

    for method_name, (cpgs, elapsed) in selections.items():
        bio = evaluate_selection(cpgs, contrast, gene_map)

        # Build the feature subset and run CV
        X_sub = X[cpgs]
        clf_summary, per_fold = classify_with_cv(
            X_sub, y, n_folds=args.n_folds, rs=args.random_state
        )

        # Tag per-fold rows with the FS method and append to the global list
        for row in per_fold:
            row["method"] = method_name
            all_per_fold_rows.append(row)

        # Build the summary row (wide format: one row per FS method)
        row = {
            "method": method_name,
            "n_cpgs": bio["n_cpgs"],
            "mean_contrast": bio["mean_contrast"],
            "n_unique_genes": bio["n_unique_genes"],
            "compress": bio["compress"],
            "one_minus_compress": bio["one_minus_compress"],
            "cpgs_per_gene": bio["cpgs_per_gene"],
        }
        for clf_name, scores in clf_summary.items():
            row[f"f1_mean_{clf_name}"] = scores["f1_mean"]
            row[f"f1_std_{clf_name}"]  = scores["f1_std"]
        row["runtime_s"] = elapsed
        summary_rows.append(row)

        f1_summary = ", ".join(
            f"{k}={v['f1_mean']:.3f}" for k, v in clf_summary.items()
        )
        print(f"[clf]   {method_name:<18s} | {f1_summary}")

    # ---- Build & save the final summary table ----
    summary = pd.DataFrame(summary_rows)
    classifier_names = ["KNN", "SVM", "LogisticRegression",
                         "DecisionTree", "NaiveBayes"]
    f1_cols = []
    for c in classifier_names:
        f1_cols += [f"f1_mean_{c}", f"f1_std_{c}"]

    # ---- Combined score: weighted combination of 3 criteria ----
    #
    # Components (each min-max normalized within the set of methods, so each
    # contributes a value in [0, 1] before weighting):
    #
    #   1. Gene compactness (1 - compress) -- WEIGHT 0.50
    #      Higher = CpGs cluster on fewer genes. The dominant criterion
    #      because biological coherence (multiple CpGs reinforcing the
    #      same gene) is the central claim of this method.
    #
    #   2. Mean contrast score              -- WEIGHT 0.25
    #      Higher = stronger between-class separation per CpG.
    #
    #   3. Mean F1 across the 5 classifiers -- WEIGHT 0.25
    #      Higher = better classification performance on average.
    #
    # combined_score = 0.50 * norm(1-compress) + 0.25 * norm(contrast)
    #                + 0.25 * norm(mean_F1_across_5_classifiers)
    #
    # Range: [0, 1]. Score is RELATIVE to the set of methods compared
    # (min-max is computed within this run), not an absolute quality.
    f1_mean_cols = [f"f1_mean_{c}" for c in classifier_names]
    avg_f1 = summary[f1_mean_cols].mean(axis=1)
    contrast_vals = summary["mean_contrast"]
    compactness = summary["one_minus_compress"]   # higher = fewer genes

    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-12:
            return pd.Series(0.0, index=s.index)
        return (s - lo) / (hi - lo)

    W_COMPACT, W_CONTRAST, W_F1 = 0.50, 0.25, 0.25
    summary["combined_score"] = (
        W_COMPACT  * _minmax(compactness)    +
        W_CONTRAST * _minmax(contrast_vals)  +
        W_F1       * _minmax(avg_f1)
    )

    cols = ["method", "n_cpgs", "mean_contrast",
            "n_unique_genes", "compress", "one_minus_compress",
            "cpgs_per_gene", *f1_cols, "combined_score", "runtime_s"]
    summary = summary[cols]

    # Sort: GA first, others by combined_score descending
    ga_row = summary[summary["method"] == "GA"]
    other = summary[summary["method"] != "GA"].sort_values(
        "combined_score", ascending=False
    )
    summary = pd.concat([ga_row, other], ignore_index=True)

    # ---- Print compact comparison table ----
    print()
    print("=" * 118)
    print(f"COMPARISON: top-{n} CpGs from each method (sorted by combined_score)")
    print("=" * 118)
    header = (
        f"{'method':<18s}  "
        f"{'comb.':>6s}  "
        f"{'contrast':>9s}  "
        f"{'genes':>5s}  "
        f"{'compr':>6s}  "
        f"{'1-com':>6s}  "
        f"{'cpg/g':>6s}  "
        f"{'KNN':>6s}  {'SVM':>6s}  {'LR':>6s}  {'DT':>6s}  {'NB':>6s}"
    )
    print(header)
    print("-" * 118)
    for _, row in summary.iterrows():
        print(
            f"{row['method']:<18s}  "
            f"{row['combined_score']:>6.3f}  "
            f"{row['mean_contrast']:>9.4f}  "
            f"{row['n_unique_genes']:>5d}  "
            f"{row['compress']:>6.3f}  "
            f"{row['one_minus_compress']:>6.3f}  "
            f"{row['cpgs_per_gene']:>6.2f}  "
            f"{row['f1_mean_KNN']:>6.3f}  "
            f"{row['f1_mean_SVM']:>6.3f}  "
            f"{row['f1_mean_LogisticRegression']:>6.3f}  "
            f"{row['f1_mean_DecisionTree']:>6.3f}  "
            f"{row['f1_mean_NaiveBayes']:>6.3f}"
        )
    print("-" * 118)
    print("Column descriptions:")
    print("  comb.    : combined_score in [0,1] (sort key). Weighted combination,")
    print("             computed AFTER min-max normalizing each component:")
    print("             0.50 * (1 - compress)  +  0.25 * mean_contrast  +  0.25 * mean_F1")
    print("             Higher = better. Compactness (fewer genes) is the dominant")
    print("             criterion (50% weight). Score is RELATIVE to this run.")
    print("  contrast : mean per-CpG contrast score (between-class separation)")
    print("  genes    : number of unique genes the selected CpGs map to")
    print("  compr    : compression ratio = unique_genes / n_cpgs (lower = better)")
    print("  1-com    : 1 - compress (gene compactness; higher = better)")
    print("  cpg/g    : average number of CpGs per unique gene (higher = better)")
    print("  KNN..NB  : mean macro F1 across CV folds for each classifier")
    print("             (KNN, SVM(RBF), Logistic Regression, Decision Tree, Naive Bayes)")

    # ---- Write outputs ----
    summary_path = results_dir / "method_summary.csv"
    per_fold_path = results_dir / "per_fold_f1.csv"

    print(f"\n[out] writing summary to {summary_path}")
    summary.to_csv(summary_path, index=False)
    print(f"[out] writing per-fold F1 scores to {per_fold_path}")
    pd.DataFrame(all_per_fold_rows)[
        ["method", "classifier", "fold", "macro_f1"]
    ].to_csv(per_fold_path, index=False)
    print(f"[out] per-method selection CSVs in {selections_dir}/")

    elapsed = time.time() - t_start
    print(f"\n[done] total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

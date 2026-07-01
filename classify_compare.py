"""
Classification Performance: Before vs After GA Feature Selection
================================================================

Compares 5 classifiers on the methylation dataset under two conditions:
  (a) all 5000 preprocessed CpGs
  (b) the GA-selected subset of CpGs

Evaluation: 10-fold stratified cross-validation. No oversampling (medical
data; synthetic samples are not appropriate). Class imbalance is handled
algorithmically via `class_weight='balanced'` where the classifier
supports it.

Classifiers (5 popular, well-understood baselines):
  1. Logistic Regression       (linear, regularized, supports class_weight)
  2. Random Forest             (non-linear ensemble, supports class_weight)
  3. Gradient Boosting         (HistGradientBoostingClassifier; sklearn's
                                fast histogram-binned boosting. Handles
                                5000-feature data efficiently. Class
                                weighting via sample_weight at fit time.)
  4. Support Vector Machine    (RBF kernel, supports class_weight)
  5. K-Nearest Neighbors       (instance-based, no class_weight; included
                                as a deliberately weight-naive baseline)

Metrics (chosen for multiclass imbalanced medical data):
  - Accuracy: overall correctness (note: misleading on imbalanced data,
    but you asked for it).
  - Macro F1: unweighted mean of per-class F1. Treats all classes equally,
    so the rare classes contribute as much as the majority class.
  - Cohen's kappa: agreement above chance. Robust to imbalance.
  - Matthews Correlation Coefficient (MCC): generalized multiclass,
    informative even when one class dominates. Range [-1, 1].

IMPORTANT methodological note (printed in output):
  The GA-selected CpGs were chosen using KW and RF rankings computed on
  the FULL dataset. This means feature selection has already "seen"
  test samples in this CV. The "after-FS" metrics will therefore be
  optimistically biased compared to a fully nested protocol. To get a
  truly leakage-free comparison, the entire selection pipeline
  (preprocessing -> ranking -> GA) would need to run inside each CV
  fold. That's a substantially bigger experiment; for an exploratory
  comparison this script is fine, but report this caveat alongside
  the numbers.

Outputs:
  - results.csv         : per-classifier x per-feature-set, mean and std
                          of every metric across the 10 CV folds
  - per_fold.csv        : raw per-fold metrics (useful for paired tests)
  - confusion.csv       : aggregated confusion matrices (one block per
                          classifier x feature-set combination)

Usage:
  python classify_compare.py \\
      --betas bvals_processed.csv \\
      --selected selected_cpgs.csv \\
      --target-col condition \\
      --n-folds 10 \\
      --random-state 42
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

# Suppress sklearn ConvergenceWarning noise from logistic regression on
# tight folds; warnings about real problems still get through.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_data(
    betas_path: Path, selected_path: Path, target_col: str
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load beta matrix, target labels, and the GA-selected CpG list."""
    print(f"[load] reading beta matrix from {betas_path}")
    df = pd.read_csv(betas_path, index_col=0)
    if target_col not in df.columns:
        sys.exit(f"[error] target column '{target_col}' not found")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    print(f"[load] full feature set: {X.shape[0]} samples x {X.shape[1]:,} CpGs")
    print(f"[load] class distribution:\n{y.value_counts().to_string()}")

    print(f"[load] reading selected CpGs from {selected_path}")
    sel_df = pd.read_csv(selected_path)
    if "feature" not in sel_df.columns:
        sys.exit(f"[error] '{selected_path}' must have a 'feature' column")
    selected = sel_df["feature"].tolist()
    print(f"[load] {len(selected):,} selected CpGs")

    missing = set(selected) - set(X.columns)
    if missing:
        print(f"[load] warning: {len(missing)} selected CpGs not in beta matrix; "
              f"dropping them from the selected set")
        selected = [c for c in selected if c in X.columns]
    return X, y, selected


# ---------------------------------------------------------------------------
# Classifier zoo
# ---------------------------------------------------------------------------
def build_classifiers(random_state: int) -> dict[str, Pipeline]:
    """Return name -> sklearn Pipeline.

    StandardScaler is included in the pipeline for distance-based and
    linear classifiers (LR, SVM, KNN). Tree-based methods (RF, GB) don't
    need scaling but including it does no harm and keeps pipelines
    uniform. Putting the scaler in the Pipeline means it is fit on the
    training fold only, then applied to the test fold -- avoiding the
    classic leakage of fitting the scaler on the entire dataset.

    `class_weight='balanced'` is set wherever the classifier supports it.
    GradientBoosting doesn't accept it as a constructor arg in sklearn,
    so we pass sample_weight at fit time (handled in evaluate_fold).
    KNN has no class_weight mechanism -- it's a deliberately naive
    baseline showing what happens without imbalance handling.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
                # multinomial is the default for >2 classes in modern sklearn,
                # but we set solver explicitly for reproducibility
                solver="lbfgs",
            )),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=4,
                random_state=random_state,
                # HistGradientBoosting is sklearn's fast boosting implementation
                # (LightGBM-style histogram binning). It handles 5000 features
                # in seconds vs minutes for the classic GradientBoosting.
                # Class weighting handled via sample_weight at fit time.
            )),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                class_weight="balanced",
                random_state=random_state,
                # gamma='scale' is the modern default (1 / (n_features * X.var()))
            )),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(
                n_neighbors=5,
                # KNN has no class_weight; included as a baseline
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------
def evaluate_fold(
    pipe: Pipeline,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    classes: np.ndarray,
) -> dict:
    """Train one pipeline on a fold and compute metrics on its test set.

    For Gradient Boosting we pass per-sample weights to the .fit() step
    via the pipeline's step__param syntax. This is the equivalent of
    class_weight='balanced' for that classifier.
    """
    fit_params = {}
    final_step = pipe.steps[-1][0]  # 'clf'
    if isinstance(pipe.named_steps[final_step], HistGradientBoostingClassifier):
        sample_weight = compute_sample_weight("balanced", y_train)
        fit_params[f"{final_step}__sample_weight"] = sample_weight

    pipe.fit(X_train, y_train, **fit_params)
    y_pred = pipe.predict(X_test)

    # Macro-averaged metrics (treat all classes equally regardless of size)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        "mcc": matthews_corrcoef(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    return metrics, cm


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------
def run_cv(
    X: pd.DataFrame, y: pd.Series, classifiers: dict[str, Pipeline],
    n_folds: int, random_state: int, feature_set_name: str,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Run stratified K-fold CV across all classifiers. Returns:
       - list of per-fold rows (with classifier, feature_set, fold, metrics)
       - dict[classifier_name] -> aggregated confusion matrix
    """
    classes = np.array(sorted(y.unique()))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    X_arr = X.values
    y_arr = y.values

    rows = []
    confusions = {name: np.zeros((len(classes), len(classes)), dtype=int)
                  for name in classifiers}

    print(f"[cv] {n_folds}-fold stratified CV on '{feature_set_name}' "
          f"({X.shape[1]:,} features)")
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_arr, y_arr), 1):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        for name, pipe in classifiers.items():
            t0 = time.time()
            metrics, cm = evaluate_fold(
                pipe, X_train, y_train, X_test, y_test, classes
            )
            elapsed = time.time() - t0
            row = {
                "feature_set": feature_set_name,
                "classifier": name,
                "fold": fold_idx,
                "fit_time_s": elapsed,
                **metrics,
            }
            rows.append(row)
            confusions[name] += cm

        # Print fold-level summary line
        fold_summary = " | ".join(
            f"{name[:12]:>12s} F1={r['macro_f1']:.3f}"
            for name, r in zip(classifiers,
                               [rows[-len(classifiers) + i] for i in range(len(classifiers))])
        )
        print(f"[cv]   fold {fold_idx:2d}/{n_folds}: {fold_summary}")

    return rows, confusions


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------
def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold metrics: mean and std for each metric."""
    metric_cols = [c for c in per_fold.columns
                   if c not in ("feature_set", "classifier", "fold", "fit_time_s")]
    grouped = per_fold.groupby(["feature_set", "classifier"])
    summary = grouped[metric_cols].agg(["mean", "std"])
    # Flatten MultiIndex columns: ('macro_f1', 'mean') -> 'macro_f1_mean'
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()
    return summary


def print_comparison_table(summary: pd.DataFrame, classes: list[str]) -> None:
    """Pretty-print a side-by-side before/after table for each metric."""
    headline_metrics = [
        "accuracy", "macro_f1", "cohen_kappa", "mcc"
    ]
    print()
    print("=" * 78)
    print("HEADLINE METRICS (mean +/- std across folds)")
    print("=" * 78)
    feature_sets = summary["feature_set"].unique()
    classifiers = summary["classifier"].unique()

    for metric in headline_metrics:
        print(f"\n{metric}")
        header = f"  {'classifier':<22s}"
        for fs in feature_sets:
            header += f" | {fs:^22s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for clf in classifiers:
            line = f"  {clf:<22s}"
            for fs in feature_sets:
                row = summary[(summary["feature_set"] == fs)
                              & (summary["classifier"] == clf)].iloc[0]
                line += f" | {row[f'{metric}_mean']:.3f} +/- {row[f'{metric}_std']:.3f}    "
            print(line)


def write_confusions(
    confusions_by_set: dict[str, dict[str, np.ndarray]],
    classes: list[str],
    output_path: Path,
) -> None:
    """Write all confusion matrices to one CSV, stacked with header rows."""
    blocks = []
    for fs, by_clf in confusions_by_set.items():
        for clf, cm in by_clf.items():
            df = pd.DataFrame(cm, index=classes, columns=classes)
            df.index.name = "true_label"
            df.insert(0, "feature_set", fs)
            df.insert(1, "classifier", clf)
            df = df.reset_index()
            blocks.append(df)
    pd.concat(blocks, ignore_index=True).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare classifier performance before/after GA feature selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--betas", type=Path, required=True,
                   help="Beta matrix CSV (samples x CpGs + condition column)")
    p.add_argument("--selected", type=Path, required=True,
                   help="GA-selected CpGs CSV (must have 'feature' column)")
    p.add_argument("--target-col", type=str, default="condition",
                   help="Name of the target column in the beta CSV")
    p.add_argument("--n-folds", type=int, default=10,
                   help="Number of stratified CV folds")
    p.add_argument("--random-state", type=int, default=42,
                   help="Random seed for fold splits and classifiers")
    p.add_argument("--output-summary", type=Path, default=Path("results.csv"),
                   help="Per-classifier x feature-set summary (mean/std of metrics)")
    p.add_argument("--output-per-fold", type=Path, default=Path("per_fold.csv"),
                   help="Raw per-fold metrics (for paired statistical tests)")
    p.add_argument("--output-confusion", type=Path, default=Path("confusion.csv"),
                   help="Aggregated confusion matrices")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    X, y, selected_cpgs = load_data(args.betas, args.selected, args.target_col)
    classes = sorted(y.unique())

    # Build the two feature sets to compare
    feature_sets = {
        f"all_features_{X.shape[1]}": X,
        f"selected_{len(selected_cpgs)}": X[selected_cpgs],
    }

    all_rows = []
    all_confusions = {}
    for fs_name, X_sub in feature_sets.items():
        classifiers = build_classifiers(args.random_state)
        rows, confusions = run_cv(
            X_sub, y, classifiers,
            n_folds=args.n_folds,
            random_state=args.random_state,
            feature_set_name=fs_name,
        )
        all_rows.extend(rows)
        all_confusions[fs_name] = confusions

    per_fold = pd.DataFrame(all_rows)
    summary = summarize(per_fold)

    # Write outputs
    print(f"\n[out] writing per-fold metrics to {args.output_per_fold}")
    per_fold.to_csv(args.output_per_fold, index=False)
    print(f"[out] writing summary to {args.output_summary}")
    summary.to_csv(args.output_summary, index=False)
    print(f"[out] writing confusion matrices to {args.output_confusion}")
    write_confusions(all_confusions, classes, args.output_confusion)

    # Pretty print the comparison
    print_comparison_table(summary, classes)

    # Methodological caveat
    print()
    print("=" * 78)
    print("CAVEAT")
    print("=" * 78)
    print(
        "The selected CpG set was chosen by a pipeline (preprocessing -> KW/RF\n"
        "ranking -> GA) that ran on the FULL dataset. The 'selected' CV scores\n"
        "below therefore include feature-selection leakage and are optimistically\n"
        "biased relative to a fully nested protocol. The 'all-features' scores\n"
        "are leakage-free (within this script). When interpreting the\n"
        "before/after gap, treat the after-FS numbers as an upper bound on\n"
        "what nested CV would give. To remove this bias, the entire selection\n"
        "pipeline would need to run inside each CV fold."
    )

    elapsed = time.time() - t_start
    print(f"\n[done] total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

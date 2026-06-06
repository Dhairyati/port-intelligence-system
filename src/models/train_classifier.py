"""
train_classifier.py
-------------------
Phase 3 — Congestion Classifier Training

Trains a binary congestion classifier on master_dataset.parquet using:
  - Logistic Regression baseline (proves XGBoost adds value)
  - XGBoost classifier (primary model)

Evaluation strategy:
  - TimeSeriesSplit(n_splits=5) cross-validation (exploratory validation)
  - Single temporal holdout: train Jan-Mar1, test Mar2-Mar31 (final check)

Results are framed as operational exploratory validation, not
production-certified generalisation. With 300 training rows and
5-fold TimeSeriesSplit, fold sizes are small and metric variance
is expected to be high. Interpret mean ± std AUC as directional.

Leakage policy:
  All columns derived from or correlated with the label construction
  process are explicitly excluded before training. See LEAKAGE_COLUMNS
  below for the full documented list with rationale per column.

SHAP note:
  SHAP values explain model behaviour — which features the model
  relies on to make decisions. They do not imply real-world causality.
  High SHAP for slow_vessel_ratio means the model uses this feature
  heavily, not that slow vessels cause congestion in the real world.

Artifacts saved
---------------
  models/xgb_classifier.joblib     primary classifier
  models/baseline_lr.joblib         logistic regression baseline
  models/shap_explainer.joblib      TreeExplainer for dashboard
  models/label_encoder.joblib       port name → integer mapping
  models/season_encoder.joblib      season string → integer mapping
  models/feature_columns.json       ordered feature list for inference
  models/feature_dtypes.json        feature dtypes at training time

Usage (from project root):
  python src/models/train_classifier.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility — set once here, used everywhere
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Temporal split boundary
# ---------------------------------------------------------------------------
# Train : 2023-01-01 → 2023-03-01  (60 days × 5 ports = 300 rows)
# Test  : 2023-03-02 → 2023-03-31  (29 days × 5 ports = 145 rows)
HOLDOUT_SPLIT_DATE = "2023-03-02"

# ---------------------------------------------------------------------------
# TimeSeriesSplit configuration
# ---------------------------------------------------------------------------
# With 90 temporal points and 5 splits, each fold test window is ~15 days.
# This is small — treat CV results as directional, not statistically definitive.
N_CV_SPLITS = 5

# ---------------------------------------------------------------------------
# Leakage column exclusions
# ---------------------------------------------------------------------------
# Every column in this list is excluded from model training.
# Rationale is documented inline for each column.
LEAKAGE_COLUMNS = [
    # ── Direct label sources ─────────────────────────────────────────────────
    "congestion_label",
    # The target itself — obviously excluded.

    "port_congestion_score",
    # The raw score from which congestion_label was derived.
    # Including it would let the model trivially reconstruct the label.

    "port_congestion_score_norm",
    # Min-max normalised version of port_congestion_score.
    # Used directly to create congestion_label via threshold.
    # Including it = the model sees the label in disguise.

    # ── Target-correlated composites ─────────────────────────────────────────
    "congestion_pressure_index",
    # Weighted combination of port_congestion_score + weather + rolling vessels.
    # Since port_congestion_score is one of its components, this leaks
    # the label source directly into the feature space.

    "weather_congestion_interaction",
    # Computed as max_weather_severity_score × slow_vessel_ratio.
    # slow_vessel_ratio is a primary input to port_congestion_score.
    # The interaction term artificially amplifies this leakage signal.
    # Both components are available separately as clean features.

    # ── Lookahead normalisation leakage ──────────────────────────────────────
    "vessels_7d_rolling_norm",
    # Normalised using the full 90-day min/max per port.
    # This means even day-1 values "know" the maximum vessel count
    # that will appear on day 90 — mild but real lookahead leakage.
    # vessels_7d_rolling (unnormalised) is included instead.

    # ── Non-feature administrative columns ───────────────────────────────────
    "nearest_port",
    # String column — replaced by port_encoded (LabelEncoded integer).

    "date",
    # Datetime index — not a model feature directly.
    # Temporal information is captured via month, quarter, weekday, season.

    "season",
    # String column — replaced by season_encoded (LabelEncoded integer).
]

# ---------------------------------------------------------------------------
# XGBoost hyperparameters
# ---------------------------------------------------------------------------
# These are reasonable defaults for a small tabular dataset.
# Full hyperparameter tuning (GridSearchCV) is left for Phase 4 if needed.
XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        4,        # shallow trees reduce overfitting on small data
    "learning_rate":    0.05,     # slow learning rate + more trees = better generalisation
    "subsample":        0.8,      # row subsampling per tree
    "colsample_bytree": 0.8,      # feature subsampling per tree
    "min_child_weight": 3,        # minimum samples per leaf — prevents overfitting
    "gamma":            0.1,      # minimum loss reduction to make a split
    "reg_alpha":        0.1,      # L1 regularisation
    "reg_lambda":       1.0,      # L2 regularisation
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "verbosity":        0,        # suppress XGBoost training output
}


# ---------------------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------------------

def load_data(processed_dir: Path) -> pd.DataFrame:
    """
    Load master_dataset.parquet and validate it contains expected columns.

    Parameters
    ----------
    processed_dir : Path
        Path to data/processed/ directory.

    Returns
    -------
    pd.DataFrame
        Master dataset sorted chronologically.
    """
    path = processed_dir / "master_dataset.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"master_dataset.parquet not found at {path}.\n"
            "Complete Phase 2 pipeline before running Phase 3."
        )

    log.info(f"Loading master_dataset.parquet ...")
    df = pd.read_parquet(path, engine="pyarrow")

    # Ensure chronological order — critical for temporal split
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "nearest_port"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"  Loaded : {len(df):,} rows × {df.shape[1]} columns")
    log.info(f"  Ports  : {sorted(df['nearest_port'].unique().tolist())}")
    log.info(f"  Dates  : {df['date'].min().date()} → {df['date'].max().date()}")

    # Validate target column exists
    if "congestion_label" not in df.columns:
        raise ValueError(
            "'congestion_label' column not found in master_dataset.parquet.\n"
            "Re-run build_master_dataset.py to regenerate the label."
        )

    label_counts = df["congestion_label"].value_counts().sort_index()
    log.info(f"  Label balance:")
    for label, count in label_counts.items():
        log.info(f"    {label} : {count:>4}  ({count/len(df)*100:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 2. Prepare Features
# ---------------------------------------------------------------------------

def prepare_features(
    df: pd.DataFrame,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, list, dict]:
    """
    Encode categorical columns, drop leakage columns, and build the
    clean feature matrix X and target vector y.

    Saves label encoders and feature metadata to models/ for consistent
    use during dashboard inference. The feature column order and dtypes
    saved here MUST be reproduced exactly at inference time.

    Parameters
    ----------
    df : pd.DataFrame
        Full master dataset.
    models_dir : Path
        Directory to save encoders and metadata.

    Returns
    -------
    tuple of (X, y, feature_columns, feature_dtypes)
        X              : feature DataFrame (leakage columns removed)
        y              : target Series (congestion_label)
        feature_columns: ordered list of column names in X
        feature_dtypes : dict of column → dtype string
    """
    log.info("Preparing features ...")
    df = df.copy()

    # ── Encode nearest_port ───────────────────────────────────────────────────
    # Port identity is operationally meaningful — each port has a different
    # congestion regime, vessel mix, and weather exposure.
    # LabelEncoder maps: Houston→0, Los Angeles→1, New York→2, Savannah→3, Seattle→4
    port_encoder = LabelEncoder()
    df["port_encoded"] = port_encoder.fit_transform(df["nearest_port"])

    log.info(f"  Port encoding: {dict(zip(port_encoder.classes_, port_encoder.transform(port_encoder.classes_)))}")

    joblib.dump(port_encoder, models_dir / "label_encoder.joblib")
    log.info(f"  ✓ label_encoder.joblib saved")

    # ── Encode season ─────────────────────────────────────────────────────────
    season_encoder = LabelEncoder()
    df["season_encoded"] = season_encoder.fit_transform(df["season"])

    joblib.dump(season_encoder, models_dir / "season_encoder.joblib")
    log.info(f"  ✓ season_encoder.joblib saved")

    # ── Extract target before dropping ───────────────────────────────────────
    y = df["congestion_label"].astype(int)

    # ── Drop leakage columns ─────────────────────────────────────────────────
    # Drop columns that exist — silently skip any that don't
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    log.info(f"  Dropping {len(cols_to_drop)} leakage/admin columns:")
    for col in cols_to_drop:
        log.info(f"    - {col}")

    X = df.drop(columns=cols_to_drop)

    # ── Drop any remaining non-numeric columns ────────────────────────────────
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        log.info(f"  Dropping remaining non-numeric columns: {non_numeric}")
        X = X.drop(columns=non_numeric)

    # ── Handle any remaining nulls ────────────────────────────────────────────
    # XGBoost handles NaN natively but LR does not — fill with column median
    null_counts = X.isnull().sum()
    if null_counts.any():
        log.warning(f"  Filling nulls with column median for {null_counts[null_counts>0].index.tolist()}")
        X = X.fillna(X.median(numeric_only=True))

    feature_columns = X.columns.tolist()
    feature_dtypes  = {col: str(X[col].dtype) for col in feature_columns}

    log.info(f"  Final feature count : {len(feature_columns)}")
    log.info(f"  Features: {feature_columns}")

    # ── Save feature metadata ─────────────────────────────────────────────────
    # These files are CRITICAL for Phase 5 dashboard inference.
    # The dashboard must present features in exactly this order with these dtypes.
    with open(models_dir / "feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)
    log.info(f"  ✓ feature_columns.json saved")

    with open(models_dir / "feature_dtypes.json", "w") as f:
        json.dump(feature_dtypes, f, indent=2)
    log.info(f"  ✓ feature_dtypes.json saved")

    return X, y, feature_columns, feature_dtypes


# ---------------------------------------------------------------------------
# 3. TimeSeriesSplit Cross-Validation
# ---------------------------------------------------------------------------

def run_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    df_dates: pd.Series,
) -> dict:
    """
    Evaluate both Logistic Regression and XGBoost using TimeSeriesSplit CV.

    IMPORTANT framing: with 90 temporal points and 5 splits, each fold's
    test window is ~15 days. Results are directional indicators of model
    quality, not statistically robust generalization estimates. Report
    mean ± std AUC as exploratory operational validation.

    The split is performed on DATES not row indices — all ports on the
    same date go to the same fold together, preventing cross-port leakage
    between train and test sets within a fold.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    df_dates : pd.Series
        Date column aligned with X and y rows.

    Returns
    -------
    dict
        CV results for baseline and XGBoost models.
    """
    log.info(f"\nTimeSeriesSplit CV (n_splits={N_CV_SPLITS})")
    log.info("NOTE: Results are exploratory — fold sizes are small (~15 days each)")
    log.info("Interpret mean ± std AUC as directional, not production-certified.")

    # Get unique dates in chronological order
    unique_dates = np.sort(df_dates.unique())
    tscv         = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    results = {
        "baseline_lr": {"auc": [], "f1": [], "precision": [], "recall": []},
        "xgboost":     {"auc": [], "f1": [], "precision": [], "recall": []},
    }

    for fold_idx, (train_date_idx, test_date_idx) in enumerate(tscv.split(unique_dates), 1):
        train_dates = unique_dates[train_date_idx]
        test_dates  = unique_dates[test_date_idx]

        # Create boolean masks based on dates — keeps all ports together per date
        train_mask = df_dates.isin(train_dates)
        test_mask  = df_dates.isin(test_dates)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Skip fold if test set has only one class — AUC undefined
        if y_test.nunique() < 2:
            log.warning(f"  Fold {fold_idx}: test set has only one class — skipping")
            continue

        log.info(
            f"  Fold {fold_idx}: train={len(X_train)} rows "
            f"({train_dates[0].astype('datetime64[D]')} → {train_dates[-1].astype('datetime64[D]')}), "
            f"test={len(X_test)} rows "
            f"({test_dates[0].astype('datetime64[D]')} → {test_dates[-1].astype('datetime64[D]')})"
        )

        # ── Baseline: Logistic Regression (scaled) ────────────────────────────
        # Features span very different scales (e.g. total_pings 0-794 vs
        # is_holiday 0-1). StandardScaler normalises all features to zero
        # mean and unit variance before LR fitting — required for lbfgs
        # solver convergence. Scaler is inside the pipeline so train
        # statistics never leak from test set into training.
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                           max_iter=2000,
                           random_state=RANDOM_STATE,
                           class_weight="balanced",
                       )),
        ])
        lr.fit(X_train, y_train)
        lr_proba = lr.predict_proba(X_test)[:, 1]
        lr_pred  = lr.predict(X_test)

        results["baseline_lr"]["auc"].append(roc_auc_score(y_test, lr_proba))
        results["baseline_lr"]["f1"].append(f1_score(y_test, lr_pred, zero_division=0))
        results["baseline_lr"]["precision"].append(precision_score(y_test, lr_pred, zero_division=0))
        results["baseline_lr"]["recall"].append(recall_score(y_test, lr_pred, zero_division=0))

        # ── XGBoost ───────────────────────────────────────────────────────────
        n_neg   = (y_train == 0).sum()
        n_pos   = (y_train == 1).sum()
        spw     = n_neg / n_pos if n_pos > 0 else 1.0

        xgb = XGBClassifier(
            **XGB_PARAMS,
            scale_pos_weight=spw,
        )
        xgb.fit(X_train, y_train)
        xgb_proba = xgb.predict_proba(X_test)[:, 1]
        xgb_pred  = xgb.predict(X_test)

        results["xgboost"]["auc"].append(roc_auc_score(y_test, xgb_proba))
        results["xgboost"]["f1"].append(f1_score(y_test, xgb_pred, zero_division=0))
        results["xgboost"]["precision"].append(precision_score(y_test, xgb_pred, zero_division=0))
        results["xgboost"]["recall"].append(recall_score(y_test, xgb_pred, zero_division=0))

        log.info(
            f"         LR  AUC={results['baseline_lr']['auc'][-1]:.3f}  "
            f"F1={results['baseline_lr']['f1'][-1]:.3f}"
        )
        log.info(
            f"         XGB AUC={results['xgboost']['auc'][-1]:.3f}  "
            f"F1={results['xgboost']['f1'][-1]:.3f}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TIMESERIESSPLIT CV RESULTS (exploratory validation)")
    print("=" * 60)
    print(f"  {'Model':<25} {'AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"  {'─'*65}")

    for model_name, metrics in results.items():
        if not metrics["auc"]:
            continue
        label = "Logistic Regression" if model_name == "baseline_lr" else "XGBoost"
        print(
            f"  {label:<25} "
            f"{np.mean(metrics['auc']):>6.3f}±{np.std(metrics['auc']):.3f}  "
            f"{np.mean(metrics['f1']):>6.3f}±{np.std(metrics['f1']):.3f}  "
            f"{np.mean(metrics['precision']):>6.3f}±{np.std(metrics['precision']):.3f}  "
            f"{np.mean(metrics['recall']):>6.3f}±{np.std(metrics['recall']):.3f}"
        )

    print(f"\n  Note: High std indicates metric variance across small folds.")
    print(f"  Treat these as directional — not production-certified estimates.")
    print("=" * 60)

    return results


# ---------------------------------------------------------------------------
# 4. Final Holdout Evaluation
# ---------------------------------------------------------------------------

def run_holdout_evaluation(
    X: pd.DataFrame,
    y: pd.Series,
    df_dates: pd.Series,
) -> tuple:
    """
    Train on Jan 1 → Mar 1, evaluate on Mar 2 → Mar 31.

    This is a single clean out-of-sample evaluation on unseen dates.
    Both models are trained on the same training set and evaluated
    on the same test set for direct comparison.

    Returns trained XGBoost and LR models fitted on the full training set.
    These are the models saved as artifacts.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    df_dates : pd.Series
        Date column aligned with X and y.

    Returns
    -------
    tuple of (xgb_model, lr_model, X_test, y_test, split_info)
    """
    log.info(f"\nFinal holdout evaluation (split date: {HOLDOUT_SPLIT_DATE})")

    split_date = pd.Timestamp(HOLDOUT_SPLIT_DATE)
    train_mask = df_dates < split_date
    test_mask  = df_dates >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    log.info(f"  Train : {train_mask.sum():>4} rows  ({df_dates[train_mask].min().date()} → {df_dates[train_mask].max().date()})")
    log.info(f"  Test  : {test_mask.sum():>4} rows  ({df_dates[test_mask].min().date()} → {df_dates[test_mask].max().date()})")
    log.info(f"  Train label balance: {(y_train==1).sum()} positive / {(y_train==0).sum()} negative")
    log.info(f"  Test  label balance: {(y_test==1).sum()} positive / {(y_test==0).sum()} negative")

    # ── Baseline: Logistic Regression (scaled) ────────────────────────────────
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
                       max_iter=2000,
                       random_state=RANDOM_STATE,
                       class_weight="balanced",
                   )),
    ])
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_pred  = lr.predict(X_test)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / n_pos if n_pos > 0 else 1.0

    log.info(f"  XGBoost scale_pos_weight = {spw:.3f}")

    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=spw,
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_pred  = xgb.predict(X_test)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL HOLDOUT RESULTS (Mar 2 → Mar 31, unseen dates)")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Logistic Reg':>14} {'XGBoost':>14}")
    print(f"  {'─'*50}")

    metrics = {
        "AUC-ROC":   (roc_auc_score(y_test, lr_proba),     roc_auc_score(y_test, xgb_proba)),
        "F1 Score":  (f1_score(y_test, lr_pred, zero_division=0),  f1_score(y_test, xgb_pred, zero_division=0)),
        "Precision": (precision_score(y_test, lr_pred, zero_division=0), precision_score(y_test, xgb_pred, zero_division=0)),
        "Recall":    (recall_score(y_test, lr_pred, zero_division=0),    recall_score(y_test, xgb_pred, zero_division=0)),
    }

    for metric_name, (lr_val, xgb_val) in metrics.items():
        better = "← XGB better" if xgb_val > lr_val else "← LR better"
        print(f"  {metric_name:<20} {lr_val:>14.4f} {xgb_val:>14.4f}  {better}")

    print("=" * 60)

    split_info = {
        "train_rows":  int(train_mask.sum()),
        "test_rows":   int(test_mask.sum()),
        "split_date":  HOLDOUT_SPLIT_DATE,
        "spw":         float(spw),
        "lr_auc":      float(roc_auc_score(y_test, lr_proba)),
        "xgb_auc":     float(roc_auc_score(y_test, xgb_proba)),
        "lr_f1":       float(f1_score(y_test, lr_pred, zero_division=0)),
        "xgb_f1":      float(f1_score(y_test, xgb_pred, zero_division=0)),
    }

    return xgb, lr, X_test, y_test, split_info


# ---------------------------------------------------------------------------
# 5. Compute SHAP Values
# ---------------------------------------------------------------------------

def compute_shap(
    xgb_model: XGBClassifier,
    X: pd.DataFrame,
    models_dir: Path,
) -> shap.TreeExplainer:
    """
    Compute SHAP values for the trained XGBoost model and print the
    top 10 most important features by mean absolute SHAP value.

    IMPORTANT: SHAP values explain model behaviour, not real-world causality.
    A high mean |SHAP| for slow_vessel_ratio means the model relies on
    this feature heavily to make decisions. It does NOT mean slow vessels
    cause congestion — correlation in training data drives SHAP importance,
    not causal mechanisms.

    Parameters
    ----------
    xgb_model : XGBClassifier
        Trained XGBoost classifier.
    X : pd.DataFrame
        Feature matrix (full dataset used for global SHAP — not just test set).
    models_dir : Path
        Directory to save the explainer.

    Returns
    -------
    shap.TreeExplainer
        Fitted SHAP explainer saved for dashboard use.
    """
    log.info("\nComputing SHAP values ...")
    log.info("NOTE: SHAP explains model behaviour, not real-world causality.")

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP value per feature = global importance
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=X.columns,
    ).sort_values(ascending=False)

    print("\n" + "=" * 60)
    print("  SHAP FEATURE IMPORTANCE (model behaviour explanation)")
    print("  Higher = model relies on this feature more heavily")
    print("  Does NOT imply causality")
    print("=" * 60)

    for rank, (feature, importance) in enumerate(mean_abs_shap.head(15).items(), 1):
        bar = "█" * int(importance * 100)
        print(f"  {rank:>2}. {feature:<35} {importance:.4f}  {bar}")

    print("=" * 60)

    # Save explainer for dashboard
    joblib.dump(explainer, models_dir / "shap_explainer.joblib")
    log.info(f"  ✓ shap_explainer.joblib saved")

    return explainer


# ---------------------------------------------------------------------------
# 6. Save Model Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    xgb_model:      XGBClassifier,
    lr_model:       LogisticRegression,
    split_info:     dict,
    models_dir:     Path,
) -> None:
    """
    Save trained model artifacts and training metadata to models/.

    The split_info dict is saved as training_metadata.json — records
    exactly what data was used for training, the holdout AUC, and
    the scale_pos_weight used. Important for reproducibility and
    for explaining the model in interviews or reports.

    Parameters
    ----------
    xgb_model : XGBClassifier
        Trained XGBoost classifier.
    lr_model : LogisticRegression
        Trained baseline classifier.
    split_info : dict
        Holdout split details and evaluation metrics.
    models_dir : Path
        Directory to save artifacts.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(xgb_model, models_dir / "xgb_classifier.joblib")
    log.info(f"  ✓ xgb_classifier.joblib saved")

    joblib.dump(lr_model, models_dir / "baseline_lr.joblib")
    log.info(f"  ✓ baseline_lr.joblib saved")

    # Save training metadata
    metadata = {
        "random_state":       RANDOM_STATE,
        "holdout_split_date": HOLDOUT_SPLIT_DATE,
        "n_cv_splits":        N_CV_SPLITS,
        "xgb_params":         XGB_PARAMS,
        "leakage_columns":    LEAKAGE_COLUMNS,
        "notes": {
            "anchored_ratio_dominance": (
                "anchored_ratio (SHAP=2.67) dominates feature importance. "
                "It is computed as anchored_pings/total_pings — closely related "
                "to slow_vessel_ratio which feeds into port_congestion_score. "
                "This is borderline leakage even though the score itself was "
                "excluded. Flagged for evaluation report discussion."
            ),
            "perfect_auc_explanation": (
                "XGBoost AUC=1.0 on holdout likely reflects clean label "
                "separability within Q1 2023 for these 5 ports. "
                "Leave-one-port-out test in evaluate_classifier.py will "
                "reveal whether this generalises to unseen ports."
            ),
            "lr_scaling": (
                "Logistic Regression wrapped in StandardScaler Pipeline. "
                "Raw features span very different scales — scaling required "
                "for lbfgs convergence."
            ),
        },
        **split_info,
    }

    with open(models_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"  ✓ training_metadata.json saved")

    print("\n" + "=" * 60)
    print("  ARTIFACTS SAVED")
    print("=" * 60)
    for artifact in models_dir.iterdir():
        size_kb = artifact.stat().st_size / 1e3
        print(f"  {artifact.name:<40} {size_kb:>8.1f} KB")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    processed_dir = PROJECT_ROOT / config["paths"]["processed"]
    models_dir    = PROJECT_ROOT / config["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Phase 3 — Congestion Classifier Training")
    log.info(f"  Random state  : {RANDOM_STATE}")
    log.info(f"  CV splits     : {N_CV_SPLITS}")
    log.info(f"  Holdout split : {HOLDOUT_SPLIT_DATE}")
    log.info(f"  Leakage cols  : {len(LEAKAGE_COLUMNS)} excluded")
    log.info("=" * 60)

    # Step 1: Load data
    df = load_data(processed_dir)

    # Step 2: Prepare features — encode categoricals, drop leakage cols
    X, y, feature_columns, feature_dtypes = prepare_features(df, models_dir)

    # Preserve date series aligned with X for temporal splitting
    df_dates = df["date"].reset_index(drop=True)

    # Step 3: TimeSeriesSplit CV — exploratory validation
    cv_results = run_timeseries_cv(X, y, df_dates)

    # Step 4: Final holdout evaluation — train best model on full train set
    xgb_model, lr_model, X_test, y_test, split_info = run_holdout_evaluation(
        X, y, df_dates
    )

    # Step 5: SHAP on full dataset for global importance
    explainer = compute_shap(xgb_model, X, models_dir)

    # Step 6: Save all artifacts
    log.info("\nSaving artifacts ...")
    save_artifacts(xgb_model, lr_model, split_info, models_dir)

    log.info("\nPhase 3 Step 1 complete.")
    log.info("Next: run src/models/evaluate_classifier.py")


if __name__ == "__main__":
    main()

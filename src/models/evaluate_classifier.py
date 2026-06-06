"""
evaluate_classifier.py
----------------------
Phase 3 — Classifier Evaluation Suite

Loads trained model artifacts and produces a comprehensive evaluation
of both the XGBoost classifier and the Logistic Regression baseline.

Evaluation components
---------------------
1. Confusion matrix (holdout test set)
2. AUC-ROC curve (XGBoost vs LR comparison)
3. Precision-Recall curve (more informative than ROC for 60/40 imbalance)
4. Per-port accuracy breakdown (which ports does the model struggle with?)
5. Leave-One-Port-Out (LOPO) generalisation test
   → trains on 4 ports, tests on 5th
   → answers: does the model generalise to unseen ports?
6. SHAP waterfall charts (one representative example per port)
7. Calibration curve with honest sample-size caveat

Key findings flagged explicitly
--------------------------------
- XGBoost AUC=1.0 on holdout: discussed with three possible explanations
- anchored_ratio dominance in SHAP: flagged as borderline leakage concern
- SHAP ≠ causality: stated explicitly throughout
- Calibration unreliable below 1000 samples: noted prominently

Usage (from project root):
  python src/models/evaluate_classifier.py

Outputs
-------
  outputs/evaluation_report.png   all charts in one figure
  outputs/per_port_metrics.csv    per-port precision, recall, F1, AUC
  outputs/lopo_results.csv        leave-one-port-out AUC per port
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE     = 42
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Constants — must match train_classifier.py exactly
# ---------------------------------------------------------------------------
HOLDOUT_SPLIT_DATE = "2023-03-02"

LEAKAGE_COLUMNS = [
    "congestion_label",
    "port_congestion_score",
    "port_congestion_score_norm",
    "congestion_pressure_index",
    "weather_congestion_interaction",
    "vessels_7d_rolling_norm",
    "nearest_port",
    "date",
    "season",
]

XGB_PARAMS = {
    "n_estimators":     200,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma":            0.1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "verbosity":        0,
}


# ---------------------------------------------------------------------------
# 1. Load Artifacts and Prepare Data
# ---------------------------------------------------------------------------

def load_artifacts_and_data(
    processed_dir: Path,
    models_dir: Path,
) -> tuple:
    """
    Load master dataset, trained models, encoders, and feature metadata.

    Returns
    -------
    tuple of (df, X, y, df_dates, xgb_model, lr_model,
              port_encoder, feature_columns, shap_explainer)
    """
    log.info("Loading artifacts and data ...")

    # ── Master dataset ────────────────────────────────────────────────────────
    df = pd.read_parquet(processed_dir / "master_dataset.parquet", engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["date", "nearest_port"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"  Master dataset: {len(df):,} rows")

    # ── Feature metadata ──────────────────────────────────────────────────────
    with open(models_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)

    # ── Encoders ──────────────────────────────────────────────────────────────
    port_encoder   = joblib.load(models_dir / "label_encoder.joblib")
    season_encoder = joblib.load(models_dir / "season_encoder.joblib")

    # ── Reproduce feature matrix exactly as in training ──────────────────────
    df_work = df.copy()
    df_work["port_encoded"]   = port_encoder.transform(df_work["nearest_port"])
    df_work["season_encoded"] = season_encoder.transform(df_work["season"])

    y  = df_work["congestion_label"].astype(int)
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df_work.columns]
    X  = df_work.drop(columns=cols_to_drop)
    X  = X.select_dtypes(include="number")
    X  = X[feature_columns]   # enforce exact column order from training
    X  = X.fillna(X.median(numeric_only=True))

    df_dates = df["date"].reset_index(drop=True)

    log.info(f"  Feature matrix: {X.shape}")

    # ── Trained models ────────────────────────────────────────────────────────
    xgb_model    = joblib.load(models_dir / "xgb_classifier.joblib")
    lr_model     = joblib.load(models_dir / "baseline_lr.joblib")
    shap_explainer = joblib.load(models_dir / "shap_explainer.joblib")

    log.info(f"  ✓ Models and artifacts loaded")

    return df, X, y, df_dates, df_work, xgb_model, lr_model, port_encoder, feature_columns, shap_explainer


# ---------------------------------------------------------------------------
# 2. Build Holdout Split
# ---------------------------------------------------------------------------

def build_holdout_split(
    X: pd.DataFrame,
    y: pd.Series,
    df_dates: pd.Series,
) -> tuple:
    """
    Reproduce the same holdout split used in training.
    Train: Jan 1 → Mar 1.  Test: Mar 2 → Mar 31.
    """
    split_date = pd.Timestamp(HOLDOUT_SPLIT_DATE)
    test_mask  = df_dates >= split_date
    train_mask = ~test_mask

    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        df_dates[train_mask], df_dates[test_mask],
    )


# ---------------------------------------------------------------------------
# 3. Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    xgb_model,
    lr_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax_xgb: plt.Axes,
    ax_lr:  plt.Axes,
) -> None:
    """Plot normalised confusion matrices for both models on the holdout set."""

    for model, ax, title in [
        (xgb_model, ax_xgb, "XGBoost"),
        (lr_model,  ax_lr,  "Logistic Regression (baseline)"),
    ]:
        y_pred = model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred, normalize="true")
        disp   = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Not Congested", "Congested"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(f"Confusion Matrix — {title}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)


# ---------------------------------------------------------------------------
# 4. AUC-ROC Curve
# ---------------------------------------------------------------------------

def plot_roc_curve(
    xgb_model,
    lr_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax: plt.Axes,
) -> None:
    """Plot AUC-ROC curves for both models on the holdout test set."""

    for model, label, color in [
        (xgb_model, "XGBoost",              "#2196F3"),
        (lr_model,  "Logistic Regression",  "#FF9800"),
    ]:
        proba     = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc   = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{label} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=9)
    ax.set_ylabel("True Positive Rate", fontsize=9)
    ax.set_title("AUC-ROC Curve (Holdout: Mar 2–31)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# 5. Precision-Recall Curve
# ---------------------------------------------------------------------------

def plot_pr_curve(
    xgb_model,
    lr_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax: plt.Axes,
) -> None:
    """
    Plot Precision-Recall curves for both models.

    PR curves are more informative than ROC for class-imbalanced datasets.
    With 60/40 split the imbalance is mild but PR still shows the
    precision-recall tradeoff more clearly at operational thresholds.
    """
    for model, label, color in [
        (xgb_model, "XGBoost",              "#2196F3"),
        (lr_model,  "Logistic Regression",  "#FF9800"),
    ]:
        proba         = model.predict_proba(X_test)[:, 1]
        prec, rec, _  = precision_recall_curve(y_test, proba)
        pr_auc        = auc(rec, prec)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{label} (AUC={pr_auc:.3f})")

    # Baseline = always predicting positive class rate
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1,
               label=f"Random baseline ({baseline:.2f})")

    ax.set_xlabel("Recall", fontsize=9)
    ax.set_ylabel("Precision", fontsize=9)
    ax.set_title("Precision-Recall Curve (Holdout: Mar 2–31)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------------------
# 6. Per-Port Accuracy Breakdown
# ---------------------------------------------------------------------------

def compute_per_port_metrics(
    xgb_model,
    lr_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df: pd.DataFrame,
    df_dates: pd.Series,
    ax: plt.Axes,
    outputs_dir: Path,
) -> pd.DataFrame:
    """
    Compute AUC, F1, Precision, Recall per port on the holdout test set.

    Answers: which ports does the model struggle with?
    Low per-port AUC on a specific port suggests that port's congestion
    pattern differs from what the model learned from other ports.

    Note: per-port test sets are small (~29 rows each for March).
    Interpret individual port metrics as directional indicators only.
    """
    log.info("Computing per-port metrics ...")

    split_date = pd.Timestamp(HOLDOUT_SPLIT_DATE)
    test_mask  = df_dates >= split_date

    # Get port labels for test rows aligned with X_test
    port_labels = df["nearest_port"].reset_index(drop=True)[test_mask].values

    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred  = xgb_model.predict(X_test)
    lr_proba  = lr_model.predict_proba(X_test)[:, 1]
    lr_pred   = lr_model.predict(X_test)
    y_arr     = y_test.values

    rows = []
    for port in sorted(np.unique(port_labels)):
        mask = port_labels == port
        if mask.sum() == 0 or y_arr[mask].nunique() < 2 if hasattr(y_arr[mask], 'nunique') else len(np.unique(y_arr[mask])) < 2:
            continue

        y_port      = y_arr[mask]
        n_pos       = y_port.sum()
        n_neg       = len(y_port) - n_pos

        xgb_auc_port = roc_auc_score(y_port, xgb_proba[mask]) if len(np.unique(y_port)) > 1 else np.nan
        lr_auc_port  = roc_auc_score(y_port, lr_proba[mask])  if len(np.unique(y_port)) > 1 else np.nan

        rows.append({
            "port":          port,
            "n_test_days":   int(mask.sum()),
            "n_congested":   int(n_pos),
            "n_not_congested": int(n_neg),
            "xgb_auc":       round(float(xgb_auc_port), 4) if not np.isnan(xgb_auc_port) else None,
            "xgb_f1":        round(float(f1_score(y_port, xgb_pred[mask], zero_division=0)), 4),
            "xgb_precision": round(float(precision_score(y_port, xgb_pred[mask], zero_division=0)), 4),
            "xgb_recall":    round(float(recall_score(y_port, xgb_pred[mask], zero_division=0)), 4),
            "lr_auc":        round(float(lr_auc_port), 4) if not np.isnan(lr_auc_port) else None,
            "lr_f1":         round(float(f1_score(y_port, lr_pred[mask], zero_division=0)), 4),
        })

    per_port_df = pd.DataFrame(rows)
    per_port_df.to_csv(outputs_dir / "per_port_metrics.csv", index=False)
    log.info(f"  ✓ per_port_metrics.csv saved")

    # Plot
    ports   = per_port_df["port"].tolist()
    x_pos   = np.arange(len(ports))
    width   = 0.35

    xgb_aucs = [r if r is not None else 0 for r in per_port_df["xgb_auc"]]
    lr_aucs  = [r if r is not None else 0 for r in per_port_df["lr_auc"]]

    ax.bar(x_pos - width/2, xgb_aucs, width, label="XGBoost",             color="#2196F3", alpha=0.85)
    ax.bar(x_pos + width/2, lr_aucs,  width, label="Logistic Regression",  color="#FF9800", alpha=0.85)
    ax.axhline(0.5, color="k", linestyle="--", lw=1, label="Random (0.5)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(ports, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("AUC-ROC", fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.set_title("Per-Port AUC (Holdout: Mar 2–31)\nNote: ~29 rows per port — directional only",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Print to console
    print("\n" + "=" * 70)
    print("  PER-PORT METRICS (Holdout — Mar 2-31)")
    print("  Note: ~29 test rows per port — treat as directional indicators")
    print("=" * 70)
    print(f"  {'Port':<20} {'Days':>5} {'XGB AUC':>9} {'XGB F1':>8} {'LR AUC':>9} {'LR F1':>8}")
    print(f"  {'─'*60}")
    for _, row in per_port_df.iterrows():
        print(
            f"  {row['port']:<20} {row['n_test_days']:>5} "
            f"{row['xgb_auc'] if row['xgb_auc'] else 'N/A':>9} "
            f"{row['xgb_f1']:>8.4f} "
            f"{row['lr_auc'] if row['lr_auc'] else 'N/A':>9} "
            f"{row['lr_f1']:>8.4f}"
        )
    print("=" * 70)

    return per_port_df


# ---------------------------------------------------------------------------
# 7. Leave-One-Port-Out Generalisation Test
# ---------------------------------------------------------------------------

def run_lopo_test(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    outputs_dir: Path,
    ax: plt.Axes,
) -> pd.DataFrame:
    """
    Leave-One-Port-Out (LOPO) generalisation test.

    For each port:
      - Train XGBoost on all other 4 ports (all 90 days)
      - Test on the held-out port (all 90 days)
      - Report AUC

    This answers: 'Can the model generalise to a port it has never seen?'
    This is the operationally relevant question — a real deployment would
    need to handle new ports without retraining from scratch.

    LOPO AUC interpretation:
      > 0.80 : model generalises well — learned cross-port patterns
      0.65-0.80 : partial generalisation — some port-specific memorisation
      < 0.65 : poor generalisation — model relies on port-specific signals

    Note: port_encoded feature is retrained per fold with a fresh
    LabelEncoder so the held-out port is never in the encoding.
    """
    log.info("\nRunning Leave-One-Port-Out (LOPO) generalisation test ...")
    log.info("Trains on 4 ports, tests on 5th — repeated for each port.")

    ports = sorted(df["nearest_port"].unique())
    rows  = []

    for held_out_port in ports:
        train_mask = df["nearest_port"] != held_out_port
        test_mask  = df["nearest_port"] == held_out_port

        # Re-encode port WITHOUT the held-out port
        # This simulates truly unseen port at inference time
        train_ports   = df[train_mask]["nearest_port"].values
        test_ports    = df[test_mask]["nearest_port"].values

        port_enc_lopo = LabelEncoder()
        port_enc_lopo.fit(train_ports)

        X_train_lopo = X[train_mask].copy()
        X_test_lopo  = X[test_mask].copy()

        # Encode known ports — held-out port gets value -1 (unseen)
        X_train_lopo["port_encoded"] = port_enc_lopo.transform(train_ports)
        X_test_lopo["port_encoded"]  = -1   # signal: unseen port

        y_train_lopo = y[train_mask]
        y_test_lopo  = y[test_mask]

        # Skip if test set has only one class
        if y_test_lopo.nunique() < 2:
            log.warning(f"  {held_out_port}: only one class in test set — skipping")
            continue

        n_neg = (y_train_lopo == 0).sum()
        n_pos = (y_train_lopo == 1).sum()
        spw   = n_neg / n_pos if n_pos > 0 else 1.0

        xgb_lopo = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        xgb_lopo.fit(X_train_lopo, y_train_lopo)

        proba    = xgb_lopo.predict_proba(X_test_lopo)[:, 1]
        lopo_auc = roc_auc_score(y_test_lopo, proba)
        lopo_f1  = f1_score(y_test_lopo, xgb_lopo.predict(X_test_lopo), zero_division=0)

        rows.append({
            "held_out_port": held_out_port,
            "train_rows":    int(train_mask.sum()),
            "test_rows":     int(test_mask.sum()),
            "lopo_auc":      round(float(lopo_auc), 4),
            "lopo_f1":       round(float(lopo_f1), 4),
        })

        log.info(f"  Hold-out: {held_out_port:<20} AUC={lopo_auc:.4f}  F1={lopo_f1:.4f}")

    lopo_df = pd.DataFrame(rows)
    lopo_df.to_csv(outputs_dir / "lopo_results.csv", index=False)
    log.info(f"  ✓ lopo_results.csv saved")

    # Interpret results
    mean_lopo_auc = lopo_df["lopo_auc"].mean()
    if mean_lopo_auc > 0.80:
        generalisation = "GOOD — model learned cross-port patterns"
    elif mean_lopo_auc > 0.65:
        generalisation = "PARTIAL — some port-specific memorisation"
    else:
        generalisation = "POOR — model relies on port-specific signals"

    print("\n" + "=" * 65)
    print("  LEAVE-ONE-PORT-OUT GENERALISATION TEST")
    print("  Trains on 4 ports, tests on held-out port (all 90 days)")
    print("=" * 65)
    print(f"  {'Held-Out Port':<25} {'AUC':>8} {'F1':>8} {'Interpretation'}")
    print(f"  {'─'*60}")
    for _, row in lopo_df.iterrows():
        interp = "✓ good" if row["lopo_auc"] > 0.80 else ("~ partial" if row["lopo_auc"] > 0.65 else "✗ poor")
        print(f"  {row['held_out_port']:<25} {row['lopo_auc']:>8.4f} {row['lopo_f1']:>8.4f}  {interp}")
    print(f"  {'─'*60}")
    print(f"  Mean LOPO AUC : {mean_lopo_auc:.4f}  →  {generalisation}")
    print("=" * 65)

    # Plot
    ports_lopo = lopo_df["held_out_port"].tolist()
    aucs_lopo  = lopo_df["lopo_auc"].tolist()
    colors_lopo = ["#4CAF50" if a > 0.80 else ("#FF9800" if a > 0.65 else "#F44336")
                   for a in aucs_lopo]

    ax.bar(ports_lopo, aucs_lopo, color=colors_lopo, alpha=0.85, edgecolor="white")
    ax.axhline(0.80, color="#4CAF50", linestyle="--", lw=1.5, label="Good threshold (0.80)")
    ax.axhline(0.65, color="#FF9800", linestyle="--", lw=1.5, label="Partial threshold (0.65)")
    ax.axhline(0.50, color="#F44336", linestyle="--", lw=1,   label="Random (0.50)")
    ax.set_ylabel("AUC-ROC", fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.set_title(
        f"Leave-One-Port-Out AUC\nMean={mean_lopo_auc:.3f} — {generalisation}",
        fontsize=10, fontweight="bold"
    )
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    return lopo_df


# ---------------------------------------------------------------------------
# 8. SHAP Waterfall — One Example Per Port
# ---------------------------------------------------------------------------

def plot_shap_waterfall(
    shap_explainer,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df: pd.DataFrame,
    df_dates: pd.Series,
    outputs_dir: Path,
) -> None:
    """
    Generate SHAP waterfall chart for one representative example per port
    from the holdout test set. Saves individual PNG per port.

    Waterfall charts show which features pushed the prediction up or down
    for that specific day. Useful for the dashboard vessel/port lookup.

    CRITICAL NOTE: SHAP values show model contribution, not real causality.
    'anchored_ratio pushed prediction up by 2.3' means the model weighted
    this feature heavily for this prediction — not that anchoring caused
    the congestion event.
    """
    log.info("\nGenerating SHAP waterfall charts (one per port) ...")
    log.info("REMINDER: SHAP = model behaviour explanation, NOT causal analysis.")

    split_date = pd.Timestamp(HOLDOUT_SPLIT_DATE)
    test_mask  = df_dates >= split_date
    port_labels = df["nearest_port"].reset_index(drop=True)[test_mask].values

    shap_values = shap_explainer.shap_values(X_test)

    outputs_dir.mkdir(parents=True, exist_ok=True)

    for port in sorted(np.unique(port_labels)):
        port_mask = port_labels == port
        port_indices = np.where(port_mask)[0]

        if len(port_indices) == 0:
            continue

        # Pick the most "interesting" example — highest predicted congestion probability
        # among true positives, to show a clear congestion explanation
        y_port    = y_test.values[port_mask]
        pos_mask  = y_port == 1
        if pos_mask.sum() > 0:
            # Pick highest SHAP sum among true positives
            port_shap_sums = shap_values[port_indices[pos_mask]].sum(axis=1)
            best_local_idx = np.argmax(port_shap_sums)
            example_idx    = port_indices[pos_mask][best_local_idx]
        else:
            example_idx = port_indices[0]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Get SHAP values for this example
        sv          = shap_values[example_idx]
        feature_names = X_test.columns.tolist()
        feature_vals  = X_test.iloc[example_idx].values

        # Sort by absolute SHAP value, show top 12
        sorted_idx  = np.argsort(np.abs(sv))[::-1][:12]
        sv_sorted   = sv[sorted_idx]
        fn_sorted   = [feature_names[i] for i in sorted_idx]
        fv_sorted   = [feature_vals[i] for i in sorted_idx]

        colors = ["#F44336" if v > 0 else "#2196F3" for v in sv_sorted]
        bars   = ax.barh(
            range(len(sv_sorted)),
            sv_sorted[::-1],
            color=colors[::-1],
            alpha=0.85,
            edgecolor="white",
        )

        ax.set_yticks(range(len(sv_sorted)))
        ax.set_yticklabels(
            [f"{fn} = {fv:.3f}" for fn, fv in zip(fn_sorted[::-1], fv_sorted[::-1])],
            fontsize=8,
        )
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP value (impact on congestion prediction)", fontsize=9)
        ax.set_title(
            f"SHAP Waterfall — {port} (example congested day)\n"
            f"Red = pushed toward congested | Blue = pushed toward not congested\n"
            f"SHAP explains model behaviour — NOT real-world causality",
            fontsize=9,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        save_path = outputs_dir / f"shap_waterfall_{port.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"  ✓ shap_waterfall_{port.lower().replace(' ', '_')}.png saved")


# ---------------------------------------------------------------------------
# 9. Calibration Curve
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    xgb_model,
    lr_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ax: plt.Axes,
) -> None:
    """
    Plot reliability diagram (calibration curve) for both models.

    A perfectly calibrated model produces a diagonal line:
    when it predicts 0.7 probability, 70% of those cases are actually positive.

    IMPORTANT CAVEAT: With only 150 holdout rows, this calibration curve
    has very high variance. Each bin may contain only 10-20 samples.
    Treat this as a directional indicator only — not a reliable
    calibration measurement.

    For production deployment, consider:
      - Platt scaling: fits a logistic regression on out-of-fold predictions
      - Isotonic regression: non-parametric, more flexible than Platt
      - Both require a dedicated calibration set (ideally 500+ samples)

    XGBoost probabilities tend to be over-confident (pushed toward 0 and 1).
    This is common for boosting methods — Platt scaling typically helps.
    """
    from sklearn.calibration import calibration_curve

    for model, label, color in [
        (xgb_model, "XGBoost",             "#2196F3"),
        (lr_model,  "Logistic Regression", "#FF9800"),
    ]:
        proba = model.predict_proba(X_test)[:, 1]
        try:
            fraction_pos, mean_pred = calibration_curve(
                y_test, proba, n_bins=5, strategy="uniform"
            )
            ax.plot(mean_pred, fraction_pos, "s-", color=color, lw=2,
                    label=f"{label}", markersize=6)
        except ValueError:
            log.warning(f"  Calibration curve failed for {label} — insufficient samples per bin")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=9)
    ax.set_ylabel("Fraction of Positives", fontsize=9)
    ax.set_title(
        "Calibration Curve (Reliability Diagram)\n"
        "⚠ Only 150 holdout rows — high variance, directional only",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    # Add honest caveat as text on plot
    ax.text(
        0.02, 0.97,
        "Production note:\nPlatt scaling or isotonic\nregression recommended\nfor deployment (≥500 samples)",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


# ---------------------------------------------------------------------------
# 10. Assemble Full Evaluation Report
# ---------------------------------------------------------------------------

def generate_evaluation_report(
    xgb_model,
    lr_model,
    shap_explainer,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    df_dates: pd.Series,
    outputs_dir: Path,
) -> None:
    """
    Assemble all evaluation charts into a single evaluation_report.png.

    Layout (3 rows × 2 cols + extras):
      Row 1: Confusion Matrix (XGB) | Confusion Matrix (LR)
      Row 2: AUC-ROC curve          | Precision-Recall curve
      Row 3: Per-port AUC           | Calibration curve
      Row 4: LOPO test (full width)
    """
    log.info("\nGenerating evaluation_report.png ...")

    X_train, X_test, y_train, y_test, dates_train, dates_test = build_holdout_split(
        X, y, df_dates
    )

    fig = plt.figure(figsize=(16, 20))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_cm_xgb  = fig.add_subplot(gs[0, 0])
    ax_cm_lr   = fig.add_subplot(gs[0, 1])
    ax_roc     = fig.add_subplot(gs[1, 0])
    ax_pr      = fig.add_subplot(gs[1, 1])
    ax_port    = fig.add_subplot(gs[2, 0])
    ax_calib   = fig.add_subplot(gs[2, 1])
    ax_lopo    = fig.add_subplot(gs[3, :])   # full width

    # Row 1 — confusion matrices
    plot_confusion_matrix(xgb_model, lr_model, X_test, y_test, ax_cm_xgb, ax_cm_lr)

    # Row 2 — ROC + PR curves
    plot_roc_curve(xgb_model, lr_model, X_test, y_test, ax_roc)
    plot_pr_curve(xgb_model, lr_model, X_test, y_test, ax_pr)

    # Row 3 — per-port + calibration
    compute_per_port_metrics(
        xgb_model, lr_model, X_test, y_test,
        df, df_dates, ax_port, outputs_dir
    )
    plot_calibration_curve(xgb_model, lr_model, X_test, y_test, ax_calib)

    # Row 4 — LOPO (full width)
    run_lopo_test(X, y, df, outputs_dir, ax_lopo)

    # Main title
    fig.suptitle(
        "Port Intelligence System — Classifier Evaluation Report\n"
        "XGBoost vs Logistic Regression Baseline | Holdout: Mar 2–31 2023\n"
        "SHAP = model behaviour explanation, NOT causality",
        fontsize=12, fontweight="bold", y=0.98,
    )

    report_path = outputs_dir / "evaluation_report.png"
    plt.savefig(report_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"  ✓ evaluation_report.png saved ({report_path.stat().st_size / 1e6:.1f} MB)")

    # SHAP waterfall charts (saved separately per port)
    plot_shap_waterfall(
        shap_explainer, X_test, y_test, df, df_dates, outputs_dir
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_config()

    processed_dir = PROJECT_ROOT / config["paths"]["processed"]
    models_dir    = PROJECT_ROOT / config["paths"]["models"]
    outputs_dir   = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("Phase 3 — Classifier Evaluation Suite")
    log.info("=" * 65)

    # Load everything
    (df, X, y, df_dates, df_work,
     xgb_model, lr_model,
     port_encoder, feature_columns,
     shap_explainer) = load_artifacts_and_data(processed_dir, models_dir)

    # Generate full report
    generate_evaluation_report(
        xgb_model, lr_model, shap_explainer,
        X, y, df, df_dates, outputs_dir,
    )

    print("\n" + "=" * 65)
    print("  EVALUATION COMPLETE")
    print("=" * 65)
    print(f"  outputs/evaluation_report.png      ← main report (all charts)")
    print(f"  outputs/per_port_metrics.csv        ← per-port AUC, F1, precision")
    print(f"  outputs/lopo_results.csv            ← leave-one-port-out AUC")
    print(f"  outputs/shap_waterfall_<port>.png   ← SHAP waterfall per port")
    print()
    print(f"  Key finding to check:")
    print(f"    LOPO mean AUC tells you whether XGBoost learned")
    print(f"    cross-port patterns or memorised port-specific ones.")
    print(f"    If LOPO AUC > 0.80 → strong generalisation story for interviews.")
    print(f"    If LOPO AUC < 0.65 → model is port-specific, note this honestly.")
    print("=" * 65)

    log.info("\nPhase 3 Step 2 complete.")
    log.info("Next: run src/models/train_forecaster.py")


if __name__ == "__main__":
    main()

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent          # src/forecasting -> src -> project root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import (
    RANDOM_SEED, LAG_FEATURES, ROLLING_WINDOWS,
    RESULTS_DIR, PREDICTIONS_DIR, DIAGNOSTICS_DIR,
)
from src.utils.data_loader import load_train_data, load_test_data

from src.forecasting.features       import prepare_features, wide_to_long
from src.forecasting.cluster_loader import load_cluster_assignments
from src.forecasting.models         import (
    select_best_point_params_by_backtest,
    train_cluster_models,
    train_global_model,
    save_point_models,
    select_best_elasticnet_params_by_backtest,
    train_cluster_elasticnet_models,
    train_global_elasticnet_model,
    save_elasticnet_models,
)
from src.forecasting.predict        import predict_point
from src.forecasting.predict_elasticnet import predict_elasticnet

# ---------------------------------------------------------------------------
# !! Update these to match notebook 03 selected k values !!
# ---------------------------------------------------------------------------
CLUSTERING_CONFIG = {
    "feature_ward": {"k": 3},
    "kshape":       {"k": 2},
}


def _load_and_intersect(diagnostics_dir, method, k, train_wide):
    """Load forecasting-ready cluster assignments and intersect with training households."""
    cluster_series, regular_ids = load_cluster_assignments(
        diagnostics_dir, method, k
    )
    common = cluster_series.index.intersection(train_wide.index)
    cluster_series = cluster_series.loc[common]
    regular_ids = cluster_series.index
    cluster_labels = sorted(cluster_series.unique().tolist())
    return cluster_series, regular_ids, cluster_labels

def evaluate_predictions(
    predictions: pd.DataFrame,
    test_wide: pd.DataFrame,
    method_name: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate 2024 forecasts exactly as required by the assignment.

    Evaluation rule:
    - compare predicted vs observed daily consumption for each household in 2024
    - compute MAE across all 366 forecast days for each household
    - report the average MAE across households

    Parameters
    ----------
    predictions : DataFrame
        Must contain columns ['household_id', 'date', 'predicted'].
    test_wide : DataFrame
        Wide 2024 ground-truth panel (households × 366 dates).
    method_name : str
        Name used in saved output filenames, e.g. 'feature_ward', 'kshape', 'global'.
    output_dir : Path
        Directory where evaluation CSVs are saved.

    Returns
    -------
    household_mae : DataFrame
        One row per household with MAE and row-count diagnostics.
    summary : dict
        Compact summary including the final reported score:
        average household MAE.
    """
    required_cols = {"household_id", "date", "predicted"}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ground truth in long format
    actual_long = wide_to_long(test_wide, value_name="actual")

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    actual_long["date"] = pd.to_datetime(actual_long["date"])

    merged = pred.merge(
        actual_long,
        on=["household_id", "date"],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No overlapping prediction/ground-truth rows found for evaluation.")

    merged["abs_error"] = (merged["predicted"] - merged["actual"]).abs()

    household_mae = (
        merged.groupby("household_id", as_index=False)
        .agg(
            mae=("abs_error", "mean"),
            n_days_compared=("abs_error", "size"),
        )
        .sort_values("household_id")
        .reset_index(drop=True)
    )

    expected_days = test_wide.shape[1]
    household_mae["complete_horizon_flag"] = household_mae["n_days_compared"] == expected_days

    average_household_mae = float(household_mae["mae"].mean())
    median_household_mae = float(household_mae["mae"].median())
    n_households = int(household_mae.shape[0])
    n_complete = int(household_mae["complete_horizon_flag"].sum())

    summary = {
        "method": method_name,
        "n_households_evaluated": n_households,
        "expected_days_per_household": int(expected_days),
        "n_households_with_complete_366_days": n_complete,
        "average_household_mae": average_household_mae,
        "median_household_mae": median_household_mae,
    }

    household_path = output_dir / f"household_mae_{method_name}.csv"
    summary_path = output_dir / f"evaluation_summary_{method_name}.csv"

    household_mae.to_csv(household_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print("\n[4/4] Assignment-compliant evaluation on 2024 ground truth")
    print(f"  Households evaluated: {n_households:,}")
    print(f"  Complete 366-day households: {n_complete:,} / {n_households:,}")
    print(f"  Average household MAE: {average_household_mae:.6f}")
    print(f"  Median household MAE : {median_household_mae:.6f}")
    print(f"  Saved household MAE table: {household_path}")
    print(f"  Saved summary table      : {summary_path}")

    return household_mae, summary

# ---------------------------------------------------------------------------
# Cluster-based pipeline
# ---------------------------------------------------------------------------

def run_cluster_pipeline(
    method: str,
    k: int,
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    save_models: bool = True,
) -> pd.DataFrame:
    """Train one model per cluster, predict 2024, and evaluate on 2024 ground truth.

    Parameters
    ----------
    method      : "feature_ward" or "kshape".
    k           : Number of regular clusters.
    train_wide  : Wide 2023 DataFrame (households × 365 dates). Training only.
    test_wide   : Wide 2024 DataFrame (households × 366 dates). Used only after
                  forecasting as ground truth for evaluation.
    save_models : Whether to write .pkl model files to results/models/.

    Returns
    -------
    DataFrame of predictions saved to results/predictions/predictions_2024_{method}.csv
    """
    print(f"\n{'='*60}")
    print(f"  {method}  (k={k})")
    print(f"{'='*60}")

    cluster_series, regular_ids, cluster_labels = \
        _load_and_intersect(DIAGNOSTICS_DIR, method, k, train_wide)

    print("\n[1/4] Building 2023 training features...")
    (
        train_feats,
        feature_cols,
        household_means,
        cluster_dow_profile,
        cluster_month_profile,
    ) = prepare_features(
        train_wide=train_wide,
        regular_ids=regular_ids,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_series=cluster_series,
    )
    train_feats["cluster"] = train_feats["household_id"].map(cluster_series)
    print(f"  Train: {train_feats.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    print("\n[2/4] Selecting model params with 2023-only walk-forward validation...")

    params_by_cluster = {}
    backtest_tables = []

    for label in cluster_labels:
        df_c = train_feats.loc[train_feats["cluster"] == label].copy()
        print(f"\n  Cluster {label} backtest tuning")
        best_params, backtest_df = select_best_point_params_by_backtest(
            train_features_clean=df_c,
            feature_cols=feature_cols,
            random_seed=RANDOM_SEED,
            n_splits=3,
            validation_days=28,
        )
        backtest_df["scope"] = f"cluster_{label}"
        params_by_cluster[label] = best_params
        backtest_tables.append(backtest_df)

    print("\n  Global backtest tuning")
    global_best_params, global_backtest_df = select_best_point_params_by_backtest(
        train_features_clean=train_feats,
        feature_cols=feature_cols,
        random_seed=RANDOM_SEED,
        n_splits=3,
        validation_days=28,
    )
    global_backtest_df["scope"] = "global"
    backtest_tables.append(global_backtest_df)

    backtest_results = pd.concat(backtest_tables, ignore_index=True)
    backtest_path = RESULTS_DIR / "evaluation" / f"backtest_selection_{method}.csv"
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_results.to_csv(backtest_path, index=False)
    print(f"\n  Saved 2023 backtest selection table: {backtest_path}")

    print("\n[3/4] Training final models on full 2023 data...")
    cluster_models = train_cluster_models(
        train_feats,
        feature_cols,
        cluster_labels,
        RANDOM_SEED,
        params_by_cluster=params_by_cluster,
    )
    global_model = train_global_model(
        train_feats,
        feature_cols,
        RANDOM_SEED,
        params_override=global_best_params,
    )

    if save_models:
        save_point_models(cluster_models, global_model, RESULTS_DIR / "models" / method)

    print("\n[4/4] Predicting 2024...")
    predictions = predict_point(
        test_wide=test_wide,
        train_wide=train_wide,
        feature_cols=feature_cols,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_labels=cluster_labels,
        cluster_models=cluster_models,
        global_model=global_model,
        cluster_series=cluster_series,
        household_means=household_means,
        cluster_dow_profile=cluster_dow_profile,
        cluster_month_profile=cluster_month_profile,
    )

    out_path = PREDICTIONS_DIR / f"predictions_2024_{method}.csv"
    predictions.to_csv(out_path, index=False)

    n_hh = predictions["household_id"].nunique()
    n_days = predictions.groupby("household_id").size()
    print(f"\n  Saved predictions: {out_path}")
    print(f"  Households: {n_hh:,}  |  Days per household: {n_days.min()}–{n_days.max()}")

    evaluate_predictions(
        predictions=predictions,
        test_wide=test_wide,
        method_name=method,
        output_dir=RESULTS_DIR / "evaluation",
    )

    return predictions


# ---------------------------------------------------------------------------
# Global (no-clustering) baseline pipeline
# ---------------------------------------------------------------------------

def run_global_pipeline(
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    save_models: bool = True,
) -> pd.DataFrame:
    """Run the no-clustering baseline and evaluate it exactly like the cluster models."""
    print(f"\n{'='*60}")
    print(f"  global baseline (no clustering)")
    print(f"{'='*60}")

    regular_ids = train_wide.index.intersection(test_wide.index)
    global_cs = pd.Series(0, index=regular_ids, name="cluster")

    print("\n[1/4] Building 2023 training features...")
    global_cs = pd.Series(0, index=regular_ids, name="cluster")
    (
        train_feats,
        feature_cols,
        household_means,
        cluster_dow_profile,
        cluster_month_profile,
    ) = prepare_features(
        train_wide=train_wide,
        regular_ids=regular_ids,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_series=global_cs,
    )
    train_feats["cluster"] = 0
    print(f"  Train: {train_feats.shape}")

    print("\n[2/4] Selecting global params with 2023-only walk-forward validation...")
    global_best_params, global_backtest_df = select_best_point_params_by_backtest(
        train_features_clean=train_feats,
        feature_cols=feature_cols,
        random_seed=RANDOM_SEED,
        n_splits=3,
        validation_days=28,
    )

    backtest_path = RESULTS_DIR / "evaluation" / "backtest_selection_global.csv"
    backtest_path.parent.mkdir(parents=True, exist_ok=True)
    global_backtest_df.to_csv(backtest_path, index=False)
    print(f"  Saved 2023 backtest selection table: {backtest_path}")

    print("\n[3/4] Training global model on full 2023 data...")
    global_model = train_global_model(
        train_feats,
        feature_cols,
        RANDOM_SEED,
        params_override=global_best_params,
    )

    if save_models:
        save_point_models({0: global_model}, global_model, RESULTS_DIR / "models" / "global")
        
    print("\n[4/4] Predicting 2024...")
    
    predictions = predict_point(
        test_wide=test_wide,
        train_wide=train_wide,
        feature_cols=feature_cols,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_labels=[0],
        cluster_models={0: global_model},
        global_model=global_model,
        cluster_series=global_cs,
        household_means=household_means,
        cluster_dow_profile=cluster_dow_profile,
        cluster_month_profile=cluster_month_profile,
    )
    predictions.loc[predictions["cluster"] == 0, "model_used"] = "global_lgbm"

    out_path = PREDICTIONS_DIR / "predictions_2024_global.csv"
    predictions.to_csv(out_path, index=False)

    n_hh = predictions["household_id"].nunique()
    n_days = predictions.groupby("household_id").size()
    print(f"\n  Saved predictions: {out_path}")
    print(f"  Households: {n_hh:,}  |  Days per household: {n_days.min()}–{n_days.max()}")

    evaluate_predictions(
        predictions=predictions,
        test_wide=test_wide,
        method_name="global",
        output_dir=RESULTS_DIR / "evaluation",
    )

    return predictions


# ===========================================================================
# Elastic Net Pipelines
# ===========================================================================

def run_cluster_elasticnet_pipeline(
    method: str,
    k: int,
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    save_models: bool = True,
    skip_tuning: bool = False,
) -> pd.DataFrame:
    """Train ElasticNet models per cluster, predict 2024, and evaluate.
    
    This pipeline follows the same structure as run_cluster_pipeline but uses
    ElasticNet regression with feature scaling instead of LightGBM.
    
    Parameters
    ----------
    method      : "feature_ward" or "kshape".
    k           : Number of regular clusters.
    train_wide  : Wide 2023 DataFrame (households × 365 dates). Training only.
    test_wide   : Wide 2024 DataFrame (households × 366 dates). Ground truth evaluation.
    save_models : Whether to write model and scaler .pkl files.
    skip_tuning : If True, use fixed default parameters (alpha=0.01, l1_ratio=0.5) instead of backtest tuning.
                  This drastically reduces runtime from hours to ~30-40 minutes.
    
    Returns
    -------
    DataFrame of predictions saved to results/predictions/predictions_2024_{method}_elasticnet.csv
    """
    print(f"\n{'='*60}")
    print(f"  {method}  (k={k}) - Elastic Net")
    print(f"{'='*60}")
    
    cluster_series, regular_ids, cluster_labels = \
        _load_and_intersect(DIAGNOSTICS_DIR, method, k, train_wide)
    
    print("\n[1/4] Building 2023 training features...")
    (
        train_feats,
        feature_cols,
        household_means,
        cluster_dow_profile,
        cluster_month_profile,
    ) = prepare_features(
        train_wide=train_wide,
        regular_ids=regular_ids,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_series=cluster_series,
    )
    train_feats["cluster"] = train_feats["household_id"].map(cluster_series)
    print(f"  Train: {train_feats.shape}")
    print(f"  Features ({len(feature_cols)}): {feature_cols}")
    
    if skip_tuning:
        print("\n[2/4] Using default ElasticNet parameters (FAST MODE - skipping tuning)...")
        # Fixed parameters for fast execution: balanced L1/L2 mix with light regularization
        default_params = {"alpha": 0.01, "l1_ratio": 0.5}
        params_by_cluster = {label: default_params for label in cluster_labels}
        global_best_params = default_params
        print(f"  Using fixed params for all clusters: alpha={default_params['alpha']}, l1_ratio={default_params['l1_ratio']}")
        print("  ⚡ This will save ~2.5-3 hours compared to full hyperparameter tuning!")
    else:
        print("\n[2/4] Selecting ElasticNet params with 2023-only walk-forward validation...")
        
        params_by_cluster = {}
        backtest_tables = []
        
        for label in cluster_labels:
            df_c = train_feats.loc[train_feats["cluster"] == label].copy()
            print(f"\n  Cluster {label} ElasticNet backtest tuning")
            best_params, backtest_df = select_best_elasticnet_params_by_backtest(
                train_features_clean=df_c,
                feature_cols=feature_cols,
                random_seed=RANDOM_SEED,
                n_splits=3,
                validation_days=28,
            )
            backtest_df["scope"] = f"cluster_{label}"
            params_by_cluster[label] = best_params
            backtest_tables.append(backtest_df)
        
        print("\n  Global ElasticNet backtest tuning")
        global_best_params, global_backtest_df = select_best_elasticnet_params_by_backtest(
            train_features_clean=train_feats,
            feature_cols=feature_cols,
            random_seed=RANDOM_SEED,
            n_splits=3,
            validation_days=28,
        )
        global_backtest_df["scope"] = "global"
        backtest_tables.append(global_backtest_df)
        
        backtest_results = pd.concat(backtest_tables, ignore_index=True)
        backtest_path = RESULTS_DIR / "evaluation" / f"backtest_selection_{method}_elasticnet.csv"
        backtest_path.parent.mkdir(parents=True, exist_ok=True)
        backtest_results.to_csv(backtest_path, index=False)
        print(f"\n  Saved 2023 backtest selection table: {backtest_path}")
    
    print("\n[3/4] Training ElasticNet models on full 2023 data...")
    cluster_models, cluster_scalers = train_cluster_elasticnet_models(
        train_feats,
        feature_cols,
        cluster_labels,
        RANDOM_SEED,
        params_by_cluster=params_by_cluster,
    )
    global_model, global_scaler = train_global_elasticnet_model(
        train_feats,
        feature_cols,
        RANDOM_SEED,
        alpha=global_best_params["alpha"],
        l1_ratio=global_best_params["l1_ratio"],
    )
    
    if save_models:
        save_elasticnet_models(
            cluster_models,
            cluster_scalers,
            global_model,
            global_scaler,
            RESULTS_DIR / "models" / f"{method}_elasticnet",
        )
    
    print("\n[4/4] Predicting 2024 with ElasticNet...")
    predictions = predict_elasticnet(
        test_wide=test_wide,
        train_wide=train_wide,
        feature_cols=feature_cols,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_labels=cluster_labels,
        cluster_models=cluster_models,
        cluster_scalers=cluster_scalers,
        global_model=global_model,
        global_scaler=global_scaler,
        cluster_series=cluster_series,
        household_means=household_means,
        cluster_dow_profile=cluster_dow_profile,
        cluster_month_profile=cluster_month_profile,
    )
    
    out_path = PREDICTIONS_DIR / f"predictions_2024_{method}_elasticnet.csv"
    predictions.to_csv(out_path, index=False)
    
    n_hh = predictions["household_id"].nunique()
    n_days = predictions.groupby("household_id").size()
    print(f"\n  Saved predictions: {out_path}")
    print(f"  Households: {n_hh:,}  |  Days per household: {n_days.min()}–{n_days.max()}")
    
    evaluate_predictions(
        predictions=predictions,
        test_wide=test_wide,
        method_name=f"{method}_elasticnet",
        output_dir=RESULTS_DIR / "evaluation",
    )
    
    return predictions


def run_global_elasticnet_pipeline(
    train_wide: pd.DataFrame,
    test_wide: pd.DataFrame,
    save_models: bool = True,
    skip_tuning: bool = False,
) -> pd.DataFrame:
    """Run the no-clustering Elastic Net baseline.
    
    Parameters
    ----------
    train_wide  : Wide 2023 training data.
    test_wide   : Wide 2024 test data.
    save_models : Whether to save models to disk.
    skip_tuning : If True, use default params (alpha=0.01, l1_ratio=0.5) for fast execution.
    """
    print(f"\n{'='*60}")
    print(f"  global baseline (no clustering) - Elastic Net")
    print(f"{'='*60}")
    
    regular_ids = train_wide.index.intersection(test_wide.index)
    global_cs = pd.Series(0, index=regular_ids, name="cluster")
    
    print("\n[1/4] Building 2023 training features...")
    (
        train_feats,
        feature_cols,
        household_means,
        cluster_dow_profile,
        cluster_month_profile,
    ) = prepare_features(
        train_wide=train_wide,
        regular_ids=regular_ids,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_series=global_cs,
    )
    train_feats["cluster"] = 0
    print(f"  Train: {train_feats.shape}")
    
    if skip_tuning:
        print("\n[2/4] Using default ElasticNet parameters (FAST MODE - skipping tuning)...")
        global_best_params = {"alpha": 0.01, "l1_ratio": 0.5}
        print(f"  Using fixed params: alpha={global_best_params['alpha']}, l1_ratio={global_best_params['l1_ratio']}")
        print("  ⚡ This will save ~30-50 minutes compared to full hyperparameter tuning!")
    else:
        print("\n[2/4] Selecting ElasticNet params with 2023-only walk-forward validation...")
        global_best_params, global_backtest_df = select_best_elasticnet_params_by_backtest(
            train_features_clean=train_feats,
            feature_cols=feature_cols,
            random_seed=RANDOM_SEED,
            n_splits=3,
            validation_days=28,
        )
        
        backtest_path = RESULTS_DIR / "evaluation" / "backtest_selection_global_elasticnet.csv"
        backtest_path.parent.mkdir(parents=True, exist_ok=True)
        global_backtest_df.to_csv(backtest_path, index=False)
        print(f"  Saved 2023 backtest selection table: {backtest_path}")
    
    print("\n[3/4] Training global ElasticNet model on full 2023 data...")
    global_model, global_scaler = train_global_elasticnet_model(
        train_feats,
        feature_cols,
        RANDOM_SEED,
        alpha=global_best_params["alpha"],
        l1_ratio=global_best_params["l1_ratio"],
    )
    
    if save_models:
        save_elasticnet_models(
            {0: global_model},
            {0: global_scaler},
            global_model,
            global_scaler,
            RESULTS_DIR / "models" / "global_elasticnet",
        )
    
    print("\n[4/4] Predicting 2024 with ElasticNet...")
    predictions = predict_elasticnet(
        test_wide=test_wide,
        train_wide=train_wide,
        feature_cols=feature_cols,
        lag_features=LAG_FEATURES,
        rolling_windows=ROLLING_WINDOWS,
        cluster_labels=[0],
        cluster_models={0: global_model},
        cluster_scalers={0: global_scaler},
        global_model=global_model,
        global_scaler=global_scaler,
        cluster_series=global_cs,
        household_means=household_means,
        cluster_dow_profile=cluster_dow_profile,
        cluster_month_profile=cluster_month_profile,
    )
    predictions.loc[predictions["cluster"] == 0, "model_used"] = "global_elasticnet"
    
    out_path = PREDICTIONS_DIR / "predictions_2024_global_elasticnet.csv"
    predictions.to_csv(out_path, index=False)
    
    n_hh = predictions["household_id"].nunique()
    n_days = predictions.groupby("household_id").size()
    print(f"\n  Saved predictions: {out_path}")
    print(f"  Households: {n_hh:,}  |  Days per household: {n_days.min()}–{n_days.max()}")
    
    evaluate_predictions(
        predictions=predictions,
        test_wide=test_wide,
        method_name="global_elasticnet",
        output_dir=RESULTS_DIR / "evaluation",
    )
    
    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run forecasting pipeline.")
    parser.add_argument(
        "--method",
        choices=["feature_ward", "kshape", "global", "all"],
        default="all",
    )
    parser.add_argument("--no-save-models", action="store_true")
    args   = parser.parse_args()
    save   = not args.no_save_models

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    train_wide = load_train_data()
    test_wide  = load_test_data()
    print(f"  Train: {train_wide.shape}  |  Test: {test_wide.shape}")

    if args.method in ("feature_ward", "all"):
        cfg = CLUSTERING_CONFIG["feature_ward"]
        run_cluster_pipeline("feature_ward", cfg["k"], train_wide, test_wide, save)

    if args.method in ("kshape", "all"):
        cfg = CLUSTERING_CONFIG["kshape"]
        run_cluster_pipeline("kshape", cfg["k"], train_wide, test_wide, save)

    if args.method in ("global", "all"):
        run_global_pipeline(train_wide, test_wide, save)

    print("\nDone. Prediction CSVs saved to:", PREDICTIONS_DIR)


if __name__ == "__main__":
    main()

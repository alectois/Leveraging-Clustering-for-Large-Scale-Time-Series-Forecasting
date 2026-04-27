import pickle
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

QUANTILES = [0.10, 0.50, 0.90]

_BASE_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)


def _point_params(random_seed: int, overrides: dict | None = None) -> dict:
    """Build point-forecast LightGBM parameters."""
    params = {
        **_BASE_PARAMS,
        "objective": "regression_l1",
        "metric": "mae",
        "random_state": random_seed,
    }
    if overrides:
        params.update(overrides)
    return params


def _quantile_params(q: float, random_seed: int) -> dict:
    return {
        **_BASE_PARAMS,
        "objective": "quantile",
        "metric": "quantile",
        "alpha": q,
        "random_state": random_seed,
    }


def _mae(y_true, y_pred) -> float:
    """Simple MAE helper."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _candidate_point_param_grid() -> list[dict]:
    """Small, readable 2023-only tuning grid.

    This is intentionally compact so it remains computationally practical
    for one global model plus one model per cluster.
    """
    return [
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
        },
        {
            "n_estimators": 700,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_child_samples": 30,
        },
        {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 127,
            "min_child_samples": 40,
        },
    ]


def _time_based_backtest_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    validation_days: int = 28,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Create rolling-origin validation windows inside 2023 only.

    Each split validates on a later block of 2023 dates. Training uses all
    earlier rows only, so the evaluation respects temporal order.
    """
    unique_dates = np.array(sorted(pd.to_datetime(df["date"]).unique()))
    if unique_dates.size < validation_days + 30:
        return []

    splits = []
    for split_idx in range(n_splits, 0, -1):
        end_pos = unique_dates.size - (split_idx - 1) * validation_days
        start_pos = end_pos - validation_days
        if start_pos <= 0:
            continue

        val_start = pd.Timestamp(unique_dates[start_pos])
        val_end = pd.Timestamp(unique_dates[end_pos - 1])
        splits.append((val_start, val_end))

    return splits


def select_best_point_params_by_backtest(
    train_features_clean: pd.DataFrame,
    feature_cols: list,
    random_seed: int = 42,
    n_splits: int = 3,
    validation_days: int = 28,
) -> tuple[dict, pd.DataFrame]:
    """Choose LightGBM point-forecast params using 2023-only walk-forward backtesting.

    Parameters
    ----------
    train_features_clean : DataFrame
        Must already be built from 2023 only.
    feature_cols : list
        Forecast feature columns.
    random_seed : int
        Random seed for LightGBM reproducibility.
    n_splits : int
        Number of rolling validation windows.
    validation_days : int
        Length of each validation block in days.

    Returns
    -------
    best_params : dict
        Best override dictionary to pass into _point_params().
    results_df : DataFrame
        Candidate-level backtest summary.
    """
    df = train_features_clean.copy()
    df["date"] = pd.to_datetime(df["date"])

    splits = _time_based_backtest_splits(
        df,
        n_splits=n_splits,
        validation_days=validation_days,
    )

    if not splits:
        fallback = _candidate_point_param_grid()[0]
        results_df = pd.DataFrame(
            [{
                "candidate_id": 0,
                "mean_mae": np.nan,
                "n_successful_splits": 0,
                **fallback,
            }]
        )
        print("  [backtest] insufficient 2023 time span for rolling validation; using default candidate 0")
        return fallback, results_df

    rows = []

    for candidate_id, param_overrides in enumerate(_candidate_point_param_grid()):
        split_maes = []

        for val_start, val_end in splits:
            train_mask = df["date"] < val_start
            valid_mask = (df["date"] >= val_start) & (df["date"] <= val_end)

            train_df = df.loc[train_mask]
            valid_df = df.loc[valid_mask]

            if train_df.empty or valid_df.empty:
                continue

            model = lgb.LGBMRegressor(
                **_point_params(random_seed=random_seed, overrides=param_overrides)
            )
            model.fit(
                train_df[feature_cols].values,
                train_df["consumption"].values,
            )
            preds = np.clip(model.predict(valid_df[feature_cols].values), 0, None)
            split_maes.append(_mae(valid_df["consumption"].values, preds))

        rows.append({
            "candidate_id": candidate_id,
            "mean_mae": float(np.mean(split_maes)) if split_maes else np.nan,
            "n_successful_splits": len(split_maes),
            **param_overrides,
        })

    results_df = pd.DataFrame(rows).sort_values(
        by=["mean_mae", "n_successful_splits", "candidate_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    best_row = results_df.iloc[0]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "learning_rate": float(best_row["learning_rate"]),
        "num_leaves": int(best_row["num_leaves"]),
        "min_child_samples": int(best_row["min_child_samples"]),
    }

    print(
        "  [backtest] selected params:",
        best_params,
        f"| mean 2023 validation MAE = {best_row['mean_mae']:.6f}"
        if pd.notna(best_row["mean_mae"]) else "| mean 2023 validation MAE = nan",
    )

    return best_params, results_df


# ---------------------------------------------------------------------------
# Point models
# ---------------------------------------------------------------------------

def train_cluster_models(
    train_features_clean,
    feature_cols: list,
    cluster_labels: list,
    random_seed: int = 42,
    params_by_cluster: dict | None = None,
) -> dict:
    """Train one MAE-optimised LightGBM model per cluster.

    Parameters
    ----------
    train_features_clean : DataFrame
        2023-only feature rows.
    feature_cols : list
        Forecast feature columns.
    cluster_labels : list
        Cluster labels to train.
    random_seed : int
        Random seed.
    params_by_cluster : dict | None
        Optional mapping: cluster_label -> parameter override dict.

    Returns
    -------
    dict
        cluster_label (int) -> trained LGBMRegressor
    """
    models = {}

    for label in cluster_labels:
        mask = train_features_clean["cluster"] == label
        df_c = train_features_clean[mask]

        overrides = None if params_by_cluster is None else params_by_cluster.get(label)
        model = lgb.LGBMRegressor(
            **_point_params(random_seed=random_seed, overrides=overrides)
        )
        model.fit(df_c[feature_cols].values, df_c["consumption"].values)
        models[label] = model
        print(
            f"  [point] cluster {label}: {len(df_c):,} train rows"
            + (f" | params={overrides}" if overrides is not None else "")
        )

    return models


def train_global_model(
    train_features_clean,
    feature_cols: list,
    random_seed: int = 42,
    params_override: dict | None = None,
) -> lgb.LGBMRegressor:
    """Train a single global MAE-optimised model on all regular households."""
    model = lgb.LGBMRegressor(
        **_point_params(random_seed=random_seed, overrides=params_override)
    )
    model.fit(
        train_features_clean[feature_cols].values,
        train_features_clean["consumption"].values,
    )
    print(
        f"  [point] global model: {len(train_features_clean):,} train rows"
        + (f" | params={params_override}" if params_override is not None else "")
    )
    return model


# ---------------------------------------------------------------------------
# Quantile models
# ---------------------------------------------------------------------------

def train_cluster_quantile_models(
    train_features_clean,
    feature_cols: list,
    cluster_labels: list,
    quantiles: Sequence[float] | None = None,
    random_seed: int = 42,
) -> dict:
    """Train q=0.10, 0.50, 0.90 quantile models per cluster.

    Returns
    -------
    dict: cluster_label -> {quantile -> trained LGBMRegressor}
    """
    if quantiles is None:
        quantiles = QUANTILES

    q_models = {label: {} for label in cluster_labels}

    for label in cluster_labels:
        mask = train_features_clean["cluster"] == label
        df_c = train_features_clean[mask]
        X    = df_c[feature_cols].values
        y    = df_c["consumption"].values
        print(f"  [quantile] cluster {label}: {len(df_c):,} rows")

        for q in quantiles:
            m = lgb.LGBMRegressor(**_quantile_params(q, random_seed))
            m.fit(X, y)
            q_models[label][q] = m
            print(f"    q={q:.2f} ✓")

    return q_models


def train_global_quantile_models(
    train_features_clean,
    feature_cols: list,
    quantiles: Sequence[float] | None = None,
    random_seed: int = 42,
) -> dict:
    """Train global quantile fallback models.

    Returns
    -------
    dict: {quantile -> trained LGBMRegressor}
    """
    if quantiles is None:
        quantiles = QUANTILES

    X = train_features_clean[feature_cols].values
    y = train_features_clean["consumption"].values
    models = {}

    for q in quantiles:
        m = lgb.LGBMRegressor(**_quantile_params(q, random_seed))
        m.fit(X, y)
        models[q] = m
        print(f"  [quantile] global q={q:.2f} ✓")

    return models


# =============================================================================
# Elastic Net Models
# =============================================================================

def _elasticnet_param_grid() -> list[dict]:
    """Parameter grid for ElasticNet hyperparameter tuning.
    
    ElasticNet combines L1 (Lasso) and L2 (Ridge) regularization:
    - alpha: overall regularization strength
    - l1_ratio: balance between L1 and L2 (1.0 = pure Lasso, 0.0 = pure Ridge)
    
    OPTIMIZED: Reduced grid for faster execution and lower memory footprint.
    """
    return [
        {"alpha": 0.1, "l1_ratio": 0.5},    # Medium regularization, equal L1/L2
        {"alpha": 0.1, "l1_ratio": 0.0},    # Pure Ridge (L2)
        {"alpha": 0.1, "l1_ratio": 1.0},    # Pure Lasso (L1)
        {"alpha": 0.1, "l1_ratio": 0.7},    # Lasso-heavy mixture
    ]


def select_best_elasticnet_params_by_backtest(
    train_features_clean: pd.DataFrame,
    feature_cols: list,
    random_seed: int = 42,
    n_splits: int = 3,
    validation_days: int = 28,
) -> tuple[dict, pd.DataFrame]:
    """Choose ElasticNet params using 2023-only walk-forward backtesting.
    
    Parameters
    ----------
    train_features_clean : DataFrame
        Must already be built from 2023 only.
    feature_cols : list
        Forecast feature columns.
    random_seed : int
        Random seed for reproducibility.
    n_splits : int
        Number of rolling validation windows.
    validation_days : int
        Length of each validation block in days.
    
    Returns
    -------
    best_params : dict
        Best alpha and l1_ratio configuration.
    results_df : DataFrame
        Candidate-level backtest summary.
    """
    df = train_features_clean.copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Memory optimization: Sample data for hyperparameter tuning if dataset is large
    MAX_TUNING_ROWS = 2_000_000  # ~2M rows max for tuning
    if len(df) > MAX_TUNING_ROWS:
        sample_frac = MAX_TUNING_ROWS / len(df)
        print(f"  [memory optimization] Sampling {sample_frac:.1%} of data ({MAX_TUNING_ROWS:,} rows) for hyperparameter tuning")
        df = df.sample(n=MAX_TUNING_ROWS, random_state=random_seed).copy()
        df = df.sort_values("date").reset_index(drop=True)
    
    splits = _time_based_backtest_splits(
        df,
        n_splits=n_splits,
        validation_days=validation_days,
    )
    
    if not splits:
        fallback = _elasticnet_param_grid()[0]
        results_df = pd.DataFrame(
            [{
                "candidate_id": 0,
                "mean_mae": np.nan,
                "n_successful_splits": 0,
                **fallback,
            }]
        )
        print("  [elasticnet backtest] insufficient 2023 time span for rolling validation; using default candidate 0")
        return fallback, results_df
    
    rows = []
    
    for candidate_id, param_config in enumerate(_elasticnet_param_grid()):
        split_maes = []
        
        for val_start, val_end in splits:
            train_mask = df["date"] < val_start
            valid_mask = (df["date"] >= val_start) & (df["date"] <= val_end)
            
            train_df = df.loc[train_mask]
            valid_df = df.loc[valid_mask]
            
            if train_df.empty or valid_df.empty:
                continue
            
            # Standardize features (ElasticNet requires scaling)
            # Use float32 to reduce memory usage by 50%
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
            X_valid_scaled = scaler.transform(valid_df[feature_cols].values.astype(np.float32))
            
            model = ElasticNet(
                alpha=param_config["alpha"],
                l1_ratio=param_config["l1_ratio"],
                max_iter=2000,
                tol=1e-4,
                random_state=random_seed,
                fit_intercept=True,
            )
            model.fit(X_train_scaled, train_df["consumption"].values)
            preds = np.clip(model.predict(X_valid_scaled), 0, None)
            split_maes.append(_mae(valid_df["consumption"].values, preds))
        
        rows.append({
            "candidate_id": candidate_id,
            "mean_mae": float(np.mean(split_maes)) if split_maes else np.nan,
            "n_successful_splits": len(split_maes),
            **param_config,
        })
    
    results_df = pd.DataFrame(rows).sort_values(
        "mean_mae", ascending=True
    ).reset_index(drop=True)
    
    best_row = results_df.iloc[0]
    best_params = {
        "alpha": best_row["alpha"],
        "l1_ratio": best_row["l1_ratio"],
    }
    
    print(f"  [elasticnet backtest] best candidate {int(best_row['candidate_id'])}: "
          f"MAE={best_row['mean_mae']:.6f} | alpha={best_params['alpha']:.3f} | "
          f"l1_ratio={best_params['l1_ratio']:.2f}")
    
    return best_params, results_df


def train_cluster_elasticnet_models(
    train_features_clean,
    feature_cols: list,
    cluster_labels: list,
    random_seed: int = 42,
    params_by_cluster: dict | None = None,
) -> tuple[dict, dict]:
    """Train one ElasticNet model per cluster with feature scaling.
    
    ElasticNet requires standardized features for optimal performance.
    Returns both models and their corresponding scalers.
    
    Parameters
    ----------
    train_features_clean : DataFrame
        2023-only feature rows.
    feature_cols : list
        Forecast feature columns.
    cluster_labels : list
        Cluster labels to train.
    random_seed : int
        Random seed.
    params_by_cluster : dict | None
        Optional mapping: cluster_label -> {"alpha": float, "l1_ratio": float}.
    
    Returns
    -------
    models : dict
        cluster_label (int) -> trained ElasticNet model
    scalers : dict
        cluster_label (int) -> fitted StandardScaler
    """
    models = {}
    scalers = {}
    
    for label in cluster_labels:
        mask = train_features_clean["cluster"] == label
        df_c = train_features_clean[mask]
        
        # Feature scaling (critical for ElasticNet)
        # Use float32 to reduce memory usage by 50%
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_c[feature_cols].values.astype(np.float32))
        
        # Get hyperparameters
        if params_by_cluster is not None and label in params_by_cluster:
            alpha = params_by_cluster[label]["alpha"]
            l1_ratio = params_by_cluster[label]["l1_ratio"]
        else:
            alpha = 0.1
            l1_ratio = 0.5
        
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=2000,
            tol=1e-4,
            random_state=random_seed,
            fit_intercept=True,
        )
        model.fit(X_scaled, df_c["consumption"].values)
        
        models[label] = model
        scalers[label] = scaler
        
        print(
            f"  [elasticnet] cluster {label}: {len(df_c):,} train rows | "
            f"alpha={alpha:.3f} | l1_ratio={l1_ratio:.2f}"
        )
    
    return models, scalers


def train_global_elasticnet_model(
    train_features_clean,
    feature_cols: list,
    random_seed: int = 42,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
) -> tuple:
    """Train a single global ElasticNet model with feature scaling.
    
    Returns
    -------
    model : ElasticNet
        Trained model
    scaler : StandardScaler
        Fitted scaler
    """
    # Use float32 to reduce memory usage by 50%
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(train_features_clean[feature_cols].values.astype(np.float32))
    
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=2000,
        tol=1e-4,
        random_state=random_seed,
        fit_intercept=True,
    )
    model.fit(X_scaled, train_features_clean["consumption"].values)
    
    print(
        f"  [elasticnet] global model: {len(train_features_clean):,} train rows | "
        f"alpha={alpha:.3f} | l1_ratio={l1_ratio:.2f}"
    )
    
    return model, scaler


def save_elasticnet_models(
    cluster_models: dict,
    cluster_scalers: dict,
    global_model,
    global_scaler,
    models_dir: Path,
) -> None:
    """Save ElasticNet models and their scalers to disk.
    
    Parameters
    ----------
    cluster_models : dict
        cluster_label -> ElasticNet model
    cluster_scalers : dict
        cluster_label -> StandardScaler
    global_model : ElasticNet
        Global baseline model
    global_scaler : StandardScaler
        Global scaler
    models_dir : Path
        Output directory
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for label, model in cluster_models.items():
        model_path = models_dir / f"elasticnet_cluster_{label}.pkl"
        scaler_path = models_dir / f"scaler_cluster_{label}.pkl"
        
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(cluster_scalers[label], f)
    
    global_model_path = models_dir / "elasticnet_global.pkl"
    global_scaler_path = models_dir / "scaler_global.pkl"
    
    with open(global_model_path, "wb") as f:
        pickle.dump(global_model, f)
    with open(global_scaler_path, "wb") as f:
        pickle.dump(global_scaler, f)
    
    print(f"  Saved {len(cluster_models)} cluster models + 1 global model to {models_dir}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_point_models(cluster_models: dict, global_model, models_dir: Path) -> None:
    """Save cluster point models and global model as pickle files."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for label, model in cluster_models.items():
        p = models_dir / f"lgbm_cluster_{label}.pkl"
        with open(p, "wb") as f:
            pickle.dump(model, f)

    with open(models_dir / "lgbm_global.pkl", "wb") as f:
        pickle.dump(global_model, f)

    print(f"  Models saved to {models_dir}")


def save_quantile_models(
    cluster_q_models: dict,
    global_q_models: dict,
    models_dir: Path,
) -> None:
    """Save quantile models as pickle files."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for label, q_dict in cluster_q_models.items():
        for q, model in q_dict.items():
            q_str = str(int(q * 100)).zfill(2)
            with open(models_dir / f"lgbm_cluster{label}_q{q_str}.pkl", "wb") as f:
                pickle.dump(model, f)

    for q, model in global_q_models.items():
        q_str = str(int(q * 100)).zfill(2)
        with open(models_dir / f"lgbm_global_q{q_str}.pkl", "wb") as f:
            pickle.dump(model, f)

    n = sum(len(v) for v in cluster_q_models.values()) + len(global_q_models)
    print(f"  {n} quantile model files saved to {models_dir}")

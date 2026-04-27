"""ElasticNet-specific prediction utilities with feature scaling support."""

from numbers import Real
import numpy as np
import pandas as pd

from src.forecasting.features import wide_to_long


def predict_elasticnet(
    test_wide: pd.DataFrame,
    train_wide: pd.DataFrame,
    feature_cols: list,
    lag_features: list,
    rolling_windows: list,
    cluster_labels: list,
    cluster_models: dict,
    cluster_scalers: dict,
    global_model,
    global_scaler,
    cluster_series: pd.Series,
    household_means: pd.Series,
    cluster_dow_profile: pd.DataFrame | None = None,
    cluster_month_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate leakage-free 2024 ElasticNet point forecasts with feature scaling.
    
    ElasticNet requires standardized features. This function applies the fitted
    scalers to features before prediction.
    
    Parameters
    ----------
    test_wide : DataFrame
        2024 test data (households × 366 dates).
    train_wide : DataFrame
        2023 training data (households × 365 dates) for lag lookback.
    feature_cols : list
        Feature column names.
    lag_features : list
        Lag day lengths.
    rolling_windows : list
        Rolling window sizes.
    cluster_labels : list
        Cluster labels.
    cluster_models : dict
        cluster_label -> trained ElasticNet model.
    cluster_scalers : dict
        cluster_label -> fitted StandardScaler.
    global_model : ElasticNet
        Global fallback model.
    global_scaler : StandardScaler
        Global scaler.
    cluster_series : Series
        household_id -> cluster_label mapping.
    household_means : Series
        household_id -> mean consumption (from 2023).
    cluster_dow_profile : DataFrame, optional
        Cluster-level day-of-week profiles.
    cluster_month_profile : DataFrame, optional
        Cluster-level month profiles.
    
    Returns
    -------
    DataFrame
        Predictions with columns: household_id, date, predicted, cluster, model_used.
    """
    results = []
    
    cluster_numeric = pd.to_numeric(cluster_series, errors="coerce")
    if cluster_numeric.isna().any():
        raise ValueError("cluster_series contains missing/non-numeric cluster labels.")
    if (cluster_numeric < 0).any():
        raise ValueError(
            "predict_elasticnet() received negative cluster labels, but forecasting "
            "expects reassigned nonnegative handoff labels only."
        )
    
    regular_ids = cluster_numeric.index.intersection(test_wide.index)
    test_dates = pd.to_datetime(test_wide.columns)
    
    hh_cluster = cluster_numeric.reindex(regular_ids).astype(int)
    
    history = {
        hh: train_wide.loc[hh].tolist()
        for hh in regular_ids.intersection(train_wide.index)
    }
    
    def _safe_float_or_nan(value: object) -> float:
        if isinstance(value, Real):
            return float(value)
        return float("nan")
    
    def _lookup_cluster_dow_mean(cluster_value, dow_value):
        if cluster_dow_profile is None or cluster_dow_profile.empty:
            return np.nan
        col = f"cluster_dow_{int(dow_value)}"
        if cluster_value not in cluster_dow_profile.index or col not in cluster_dow_profile.columns:
            return np.nan
        v = cluster_dow_profile.loc[cluster_value, col]
        return _safe_float_or_nan(v)
    
    def _lookup_cluster_month_mean(cluster_value, month_value):
        if cluster_month_profile is None or cluster_month_profile.empty:
            return np.nan
        col = f"cluster_month_{int(month_value)}"
        if cluster_value not in cluster_month_profile.index or col not in cluster_month_profile.columns:
            return np.nan
        v = cluster_month_profile.loc[cluster_value, col]
        return _safe_float_or_nan(v)
    
    for date in test_dates:
        day_rows = []
        
        for hh in regular_ids:
            if hh not in history:
                history[hh] = []
            
            hist = history[hh]
            hh_mean = float(household_means.get(hh, 0.0))
            cluster_value = int(hh_cluster.get(hh, 0))
            
            row = {
                "household_id": hh,
                "date": date,
                "day_of_week": date.dayofweek,
                "month": date.month,
                "day_of_year": date.dayofyear,
                "is_weekend": int(date.dayofweek >= 5),
                "week_of_year": int(date.isocalendar().week),
                "day_of_year_sin": float(np.sin(2 * np.pi * date.dayofyear / 366.0)),
                "day_of_year_cos": float(np.cos(2 * np.pi * date.dayofyear / 366.0)),
                "week_of_year_sin": float(np.sin(2 * np.pi * int(date.isocalendar().week) / 53.0)),
                "week_of_year_cos": float(np.cos(2 * np.pi * int(date.isocalendar().week) / 53.0)),
                "household_mean": hh_mean,
                "cluster": cluster_value,
            }
            
            for lag in lag_features:
                row[f"lag_{lag}"] = hist[-lag] if len(hist) >= lag else 0.0
            
            for window in rolling_windows:
                vals = hist[-window:] if len(hist) >= window else hist
                row[f"rolling_mean_{window}"] = float(np.mean(vals)) if len(vals) > 0 else 0.0
                row[f"rolling_std_{window}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            
            row["cluster_dow_mean"] = _lookup_cluster_dow_mean(cluster_value, date.dayofweek)
            row["cluster_month_mean"] = _lookup_cluster_month_mean(cluster_value, date.month)
            row["household_minus_cluster_dow_mean"] = hh_mean - row["cluster_dow_mean"]
            row["household_minus_cluster_month_mean"] = hh_mean - row["cluster_month_mean"]
            
            day_rows.append(row)
        
        day_df = pd.DataFrame(day_rows)
        
        # Predict with ElasticNet models (requires scaling)
        for label in cluster_labels:
            mask = day_df["cluster"] == label
            if mask.sum() == 0:
                continue
            
            X = day_df.loc[mask, feature_cols].values.astype(np.float32)
            
            # Apply the scaler for this cluster
            scaler = cluster_scalers.get(label, global_scaler)
            X_scaled = scaler.transform(X)
            
            model = cluster_models.get(label, global_model)
            preds = np.clip(model.predict(X_scaled), 0, None)
            
            out = day_df.loc[mask, ["household_id", "date"]].copy()
            out["predicted"] = preds
            out["cluster"] = label
            out["model_used"] = "cluster_elasticnet"
            results.append(out)
            
            for hh, pred in zip(out["household_id"].values, preds):
                history[hh].append(float(pred))
        
        # Handle unmapped households with global fallback
        unmapped = ~day_df["cluster"].isin(cluster_labels)
        if unmapped.sum() > 0:
            X = day_df.loc[unmapped, feature_cols].values.astype(np.float32)
            X_scaled = global_scaler.transform(X)
            preds = np.clip(global_model.predict(X_scaled), 0, None)
            
            out = day_df.loc[unmapped, ["household_id", "date"]].copy()
            out["predicted"] = preds
            out["cluster"] = -99
            out["model_used"] = "global_elasticnet_fallback"
            results.append(out)
            
            for hh, pred in zip(out["household_id"].values, preds):
                history[hh].append(float(pred))
    
    all_preds = pd.concat(results, ignore_index=True) if results else pd.DataFrame(
        columns=["household_id", "date", "predicted", "cluster", "model_used"]
    )
    all_preds = all_preds.sort_values(["household_id", "date"]).reset_index(drop=True)
    return all_preds

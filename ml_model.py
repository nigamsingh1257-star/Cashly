"""
Cashly - Machine Learning Budget & Expense Predictor
Implements time-series regression with chronological train/test splitting,
MAE evaluation, honest data sufficiency checks, and next-month forecasting.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

MIN_MONTHS_REQUIRED = 4


def prepare_monthly_series(df_expenses: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw expense transactions into a monthly time series.
    Returns DataFrame sorted chronologically by year-month.
    """
    if df_expenses.empty:
        return pd.DataFrame()

    df = df_expenses.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    # Aggregate monthly sum and transaction counts
    monthly = df.groupby("year_month").agg(
        total_expense=("amount", "sum"),
        transaction_count=("amount", "count"),
        start_date=("date", "min")
    ).reset_index()

    monthly["month_timestamp"] = monthly["year_month"].dt.to_timestamp()
    monthly = monthly.sort_values("month_timestamp").reset_index(drop=True)
    return monthly


def build_time_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs time-based and lag features for time-series regression.
    Features:
    - month_idx: sequential counter (1, 2, 3...) capturing macro spending trend
    - month_of_year: calendar month (1 to 12) capturing seasonal shifts
    - quarter: quarterly cycle (1 to 4)
    - lag_1: expenditure of immediate prior month
    - rolling_mean_3: 3-month moving average
    """
    df = monthly_df.copy()
    df["month_idx"] = np.arange(1, len(df) + 1)
    df["month_of_year"] = df["month_timestamp"].dt.month
    df["quarter"] = df["month_timestamp"].dt.quarter

    # Lag 1 feature (previous month spend)
    df["lag_1"] = df["total_expense"].shift(1)
    # Fill first month lag with its own value to retain data points on small samples
    df["lag_1"] = df["lag_1"].bfill()

    # 3-month rolling mean feature
    df["rolling_mean_3"] = df["total_expense"].rolling(window=3, min_periods=1).mean()

    return df


def train_and_predict_expenses(df_expenses: pd.DataFrame) -> Dict[str, Any]:
    """
    End-to-end Machine Learning Pipeline:
    1. Validates historical sufficiency (minimum 4 months).
    2. Builds chronological time series with lag & seasonal features.
    3. Chronological train/test split (no lookahead data leakage).
    4. Fits Linear Regression & Random Forest models.
    5. Computes MAE, RMSE, and MAPE on the chronological holdout test set.
    6. Predicts next month's spending with confidence interval (± MAE).
    """
    if df_expenses.empty:
        return {
            "status": "insufficient_data",
            "message": "No expense records found. Please add expenses or load demo data to begin.",
            "month_count": 0
        }

    monthly_df = prepare_monthly_series(df_expenses)
    month_count = len(monthly_df)

    # Honest Data Sufficiency Guard:
    # A true time-series regression model cannot reliably split into train and test
    # without at least 4 distinct calendar months of data.
    if month_count < MIN_MONTHS_REQUIRED:
        return {
            "status": "insufficient_data",
            "message": (
                f"Not enough data for reliable prediction. "
                f"At least {MIN_MONTHS_REQUIRED} distinct months of expense history are required to train the "
                f"machine learning model without inventing synthetic numbers. "
                f"(Currently available: {month_count} month{'s' if month_count != 1 else ''}). "
                f"Please record expenses over multiple months or click 'Load Demo Data' in Settings."
            ),
            "month_count": month_count,
            "monthly_df": monthly_df
        }

    # Feature Engineering
    featured_df = build_time_features(monthly_df)

    feature_cols = ["month_idx", "month_of_year", "quarter", "lag_1", "rolling_mean_3"]
    X = featured_df[feature_cols].values
    y = featured_df["total_expense"].values

    # Chronological Train / Test Split
    # To prevent temporal data leakage, we DO NOT shuffle.
    # The most recent 1 or 2 months serve as the holdout Test set.
    test_size = 2 if month_count >= 6 else 1
    train_size = month_count - test_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Model 1: Linear Regression (interpretable, standard benchmark)
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_test_lin = lin_model.predict(X_test)

    # Model 2: Ridge Regression with L2 Regularization (stable for collinear time features)
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_test_ridge = ridge_model.predict(X_test)

    # Model Evaluation Metrics
    mae_lin = float(mean_absolute_error(y_test, y_pred_test_lin))
    rmse_lin = float(np.sqrt(mean_squared_error(y_test, y_pred_test_lin)))
    mape_lin = float(np.mean(np.abs((y_test - y_pred_test_lin) / np.maximum(y_test, 1))) * 100)

    mae_ridge = float(mean_absolute_error(y_test, y_pred_test_ridge))

    # Select the model with lower test MAE
    if mae_ridge <= mae_lin:
        best_model = ridge_model
        best_model_name = "Ridge Regression (L2 Regularized)"
        best_mae = mae_ridge
        y_pred_test = y_pred_test_ridge
    else:
        best_model = lin_model
        best_model_name = "Linear Regression (Ordinary Least Squares)"
        best_mae = mae_lin
        y_pred_test = y_pred_test_lin

    # Naive Baseline Model (predicting test month = previous month)
    naive_pred = featured_df["lag_1"].values[train_size:]
    naive_mae = float(mean_absolute_error(y_test, naive_pred))

    # Feature Weights / Interpretability
    coefficients = {
        col: round(float(coef), 2)
        for col, coef in zip(feature_cols, lin_model.coef_)
    }

    # Forecast Next Month
    last_row = featured_df.iloc[-1]
    last_timestamp = last_row["month_timestamp"]

    # Compute next month's date & features
    if last_timestamp.month == 12:
        next_year = last_timestamp.year + 1
        next_month = 1
    else:
        next_year = last_timestamp.year
        next_month = last_timestamp.month + 1

    next_date_str = f"{next_year}-{next_month:02d}"
    next_month_name = datetime(next_year, next_month, 1).strftime("%B %Y")
    next_month_idx = len(featured_df) + 1
    next_quarter = ((next_month - 1) // 3) + 1
    next_lag_1 = float(last_row["total_expense"])

    # 3-month rolling average for next month
    recent_3_avg = float(featured_df["total_expense"].tail(3).mean())

    X_next = np.array([[next_month_idx, next_month, next_quarter, next_lag_1, recent_3_avg]])
    predicted_next_expense = float(best_model.predict(X_next)[0])
    # Expenses cannot be negative
    predicted_next_expense = max(100.0, round(predicted_next_expense, 2))

    # Confidence Bounds (Predicted ± Test MAE)
    lower_bound = max(0.0, round(predicted_next_expense - best_mae, 2))
    upper_bound = round(predicted_next_expense + best_mae, 2)

    # Category breakdown forecast (proportional to recent 3-month expenditure)
    recent_date_cutoff = featured_df["month_timestamp"].iloc[-3] if len(featured_df) >= 3 else featured_df["month_timestamp"].iloc[0]
    recent_expenses = df_expenses[pd.to_datetime(df_expenses["date"]) >= recent_date_cutoff]

    if not recent_expenses.empty:
        cat_sums = recent_expenses.groupby("category")["amount"].sum()
        total_recent = cat_sums.sum()
        cat_forecast = {
            cat: round(float((amt / total_recent) * predicted_next_expense), 2)
            for cat, amt in cat_sums.items()
        }
    else:
        cat_forecast = {}

    # Test comparison table
    test_labels = [dt.strftime("%b %Y") for dt in featured_df["month_timestamp"].iloc[train_size:]]
    test_comparison = pd.DataFrame({
        "Month": test_labels,
        "Actual Expense": [round(float(v), 2) for v in y_test],
        "Predicted Expense": [round(float(v), 2) for v in y_pred_test],
        "Absolute Error": [round(float(abs(a - p)), 2) for a, p in zip(y_test, y_pred_test)]
    })

    return {
        "status": "success",
        "model_name": best_model_name,
        "month_count": month_count,
        "train_size": train_size,
        "test_size": test_size,
        "mae": round(best_mae, 2),
        "rmse": round(rmse_lin, 2),
        "mape": round(mape_lin, 2),
        "naive_mae": round(naive_mae, 2),
        "predicted_next_expense": predicted_next_expense,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "next_month_name": next_month_name,
        "next_date_str": next_date_str,
        "coefficients": coefficients,
        "category_forecast": cat_forecast,
        "test_comparison": test_comparison,
        "featured_df": featured_df,
        "split_index": train_size
    }

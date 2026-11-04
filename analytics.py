"""
Cashly - Data Analytics & Plotly Visualizations
Provides interactive charts styled with modern Dark Blue & Lavender themes.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Color Palette: Dark Blue, Indigo, and Lavender Harmony
COLOR_PALETTE = [
    "#4F46E5",  # Indigo
    "#818CF8",  # Soft Lavender
    "#6366F1",  # Vibrant Iris
    "#A5B4FC",  # Pale Lavender
    "#312E81",  # Deep Midnight Blue
    "#C084FC",  # Lilac / Orchid
    "#38BDF8",  # Sky Blue
    "#F43F5E",  # Rose Accent
]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, -apple-system, sans-serif", color="#1E293B"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)


def create_category_donut_chart(df: pd.DataFrame, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders an interactive Donut Chart for category-wise expense distribution.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No expense data available", showarrow=False, font=dict(size=14, color="#64748B"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    cat_summary = df.groupby("category")["amount"].sum().reset_index()
    cat_summary = cat_summary.sort_values(by="amount", ascending=False)

    fig = go.Figure(data=[
        go.Pie(
            labels=cat_summary["category"],
            values=cat_summary["amount"],
            hole=0.55,
            marker=dict(colors=COLOR_PALETTE, line=dict(color="#FFFFFF", width=2)),
            textinfo="percent+label",
            textposition="inside",
            hovertemplate="<b>%{label}</b><br>Total: " + currency_symbol + "%{value:,.2f}<br>Share: %{percent}<extra></extra>"
        )
    ])

    total_spend = cat_summary["amount"].sum()
    fig.add_annotation(
        text=f"Total<br><b>{currency_symbol}{total_spend:,.0f}</b>",
        x=0.5, y=0.5,
        font=dict(size=14, color="#1E1B4B", family="sans-serif"),
        showarrow=False
    )

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        showlegend=True,
        height=360
    )
    return fig


def create_monthly_trend_chart(df: pd.DataFrame, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders a smooth line chart with gradient area for monthly expense trends.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No expense data available", showarrow=False, font=dict(size=14, color="#64748B"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy["month_str"] = df_copy["date"].dt.strftime("%Y-%m")
    df_copy["month_label"] = df_copy["date"].dt.strftime("%b %Y")

    monthly = df_copy.groupby(["month_str", "month_label"])["amount"].sum().reset_index()
    monthly = monthly.sort_values("month_str")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["month_label"],
        y=monthly["amount"],
        mode="lines+markers",
        line=dict(color="#4F46E5", width=3, shape="spline"),
        marker=dict(size=8, color="#1E1B4B", line=dict(color="#FFFFFF", width=2)),
        fill="tozeroy",
        fillcolor="rgba(99, 102, 241, 0.12)",
        name="Monthly Spending",
        hovertemplate="<b>%{x}</b><br>Expenses: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis=dict(title="", showgrid=False, tickfont=dict(size=11, color="#64748B")),
        yaxis=dict(title=f"Total Amount ({currency_symbol})", showgrid=True, gridcolor="#F1F5F9", tickfont=dict(size=11, color="#64748B")),
        height=340
    )
    return fig


def create_income_vs_expense_chart(df: pd.DataFrame, monthly_income: float, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders a grouped comparison bar chart for Monthly Income vs Total Expenses.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No expense data available", showarrow=False, font=dict(size=14, color="#64748B"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy["month_str"] = df_copy["date"].dt.strftime("%Y-%m")
    df_copy["month_label"] = df_copy["date"].dt.strftime("%b %Y")

    monthly = df_copy.groupby(["month_str", "month_label"])["amount"].sum().reset_index()
    monthly = monthly.sort_values("month_str")

    fig = go.Figure()

    # Bar 1: Income
    fig.add_trace(go.Bar(
        x=monthly["month_label"],
        y=[monthly_income] * len(monthly),
        name="Income",
        marker_color="#10B981",  # Emerald Green
        hovertemplate="<b>%{x}</b><br>Income: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    # Bar 2: Expenses
    fig.add_trace(go.Bar(
        x=monthly["month_label"],
        y=monthly["amount"],
        name="Expenses",
        marker_color="#6366F1",  # Indigo/Lavender
        hovertemplate="<b>%{x}</b><br>Expenses: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        barmode="group",
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title=f"Amount ({currency_symbol})", showgrid=True, gridcolor="#F1F5F9"),
        height=340
    )
    return fig


def create_top_categories_bar_chart(df: pd.DataFrame, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders a horizontal ranking bar chart of top spending categories.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No expense data available", showarrow=False, font=dict(size=14, color="#64748B"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    cat_sums = df.groupby("category")["amount"].sum().reset_index()
    cat_sums = cat_sums.sort_values("amount", ascending=True)

    fig = go.Figure(go.Bar(
        x=cat_sums["amount"],
        y=cat_sums["category"],
        orientation="h",
        marker=dict(
            color=cat_sums["amount"],
            colorscale=[[0, "#A5B4FC"], [0.5, "#6366F1"], [1.0, "#312E81"]],
            showscale=False
        ),
        text=cat_sums["amount"].apply(lambda x: f"{currency_symbol}{x:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b>: " + currency_symbol + "%{x:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis=dict(title=f"Spending ({currency_symbol})", showgrid=True, gridcolor="#F1F5F9"),
        yaxis=dict(title="", showgrid=False),
        height=330
    )
    return fig


def create_daily_spending_chart(df: pd.DataFrame, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders daily spending timeline with category color tagging.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No expense data available", showarrow=False, font=dict(size=14, color="#64748B"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    df_copy = df.copy()
    df_copy["date_str"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m-%d")
    daily = df_copy.groupby("date_str")["amount"].sum().reset_index()
    daily = daily.sort_values("date_str").tail(45)  # Last 45 days for readability

    # Rolling 7-day average
    daily["rolling_7"] = daily["amount"].rolling(window=7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["date_str"],
        y=daily["amount"],
        name="Daily Total",
        marker_color="#818CF8",
        opacity=0.75,
        hovertemplate="<b>%{x}</b><br>Daily Spend: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=daily["date_str"],
        y=daily["rolling_7"],
        mode="lines",
        name="7-Day Moving Avg",
        line=dict(color="#1E1B4B", width=2.5),
        hovertemplate="<b>7-Day Avg</b>: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis=dict(title="", showgrid=False, tickangle=-45),
        yaxis=dict(title=f"Amount ({currency_symbol})", showgrid=True, gridcolor="#F1F5F9"),
        height=320
    )
    return fig


def create_day_of_week_chart(df: pd.DataFrame, currency_symbol: str = "₹") -> go.Figure:
    """
    Analyzes which days of the week experience the heaviest spending.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No expense data available", showarrow=False, font=dict(size=14, color="#64748B"))
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy["day_name"] = df_copy["date"].dt.day_name()
    df_copy["day_num"] = df_copy["date"].dt.dayofweek

    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow = df_copy.groupby(["day_num", "day_name"]).agg(
        total_spend=("amount", "sum"),
        avg_spend=("amount", "mean"),
        count=("amount", "count")
    ).reset_index().sort_values("day_num")

    colors = ["#4F46E5" if d in ["Saturday", "Sunday"] else "#A5B4FC" for d in dow["day_name"]]

    fig = go.Figure(go.Bar(
        x=dow["day_name"],
        y=dow["total_spend"],
        marker_color=colors,
        text=dow["total_spend"].apply(lambda x: f"{currency_symbol}{x:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Total Spend: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        xaxis=dict(title="", showgrid=False),
        yaxis=dict(title=f"Total Spent ({currency_symbol})", showgrid=True, gridcolor="#F1F5F9"),
        height=310
    )
    return fig


def create_ml_forecast_chart(ml_results: dict, currency_symbol: str = "₹") -> go.Figure:
    """
    Renders chronological Train/Test split visualization and Next-Month ML forecast with confidence band.
    """
    fig = go.Figure()
    if ml_results.get("status") != "success":
        fig.add_annotation(
            text="Insufficient data to render ML forecast visualization.",
            showarrow=False,
            font=dict(size=14, color="#64748B")
        )
        fig.update_layout(**LAYOUT_DEFAULTS)
        return fig

    featured_df = ml_results["featured_df"]
    split_idx = ml_results["split_index"]

    # Historical timeline dates
    dates = [dt.strftime("%b %Y") for dt in featured_df["month_timestamp"]]
    train_dates = dates[:split_idx]
    train_actuals = featured_df["total_expense"].iloc[:split_idx].tolist()

    test_dates = dates[split_idx:]
    test_actuals = featured_df["total_expense"].iloc[split_idx:].tolist()
    test_preds = ml_results["test_comparison"]["Predicted Expense"].tolist()

    next_date = ml_results["next_month_name"]
    next_pred = ml_results["predicted_next_expense"]
    lower_bound = ml_results["lower_bound"]
    upper_bound = ml_results["upper_bound"]

    # 1. Historical Training Data
    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_actuals,
        mode="lines+markers",
        name="Training Data (Actual)",
        line=dict(color="#312E81", width=3),
        marker=dict(size=7, color="#312E81"),
        hovertemplate="<b>Training Actual</b>: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    # 2. Holdout Test Actual
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_actuals,
        mode="lines+markers",
        name="Test Data (Actual)",
        line=dict(color="#10B981", width=3),
        marker=dict(size=8, color="#10B981"),
        hovertemplate="<b>Test Actual</b>: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    # 3. Holdout Test Predictions
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_preds,
        mode="lines+markers",
        name="Test Prediction (Evaluated)",
        line=dict(color="#F59E0B", width=2.5, dash="dash"),
        marker=dict(size=8, color="#F59E0B", symbol="diamond"),
        hovertemplate="<b>Model Prediction</b>: " + currency_symbol + "%{y:,.2f}<extra></extra>"
    ))

    # 4. Next Month Predicted Point
    all_dates = dates + [next_date]
    pred_line_x = [dates[-1], next_date]
    pred_line_y = [featured_df["total_expense"].iloc[-1], next_pred]

    fig.add_trace(go.Scatter(
        x=pred_line_x,
        y=pred_line_y,
        mode="lines",
        line=dict(color="#6366F1", width=2.5, dash="dot"),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=[next_date],
        y=[next_pred],
        mode="markers+text",
        name="Next Month ML Forecast",
        marker=dict(size=14, color="#6366F1", symbol="star"),
        text=[f"Predicted: {currency_symbol}{next_pred:,.0f}"],
        textposition="top center",
        hovertemplate="<b>Next Month Forecast</b>: " + currency_symbol + "%{y:,.2f}<br>Range: " + currency_symbol + f"{lower_bound:,.0f} - {currency_symbol}{upper_bound:,.0f}<extra></extra>"
    ))

    # Error bar annotation
    fig.add_trace(go.Scatter(
        x=[next_date, next_date],
        y=[lower_bound, upper_bound],
        mode="lines",
        name="Confidence Interval (± MAE)",
        line=dict(color="#A5B4FC", width=6),
        opacity=0.6,
        hoverinfo="skip"
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="<b>Chronological Train/Test Split & ML Forecast</b>", font=dict(size=14, color="#1E1B4B")),
        xaxis=dict(title="Timeline", showgrid=False),
        yaxis=dict(title=f"Monthly Spending ({currency_symbol})", showgrid=True, gridcolor="#F1F5F9"),
        height=380
    )
    return fig

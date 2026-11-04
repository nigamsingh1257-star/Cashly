"""
Cashly - Automated Verification & Test Suite
Tests SQLite persistence, demo data generation, ML regression pipeline,
Plotly analytics rendering, and budget insights.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure spendwise_ai is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Safe encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from database import (
    init_db,
    add_expense,
    update_expense,
    delete_expense,
    get_expenses_df,
    get_expense_by_id,
    count_expenses,
    clear_all_expenses,
    get_setting,
    set_setting,
    get_category_budgets,
    set_category_budget,
    CATEGORIES,
    DEFAULT_DB_PATH
)
from sample_data import load_demo_dataset, generate_sample_expenses
from ml_model import train_and_predict_expenses
from analytics import (
    create_category_donut_chart,
    create_monthly_trend_chart,
    create_income_vs_expense_chart,
    create_top_categories_bar_chart,
    create_daily_spending_chart,
    create_day_of_week_chart,
    create_ml_forecast_chart
)
from insights import (
    compute_budget_health,
    analyze_category_breakdown,
    calculate_50_30_20_split,
    generate_smart_suggestions
)


def run_tests():
    print("=========================================")
    print("🧪 Running Cashly Test Suite")
    print("=========================================")

    # Test 1: Database Initialization
    print("\n[Test 1] Testing Database Initialization...")
    init_db()
    assert count_expenses() >= 0, "Database init failed!"
    print("  -> ✅ Database tables initialized successfully.")

    # Test 2: Clear Database & Check Empty State
    print("\n[Test 2] Testing Empty State & Clear DB...")
    clear_all_expenses()
    assert count_expenses() == 0, "Expenses should be 0 after clear!"
    df_empty = get_expenses_df()
    assert df_empty.empty, "DataFrame should be empty!"
    print("  -> ✅ Database cleared and verified empty.")

    # Test 3: ML Guard on Insufficient Data (< 4 months)
    print("\n[Test 3] Testing ML Insufficient Data Guard...")
    ml_res_empty = train_and_predict_expenses(df_empty)
    assert ml_res_empty["status"] == "insufficient_data", "ML should refuse when empty!"
    
    # Add only 1 record
    add_expense("2026-11-01", 500.0, "Food", "Groceries at market")
    df_single = get_expenses_df()
    ml_res_single = train_and_predict_expenses(df_single)
    assert ml_res_single["status"] == "insufficient_data", "ML should refuse with only 1 month!"
    print(f"  -> ✅ ML correctly refused prediction with honest message: '{ml_res_single['message'][:50]}...'")

    # Test 4: CRUD Operations
    print("\n[Test 4] Testing CRUD Operations...")
    exp_id = add_expense("2026-11-02", 1250.0, "Shopping", "Winter jacket", "Credit Card")
    assert exp_id > 0, "Failed to insert expense!"
    rec = get_expense_by_id(exp_id)
    assert rec["amount"] == 1250.0, "Inserted amount mismatch!"
    assert rec["category"] == "Shopping", "Category mismatch!"

    # Update
    updated = update_expense(exp_id, "2026-11-02", 1150.0, "Shopping", "Winter jacket (discounted)", "Credit Card")
    assert updated, "Failed to update expense!"
    rec_up = get_expense_by_id(exp_id)
    assert rec_up["amount"] == 1150.0, "Updated amount mismatch!"

    # Delete
    deleted = delete_expense(exp_id)
    assert deleted, "Failed to delete expense!"
    assert get_expense_by_id(exp_id) is None, "Expense should no longer exist!"
    print("  -> ✅ Insert, Retrieve, Update, and Delete operations fully verified.")

    # Test 5: Loading Realistic Demo Dataset
    print("\n[Test 5] Testing Realistic Demo Dataset Generation...")
    count = load_demo_dataset()
    print(f"  -> Loaded {count} demo records.")
    assert count > 50, f"Expected > 50 demo records, got {count}"
    df_demo = get_expenses_df()
    assert not df_demo.empty, "Demo DataFrame should not be empty!"
    months_covered = df_demo["date"].dt.to_period("M").nunique()
    print(f"  -> Spanning {months_covered} distinct chronological months.")
    assert months_covered >= 6, "Demo data should cover at least 6 months for ML!"
    print("  -> ✅ Demo data generation passed.")

    # Test 6: Machine Learning Pipeline
    print("\n[Test 6] Testing ML Time-Series Pipeline with Demo Data...")
    ml_res = train_and_predict_expenses(df_demo)
    assert ml_res["status"] == "success", f"ML failed with message: {ml_res.get('message')}"
    print(f"  -> Model Chosen: {ml_res['model_name']}")
    print(f"  -> Chronological Split: Train={ml_res['train_size']} months, Test={ml_res['test_size']} months")
    print(f"  -> Holdout Test MAE: ₹{ml_res['mae']:,.2f}")
    print(f"  -> Naive Baseline MAE: ₹{ml_res['naive_mae']:,.2f}")
    print(f"  -> Forecast for {ml_res['next_month_name']}: ₹{ml_res['predicted_next_expense']:,.2f} (Range: ₹{ml_res['lower_bound']:,.2f} – ₹{ml_res['upper_bound']:,.2f})")
    assert ml_res["predicted_next_expense"] > 0, "Predicted expense must be positive!"
    assert ml_res["mae"] >= 0, "MAE must be non-negative!"
    print("  -> ✅ ML pipeline, chronological split, and evaluation verified.")

    # Test 7: Plotly Visualizations Rendering
    print("\n[Test 7] Testing Plotly Visualizations...")
    fig1 = create_category_donut_chart(df_demo)
    fig2 = create_monthly_trend_chart(df_demo)
    fig3 = create_income_vs_expense_chart(df_demo, monthly_income=45000)
    fig4 = create_top_categories_bar_chart(df_demo)
    fig5 = create_daily_spending_chart(df_demo)
    fig6 = create_day_of_week_chart(df_demo)
    fig7 = create_ml_forecast_chart(ml_res)
    assert all([fig1, fig2, fig3, fig4, fig5, fig6, fig7]), "All figures must create successfully!"
    print("  -> ✅ All 7 Plotly charts rendered without exceptions.")

    # Test 8: Budget Health & Smart Insights
    print("\n[Test 8] Testing Budget Health & Insights...")
    health = compute_budget_health(current_month_spent=32000, monthly_budget=35000)
    assert health["utilization_pct"] > 0, "Utilization should be computed!"
    assert not health["is_exceeded"], "32k of 35k should not be exceeded!"

    health_exceeded = compute_budget_health(current_month_spent=42000, monthly_budget=35000)
    assert health_exceeded["is_exceeded"], "42k of 35k must be marked as exceeded!"

    split = calculate_50_30_20_split(45000, df_demo)
    assert "savings_pct" in split, "Savings % must be calculated!"

    suggestions = generate_smart_suggestions(
        df_all=df_demo,
        df_current=df_demo.head(30),
        monthly_income=45000,
        monthly_budget=35000,
        category_budgets=get_category_budgets(),
        ml_forecast=ml_res
    )
    assert len(suggestions) > 0, "Should generate at least one suggestion!"
    types_found = {s["type"] for s in suggestions}
    print(f"  -> Generated {len(suggestions)} suggestions across types: {types_found}")
    assert "Rule-Based" in types_found, "Must include Rule-Based suggestions!"
    assert "Machine Learning" in types_found, "Must include Machine Learning suggestions!"
    print("  -> ✅ Rule-Based and ML suggestions properly distinguished.")

    print("\n=========================================")
    print("🎉 ALL TESTS PASSED SUCCESSFULLY (8/8)!")
    print("=========================================")


if __name__ == "__main__":
    run_tests()

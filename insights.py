"""
Cashly - Smart Budget Insights & Recommendation Engine
Calculates budget utilization, detects spending anomalies, and generates
practical savings suggestions while clearly distinguishing Rule-Based insights from ML forecasts.
"""

from typing import Dict, Any, List
import pandas as pd
from datetime import datetime


def compute_budget_health(current_month_spent: float, monthly_budget: float) -> Dict[str, Any]:
    """
    Computes overall budget utilization percentage and risk status.
    """
    if monthly_budget <= 0:
        return {
            "utilization_pct": 0.0,
            "status": "No Budget Set",
            "color": "#64748B",
            "message": "Set a monthly budget to unlock spending pace tracking.",
            "is_exceeded": False
        }

    utilization = (current_month_spent / monthly_budget) * 100

    if utilization >= 100:
        status = "Budget Exceeded"
        color = "#EF4444"  # Red
        message = f"🚨 Alert: You have exceeded your monthly budget by {utilization - 100:.1f}%."
        is_exceeded = True
    elif utilization >= 85:
        status = "Approaching Limit"
        color = "#F59E0B"  # Amber
        message = f"⚠️ Warning: You have consumed {utilization:.1f}% of your budget with time remaining."
        is_exceeded = False
    elif utilization >= 70:
        status = "Moderate Spending"
        color = "#6366F1"  # Indigo
        message = f"ℹ️ Notice: Spending pace is moderate ({utilization:.1f}% used)."
        is_exceeded = False
    else:
        status = "Healthy / On Track"
        color = "#10B981"  # Emerald
        message = f"✅ Excellent: Spending is well within limits ({utilization:.1f}% consumed)."
        is_exceeded = False

    return {
        "utilization_pct": round(utilization, 1),
        "status": status,
        "color": color,
        "message": message,
        "is_exceeded": is_exceeded,
        "remaining_budget": max(0.0, monthly_budget - current_month_spent),
        "overspent_amount": max(0.0, current_month_spent - monthly_budget)
    }


def analyze_category_breakdown(df_current_month: pd.DataFrame, category_budgets: dict) -> Dict[str, Any]:
    """
    Identifies top spending categories and checks category budget limits.
    """
    if df_current_month.empty:
        return {
            "top_category": None,
            "top_amount": 0.0,
            "top_share": 0.0,
            "over_budget_categories": [],
            "approaching_categories": []
        }

    cat_totals = df_current_month.groupby("category")["amount"].sum()
    total_spent = cat_totals.sum()

    top_cat = cat_totals.idxmax()
    top_amt = float(cat_totals.max())
    top_share = round((top_amt / total_spent) * 100, 1) if total_spent > 0 else 0.0

    over_budget_cats = []
    approaching_cats = []

    for cat, spent in cat_totals.items():
        bgt = category_budgets.get(cat, 0.0)
        if bgt > 0:
            cat_util = (spent / bgt) * 100
            if cat_util >= 100:
                over_budget_cats.append({
                    "category": cat,
                    "spent": round(float(spent), 2),
                    "budget": bgt,
                    "utilization": round(cat_util, 1),
                    "diff": round(float(spent - bgt), 2)
                })
            elif cat_util >= 85:
                approaching_cats.append({
                    "category": cat,
                    "spent": round(float(spent), 2),
                    "budget": bgt,
                    "utilization": round(cat_util, 1),
                    "remaining": round(float(bgt - spent), 2)
                })

    return {
        "top_category": top_cat,
        "top_amount": top_amt,
        "top_share": top_share,
        "total_spent": total_spent,
        "over_budget_categories": over_budget_cats,
        "approaching_categories": approaching_cats
    }


def calculate_50_30_20_split(monthly_income: float, current_expenses: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculates adherence to the classic 50/30/20 financial rule:
    - 50% Needs: Bills, Health, Education, Basic Groceries
    - 30% Wants: Shopping, Entertainment, Dining/Travel
    - 20% Savings: Remaining Income
    """
    if monthly_income <= 0 or current_expenses.empty:
        return {}

    needs_cats = ["Bills", "Health", "Education", "Food"]
    wants_cats = ["Shopping", "Entertainment", "Travel", "Others"]

    cat_sums = current_expenses.groupby("category")["amount"].sum()

    needs_spent = float(cat_sums[cat_sums.index.isin(needs_cats)].sum())
    wants_spent = float(cat_sums[cat_sums.index.isin(wants_cats)].sum())
    total_spent = needs_spent + wants_spent

    savings = max(0.0, monthly_income - total_spent)
    savings_pct = round((savings / monthly_income) * 100, 1)
    needs_pct = round((needs_spent / monthly_income) * 100, 1)
    wants_pct = round((wants_spent / monthly_income) * 100, 1)

    return {
        "needs_spent": round(needs_spent, 2),
        "needs_pct": needs_pct,
        "needs_target_pct": 50.0,
        "wants_spent": round(wants_spent, 2),
        "wants_pct": wants_pct,
        "wants_target_pct": 30.0,
        "savings": round(savings, 2),
        "savings_pct": savings_pct,
        "savings_target_pct": 20.0
    }


def generate_smart_suggestions(
    df_all: pd.DataFrame,
    df_current: pd.DataFrame,
    monthly_income: float,
    monthly_budget: float,
    category_budgets: dict,
    ml_forecast: dict = None,
    currency_symbol: str = "₹"
) -> List[Dict[str, str]]:
    """
    Generates actionable financial advice and explicitly tags each suggestion
    as either [Rule-Based Heuristic] or [Machine Learning Predictive].
    """
    suggestions = []

    if df_current.empty and df_all.empty:
        suggestions.append({
            "type": "Rule-Based",
            "tag": "🏷️ Rule-Based Financial Standard",
            "badge_color": "secondary",
            "title": "Start By Recording Transactions",
            "content": "Add your first set of real expenses or load Demo Data from Settings to unlock personalized saving recommendations."
        })
        return suggestions

    cat_analysis = analyze_category_breakdown(df_current, category_budgets)
    current_spent = cat_analysis.get("total_spent", 0.0)

    # 1. Budget Utilization Suggestion (Rule-Based)
    if monthly_budget > 0:
        util = (current_spent / monthly_budget) * 100
        if util > 100:
            exceeded = current_spent - monthly_budget
            suggestions.append({
                "type": "Rule-Based",
                "tag": "🏷️ Rule-Based Financial Alert",
                "badge_color": "danger",
                "title": "Immediate Budget Reset Required",
                "content": f"You are currently {currency_symbol}{exceeded:,.2f} over your total monthly budget. Pause all non-essential discretionary spending (Shopping, Entertainment) for the remainder of this cycle."
            })
        elif util >= 85:
            remaining = monthly_budget - current_spent
            suggestions.append({
                "type": "Rule-Based",
                "tag": "🏷️ Rule-Based Financial Alert",
                "badge_color": "warning",
                "title": "Approaching Budget Threshold",
                "content": f"You have consumed {util:.1f}% of your monthly limit with {currency_symbol}{remaining:,.2f} remaining. Limit daily discretionary spending to maintain your savings goals."
            })
        else:
            remaining = monthly_budget - current_spent
            suggestions.append({
                "type": "Rule-Based",
                "tag": "🏷️ Rule-Based Financial Benchmark",
                "badge_color": "success",
                "title": "Budget Pacing is Healthy",
                "content": f"You have utilized {util:.1f}% of your planned budget with {currency_symbol}{remaining:,.2f} available in reserve. Maintain this current rate to achieve a positive monthly surplus."
            })

    # 2. Highest Spending Category Insight (Rule-Based)
    if cat_analysis.get("top_category"):
        top_cat = cat_analysis["top_category"]
        top_share = cat_analysis["top_share"]
        top_amt = cat_analysis["top_amount"]

        tips = {
            "Food": "Consider weekly meal planning and cooking at home 2 more days per week; this typically reduces dining outlays by ~15-20%.",
            "Shopping": "Implement the 48-hour delay rule before checkout to curb non-essential impulse purchases.",
            "Travel": "Review commute costs; carpooling or purchasing monthly public transit passes can reduce travel outlays.",
            "Entertainment": "Audit recurring subscriptions and cancel streaming platforms you haven't actively used this month.",
            "Bills": "Check for off-peak electricity rates or renegotiate broadband/postpaid service plans.",
            "Education": "Explore student discounts and open-source learning repositories to optimize educational material costs.",
            "Health": "Look into generic pharmaceutical equivalents and annual preventive checkup insurance benefits.",
            "Others": "Review individual line items to spot recurring miscellaneous leakages."
        }
        tip_detail = tips.get(top_cat, "Review individual line items to spot recurring small expenses.")
        suggestions.append({
            "type": "Rule-Based",
            "tag": "🏷️ Rule-Based Category Optimization",
            "badge_color": "info",
            "title": f"Primary Cost Driver: {top_cat} ({top_share}% of expenses)",
            "content": f"You spent {currency_symbol}{top_amt:,.2f} on {top_cat} this period. {tip_detail}"
        })

    # 3. 50/30/20 Rule Suggestions (Rule-Based)
    rule_data = calculate_50_30_20_split(monthly_income, df_current)
    if rule_data:
        if rule_data["savings_pct"] < 20.0:
            gap = round(20.0 - rule_data["savings_pct"], 1)
            suggestions.append({
                "type": "Rule-Based",
                "tag": "🏷️ Rule-Based 50/30/20 Framework",
                "badge_color": "warning",
                "title": "Below Recommended 20% Savings Rate",
                "content": f"Your current savings rate is {rule_data['savings_pct']}%, which is {gap}% below the standard 20% benchmark. Transfer savings to an automated deposit at the start of each month."
            })
        else:
            suggestions.append({
                "type": "Rule-Based",
                "tag": "🏷️ Rule-Based 50/30/20 Framework",
                "badge_color": "success",
                "title": "50/30/20 Rule Compliant",
                "content": f"Excellent discipline! Your savings rate of {rule_data['savings_pct']}% exceeds the 20% standard recommended by financial planners."
            })
        if rule_data["wants_pct"] > 35.0:
            suggestions.append({
                "type": "Rule-Based",
                "tag": "🏷️ Rule-Based 50/30/20 Framework",
                "badge_color": "info",
                "title": "Discretionary 'Wants' Exceeding 30%",
                "content": f"Non-essential wants account for {rule_data['wants_pct']}% of your monthly income. Keeping wants under 30% creates a buffer for unforeseen emergencies."
            })

    # 4. Weekend Spending Spike Detection (Rule-Based)
    if not df_all.empty:
        df_all_copy = df_all.copy()
        df_all_copy["is_weekend"] = pd.to_datetime(df_all_copy["date"]).dt.weekday.isin([5, 6])
        avg_weekend = df_all_copy[df_all_copy["is_weekend"]]["amount"].mean()
        avg_weekday = df_all_copy[~df_all_copy["is_weekend"]]["amount"].mean()

        if avg_weekend > (avg_weekday * 1.5):
            ratio = round(avg_weekend / max(avg_weekday, 1.0), 1)
            suggestions.append({
                "type": "Rule-Based",
                "tag": "🏷️ Rule-Based Behavioral Pattern",
                "badge_color": "info",
                "title": "Weekend Expense Surge Detected",
                "content": f"Your average weekend transaction ({currency_symbol}{avg_weekend:,.0f}) is {ratio}x higher than weekday transactions ({currency_symbol}{avg_weekday:,.0f}). Set a fixed weekend entertainment cap."
            })

    # 5. Machine Learning Predictive Suggestions (ML/AI)
    if ml_forecast and ml_forecast.get("status") == "success":
        pred_spend = ml_forecast["predicted_next_expense"]
        next_month = ml_forecast["next_month_name"]
        mae = ml_forecast["mae"]

        if monthly_budget > 0 and pred_spend > monthly_budget:
            pred_diff = pred_spend - monthly_budget
            suggestions.append({
                "type": "Machine Learning",
                "tag": "🤖 Machine Learning Predictive Forecast",
                "badge_color": "primary",
                "title": f"Forecasted Budget Deficit for {next_month}",
                "content": (
                    f"Our Scikit-Learn regression model forecasts expenses of {currency_symbol}{pred_spend:,.2f} "
                    f"(±{currency_symbol}{mae:,.2f}) next month based on historical momentum and seasonal trends. "
                    f"This exceeds your planned budget by {currency_symbol}{pred_diff:,.2f}. "
                    f"Proactively plan cutbacks before the month begins to prevent a cash deficit."
                )
            })
        else:
            suggestions.append({
                "type": "Machine Learning",
                "tag": "🤖 Machine Learning Predictive Forecast",
                "badge_color": "primary",
                "title": f"Projected Sustainable Spend for {next_month}",
                "content": (
                    f"Our trained regression model forecasts next month's total expenditure at "
                    f"{currency_symbol}{pred_spend:,.2f} (Model MAE: {currency_symbol}{mae:,.2f}). "
                    f"Spending trajectory remains aligned with your historical patterns."
                )
            })

    return suggestions

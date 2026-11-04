"""
Cashly – Track Smart. Save More.
A comprehensive Data Science Web Application built with Streamlit, SQLite, Plotly & Scikit-Learn.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import io
import sys

# Ensure application directory is in sys.path for cross-platform imports on Linux & Streamlit Cloud
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Import modular backend components
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
    PAYMENT_METHODS,
    DEFAULT_DB_PATH
)
from sample_data import load_demo_dataset
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

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cashly – Track Smart. Save More.",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database tables on startup
init_db()

# ---------------------------------------------------------
# Custom Modern CSS Styling (Dark Blue, Lavender, Clean White)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Custom Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4F46E5 100%);
        border-radius: 18px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.25);
    }
    .hero-title {
        font-size: 26px;
        font-weight: 800;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #C7D2FE;
        font-weight: 500;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E0E7FF;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.12);
    }
    .kpi-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 800;
        color: #1E1B4B;
        letter-spacing: -0.5px;
    }
    .kpi-subtext {
        font-size: 12px;
        margin-top: 6px;
        font-weight: 500;
    }

    /* Pill Badges */
    .badge-rule {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-ml {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-demo {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        display: inline-block;
    }

    /* Insight Card */
    .insight-card {
        background: #FFFFFF;
        border-left: 5px solid #6366F1;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    /* Streamlit overrides */
    div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #1E1B4B !important;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stDownloadButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0 20px 0;">
                <div style="font-size: 40px; margin-bottom: 4px;">💳</div>
                <h2 style="margin: 0; color: #1E1B4B; font-weight: 800; font-size: 24px; letter-spacing: -0.5px;">Cashly</h2>
                <p style="margin: 0; color: #6366F1; font-size: 12px; font-weight: 700; letter-spacing: 0.5px;">TRACK SMART. SAVE MORE.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Main Navigation
        selected_page = st.radio(
            "Navigation",
            options=[
                "📊 Dashboard",
                "💳 Expense Manager",
                "📈 Data Analytics",
                "🤖 ML Budget Predictor",
                "💡 Smart Budget Insights",
                "📑 Reports & Export",
                "⚙️ Settings & Demo Data"
            ],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("---")

        # Global Currency & Profile Status
        curr_symbol = get_setting("currency_symbol", "₹")
        monthly_income = float(get_setting("monthly_income", "45000"))
        monthly_budget = float(get_setting("monthly_budget", "35000"))
        is_demo = get_setting("is_demo_loaded", "0") == "1"
        rec_count = count_expenses()

        st.markdown("#### ⚙️ Quick Overview")
        if is_demo:
            st.markdown('<span class="badge-demo">⚡ Demo Mode Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-rule">📁 Personal User Records</span>', unsafe_allow_html=True)

        st.caption(f"Total Transactions: **{rec_count} records**")
        st.caption(f"Monthly Income: **{curr_symbol}{monthly_income:,.0f}**")
        st.caption(f"Monthly Budget: **{curr_symbol}{monthly_budget:,.0f}**")

        st.markdown("---")
        with st.expander("☁️ Cloud Deployment Notice", expanded=False):
            st.markdown(
                """
                <div style="font-size: 11.5px; color: #475569; line-height: 1.4;">
                    <b>Shared Public Demo:</b><br>
                    • Running on Streamlit Community Cloud.<br>
                    • SQLite database is <i>ephemeral</i> and resets upon container reboots.<br>
                    • No login authentication is implemented; do not enter sensitive bank details.<br>
                    • For permanent private tracking, run locally or connect a persistent cloud database (e.g. Supabase).
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            """
            <div style="font-size: 11px; color: #64748B; text-align: center; margin-top: 10px;">
                <b>Cashly v1.0</b><br>
                Track Smart. Save More.<br>
                Streamlit • SQLite • Scikit-Learn • Plotly
            </div>
            """,
            unsafe_allow_html=True
        )

    return selected_page, curr_symbol, monthly_income, monthly_budget


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def get_current_month_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """Filters expenses for the current active calendar month."""
    if df.empty:
        return df
    today = date.today()
    curr_ym = today.strftime("%Y-%m")
    df_copy = df.copy()
    df_copy["ym"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m")
    curr_df = df_copy[df_copy["ym"] == curr_ym]
    
    # If user hasn't added expenses for current month yet, fall back to the most recent month in dataset
    if curr_df.empty and not df.empty:
        latest_ym = df_copy["ym"].max()
        return df_copy[df_copy["ym"] == latest_ym]
    return curr_df


# ---------------------------------------------------------
# Page 1: Dashboard
# ---------------------------------------------------------
def render_dashboard(curr_symbol: str, monthly_income: float, monthly_budget: float):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">👋 Welcome to Cashly</div>
            <div class="hero-subtitle">Track Smart. Save More. – Intelligent Personal Finance & Budget Prediction</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = get_expenses_df()
    curr_month_df = get_current_month_expenses(df)
    cat_budgets = get_category_budgets()

    # Helpful empty state callout for first-time public cloud visitors
    if df.empty:
        st.info("👋 **Welcome to Cashly!** Your expense database is currently empty. To explore the Machine Learning forecasting model and interactive analytics right away, click **Load Realistic Demo Data** below.")
        col_load, _ = st.columns([1, 2])
        with col_load:
            if st.button("🚀 Load Realistic Demo Data (7 Months)", type="primary", key="dash_empty_load"):
                load_demo_dataset()
                st.success("Demo dataset loaded!")
                st.rerun()

    # Determine reference month display
    if not curr_month_df.empty:
        ref_month_name = pd.to_datetime(curr_month_df["date"].iloc[0]).strftime("%B %Y")
    else:
        ref_month_name = datetime.now().strftime("%B %Y")

    total_spent = curr_month_df["amount"].sum() if not curr_month_df.empty else 0.0
    balance = monthly_income - total_spent
    savings_pct = (balance / monthly_income * 100) if monthly_income > 0 else 0.0
    budget_health = compute_budget_health(total_spent, monthly_budget)

    # Budget Alert Notification Banner if needed
    if budget_health["is_exceeded"]:
        st.error(f"🚨 **Budget Alert:** You have exceeded your {ref_month_name} budget by {curr_symbol}{budget_health['overspent_amount']:,.2f} ({budget_health['utilization_pct']}% utilized).")
    elif budget_health["utilization_pct"] >= 85:
        st.warning(f"⚠️ **Budget Warning:** You have consumed {budget_health['utilization_pct']}% of your {ref_month_name} budget. {curr_symbol}{budget_health['remaining_budget']:,.2f} remaining.")

    # KPI Summary Cards (5 Key Metrics)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">💸 {ref_month_name} Spend</div>
                <div class="kpi-value">{curr_symbol}{total_spent:,.0f}</div>
                <div class="kpi-subtext" style="color: #6366F1;">{len(curr_month_df)} transactions</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">💰 Monthly Income</div>
                <div class="kpi-value">{curr_symbol}{monthly_income:,.0f}</div>
                <div class="kpi-subtext" style="color: #10B981;">Fixed Monthly Target</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        bal_color = "#10B981" if balance >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">⚖️ Net Balance</div>
                <div class="kpi-value" style="color: {bal_color};">{curr_symbol}{balance:,.0f}</div>
                <div class="kpi-subtext" style="color: {bal_color};">{'Surplus' if balance >= 0 else 'Deficit'}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        sav_color = "#10B981" if savings_pct >= 20 else "#F59E0B"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">📈 Savings Rate</div>
                <div class="kpi-value" style="color: {sav_color};">{savings_pct:.1f}%</div>
                <div class="kpi-subtext" style="color: {sav_color};">Benchmark: 20%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col5:
        util_color = budget_health["color"]
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">🎯 Budget Used</div>
                <div class="kpi-value" style="color: {util_color};">{budget_health['utilization_pct']}%</div>
                <div class="kpi-subtext" style="color: {util_color};">{budget_health['status']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Visual Budget Progress Bar
    util_clamped = min(1.0, total_spent / monthly_budget) if monthly_budget > 0 else 0.0
    st.progress(util_clamped, text=f"Budget Pace: {curr_symbol}{total_spent:,.0f} spent of {curr_symbol}{monthly_budget:,.0f} limit")

    st.write("")

    # Visualizations Row
    chart_col1, chart_col2 = st.columns([1, 1])

    with chart_col1:
        st.markdown(f"##### 🍩 Spending by Category ({ref_month_name})")
        fig_donut = create_category_donut_chart(curr_month_df, curr_symbol)
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        st.markdown("##### 📈 Multi-Month Spending Trend")
        fig_trend = create_monthly_trend_chart(df, curr_symbol)
        st.plotly_chart(fig_trend, use_container_width=True)

    # ML Next Month Forecast Snapshot Card
    ml_res = train_and_predict_expenses(df)
    st.markdown("##### 🤖 ML Next-Month Spend Forecast Preview")
    
    if ml_res.get("status") == "success":
        m_col1, m_col2, m_col3 = st.columns([2, 1, 1])
        with m_col1:
            st.info(
                f"🔮 **Scikit-Learn Forecast for {ml_res['next_month_name']}:** "
                f"**{curr_symbol}{ml_res['predicted_next_expense']:,.2f}** "
                f"(Range: {curr_symbol}{ml_res['lower_bound']:,.2f} – {curr_symbol}{ml_res['upper_bound']:,.2f}). "
                f"Model MAE accuracy: **{curr_symbol}{ml_res['mae']:,.2f}**."
            )
        with m_col2:
            st.metric("Model Algorithm", "Linear / Ridge", help="Trained with chronological time features")
        with m_col3:
            st.metric("Test Set MAE", f"{curr_symbol}{ml_res['mae']:,.0f}", delta=f"-{ml_res['naive_mae'] - ml_res['mae']:,.0f} vs naive", delta_color="inverse")
    else:
        st.caption(f"ℹ️ {ml_res.get('message', 'Not enough data for reliable ML forecasting.')}")

    # Recent Transactions Snapshot
    st.markdown("##### 🕒 Recent Transactions")
    if not df.empty:
        recent_display = df.head(8).copy()
        recent_display["date"] = pd.to_datetime(recent_display["date"]).dt.strftime("%d %b %Y")
        recent_display["amount"] = recent_display["amount"].apply(lambda x: f"{curr_symbol}{x:,.2f}")
        recent_display = recent_display.rename(columns={
            "id": "ID", "date": "Date", "category": "Category",
            "amount": "Amount", "description": "Description", "payment_method": "Payment Method"
        })
        st.dataframe(recent_display[["ID", "Date", "Category", "Amount", "Description", "Payment Method"]], use_container_width=True, hide_index=True)
    else:
        st.info("No expense transactions recorded yet. Head over to **Expense Manager** to add your first expense or load Demo Data from **Settings**!")


# ---------------------------------------------------------
# Page 2: Expense Manager (Add, Edit, Delete, Search & Filter)
# ---------------------------------------------------------
def render_expense_manager(curr_symbol: str):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">💳 Expense Manager</div>
            <div class="hero-subtitle">Add new expenses, modify historical records, and filter your financial ledger</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    tab_add, tab_edit_delete, tab_filter = st.tabs([
        "➕ Add New Expense",
        "✏️ Edit & Delete Expenses",
        "🔍 Search & Filter Ledger"
    ])

    # TAB 1: ADD EXPENSE
    with tab_add:
        st.markdown("#### Enter New Expense Details")
        with st.form("add_expense_form", clear_on_submit=True):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                exp_date = st.date_input("Transaction Date", value=date.today())
                amount = st.number_input(f"Amount ({curr_symbol})", min_value=1.0, max_value=1000000.0, value=250.0, step=10.0)
                category = st.selectbox("Expense Category", options=CATEGORIES, index=0)

            with fcol2:
                payment_method = st.selectbox("Payment Method", options=PAYMENT_METHODS, index=0)
                description = st.text_input("Description / Merchant", placeholder="e.g. Grocery items at supermarket, Metro pass, Netflix")

            submit_add = st.form_submit_button("💾 Save Expense Record", use_container_width=True)

            if submit_add:
                if not description.strip():
                    st.error("Please provide a description or merchant name.")
                else:
                    new_id = add_expense(
                        exp_date=exp_date,
                        amount=amount,
                        category=category,
                        description=description,
                        payment_method=payment_method
                    )
                    st.success(f"✅ Expense of {curr_symbol}{amount:,.2f} in **{category}** successfully recorded (ID: #{new_id})!")
                    st.rerun()

    # TAB 2: EDIT & DELETE
    with tab_edit_delete:
        st.markdown("#### Modify or Remove an Existing Record")
        df_all = get_expenses_df()

        if df_all.empty:
            st.info("No expense records available to edit or delete.")
        else:
            options_list = [
                f"#{row['id']} | {row['date'].strftime('%Y-%m-%d')} | {curr_symbol}{row['amount']:,.2f} | {row['category']} - {row['description']}"
                for _, row in df_all.iterrows()
            ]
            selected_option = st.selectbox("Select Expense Record to Modify", options=options_list)
            selected_id = int(selected_option.split("|")[0].replace("#", "").strip())

            record = get_expense_by_id(selected_id)
            if record:
                rec_date = datetime.strptime(record["date"], "%Y-%m-%d").date()
                cat_idx = CATEGORIES.index(record["category"]) if record["category"] in CATEGORIES else 0
                pay_idx = PAYMENT_METHODS.index(record["payment_method"]) if record["payment_method"] in PAYMENT_METHODS else 0

                st.write("---")
                col_edit1, col_edit2 = st.columns(2)
                with col_edit1:
                    edit_date = st.date_input("Updated Date", value=rec_date, key="edit_d")
                    edit_amount = st.number_input(f"Updated Amount ({curr_symbol})", min_value=1.0, value=float(record["amount"]), key="edit_a")
                    edit_category = st.selectbox("Updated Category", options=CATEGORIES, index=cat_idx, key="edit_c")

                with col_edit2:
                    edit_payment = st.selectbox("Updated Payment Method", options=PAYMENT_METHODS, index=pay_idx, key="edit_p")
                    edit_desc = st.text_input("Updated Description", value=record["description"], key="edit_desc")

                st.write("")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("💾 Update Record", use_container_width=True, type="primary"):
                        if not edit_desc.strip():
                            st.error("Description cannot be empty.")
                        else:
                            success = update_expense(
                                expense_id=selected_id,
                                exp_date=edit_date,
                                amount=edit_amount,
                                category=edit_category,
                                description=edit_desc,
                                payment_method=edit_payment
                            )
                            if success:
                                st.success(f"Expense #{selected_id} updated successfully!")
                                st.rerun()

                with btn_col2:
                    if st.button("🗑️ Delete Record", use_container_width=True):
                        success = delete_expense(selected_id)
                        if success:
                            st.warning(f"Expense #{selected_id} deleted.")
                            st.rerun()

    # TAB 3: SEARCH & FILTER
    with tab_filter:
        st.markdown("#### Search & Filter Expense Transactions")
        scol1, scol2, scol3 = st.columns(3)
        with scol1:
            search_query = st.text_input("🔍 Search Keyword", placeholder="Search description or category...")
        with scol2:
            filter_cats = st.multiselect("Filter by Categories", options=CATEGORIES, default=[])
        with scol3:
            today = date.today()
            date_filter = st.date_input("Filter Date Range", value=(today - timedelta(days=90), today))

        start_d = date_filter[0] if isinstance(date_filter, tuple) and len(date_filter) > 0 else None
        end_d = date_filter[1] if isinstance(date_filter, tuple) and len(date_filter) > 1 else None

        filtered_df = get_expenses_df(
            start_date=start_d,
            end_date=end_d,
            categories=filter_cats if filter_cats else None,
            search=search_query
        )

        st.write("")
        st.markdown(f"**Showing {len(filtered_df)} Matching Transactions** (Total: **{curr_symbol}{filtered_df['amount'].sum() if not filtered_df.empty else 0:,.2f}**)")

        if not filtered_df.empty:
            display_df = filtered_df.copy()
            display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
            display_df["formatted_amount"] = display_df["amount"].apply(lambda x: f"{curr_symbol}{x:,.2f}")
            st.dataframe(
                display_df[["id", "date", "category", "formatted_amount", "description", "payment_method"]].rename(columns={
                    "id": "ID", "date": "Date", "category": "Category", "formatted_amount": "Amount",
                    "description": "Description", "payment_method": "Payment Method"
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No records matched your search filters.")


# ---------------------------------------------------------
# Page 3: Data Analytics
# ---------------------------------------------------------
def render_analytics(curr_symbol: str, monthly_income: float):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">📈 Data Analytics & Spending Patterns</div>
            <div class="hero-subtitle">Interactive visual breakdowns, temporal patterns, and income vs expense ratios</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = get_expenses_df()
    if df.empty:
        st.info("No expenses found. Please record expenses or load demo data to view analytics.")
        return

    # Visual Filters
    f1, f2 = st.columns([2, 1])
    with f1:
        date_range_choice = st.selectbox(
            "Analytics Timeframe",
            ["All Historical Data", "Last 30 Days", "Last 90 Days", "Last 180 Days"]
        )
    with f2:
        category_focus = st.multiselect("Focus Categories (Optional)", options=CATEGORIES, default=[])

    # Apply timeframe filter
    df_chart = df.copy()
    today = date.today()
    if date_range_choice == "Last 30 Days":
        df_chart = df_chart[pd.to_datetime(df_chart["date"]).dt.date >= (today - timedelta(days=30))]
    elif date_range_choice == "Last 90 Days":
        df_chart = df_chart[pd.to_datetime(df_chart["date"]).dt.date >= (today - timedelta(days=90))]
    elif date_range_choice == "Last 180 Days":
        df_chart = df_chart[pd.to_datetime(df_chart["date"]).dt.date >= (today - timedelta(days=180))]

    if category_focus:
        df_chart = df_chart[df_chart["category"].isin(category_focus)]

    # Row 1: Donut Chart & Monthly Trend
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        st.markdown("##### 🍩 Category Spending Distribution")
        fig_donut = create_category_donut_chart(df_chart, curr_symbol)
        st.plotly_chart(fig_donut, use_container_width=True)

    with r1_col2:
        st.markdown("##### 📈 Monthly Spending Trajectory")
        fig_trend = create_monthly_trend_chart(df_chart, curr_symbol)
        st.plotly_chart(fig_trend, use_container_width=True)

    # Row 2: Income vs Expenses & Top Spending Ranking
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        st.markdown("##### ⚖️ Monthly Income vs Expenses Comparison")
        fig_inc_exp = create_income_vs_expense_chart(df_chart, monthly_income, curr_symbol)
        st.plotly_chart(fig_inc_exp, use_container_width=True)

    with r2_col2:
        st.markdown("##### 🏆 Ranked Top Spending Categories")
        fig_top = create_top_categories_bar_chart(df_chart, curr_symbol)
        st.plotly_chart(fig_top, use_container_width=True)

    # Row 3: Daily Timeline & Day of Week Behavioral Analysis
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        st.markdown("##### 📅 Daily Spending Rhythm & 7-Day Moving Average")
        fig_daily = create_daily_spending_chart(df_chart, curr_symbol)
        st.plotly_chart(fig_daily, use_container_width=True)

    with r3_col2:
        st.markdown("##### 🗓️ Day of Week Spending Analysis (Weekend vs Weekday)")
        fig_dow = create_day_of_week_chart(df_chart, curr_symbol)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Tabular Breakdown
    st.markdown("##### 📊 Category Performance Statistics")
    cat_stats = df_chart.groupby("category").agg(
        Total_Spend=("amount", "sum"),
        Transactions=("amount", "count"),
        Average_Ticket=("amount", "mean"),
        Max_Transaction=("amount", "max")
    ).reset_index()

    cat_stats["Share"] = (cat_stats["Total_Spend"] / cat_stats["Total_Spend"].sum() * 100).round(1).astype(str) + "%"
    cat_stats["Total_Spend"] = cat_stats["Total_Spend"].apply(lambda x: f"{curr_symbol}{x:,.2f}")
    cat_stats["Average_Ticket"] = cat_stats["Average_Ticket"].apply(lambda x: f"{curr_symbol}{x:,.2f}")
    cat_stats["Max_Transaction"] = cat_stats["Max_Transaction"].apply(lambda x: f"{curr_symbol}{x:,.2f}")

    st.dataframe(cat_stats.rename(columns={
        "category": "Category", "Total_Spend": "Total Spend", "Transactions": "Transactions Count",
        "Average_Ticket": "Average per Expense", "Max_Transaction": "Largest Item", "Share": "% of Total"
    }), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Page 4: ML Budget Predictor (Rigorous ML Pipeline & Evaluation)
# ---------------------------------------------------------
def render_ml_predictor(curr_symbol: str, monthly_budget: float):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">🤖 Machine Learning Budget Predictor</div>
            <div class="hero-subtitle">Time-series regression model with chronological train/test splitting and MAE evaluation</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = get_expenses_df()
    ml_results = train_and_predict_expenses(df)

    # Honest Data Sufficiency Guard Check
    if ml_results.get("status") != "success":
        st.warning(f"⚠️ **Data Sufficiency Notice:** {ml_results.get('message')}")
        st.markdown(
            """
            > **Why this matters in Data Science:**
            > In time-series machine learning, splitting data chronologically requires enough sequential periods 
            > to establish lag features (e.g. `lag_1`, `rolling_mean`) and validate against an unseen holdout period without 
            > lookahead data leakage. A minimum of **4 calendar months** is mathematically necessary for an honest evaluation.
            """
        )
        if st.button("🚀 Load Realistic Demo Data (7 Months)", type="primary"):
            load_demo_dataset()
            st.success("Loaded realistic multi-month demo dataset! Re-evaluating ML model...")
            st.rerun()
        return

    # Success: ML Model Results
    st.markdown(
        f"""
        <div style="background: white; border: 1px solid #E0E7FF; border-radius: 14px; padding: 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span class="badge-ml">🤖 Scikit-Learn Model Active</span>
                    <h3 style="margin: 8px 0 4px 0; color: #1E1B4B; font-weight: 800;">Next-Month Forecast: {ml_results['next_month_name']}</h3>
                    <p style="margin: 0; color: #64748B; font-size: 14px;">Trained on {ml_results['month_count']} sequential months using chronological validation.</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 32px; font-weight: 800; color: #4F46E5;">{curr_symbol}{ml_results['predicted_next_expense']:,.2f}</div>
                    <div style="font-size: 13px; color: #6366F1; font-weight: 600;">Expected Range: {curr_symbol}{ml_results['lower_bound']:,.0f} – {curr_symbol}{ml_results['upper_bound']:,.0f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4 Key Machine Learning Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Holdout Test MAE",
            value=f"{curr_symbol}{ml_results['mae']:,.2f}",
            help="Mean Absolute Error on the unseen chronological holdout test set"
        )
    with col2:
        st.metric(
            label="Root Mean Squared Error",
            value=f"{curr_symbol}{ml_results['rmse']:,.2f}",
            help="Penalizes larger prediction deviations"
        )
    with col3:
        st.metric(
            label="Mean Abs % Error (MAPE)",
            value=f"{ml_results['mape']:.1f}%",
            help="Relative percentage deviation on test set"
        )
    with col4:
        diff_naive = ml_results["naive_mae"] - ml_results["mae"]
        st.metric(
            label="Baseline Naive Comparison",
            value=f"{curr_symbol}{ml_results['naive_mae']:,.2f}",
            delta=f"{curr_symbol}{diff_naive:,.2f} better than guessing",
            delta_color="normal"
        )

    # Interactive Forecast Visualization
    st.markdown("##### 📈 Chronological Split & ML Forecasting Visualizer")
    fig_ml = create_ml_forecast_chart(ml_results, curr_symbol)
    st.plotly_chart(fig_ml, use_container_width=True)

    # Deep Dive: Chronological Holdout Evaluation & Feature Importance
    d_col1, d_col2 = st.columns(2)

    with d_col1:
        st.markdown("##### 🧪 Chronological Holdout Test Evaluation")
        st.caption("Validating against the most recent historical month(s) without lookahead data leakage:")
        st.dataframe(
            ml_results["test_comparison"].style.format({
                "Actual Expense": f"{curr_symbol}{{:,.2f}}",
                "Predicted Expense": f"{curr_symbol}{{:,.2f}}",
                "Absolute Error": f"{curr_symbol}{{:,.2f}}"
            }),
            use_container_width=True,
            hide_index=True
        )

    with d_col2:
        st.markdown("##### 🔬 Regression Feature Coefficients")
        st.caption("How each time feature mathematically influences the predicted spend:")
        coef_df = pd.DataFrame([
            {"Feature": "Sequential Month Index (Trend)", "Coefficient": ml_results["coefficients"].get("month_idx", 0.0), "Impact": "Secular spending trajectory"},
            {"Feature": "Calendar Month (Seasonality)", "Coefficient": ml_results["coefficients"].get("month_of_year", 0.0), "Impact": "Annual cyclical variation"},
            {"Feature": "Quarterly Cycle", "Coefficient": ml_results["coefficients"].get("quarter", 0.0), "Impact": "Quarterly spending trends"},
            {"Feature": "Prior Month Spend (Lag 1)", "Coefficient": ml_results["coefficients"].get("lag_1", 0.0), "Impact": "Immediate spending momentum"},
            {"Feature": "3-Month Moving Average", "Coefficient": ml_results["coefficients"].get("rolling_mean_3", 0.0), "Impact": "Smooth historical baseline"}
        ])
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    # Category Projections
    if ml_results.get("category_forecast"):
        st.markdown("##### 🏷️ Projected Category Breakdown for Next Month")
        cat_p_df = pd.DataFrame([
            {"Category": cat, "Projected Spend": f"{curr_symbol}{amt:,.2f}", "Raw": amt}
            for cat, amt in ml_results["category_forecast"].items()
        ]).sort_values("Raw", ascending=False)
        st.dataframe(cat_p_df[["Category", "Projected Spend"]], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Page 5: Smart Budget Insights
# ---------------------------------------------------------
def render_insights(curr_symbol: str, monthly_income: float, monthly_budget: float):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">💡 Smart Budget Insights & Advice</div>
            <div class="hero-subtitle">Automated financial heuristics and machine learning intelligence to optimize savings</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df_all = get_expenses_df()
    df_curr = get_current_month_expenses(df_all)
    cat_budgets = get_category_budgets()
    ml_res = train_and_predict_expenses(df_all)

    # 1. Budget Health Overview
    curr_spent = df_curr["amount"].sum() if not df_curr.empty else 0.0
    budget_health = compute_budget_health(curr_spent, monthly_budget)

    st.markdown("##### 🎯 Current Budget Status")
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        st.metric("Total Monthly Budget", f"{curr_symbol}{monthly_budget:,.0f}")
    with b_col2:
        st.metric("Spent So Far", f"{curr_symbol}{curr_spent:,.0f}", delta=f"{budget_health['utilization_pct']}% utilized")
    with b_col3:
        st.metric("Remaining Buffer", f"{curr_symbol}{budget_health['remaining_budget']:,.0f}", delta=budget_health["status"], delta_color="inverse" if budget_health["is_exceeded"] else "normal")

    st.write("")

    # 2. 50/30/20 Financial Framework Breakdown
    st.markdown("##### ⚖️ 50/30/20 Financial Health Benchmark")
    rule_info = calculate_50_30_20_split(monthly_income, df_curr)
    if rule_info:
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">🏠 Needs (Target: 50%)</div>
                    <div class="kpi-value">{rule_info['needs_pct']}%</div>
                    <div class="kpi-subtext">{curr_symbol}{rule_info['needs_spent']:,.0f} (Bills, Food, Health, Education)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">🛍️ Wants (Target: 30%)</div>
                    <div class="kpi-value">{rule_info['wants_pct']}%</div>
                    <div class="kpi-subtext">{curr_symbol}{rule_info['wants_spent']:,.0f} (Shopping, Travel, Entertainment)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with r3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-label">💰 Savings (Target: 20%)</div>
                    <div class="kpi-value">{rule_info['savings_pct']}%</div>
                    <div class="kpi-subtext">{curr_symbol}{rule_info['savings']:,.0f} Net Savings Buffer</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("Set your monthly income in Settings to enable the 50/30/20 financial benchmark.")

    st.write("")

    # 3. Actionable Suggestions (Rule-Based vs ML explicitly tagged!)
    st.markdown("##### 🧠 Actionable Financial Recommendations")
    st.caption("Every recommendation below is explicitly classified by origin (Deterministic Financial Standard vs ML Model):")

    suggestions = generate_smart_suggestions(
        df_all=df_all,
        df_current=df_curr,
        monthly_income=monthly_income,
        monthly_budget=monthly_budget,
        category_budgets=cat_budgets,
        ml_forecast=ml_res,
        currency_symbol=curr_symbol
    )

    for sug in suggestions:
        is_ml = sug["type"] == "Machine Learning"
        badge_class = "badge-ml" if is_ml else "badge-rule"
        border_color = "#3B82F6" if is_ml else "#8B5CF6"

        st.markdown(
            f"""
            <div style="background: white; border-left: 5px solid {border_color}; border-radius: 12px; padding: 18px; margin-bottom: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="margin-bottom: 6px;">
                    <span class="{badge_class}">{sug['tag']}</span>
                </div>
                <div style="font-weight: 700; font-size: 16px; color: #1E1B4B; margin-bottom: 4px;">{sug['title']}</div>
                <div style="font-size: 13.5px; color: #475569; line-height: 1.5;">{sug['content']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Category Budget Compliance Table
    st.markdown("##### 📋 Category Budget Compliance")
    cat_analysis = analyze_category_breakdown(df_curr, cat_budgets)

    cat_rows = []
    cat_sums = df_curr.groupby("category")["amount"].sum() if not df_curr.empty else {}
    for cat in CATEGORIES:
        spent = float(cat_sums.get(cat, 0.0))
        bgt = float(cat_budgets.get(cat, 0.0))
        util = round((spent / bgt * 100), 1) if bgt > 0 else 0.0
        status_text = "🚨 Over Budget" if util >= 100 else ("⚠️ Approaching" if util >= 85 else "✅ Normal")
        cat_rows.append({
            "Category": cat,
            "Spent": f"{curr_symbol}{spent:,.2f}",
            "Budget": f"{curr_symbol}{bgt:,.2f}",
            "Utilization": f"{util}%",
            "Status": status_text
        })

    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Page 6: Reports & Export
# ---------------------------------------------------------
def render_reports(curr_symbol: str, monthly_income: float, monthly_budget: float):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">📑 Reports & Data Export</div>
            <div class="hero-subtitle">Generate monthly statements, filter custom periods, and export CSV files</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = get_expenses_df()
    if df.empty:
        st.info("No expense data available to generate reports.")
        return

    rep_tab1, rep_tab2 = st.tabs(["📅 Monthly Statement Summary", "📥 Custom Export Ledger"])

    with rep_tab1:
        st.markdown("#### Monthly Expense Statement")
        df_copy = df.copy()
        df_copy["ym"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m")
        unique_months = sorted(df_copy["ym"].unique(), reverse=True)

        selected_ym = st.selectbox("Select Billing Month", options=unique_months)
        month_records = df_copy[df_copy["ym"] == selected_ym]

        m_spent = month_records["amount"].sum()
        m_count = len(month_records)
        m_avg = month_records["amount"].mean() if m_count > 0 else 0.0
        m_max_item = month_records.loc[month_records["amount"].idxmax()] if m_count > 0 else None

        # Executive Metrics
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("Total Statement Spend", f"{curr_symbol}{m_spent:,.2f}")
        with rc2:
            st.metric("Total Transactions", f"{m_count} items")
        with rc3:
            st.metric("Average Transaction", f"{curr_symbol}{m_avg:,.2f}")
        with rc4:
            if m_max_item is not None:
                st.metric("Largest Single Item", f"{curr_symbol}{m_max_item['amount']:,.2f}", help=f"{m_max_item['category']}: {m_max_item['description']}")

        st.write("---")
        st.markdown(f"##### Category Breakdown for {selected_ym}")
        cat_breakdown = month_records.groupby("category").agg(
            Total=("amount", "sum"),
            Count=("amount", "count"),
            Avg=("amount", "mean")
        ).reset_index()
        cat_breakdown["Share %"] = (cat_breakdown["Total"] / m_spent * 100).round(1).astype(str) + "%"
        cat_breakdown["Total"] = cat_breakdown["Total"].apply(lambda x: f"{curr_symbol}{x:,.2f}")
        cat_breakdown["Avg"] = cat_breakdown["Avg"].apply(lambda x: f"{curr_symbol}{x:,.2f}")
        st.dataframe(cat_breakdown, use_container_width=True, hide_index=True)

        # Download Month CSV
        csv_buffer = io.StringIO()
        month_records.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"📥 Download {selected_ym} Statement (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"cashly_statement_{selected_ym}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with rep_tab2:
        st.markdown("#### Custom Ledger Export")
        c1, c2 = st.columns(2)
        with c1:
            exp_date_range = st.date_input("Filter Range", value=(date.today() - timedelta(days=60), date.today()))
        with c2:
            exp_cats = st.multiselect("Filter Categories", options=CATEGORIES, default=CATEGORIES)

        s_d = exp_date_range[0] if isinstance(exp_date_range, tuple) and len(exp_date_range) > 0 else None
        e_d = exp_date_range[1] if isinstance(exp_date_range, tuple) and len(exp_date_range) > 1 else None

        export_df = get_expenses_df(start_date=s_d, end_date=e_d, categories=exp_cats)
        st.write(f"**Found {len(export_df)} records matching export criteria.**")

        if not export_df.empty:
            full_csv = io.StringIO()
            export_df.to_csv(full_csv, index=False)
            st.download_button(
                label="📥 Download Filtered Ledger as CSV",
                data=full_csv.getvalue(),
                file_name="cashly_ledger_export.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True
            )


# ---------------------------------------------------------
# Page 7: Settings & Demo Data
# ---------------------------------------------------------
def render_settings(curr_symbol: str, monthly_income: float, monthly_budget: float):
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-title">⚙️ Settings & Demo Data Controls</div>
            <div class="hero-subtitle">Customize budget limits, currency preferences, and toggle sample datasets</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    t1, t2, t3 = st.tabs(["💰 Income & Budget Targets", "🏷️ Category Budgets", "🔄 Demo Data & Database Management"])

    with t1:
        st.markdown("#### General Financial Configuration")
        with st.form("settings_budget_form"):
            new_curr = st.selectbox(
                "Preferred Currency Symbol",
                options=["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)", "¥ (JPY)"],
                index=0 if curr_symbol == "₹" else 1
            )
            new_inc = st.number_input(f"Expected Monthly Income ({curr_symbol})", min_value=100.0, value=monthly_income, step=1000.0)
            new_bgt = st.number_input(f"Monthly Spending Budget ({curr_symbol})", min_value=100.0, value=monthly_budget, step=1000.0)

            save_gen = st.form_submit_button("💾 Save Financial Targets", type="primary")
            if save_gen:
                symbol_only = new_curr.split()[0]
                set_setting("currency_symbol", symbol_only)
                set_setting("monthly_income", str(new_inc))
                set_setting("monthly_budget", str(new_bgt))
                st.success("Financial preferences saved successfully!")
                st.rerun()

    with t2:
        st.markdown("#### Category-Specific Spending Limits")
        current_cat_budgets = get_category_budgets()

        with st.form("cat_budget_form"):
            cols = st.columns(2)
            new_vals = {}
            for idx, cat in enumerate(CATEGORIES):
                col = cols[idx % 2]
                with col:
                    existing = current_cat_budgets.get(cat, 5000.0)
                    new_vals[cat] = st.number_input(f"{cat} Budget ({curr_symbol})", min_value=0.0, value=existing, step=500.0)

            save_cats = st.form_submit_button("💾 Update Category Limits", type="primary")
            if save_cats:
                for cat, amt in new_vals.items():
                    set_category_budget(cat, amt)
                st.success("Category budgets updated!")
                st.rerun()

    with t3:
        st.markdown("#### Dataset Control & Persistence")
        st.write(
            """
            Cashly uses local **SQLite database storage** (`spendwise.db`), which persists across application restarts when run locally.
            You can load realistic multi-month Demo Data to explore analytics and ML models, or clear all data to start tracking your own personal finances.
            """
        )
        st.info(
            """
            ℹ️ **Streamlit Community Cloud Notice:**
            Cloud container instances are ephemeral. On Streamlit Cloud, SQLite records reset when the container sleeps or reboots.
            In addition, all public visitors share the same demonstration state without authentication.
            For production multi-user persistence with individual logins, connect an external cloud database (such as **Supabase PostgreSQL**, **Neon**, or **Turso LibSQL**).
            """
        )

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("##### 🚀 Realistic Demo Data")
            st.caption("Populates SQLite with 7 months of realistic seasonal expenses across all 8 categories (ideal for presentations & ML evaluation).")
            if st.button("Load 7-Month Demo Dataset", use_container_width=True, type="primary"):
                count = load_demo_dataset()
                st.success(f"Successfully generated and loaded {count} realistic expense records!")
                st.rerun()

        with d2:
            st.markdown("##### 🗑️ Clear Database")
            st.caption("Permanently deletes all stored expense records so you can track your own real-world finances from scratch.")
            if st.button("Delete All Expenses (Start Fresh)", use_container_width=True):
                clear_all_expenses()
                st.warning("Database cleared! All transaction records have been reset.")
                st.rerun()


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    selected_page, curr_symbol, monthly_income, monthly_budget = render_sidebar()

    if selected_page == "📊 Dashboard":
        render_dashboard(curr_symbol, monthly_income, monthly_budget)
    elif selected_page == "💳 Expense Manager":
        render_expense_manager(curr_symbol)
    elif selected_page == "📈 Data Analytics":
        render_analytics(curr_symbol, monthly_income)
    elif selected_page == "🤖 ML Budget Predictor":
        render_ml_predictor(curr_symbol, monthly_budget)
    elif selected_page == "💡 Smart Budget Insights":
        render_insights(curr_symbol, monthly_income, monthly_budget)
    elif selected_page == "📑 Reports & Export":
        render_reports(curr_symbol, monthly_income, monthly_budget)
    elif selected_page == "⚙️ Settings & Demo Data":
        render_settings(curr_symbol, monthly_income, monthly_budget)


if __name__ == "__main__":
    main()

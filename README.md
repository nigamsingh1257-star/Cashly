# 💳 Cashly – Track Smart. Save More.

> A complete, professional, beginner-friendly Data Science web application built with **Streamlit**, **SQLite**, **Plotly**, and **Scikit-Learn**.

---

## 🌟 Executive Summary

**Cashly** is an intelligent personal finance platform designed to bridge everyday expense tracking with genuine Data Science and Machine Learning. Beyond basic bookkeeping, Cashly analyzes multi-month spending patterns, monitors budget pace, evaluates financial health against the standard 50/30/20 rule, and forecasts next month's spending using a time-series regression model with chronological train/test validation.

Built as an end-to-end college mini-project achievable in 1–3 days, Cashly requires **zero paid APIs**, runs 100% locally with persistent SQLite storage, and adheres to strict Data Science best practices (preventing temporal data leakage and avoiding synthetic number hallucination).

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Streamlit** | Interactive modern web dashboard UI |
| **SQLite3** | Local persistent ACID transaction storage |
| **Pandas & NumPy** | Data wrangling, aggregation, and feature engineering |
| **Plotly Express / Graph Objects** | Interactive, publication-ready data visualizations |
| **Scikit-Learn** | Linear & Ridge regression models, evaluation metrics (MAE, RMSE, MAPE) |

---

## 🚀 Key Features

### 1. 📊 Executive Financial Dashboard
- **Monthly Spending**: Real-time aggregation of current month expenditures.
- **Monthly Income & Net Balance**: Displays whether the user is running a surplus or deficit.
- **Savings Percentage**: Tracks percentage of income saved against the standard 20% benchmark.
- **Budget Utilization & Pace**: Color-coded progress bar and risk status (*Healthy*, *Moderate*, *Approaching Limit*, *Over Budget*).
- **Recent Transactions Ledger**: Formatted recent entries with category tags.
- **ML Forecast Preview**: Instant snapshot of next month's projected expenses.

### 2. 💳 Robust Expense Management (CRUD + Search)
- **Add Expense**: Intuitive form with Date Picker, Numeric Amount, Category dropdown, Description, and Payment Method.
- **8 Standard Expense Categories**: `Food`, `Shopping`, `Travel`, `Bills`, `Education`, `Health`, `Entertainment`, `Others`.
- **Edit & Delete Records**: Seamlessly select existing entries by ID to update amounts, descriptions, or remove accidental records.
- **Search & Filter**: Search transactions by keyword in description or filter by date ranges and categories.
- **ACID Persistence**: All changes persist permanently in local SQLite (`spendwise.db`).

### 3. 📈 Interactive Data Analytics (Dark Blue & Lavender Theme)
- **Category Donut Chart**: Proportional breakdown of expenses with interactive tooltips and center total.
- **Monthly Trend Spline Chart**: Multi-month spending trajectory with gradient area fill.
- **Income vs Expenses Comparison**: Grouped bar charts showing monthly earnings vs outlays.
- **Top Spending Categories**: Horizontal ranking bar chart highlighting primary cost drivers.
- **Daily Spending Timeline**: Daily transaction volume coupled with a 7-day moving average.
- **Day of Week Behavioral Analysis**: Identifies spending surges across weekdays vs weekends.

### 4. 🤖 Machine Learning Budget Predictor (Rigorous Data Science)
- **Time-Series Regression**: Uses sequential time steps, calendar seasonality, quarterly cycles, lag-1 momentum, and rolling 3-month moving averages.
- **Chronological Train/Test Split**: Prevents lookahead data leakage by strictly training on historical months and testing on unseen recent months.
- **Evaluation via MAE & RMSE**: Evaluates model performance using Mean Absolute Error (MAE) and compares against a naive moving-average baseline.
- **Honest Data Sufficiency Guard**: If fewer than 4 distinct months of expense history exist, the model transparently displays:
  > *"⚠️ Not enough data for reliable prediction. At least 4 months of expense history is required to train the ML model without inventing synthetic numbers."*
- **Next-Month Spending Forecast**: Predicts upcoming monthly expenses with a confidence band (`± MAE`).
- **Category Projections**: Mathematically estimates category-level requirements for the upcoming month.

### 5. 💡 Smart Budget Insights & 50/30/20 Framework
- **Deterministic Budget Alerts**: Triggered when expenditures cross 85% and 100% of limits.
- **Category Limit Tracking**: Tracks adherence against individual category budgets.
- **50/30/20 Rule Analysis**: Evaluates spending split between *Needs* (50%), *Wants* (30%), and *Savings* (20%).
- **Explicit Classification**: Every suggestion is clearly distinguished by origin:
  - 🏷️ `[Rule-Based Financial Standard]` (Heuristic guidelines)
  - 🤖 `[Machine Learning Predictive Forecast]` (Statistical regression)

### 6. 📑 Reports & CSV Export
- Generate monthly financial statements with transaction summaries and top expense item.
- Download formatted CSV reports with custom date and category filters.
- One-click export for spreadsheet software (Excel, Google Sheets).

### 7. ⚙️ Settings & Realistic Demo Data
- Configure Monthly Income, Overall Budget, and Individual Category Budgets.
- Select preferred currency (`₹`, `$`, `€`, `£`, `¥`).
- **Load Realistic Demo Data**: One-click generation of 7 months of realistic seasonal expenses across all 8 categories (perfect for presentations and viva demos).
- **Clear Database**: Reset records to start fresh with personal tracking.

---

## 📁 Project Architecture

```text
spendwise_ai/
│
├── .streamlit/
│   └── config.toml          # Custom modern UI theme (Indigo, Lavender, Slate)
│
├── data/
│   └── spendwise.db         # Persistent SQLite database (created on first run)
│
├── app.py                   # Main Streamlit web application & multi-page router
├── database.py              # SQLite CRUD operations, settings, and category budgets
├── sample_data.py           # Realistic 7-month seasonal demo data generator
├── ml_model.py              # Scikit-Learn time-series regression & evaluation pipeline
├── analytics.py             # Plotly charts with modern Dark Blue/Lavender styling
├── insights.py              # Budget alerts, 50/30/20 engine, and rule-based advice
│
├── requirements.txt         # Project dependencies
├── run.bat                  # One-click Windows launch script
├── .gitignore               # Clean git exclusions
└── README.md                # Complete documentation & viva guide
```

---

## ⚡ Quick Start Guide

### Prerequisites
Make sure Python 3.10 or higher is installed on your computer.

### Step 1: Navigate to the Project Folder
Open your terminal or PowerShell and change into the `spendwise_ai` directory:
```bash
cd spendwise_ai
```

### Step 2: (Recommended) Create a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```
*(On Windows, you can also simply double-click `run.bat`!)*

The application will automatically open in your default browser at `http://localhost:8501`.

---

## 📤 How to Upload This Project to GitHub

Follow these simple steps to upload **Cashly** to your GitHub profile:

### 1. Initialize Git in the Project Folder
Open PowerShell / Terminal inside `spendwise_ai`:
```bash
cd spendwise_ai
git init
```

### 2. Add and Commit Files
```bash
git add .
git commit -m "feat: Initial commit of Cashly - Track Smart. Save More."
```

### 3. Rename Main Branch
```bash
git branch -M main
```

### 4. Create a New Repository on GitHub
1. Go to [github.com](https://github.com) and log in.
2. Click the **+** icon in the top right and select **New repository**.
3. Name your repository: `cashly` (or `Cashly`).
4. Set visibility to **Public**.
5. **Do not** check "Add a README file" (we already have a complete one).
6. Click **Create repository**.

### 5. Link Local Repository and Push
Copy your repository URL from GitHub and run:
```bash
git remote add origin https://github.com/<YOUR-USERNAME>/cashly.git
git push -u origin main
```

---

## ☁️ How to Deploy on Streamlit Community Cloud (Free)

Follow these steps to deploy Cashly publicly for free:

### 1. Sign In to Streamlit Community Cloud
1. Visit [share.streamlit.io](https://share.streamlit.io).
2. Log in using your **GitHub account**.

### 2. Deploy the App
1. Click the **"New app"** button.
2. Select your repository: `<YOUR-USERNAME>/cashly`.
3. Set **Branch**: `main`.
4. Set **Main file path**: `app.py` *(or `spendwise_ai/app.py` if your repository includes the outer folder)*.
5. Under **App URL**, customize your free subdomain if desired (e.g. `cashly.streamlit.app`).
6. Click **"Deploy!"**.

Streamlit Cloud will provision a Linux container, install `requirements.txt`, and launch your application with a public HTTPS link in 1–2 minutes!

---

## 🔒 Storage Architecture & Public Deployment Notes

### 1. Ephemeral SQLite Behavior on Streamlit Cloud
- **How it works:** Streamlit Community Cloud operates on stateless, ephemeral Linux containers.
- **Limitation:** Any records saved to the local SQLite database (`spendwise.db`) exist only for the lifetime of that container. When the app reboots, sleeps from inactivity, or is updated via a Git push, the container disk resets to the clean Git commit state.
- **Security & Privacy:** This public web deployment is configured as a **Shared Demonstration Instance** without individual user authentication. All public visitors share the same demo state. **Do not input real banking credentials, passwords, or confidential account numbers.**

### 2. Recommended Cloud Databases for Permanent Persistence
To upgrade Cashly to permanent, user-isolated cloud storage:
- **[Supabase](https://supabase.com/) (PostgreSQL)**: Generous free tier with built-in user authentication, Row-Level Security (RLS), and native Python client (`supabase-py`).
- **[Neon](https://neon.tech/) (Serverless Postgres)**: Free serverless PostgreSQL compatible with standard `SQLAlchemy` or `psycopg2`.
- **[Turso](https://turso.tech/) (LibSQL / Cloud SQLite)**: Serverless SQLite database in the cloud, allowing you to retain the exact same SQLite syntax via `libsql-client`.
- **Streamlit Google Sheets**: Simple zero-cost storage using `st.connection("gsheets")`.

---

## 🎓 College Mini-Project Viva & Presentation Guide

When presenting this project to professors or evaluators, highlight these core Data Science strengths:

1. **Why Chronological Splitting instead of Random Train-Test Split?**
   - *Answer:* In financial and time-series data, random shuffling causes **lookahead data leakage** (training on future information to predict past events). Chronological splitting strictly trains on earlier months to predict later holdout months.

2. **Why Mean Absolute Error (MAE) instead of Mean Squared Error (MSE)?**
   - *Answer:* MAE is expressed in the exact same monetary units as the expense currency (e.g., "our model has an average error of ₹1,200 per month"), making it intuitive and interpretable for end users, while being robust against occasional outlier expenses.

3. **How does the Data Sufficiency Check uphold Scientific Integrity?**
   - *Answer:* Many amateur apps fake predictions when given 2 transactions. Cashly enforces a 4-month minimum threshold, transparently alerting users when statistical power is insufficient rather than fabricating misleading numbers.

4. **What is the difference between Rule-Based Insights and ML Predictions?**
   - *Answer:* Rule-based insights apply established financial formulas (such as the Elizabeth Warren 50/30/20 standard and budget thresholds). The ML model learns statistical patterns (secular trends, monthly seasonality, and spending momentum) directly from historical spending data.

---

## 📄 License
This project is open-source and free to use for educational and personal finance tracking purposes.

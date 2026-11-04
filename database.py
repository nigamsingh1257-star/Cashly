"""
Cashly - SQLite Database Manager
Handles all database operations for expenses, budgets, and user settings.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime, date
import pandas as pd

# Default database path inside spendwise_ai/data/
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "spendwise.db"

CATEGORIES = [
    "Food",
    "Shopping",
    "Travel",
    "Bills",
    "Education",
    "Health",
    "Entertainment",
    "Others"
]

PAYMENT_METHODS = ["UPI / Online", "Credit Card", "Debit Card", "Cash", "Net Banking"]


def get_db_connection(db_path: Path = DEFAULT_DB_PATH):
    """Create and return a SQLite database connection with row factory enabled."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH):
    """Initialize database tables for expenses and user settings."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Table: expenses
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount > 0),
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            payment_method TEXT DEFAULT 'UPI / Online',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table: settings (monthly income, overall budget, currency, etc.)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Table: category_budgets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS category_budgets (
            category TEXT PRIMARY KEY,
            budget_amount REAL NOT NULL DEFAULT 0.0
        )
    """)

    conn.commit()

    # Insert default settings if not already present
    default_settings = {
        "monthly_income": "45000",
        "monthly_budget": "35000",
        "currency_symbol": "₹",
        "currency_code": "INR",
        "is_demo_loaded": "0"
    }

    for key, val in default_settings.items():
        cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, val))

    # Initialize category budgets defaults
    default_cat_budgets = {
        "Food": 10000.0,
        "Shopping": 6000.0,
        "Travel": 4000.0,
        "Bills": 5000.0,
        "Education": 3000.0,
        "Health": 3000.0,
        "Entertainment": 2500.0,
        "Others": 1500.0
    }
    for cat, bgt in default_cat_budgets.items():
        cursor.execute("INSERT OR IGNORE INTO category_budgets (category, budget_amount) VALUES (?, ?)", (cat, bgt))

    conn.commit()
    conn.close()


def add_expense(exp_date, amount: float, category: str, description: str, payment_method: str = "UPI / Online", db_path: Path = DEFAULT_DB_PATH) -> int:
    """Add a new expense record to the database."""
    if isinstance(exp_date, (datetime, date)):
        exp_date = exp_date.strftime("%Y-%m-%d")

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO expenses (date, amount, category, description, payment_method)
        VALUES (?, ?, ?, ?, ?)
    """, (str(exp_date), float(amount), str(category), str(description).strip(), str(payment_method)))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id


def update_expense(expense_id: int, exp_date, amount: float, category: str, description: str, payment_method: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Update an existing expense by ID."""
    if isinstance(exp_date, (datetime, date)):
        exp_date = exp_date.strftime("%Y-%m-%d")

    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE expenses
        SET date = ?, amount = ?, category = ?, description = ?, payment_method = ?
        WHERE id = ?
    """, (str(exp_date), float(amount), str(category), str(description).strip(), str(payment_method), int(expense_id)))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def delete_expense(expense_id: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Delete an expense record by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expenses WHERE id = ?", (int(expense_id),))
    conn.commit()
    rows_affected = cursor.rowcount
    conn.close()
    return rows_affected > 0


def get_expenses_df(start_date=None, end_date=None, categories=None, search=None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve filtered expenses as a pandas DataFrame."""
    conn = get_db_connection(db_path)
    query = "SELECT id, date, amount, category, description, payment_method, created_at FROM expenses WHERE 1=1"
    params = []

    if start_date:
        query += " AND date >= ?"
        params.append(str(start_date))

    if end_date:
        query += " AND date <= ?"
        params.append(str(end_date))

    if categories and len(categories) > 0:
        placeholders = ",".join(["?"] * len(categories))
        query += f" AND category IN ({placeholders})"
        params.extend(categories)

    if search and search.strip():
        query += " AND (description LIKE ? OR category LIKE ?)"
        term = f"%{search.strip()}%"
        params.extend([term, term])

    query += " ORDER BY date DESC, id DESC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    return df


def get_expense_by_id(expense_id: int, db_path: Path = DEFAULT_DB_PATH):
    """Retrieve single expense dict by ID."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM expenses WHERE id = ?", (int(expense_id),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def count_expenses(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Get total count of expense records."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM expenses")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def clear_all_expenses(db_path: Path = DEFAULT_DB_PATH):
    """Delete all expenses from the database (resets demo or user data)."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM expenses")
    cursor.execute("UPDATE settings SET value = '0' WHERE key = 'is_demo_loaded'")
    conn.commit()
    conn.close()


def get_setting(key: str, default: str = "", db_path: Path = DEFAULT_DB_PATH) -> str:
    """Get a configuration setting value."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str, db_path: Path = DEFAULT_DB_PATH):
    """Set or update a configuration setting."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
    conn.commit()
    conn.close()


def get_category_budgets(db_path: Path = DEFAULT_DB_PATH) -> dict:
    """Retrieve all category budgets as a dictionary {category: budget_amount}."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT category, budget_amount FROM category_budgets")
    rows = cursor.fetchall()
    conn.close()
    return {row["category"]: float(row["budget_amount"]) for row in rows}


def set_category_budget(category: str, amount: float, db_path: Path = DEFAULT_DB_PATH):
    """Set or update budget for a single category."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO category_budgets (category, budget_amount)
        VALUES (?, ?)
    """, (category, float(amount)))
    conn.commit()
    conn.close()

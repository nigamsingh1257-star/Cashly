"""
Cashly - Realistic Demo Data Generator
Generates multi-month realistic personal expense datasets for Data Science analytics & ML training.
"""

import random
from datetime import datetime, timedelta, date
from pathlib import Path
import pandas as pd
from database import add_expense, set_setting, clear_all_expenses, DEFAULT_DB_PATH

# Seed for reproducible realistic demo data
DEMO_RANDOM_SEED = 42

SAMPLE_TEMPLATES = {
    "Food": [
        ("Weekly Supermarket Grocery run", (800, 2400), ["Credit Card", "UPI / Online"]),
        ("Fresh vegetables & fruits market", (250, 650), ["Cash", "UPI / Online"]),
        ("Dinner with college friends at bistro", (600, 1500), ["Credit Card", "UPI / Online"]),
        ("Morning coffee & bagel at cafe", (120, 280), ["UPI / Online", "Cash"]),
        ("Online food delivery (Swiggy/Zomato)", (300, 750), ["UPI / Online"]),
        ("Office team lunch buffet", (450, 950), ["UPI / Online", "Debit Card"]),
        ("Bakery treats & sourdough bread", (180, 420), ["UPI / Online", "Cash"]),
    ],
    "Shopping": [
        ("Amazon electronics & USB-C hub", (700, 2200), ["Credit Card", "UPI / Online"]),
        ("Casual t-shirt & denim jeans", (900, 2500), ["Credit Card", "Debit Card"]),
        ("Home kitchen essentials & organizer", (450, 1200), ["UPI / Online"]),
        ("Running shoes replacement", (1800, 3800), ["Credit Card"]),
        ("Personal care & grooming kit", (350, 950), ["UPI / Online"]),
    ],
    "Travel": [
        ("Weekly metro commuter pass", (300, 600), ["UPI / Online", "Debit Card"]),
        ("Uber cab ride across town", (220, 580), ["UPI / Online"]),
        ("Vehicle petrol & fuel tank refill", (1200, 2500), ["Credit Card", "Debit Card"]),
        ("Weekend highway toll charges", (150, 350), ["UPI / Online", "Cash"]),
        ("Train ticket for hometown visit", (650, 1400), ["UPI / Online"]),
    ],
    "Bills": [
        ("Monthly high-speed fiber broadband", (799, 1199), ["UPI / Online", "Credit Card"]),
        ("Residential electricity utility bill", (1400, 2800), ["Net Banking", "UPI / Online"]),
        ("Mobile postpaid 5G recharge plan", (399, 799), ["UPI / Online"]),
        ("LPG cooking gas cylinder refill", (850, 1050), ["UPI / Online", "Cash"]),
        ("Water utility & society maintenance", (600, 1500), ["Net Banking", "UPI / Online"]),
    ],
    "Education": [
        ("Online Data Science specialization course", (999, 2499), ["Credit Card", "UPI / Online"]),
        ("Python & Machine Learning reference book", (450, 900), ["UPI / Online"]),
        ("College laboratory stationery & printouts", (150, 450), ["Cash", "UPI / Online"]),
        ("Cloud computing student lab credits", (350, 850), ["Credit Card"]),
    ],
    "Health": [
        ("Monthly fitness center / gym fee", (1200, 2000), ["UPI / Online", "Debit Card"]),
        ("Pharmacy prescription medicines & vitamins", (350, 1100), ["UPI / Online", "Cash"]),
        ("Routine dental checkup & cleaning", (800, 1500), ["UPI / Online", "Credit Card"]),
        ("Protein powder & health supplements", (1400, 2600), ["Credit Card", "UPI / Online"]),
    ],
    "Entertainment": [
        ("Netflix 4K Ultra subscription", (649, 649), ["Credit Card"]),
        ("Spotify Premium family music stream", (179, 179), ["Credit Card", "UPI / Online"]),
        ("Weekend IMAX movie tickets & popcorn", (550, 1200), ["Credit Card", "UPI / Online"]),
        ("Board game cafe evening with peers", (350, 750), ["UPI / Online"]),
    ],
    "Others": [
        ("Birthday present for close friend", (600, 1600), ["UPI / Online"]),
        ("Home plumbing hardware repair", (300, 750), ["Cash", "UPI / Online"]),
        ("Dry cleaning & winter jacket laundry", (250, 600), ["Cash", "UPI / Online"]),
    ]
}


def generate_sample_expenses(months_back: int = 7, db_path: Path = DEFAULT_DB_PATH) -> int:
    """
    Populates SQLite database with realistic multi-month expense records.
    Covers the last `months_back` months up to today.
    Returns the total count of inserted demo records.
    """
    random.seed(DEMO_RANDOM_SEED)
    today = date.today()
    start_date = today - timedelta(days=months_back * 30)

    inserted_count = 0
    current_day = start_date

    while current_day <= today:
        # Determine number of transactions for today (0 to 3 transactions)
        # Weekends (Saturday=5, Sunday=6) have slightly higher chance of transactions
        is_weekend = current_day.weekday() in (5, 6)
        num_transactions = random.choices([0, 1, 2, 3], weights=[0.25, 0.40, 0.25, 0.10] if not is_weekend else [0.10, 0.35, 0.35, 0.20])[0]

        # Monthly bill check on the 1st - 5th of each month
        if current_day.day in (2, 3, 5):
            # Schedule broadband or electricity
            if current_day.day == 2:
                tmpl = SAMPLE_TEMPLATES["Bills"][0]  # broadband
                amount = round(random.uniform(*tmpl[1]), 2)
                method = random.choice(tmpl[2])
                add_expense(current_day, amount, "Bills", tmpl[0], method, db_path=db_path)
                inserted_count += 1
            elif current_day.day == 5:
                tmpl = SAMPLE_TEMPLATES["Bills"][1]  # electricity
                amount = round(random.uniform(*tmpl[1]), 2)
                method = random.choice(tmpl[2])
                add_expense(current_day, amount, "Bills", tmpl[0], method, db_path=db_path)
                inserted_count += 1

        for _ in range(num_transactions):
            # Select category with realistic weights
            cat_weights = {
                "Food": 0.35,
                "Shopping": 0.15,
                "Travel": 0.18,
                "Bills": 0.08,
                "Education": 0.07,
                "Health": 0.08,
                "Entertainment": 0.06,
                "Others": 0.03
            }
            cat = random.choices(list(cat_weights.keys()), weights=list(cat_weights.values()))[0]
            options = SAMPLE_TEMPLATES[cat]
            tmpl = random.choice(options)

            # Weekend multiplier on food/entertainment
            multiplier = 1.15 if (is_weekend and cat in ["Food", "Entertainment"]) else 1.0
            amount = round(random.uniform(*tmpl[1]) * multiplier, 2)
            method = random.choice(tmpl[2])

            add_expense(current_day, amount, cat, tmpl[0], method, db_path=db_path)
            inserted_count += 1

        current_day += timedelta(days=1)

    set_setting("is_demo_loaded", "1", db_path=db_path)
    return inserted_count


def load_demo_dataset(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Clears existing data and loads fresh realistic demo dataset."""
    clear_all_expenses(db_path)
    count = generate_sample_expenses(months_back=7, db_path=db_path)
    return count

"""
Synthetic Banking/UPI Transaction Dataset Generator
Generates 100,000 realistic transactions with ~5% fraud rate
"""

import numpy as np
import pandas as pd
import uuid
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRANSACTION_TYPES = ["UPI", "NEFT", "IMPS", "RTGS", "Card", "NetBanking"]
MERCHANT_CATEGORIES = [
    "Grocery", "Electronics", "Restaurant", "Travel", "Entertainment",
    "Healthcare", "Clothing", "Fuel", "Education", "Utilities",
    "ATM_Withdrawal", "Online_Shopping", "Jewellery", "Unknown"
]
LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow",
    "Surat", "Patna", "Bhopal", "Indore", "Foreign"
]


def generate_user_profiles(n_users: int = 500) -> dict:
    """Generate user spending profiles."""
    profiles = {}
    for i in range(1, n_users + 1):
        avg_amount = random.uniform(500, 5000)
        profiles[i] = {
            "user_id": i,
            "name": f"User_{i:04d}",
            "account_number": f"ACCT{random.randint(1000000000, 9999999999)}",
            "home_location": random.choice(LOCATIONS[:-1]),
            "avg_transaction_amount": avg_amount,
            "primary_device": str(uuid.uuid4())[:8],
        }
    return profiles


def is_unusual_time(hour: int) -> int:
    """Flag transactions between 12 AM - 5 AM as unusual."""
    return 1 if 0 <= hour <= 5 else 0


def generate_transaction(
    transaction_id: str,
    user_id: int,
    profile: dict,
    timestamp: datetime,
    is_fraud: bool,
    user_daily_counts: dict,
    user_weekly_counts: dict,
    user_monthly_counts: dict,
) -> dict:
    """Generate a single transaction record."""
    hour = timestamp.hour
    day_key = timestamp.strftime("%Y-%m-%d")
    week_key = timestamp.strftime("%Y-W%W")
    month_key = timestamp.strftime("%Y-%m")

    uid = str(user_id)
    daily_count = user_daily_counts.get(f"{uid}_{day_key}", 0) + 1
    weekly_count = user_weekly_counts.get(f"{uid}_{week_key}", 0) + 1
    monthly_count = user_monthly_counts.get(f"{uid}_{month_key}", 0) + 1

    if is_fraud:
        # Fraud patterns
        fraud_pattern = random.choice(["high_amount", "unusual_time", "location_change", "velocity"])
        avg = profile["avg_transaction_amount"]

        if fraud_pattern == "high_amount":
            amount = avg * random.uniform(15, 50)
        elif fraud_pattern == "unusual_time":
            amount = avg * random.uniform(3, 10)
            hour = random.randint(0, 4)
            timestamp = timestamp.replace(hour=hour)
        elif fraud_pattern == "location_change":
            amount = avg * random.uniform(2, 8)
        else:  # velocity
            amount = avg * random.uniform(1.5, 5)
            daily_count = random.randint(10, 25)

        location = random.choice(LOCATIONS)
        device_id = str(uuid.uuid4())[:8] if random.random() > 0.3 else profile["primary_device"]
        merchant_category = random.choice(["Unknown", "Jewellery", "ATM_Withdrawal", "Electronics", "Online_Shopping"])
        transaction_type = random.choice(TRANSACTION_TYPES)
    else:
        # Normal transaction
        amount = abs(np.random.normal(profile["avg_transaction_amount"], profile["avg_transaction_amount"] * 0.3))
        amount = max(10, min(amount, profile["avg_transaction_amount"] * 5))
        location = profile["home_location"] if random.random() > 0.1 else random.choice(LOCATIONS[:-1])
        device_id = profile["primary_device"] if random.random() > 0.05 else str(uuid.uuid4())[:8]
        merchant_category = random.choice(MERCHANT_CATEGORIES[:-3])
        transaction_type = random.choices(TRANSACTION_TYPES, weights=[40, 15, 20, 5, 15, 5])[0]

    avg_amount = profile["avg_transaction_amount"]
    amount_deviation = (amount - avg_amount) / (avg_amount + 1e-6)
    device_change_flag = 1 if device_id != profile["primary_device"] else 0
    unusual_location_flag = 1 if location == "Foreign" or (location != profile["home_location"] and is_fraud) else 0

    return {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "transaction_amount": round(amount, 2),
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "transaction_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_hour": hour,
        "location": location,
        "device_id": device_id,
        "avg_transaction_amount": round(avg_amount, 2),
        "daily_transaction_count": daily_count,
        "weekly_transaction_count": weekly_count,
        "monthly_transaction_count": monthly_count,
        "amount_deviation_from_average": round(amount_deviation, 4),
        "transaction_velocity": daily_count,
        "unusual_time_flag": is_unusual_time(hour),
        "unusual_location_flag": unusual_location_flag,
        "device_change_flag": device_change_flag,
        "fraud_label": 1 if is_fraud else 0,
    }


def generate_dataset(n_records: int = 100_000, fraud_rate: float = 0.05, seed: int = 42) -> pd.DataFrame:
    """Generate complete synthetic transaction dataset."""
    np.random.seed(seed)
    random.seed(seed)

    logger.info(f"Generating {n_records:,} transactions with {fraud_rate*100:.0f}% fraud rate...")
    profiles = generate_user_profiles(500)
    user_ids = list(profiles.keys())

    n_fraud = int(n_records * fraud_rate)
    n_normal = n_records - n_fraud
    fraud_flags = [False] * n_normal + [True] * n_fraud
    random.shuffle(fraud_flags)

    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    user_daily_counts: dict = {}
    user_weekly_counts: dict = {}
    user_monthly_counts: dict = {}

    records = []
    for i, is_fraud in enumerate(fraud_flags):
        transaction_id = f"TXN{str(uuid.uuid4()).replace('-','')[:12].upper()}"
        user_id = random.choice(user_ids)
        profile = profiles[user_id]
        days_offset = random.randint(0, date_range)
        hours_offset = random.randint(0, 23)
        minutes_offset = random.randint(0, 59)
        timestamp = start_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)

        uid = str(user_id)
        day_key = timestamp.strftime("%Y-%m-%d")
        week_key = timestamp.strftime("%Y-W%W")
        month_key = timestamp.strftime("%Y-%m")

        record = generate_transaction(
            transaction_id, user_id, profile, timestamp, is_fraud,
            user_daily_counts, user_weekly_counts, user_monthly_counts,
        )
        records.append(record)

        user_daily_counts[f"{uid}_{day_key}"] = user_daily_counts.get(f"{uid}_{day_key}", 0) + 1
        user_weekly_counts[f"{uid}_{week_key}"] = user_weekly_counts.get(f"{uid}_{week_key}", 0) + 1
        user_monthly_counts[f"{uid}_{month_key}"] = user_monthly_counts.get(f"{uid}_{month_key}", 0) + 1

        if (i + 1) % 10000 == 0:
            logger.info(f"  Generated {i+1:,} / {n_records:,} records...")

    df = pd.DataFrame(records)
    actual_fraud_rate = df["fraud_label"].mean()
    logger.info(f"Dataset generated: {len(df):,} records, fraud rate: {actual_fraud_rate*100:.2f}%")
    return df


if __name__ == "__main__":
    df = generate_dataset(100_000, 0.05)
    output_path = "transactions.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    print("\nDataset Summary:")
    print(df.describe())
    print(f"\nFraud distribution:\n{df['fraud_label'].value_counts()}")
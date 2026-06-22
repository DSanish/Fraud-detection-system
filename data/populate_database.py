import sys
import os
import pandas as pd

# Project root add karo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection import SessionLocal
from database.models import User, Transaction

print("Loading CSV...")

df = pd.read_csv("transactions.csv")

db = SessionLocal()

print(f"Found {len(df)} records")

# -----------------------------
# Users Insert
# -----------------------------
print("Creating users...")

existing_users = {u.user_id for u in db.query(User).all()}

for uid in df["user_id"].unique():

    if uid not in existing_users:

        user = User(
            user_id=int(uid),
            name=f"User_{uid}",
            account_number=f"ACC{uid:06d}",
            email=f"user{uid}@mail.com",
            phone="9999999999"
        )

        db.add(user)

db.commit()

print("Users inserted")

# -----------------------------
# Transactions Insert
# -----------------------------
print("Creating transactions...")

count = 0

for _, row in df.iterrows():

    txn = Transaction(
        transaction_id=row["transaction_id"],
        user_id=int(row["user_id"]),
        amount=float(row["transaction_amount"]),
        transaction_type=row["transaction_type"],
        merchant_category=row["merchant_category"],
        location=row["location"],
        device_id=row["device_id"],
        timestamp=pd.to_datetime(row["transaction_time"]),
        prediction="Fraud" if row["fraud_label"] == 1 else "Legitimate",
        fraud_probability=0.95 if row["fraud_label"] == 1 else 0.05,
        risk_level="High" if row["fraud_label"] == 1 else "Low",
        avg_transaction_amount=float(row["avg_transaction_amount"]),
        daily_transaction_count=int(row["daily_transaction_count"]),
        unusual_time_flag=bool(row["unusual_time_flag"]),
        unusual_location_flag=bool(row["unusual_location_flag"]),
        device_change_flag=bool(row["device_change_flag"]),
        is_confirmed_fraud=bool(row["fraud_label"])
    )

    db.add(txn)

    count += 1

    if count % 1000 == 0:
        db.commit()
        print(f"{count} inserted...")

db.commit()

print("===================================")
print("Database populated successfully.")
print("===================================")

db.close()
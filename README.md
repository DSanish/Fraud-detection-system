# 🛡️ AI-Powered Banking Fraud Detection System

> A production-ready AI-powered Banking Fraud Detection System built with **FastAPI, XGBoost, PostgreSQL (Neon Cloud), SQLAlchemy, and Streamlit** for real-time fraud prediction, monitoring, and analytics.

---

# 🚀 Live Demo

### 🌐 Live Backend API
https://fraud-detection-system-p2jr.onrender.com

### 📑 Swagger Documentation
https://fraud-detection-system-p2jr.onrender.com/docs

### 📊 Live Dashboard
https://fraud-dashboard-5qfa.onrender.com/

---

# 📌 Project Overview

Financial fraud is increasing rapidly in digital banking and UPI transactions. This project uses Machine Learning to detect fraudulent transactions in real time.

The system predicts fraud probability using an XGBoost model and provides an interactive dashboard for monitoring transactions, fraud alerts, analytics, and model performance.

---

# ✨ Features

✅ Real-time Fraud Prediction

✅ FastAPI REST API

✅ XGBoost Machine Learning Model

✅ PostgreSQL (Neon Cloud Database)

✅ SQLAlchemy ORM

✅ Interactive Streamlit Dashboard

✅ Fraud Alerts Engine

✅ Live Transaction Monitoring

✅ Model Performance Metrics

✅ Risk Level Classification

✅ REST API Documentation (Swagger)

✅ Cloud Deployment on Render

---

# 🏗️ System Architecture

```
                 User
                   │
                   ▼
        Streamlit Dashboard
                   │
                   ▼
             FastAPI Backend
                   │
     ┌─────────────┴─────────────┐
     ▼                           ▼
 XGBoost Model              PostgreSQL
 Fraud Prediction          Transaction DB
     │                           │
     └─────────────┬─────────────┘
                   ▼
             Alert Engine
                   │
                   ▼
         Fraud Detection Result
```

---

# ⚙️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| FastAPI | REST API |
| Streamlit | Dashboard |
| PostgreSQL | Database |
| Neon Cloud | Cloud Database |
| SQLAlchemy | ORM |
| XGBoost | Machine Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Plotly | Charts |
| Git | Version Control |
| GitHub | Source Code |
| Render | Deployment |

---

# 📁 Project Structure

```
fraud_detection_system/

│
├── api/
│   ├── routes/
│   ├── schemas/
│   ├── services/
│   └── main.py
│
├── alerts/
│
├── dashboard/
│   └── app.py
│
├── database/
│   ├── connection.py
│   ├── crud.py
│   └── models.py
│
├── training/
│
├── data/
│
├── models/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Database Schema

### Users

- user_id
- name
- account_number
- email
- phone

### Transactions

- transaction_id
- amount
- merchant_category
- prediction
- fraud_probability
- risk_level

### Alerts

- alert_id
- transaction_id
- fraud_probability
- risk_level
- status

---

# 🔥 REST API Endpoints

## Prediction

POST

```
/api/v1/predict
```

Predicts fraud probability.

---

## Health

GET

```
/api/v1/health
```

Returns API and database status.

---

## Metrics

GET

```
/api/v1/metrics
```

Returns model performance.

---

## Transactions

GET

```
/api/v1/transactions
```

Returns latest transactions.

---

## Alerts

GET

```
/api/v1/alerts
```

Returns fraud alerts.

---

# 🤖 Machine Learning Model

Model Used

- XGBoost Classifier

Features

- Transaction Amount
- Transaction Type
- Merchant Category
- Transaction Hour
- Daily Transaction Count
- Weekly Transaction Count
- Monthly Transaction Count
- Device Change Flag
- Unusual Location Flag
- Unusual Time Flag

Output

- Fraud Probability
- Prediction
- Risk Level

---

# 📈 Model Performance

| Metric | Score |
|---------|--------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |
| ROC-AUC | 100% |

---

# ☁️ Deployment

Backend

- Render

Database

- PostgreSQL (Neon Cloud)

Dashboard

- Streamlit

---

# ▶️ Installation

Clone repository

```bash
git clone https://github.com/DSanish/Fraud-detection-system.git
```

Move into project

```bash
cd Fraud-detection-system
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run API

```bash
uvicorn api.main:app --reload
```

Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 🎯 Future Improvements

- JWT Authentication
- Login / Signup
- User Management
- Admin Dashboard
- Email Notifications
- SMS Alerts
- WebSocket Live Monitoring
- Docker Deployment
- Kubernetes
- CI/CD Pipeline
- Power BI Integration
- Explainable AI (SHAP)
- Audit Logs
- Fraud Investigation Workflow

---

# 👨‍💻 Author

**Anish kumar**

GitHub

https://github.com/DSanish

LinkedIn

https://www.linkedin.com/in/anish-kumar-072022275/

---
## Docker Support

Run the complete application using Docker Compose.

```bash
docker compose up --build


# ⭐ If you like this project, don't forget to Star this repository.

"""
Prediction Router — /predict, /health, /metrics, /transactions, /alerts
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional


from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from api.schemas.transaction import (
    TransactionRequest, PredictionResponse, HealthResponse, MetricsResponse
)
from api.services.prediction_service import prediction_service
from alerts.alert_engine import alert_engine
from database.connection import get_db
from database import crud

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(
    request: TransactionRequest,
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """
    Predict fraud probability for a banking/UPI transaction.

    - Runs XGBoost model inference
    - Generates alert if fraud probability > 0.50
    - Persists transaction and alert to PostgreSQL
    """
    try:
        txn_data = request.dict()
        if not txn_data.get("transaction_id"):
            txn_data["transaction_id"] = f"TXN_{uuid.uuid4().hex[:12].upper()}"

        # ML prediction
        result = prediction_service.predict(txn_data)

        # Alert evaluation
        alert_result = alert_engine.evaluate(
            transaction_data=txn_data,
            fraud_probability=result["fraud_probability"],
            prediction=result["prediction"],
        )

        # Persist transaction
        try:
            transaction_record = {
                "transaction_id": result["transaction_id"],
                "user_id": request.user_id or 1,
                "amount": request.transaction_amount,
                "transaction_type": request.transaction_type,
                "merchant_category": request.merchant_category,
                "prediction": result["prediction"],
                "fraud_probability": result["fraud_probability"],
                "risk_level": result["risk_level"],
                "avg_transaction_amount": request.avg_transaction_amount,
                "daily_transaction_count": request.daily_transaction_count,
                "unusual_time_flag": bool(request.unusual_time_flag),
                "unusual_location_flag": bool(request.unusual_location_flag),
                "device_change_flag": bool(request.device_change_flag),
            }
            logger.info(f"Transaction Record: {transaction_record}")
            crud.create_transaction(db, transaction_record)
            

            # Persist alert if triggered
            if alert_result.should_alert and alert_result.risk_level:
                crud.create_alert(
                    db=db,
                    transaction_id=result["transaction_id"],
                    risk_level=alert_result.risk_level,
                    fraud_probability=result["fraud_probability"],
                    message=alert_result.message,
                )
        except Exception as db_err:
            logger.warning(f"DB write failed (non-fatal): {db_err}")

        return PredictionResponse(
            transaction_id=result["transaction_id"],
            fraud_probability=result["fraud_probability"],
            prediction=result["prediction"],
            risk_level=result["risk_level"],
            alert_triggered=alert_result.should_alert,
            alert_message=alert_result.message if alert_result.should_alert else None,
            processing_time_ms=result["processing_time_ms"],
            timestamp=result["timestamp"],
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


# @router.get("/health", response_model=HealthResponse, tags=["System"])
# async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
#     """Health check for load balancers and monitoring."""
#     db_connected = False
#     try:
#         db.execute("SELECT 1")
#         db_connected = True
#     except Exception:
#         pass

#     return HealthResponse(
#         status="healthy" if prediction_service.is_ready and db_connected else "degraded",
#         model_loaded=prediction_service.is_ready,
#         database_connected=db_connected,
#         version="1.0.0",
#         timestamp=datetime.utcnow(),
#     )
@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    """Health check for load balancers and monitoring."""
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception as e:
        print("DATABASE ERROR:", e)

    return HealthResponse(
        status="healthy" if prediction_service.is_ready and db_connected else "degraded",
        model_loaded=prediction_service.is_ready,
        database_connected=db_connected,
        version="1.0.0",
        timestamp=datetime.utcnow(),
    )


@router.get("/metrics", tags=["Model"])
async def get_model_metrics(db: Session = Depends(get_db)):
    """
    Dashboard Metrics
    """

    stats = crud.get_dashboard_stats(db)

    model = prediction_service.get_metrics()

    return {
        "total_transactions": stats["total_transactions"],
        "fraud_count": stats["fraud_count"],
        "fraud_rate": stats["fraud_rate"],

        "accuracy": model["accuracy"],
        "precision": model["precision"],
        "recall": model["recall"],
        "f1_score": model["f1_score"],
        "roc_auc": model["roc_auc"],

        "model_version": model["model_version"]
    }

@router.get("/transactions", tags=["Data"])
async def list_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    risk_level: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """List recent transactions with optional risk_level filter."""
    if risk_level:
        txns = crud.get_transactions_by_risk(db, risk_level=risk_level, limit=limit)
    else:
        txns = crud.get_recent_transactions(db, limit=limit)
    return [
        {
            "transaction_id": t.transaction_id,
            "amount": t.amount,
            "prediction": t.prediction,
            "fraud_probability": t.fraud_probability,
            "risk_level": t.risk_level,
            "timestamp": t.timestamp,
        }
        for t in txns
    ]


@router.get("/alerts", tags=["Alerts"])
async def list_alerts(
    limit: int = Query(default=50, ge=1, le=200),
    resolved: Optional[bool] = Query(default=None),
    db: Session = Depends(get_db),
):
    """List fraud alerts with optional resolved filter."""
    alerts = crud.get_recent_alerts(db, limit=limit, resolved=resolved)
    return [
        {
            "alert_id": a.alert_id,
            "transaction_id": a.transaction_id,
            "risk_level": a.risk_level,
            "fraud_probability": a.fraud_probability,
            "message": a.alert_message,
            "is_resolved": a.is_resolved,
            "created_at": a.created_at,
        }
        for a in alerts
    ]


@router.get("/stats", tags=["Data"])
async def get_stats(db: Session = Depends(get_db)):
    """Return dashboard statistics."""
    return crud.get_dashboard_stats(db)


@router.patch("/alerts/{alert_id}/resolve", tags=["Alerts"])
async def resolve_alert(
    alert_id: int,
    resolved_by: str = Query(default="admin"),
    db: Session = Depends(get_db),
):
    """Mark a fraud alert as resolved."""
    alert = crud.resolve_alert(db, alert_id=alert_id, resolved_by=resolved_by)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return {"message": f"Alert {alert_id} resolved", "resolved_at": alert.resolved_at}
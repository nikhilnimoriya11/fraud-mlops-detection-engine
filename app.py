import json
import math
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Real-Time Fraud & Anomaly Detection Engine",
    description="Low-latency inference API with statistical data drift monitoring.",
    version="1.0.0"
)

# Load reference data
try:
    with open("reference_data.json", "r") as f:
        reference_data = json.load(f)
except FileNotFoundError:
    reference_data = []

production_logs = []

class Transaction(BaseModel):
    v1: float
    v2: float
    amount: float

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

@app.get("/")
def health_check():
    return {"status": "online", "model_version": "1.0.0"}

@app.post("/predict")
def predict_fraud(tx: Transaction):
    # Pure mathematical linear decision boundary for fraud scoring
    # Weights: v1 (+1.5), v2 (-1.8), amount (+0.01), bias (-3.0)
    z = (1.5 * tx.v1) + (-1.8 * tx.v2) + (0.01 * tx.amount) - 3.0
    prob = sigmoid(z)
    is_fraud = prob > 0.5
    
    # Store payload for drift monitoring
    production_logs.append({"v1": tx.v1, "v2": tx.v2, "amount": tx.amount})
    
    return {
        "is_fraud": is_fraud,
        "fraud_probability": round(prob, 4),
        "status": "FLAGGED_HIGH_RISK" if is_fraud else "APPROVED"
    }

@app.get("/drift-status")
def check_drift():
    if len(production_logs) < 5:
        return {
            "status": "insufficient_data",
            "message": f"Requires at least 5 logged requests. Currently logged: {len(production_logs)}"
        }
    
    # Calculate baseline reference stats
    ref_count = len(reference_data)
    ref_v1_mean = sum(x["v1"] for x in reference_data) / ref_count
    ref_v2_mean = sum(x["v2"] for x in reference_data) / ref_count
    ref_amt_mean = sum(x["amount"] for x in reference_data) / ref_count
    
    # Calculate production logs stats
    prod_count = len(production_logs)
    prod_v1_mean = sum(x["v1"] for x in production_logs) / prod_count
    prod_v2_mean = sum(x["v2"] for x in production_logs) / prod_count
    prod_amt_mean = sum(x["amount"] for x in production_logs) / prod_count

    drift_results = {
        "v1": {"drift_detected": abs(prod_v1_mean - ref_v1_mean) > 1.5},
        "v2": {"drift_detected": abs(prod_v2_mean - ref_v2_mean) > 1.5},
        "amount": {"drift_detected": abs(prod_amt_mean - ref_amt_mean) > 30.0}
    }
    
    any_drift = any(v["drift_detected"] for v in drift_results.values())

    return {
        "dataset_drift_detected": any_drift,
        "feature_drift_metrics": drift_results,
        "total_logged_requests": prod_count
    }
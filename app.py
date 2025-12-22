from fastapi import FastAPI, HTTPException
from datetime import datetime
import pickle
import os
import pandas as pd

app = FastAPI(title="Inventory Forecast API")

PRODUCT_ID_TO_INDEX = {
    20: 0,
    120: 1,
    150: 2,
    153: 3,
    165: 4,
    298: 5,
    310: 6,
    375: 7,
    437: 8,
    448: 9
}

PROPHET_MODELS = {}
QTY_SCALER = None


@app.on_event("startup")
def load_models():
    global QTY_SCALER

    # Load quantity scaler
    with open("models/qty_scaler.pkl", "rb") as f:
        QTY_SCALER = pickle.load(f)

    # Load Prophet models only
    for product_id, idx in PRODUCT_ID_TO_INDEX.items():
        prophet_path = f"models/prophet_model_{idx}.pkl"

        if not os.path.exists(prophet_path):
            raise RuntimeError(f"Missing model file: {prophet_path}")

        with open(prophet_path, "rb") as f:
            PROPHET_MODELS[idx] = pickle.load(f)

    print("Prophet models and scaler loaded successfully")


@app.get("/")
def root():
    return {"message": "Inventory Forecast API is running"}


@app.get("/predict")
def predict(product_id: int, date: str):
    """
    Predict quantity for a given product_id and date (YYYY-MM-DD)
    """

    # Validate product_id
    if product_id not in PRODUCT_ID_TO_INDEX:
        raise HTTPException(status_code=400, detail="Invalid product_id")

    product_index = PRODUCT_ID_TO_INDEX[product_id]

    # Parse date
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Date must be in YYYY-MM-DD format"
        )

    # Get Prophet model
    model = PROPHET_MODELS.get(product_index)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # -------------------------
    # PREDICTION LOGIC (PROPHET)
    # -------------------------
    history_df = model.history.copy()
    history_df["ds"] = pd.to_datetime(history_df["ds"]).dt.date
    last_date = history_df["ds"].max()

    # Historical prediction
    if target_date <= last_date:
        full_pred = model.predict(model.history)
        full_pred["ds"] = pd.to_datetime(full_pred["ds"]).dt.date
        value = full_pred.loc[full_pred["ds"] == target_date, "yhat"]

        if len(value) > 0:
            scaled_pred = value.values[0]
        else:
            scaled_pred = history_df["y"].iloc[-1]

    # Future prediction
    else:
        steps = (target_date - last_date).days
        future_df = model.make_future_dataframe(
            periods=steps,
            freq="D",
            include_history=False
        )
        forecast = model.predict(future_df)
        scaled_pred = forecast["yhat"].iloc[-1]

    # Inverse scale
    real_pred = QTY_SCALER.inverse_transform([[scaled_pred]])[0][0]
    real_pred = max(1, int(round(real_pred)))

    return {
        "product_id": product_id,
        "date": date,
        "model_used": "prophet",
        "predicted_quantity": real_pred
    }

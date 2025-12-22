import os
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =========================
# LOAD SCALER
# =========================
scaler_path = os.path.join(MODELS_DIR, "qty_scaler.pkl")
with open(scaler_path, "rb") as f:
    qty_scaler = pickle.load(f)

# =========================
# LOAD PROPHET MODELS
# =========================
final_models = {}

for file in os.listdir(MODELS_DIR):
    if file.startswith("prophet_model_") and file.endswith(".pkl"):
        pid = int(file.replace("prophet_model_", "").replace(".pkl", ""))
        with open(os.path.join(MODELS_DIR, file), "rb") as f:
            final_models[pid] = pickle.load(f)

# =========================
# PREDICTION FUNCTION
# =========================
def predict_quantity(product_id, date_str):

    model = final_models[product_id]
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    history_df = model.history.copy()
    history_df["ds"] = pd.to_datetime(history_df["ds"]).dt.date
    last_date = history_df["ds"].max()

    # -------------------------
    # HISTORICAL DATE
    # -------------------------
    if target_date <= last_date:
        full_pred = model.predict(model.history)
        full_pred["ds"] = pd.to_datetime(full_pred["ds"]).dt.date

        value = full_pred.loc[full_pred["ds"] == target_date, "yhat"]
        if len(value) > 0:
            scaled_pred = value.values[0]
        else:
            scaled_pred = history_df["y"].iloc[-1]  # fallback

    # -------------------------
    # FUTURE DATE
    # -------------------------
    else:
        steps = (target_date - last_date).days
        future_df = model.make_future_dataframe(
            periods=steps,
            freq="D",
            include_history=False
        )
        forecast = model.predict(future_df)
        scaled_pred = forecast["yhat"].iloc[-1]

    # -------------------------
    # INVERSE SCALE
    # -------------------------
    real_pred = qty_scaler.inverse_transform([[scaled_pred]])[0][0]
    real_pred = max(1, int(round(real_pred)))

    return real_pred


# =========================
# LOCAL TEST
# =========================
if __name__ == "__main__":
    pid = 3
    date = "2024-01-01"
    pred = predict_quantity(pid, date)
    print(f"Predicted quantity for Product {pid} on {date}: {pred}")

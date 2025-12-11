# Prediction.py
import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

scaler_path = os.path.join(MODELS_DIR, "qty_scaler.pkl")
with open(scaler_path, "rb") as f:
    qty_scaler = pickle.load(f)

final_models = {}
PROPHET_PRODUCTS = [0, 1, 2, 6, 7, 8, 9]
SARIMA_PRODUCTS = [3, 4, 5]

for pid in PROPHET_PRODUCTS:
    model_path = os.path.join(MODELS_DIR, f"prophet_model_{pid}.pkl")
    with open(model_path, "rb") as f:
        final_models[pid] = ("prophet", pickle.load(f))

for pid in SARIMA_PRODUCTS:
    model_path = os.path.join(MODELS_DIR, f"sarima_model_{pid}.pkl")
    with open(model_path, "rb") as f:
        final_models[pid] = ("sarima", pickle.load(f))

from datetime import datetime, timedelta

def predict_quantity(product_id, date_str):
    model_type, model = final_models[product_id]
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    if model_type == "prophet":
        # Compute days ahead from last training date
        last_train_date = model.history['ds'].max().date()
        steps_ahead = (target_date - last_train_date).days
        if steps_ahead < 1:
            steps_ahead = 1
        future = model.make_future_dataframe(periods=steps_ahead, freq='D', include_history=False)
        forecast = model.predict(future)
        scaled_pred = forecast['yhat'].values[-1]

    elif model_type == "sarima":
        # Compute steps ahead
        # SARIMA last training date assumed same as index length
        steps_ahead = (target_date - model.arima_res_.index[-1].date()).days if hasattr(model.arima_res_, 'index') else 1
        steps_ahead = max(1, steps_ahead)
        scaled_pred = model.predict(n_periods=steps_ahead)[-1]

    # Inverse scale to real units
    real_pred = qty_scaler.inverse_transform(np.array([[scaled_pred]])).flatten()[0]
    real_pred = max(1, int(round(real_pred)))  # Clip to min 1 and round

    return real_pred

if __name__ == "__main__":
    pid = 0
    date = "2025-12-05"
    pred = predict_quantity(pid, date)
    print(f"Predicted quantity for Product {pid} on {date}: {pred}")

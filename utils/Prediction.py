# Prediction.py
import os
import pickle
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load qty scaler
scaler_path = os.path.join(MODELS_DIR, "qty_scaler.pkl")
with open(scaler_path, "rb") as f:
    qty_scaler = pickle.load(f)

# Load saved models
final_models = {}
PROPHET_PRODUCTS = [0, 1, 2, 6, 7, 8, 9]
SARIMA_PRODUCTS = [3, 4, 5]

for pid in PROPHET_PRODUCTS:
    with open(os.path.join(MODELS_DIR, f"prophet_model_{pid}.pkl"), "rb") as f:
        final_models[pid] = ("prophet", pickle.load(f))

for pid in SARIMA_PRODUCTS:
    with open(os.path.join(MODELS_DIR, f"sarima_model_{pid}.pkl"), "rb") as f:
        final_models[pid] = ("sarima", pickle.load(f))


def predict_quantity(product_id, date_str):

    model_type, model = final_models[product_id]
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    if model_type == "prophet":

        history_df = model.history.copy()
        history_df['ds'] = pd.to_datetime(history_df['ds']).dt.date

        last_date = history_df['ds'].max()

        if target_date <= last_date:
            full_pred = model.predict(model.history)
            full_pred['ds'] = pd.to_datetime(full_pred['ds']).dt.date
            value = full_pred.loc[full_pred['ds'] == target_date, 'yhat']
            if len(value) > 0:
                scaled_pred = value.values[0]
            else:
                scaled_pred = history_df['y'].iloc[-1]  # fallback
        else:
            # Future prediction
            steps = (target_date - last_date).days
            future_df = model.make_future_dataframe(periods=steps, freq='D', include_history=False)
            forecast = model.predict(future_df)
            scaled_pred = forecast['yhat'].iloc[-1]

    
    else:

        # Extract last train date if available
        try:
            sarima_index = model.arima_res_.data.dates
            last_date = sarima_index[-1].date()
        except:
            last_date = datetime.today().date()

        # SARIMA CANNOT give historical prediction → use future only
        steps = (target_date - last_date).days
        steps = max(1, steps)

        scaled_pred = model.predict(n_periods=steps)[-1]


    real_pred = qty_scaler.inverse_transform([[scaled_pred]])[0][0]
    real_pred = max(1, int(round(real_pred)))

    return real_pred


if __name__ == "__main__":
    pid = 0
    date = "2024-01-01"
    pred = predict_quantity(pid, date)
    print(f"Predicted quantity for Product {pid} on {date}: {pred}")

from utils.Prediction import predict_quantity
from datetime import datetime

print("=== Inventory Forecast ===")

# Validate Product ID
while True:
    try:
        product_id = int(input("Enter Product ID (0-9): "))
        if product_id < 0 or product_id > 9:
            raise ValueError
        break
    except ValueError:
        print("Invalid Product ID! Enter an integer between 0 and 9.")

# Validate Date
while True:
    target_date = input("Enter date (YYYY-MM-DD): ")
    try:
        datetime.strptime(target_date, "%Y-%m-%d")
        break
    except ValueError:
        print("Invalid date format! Use YYYY-MM-DD.")

#  Predict
predicted_qty = predict_quantity(product_id, target_date)
print(f"\nPredicted quantity for Product {product_id} on {target_date}: {predicted_qty}")

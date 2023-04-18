import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load data from the JSON file
data_file = "./dump/evaluation_results_20230417_224319.json"
with open(data_file, 'r') as file:
    data = json.load(file)

# Extract actual and predicted values
def convert_value(val):
    try:
        return float(val)
    except ValueError:
        return val
actual_values = [list(map(convert_value, entry['actual'].strip('[]').split(', '))) for entry in data]
predicted_values = [list(map(convert_value, entry['response'].strip('[]').split(', '))) for entry in data]

# Check if the predicted date is correct
def check_dates(actual_values, predicted_values):
    correct_dates = 0
    for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
        if actual[0] == predicted[0]:
            correct_dates += 1
    return correct_dates

# Check if the predicted direction is correct
def check_direction(actual_values, predicted_values):
    correct_direction = 0
    for actual, predicted in zip(actual_values, predicted_values):
        actual_direction = np.sign(actual[7])
        predicted_direction = np.sign(predicted[7])
        if actual_direction == predicted_direction:
            correct_direction += 1
    return correct_direction

# Calculate regression metrics for each value (open, high, low, close, and volume)
metrics = {}
for i, name in enumerate(["Open", "High", "Low", "Close", "Volume"]):
    actual_col = [row[i + 1] for row in actual_values]
    predicted_col = [row[i + 1] for row in predicted_values]
    
    mae = mean_absolute_error(actual_col, predicted_col)
    mse = mean_squared_error(actual_col, predicted_col)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_col, predicted_col)

    metrics[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

for name, values in metrics.items():
    print(f"{name}:")
    for metric, value in values.items():
        print(f"  {metric}: {value}")
    print()

correct_dates = check_dates(actual_values, predicted_values)
correct_direction = check_direction(actual_values, predicted_values)

print(f"Correct Dates: {correct_dates}/{len(actual_values)}")
print(f"Correct Direction: {correct_direction}/{len(actual_values)}")


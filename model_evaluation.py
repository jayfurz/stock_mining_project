import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data from the JSON file
data_file = "./dump/evaluation_results_20230418_134035.json"
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

# Function to calculate Mean Absolute Percentage Error
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_changes(predicted_values):
    changes = []
    for row in predicted_values:
        open_price = row[1]
        close_price = row[4]
        absolute_change = close_price - open_price
        percent_change = absolute_change / open_price 
        changes.append((percent_change, absolute_change))
    return changes

def check_changes(actual_values, predicted_values):
    correct_percent_changes = 0
    correct_absolute_changes = 0
    total = len(actual_values)

    predicted_changes = calculate_changes(predicted_values)

    for i in range(total):
        actual_percent_change = actual_values[i][-2]
        actual_absolute_change = actual_values[i][-1]

        predicted_percent_change = predicted_changes[i][0]
        predicted_absolute_change = predicted_changes[i][1]

        if round(predicted_values[i][-2], 3) == round(predicted_percent_change, 3):
            correct_percent_changes += 1

        if round(predicted_values[i][-1], 1) == round(predicted_absolute_change, 1):
            correct_absolute_changes += 1

    return correct_percent_changes, correct_absolute_changes, total

predicted_changes = calculate_changes(predicted_values)

for i in range(len(predicted_values)):
    predicted_values[i].extend(predicted_changes[i][:-2])
# Calculate regression metrics for each value (open, high, low, close, and volume)
metrics = {}
for i, name in enumerate(["Open", "High", "Low", "Close", "AdjClose", "Volume", "PercentChange", "AbsChange"]):
    actual_col = [row[i + 1] for row in actual_values]
    predicted_col = [row[i + 1] for row in predicted_values]
    
    mae = mean_absolute_error(actual_col, predicted_col)
    mse = mean_squared_error(actual_col, predicted_col)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_col, predicted_col)
    mape = mean_absolute_percentage_error(actual_col, predicted_col)

    metrics[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2, "MAPE": mape}

for name, values in metrics.items():
    print(f"{name}:")
    for metric, value in values.items():
        print(f"  {metric}: {value}")
    print()

correct_dates = check_dates(actual_values, predicted_values)
correct_direction = check_direction(actual_values, predicted_values)

print(f"Correct Dates: {correct_dates}/{len(actual_values)}")
print(f"Correct Direction: {correct_direction}/{len(actual_values)}")

correct_percent_changes, correct_absolute_changes, total = check_changes(actual_values, predicted_values)

print(f"Correct percent changes: {correct_percent_changes}/{total} ({(correct_percent_changes / total) * 100:.2f}%)")
print(f"Correct absolute changes: {correct_absolute_changes}/{total} ({(correct_absolute_changes / total) * 100:.2f}%)")

actual_close_prices = [actual[3] for actual in actual_values]
predicted_close_prices = [predicted[3] for predicted in predicted_values]
actual_daily_returns = pd.Series(actual_close_prices).pct_change()
predicted_daily_returns = pd.Series(predicted_close_prices).pct_change()
actual_cumulative_returns = (1 + actual_daily_returns).cumprod() - 1
predicted_cumulative_returns = (1 + predicted_daily_returns).cumprod() - 1
annual_risk_free_rate = 0.03
daily_risk_free_rate = (1 + annual_risk_free_rate) ** (1 / 252) - 1
actual_excess_returns = actual_daily_returns - daily_risk_free_rate
predicted_excess_returns = predicted_daily_returns - daily_risk_free_rate
actual_sharpe_ratio = np.mean(actual_excess_returns) / np.std(actual_excess_returns)
predicted_sharpe_ratio = np.mean(predicted_excess_returns) / np.std(predicted_excess_returns)
print(f"Actual sharpe ratio: {actual_sharpe_ratio}")
print(f"Predicted sharpe ratio: {predicted_sharpe_ratio}")

plt.plot(actual_cumulative_returns, label="Actual Cumulative Returns")
plt.plot(predicted_cumulative_returns, label="Predicted Cumulative Returns")
plt.xlabel("Days")
plt.ylabel("Cumulative Returns")
plt.legend()
plt.show()


import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def generate_sequences(self, sequence_length=5):
        x_data, y_data = [], []

        for i in range(0,len(self.data) - sequence_length,sequence_length):
            x_data.append(self.data.iloc[i : i + sequence_length - 1].values)
            y_data.append(self.data.iloc[i + sequence_length - 1].values)

        return np.array(x_data), np.array(y_data)

# Load the preprocessed NASDAQ data
nasdaq_data = pd.read_csv("nasdaq_data.csv")
# Load the preprocessed NYSE data
nyse_data = pd.read_csv("nyse_data.csv")

# Filter out any rows where the open, close, high, and low values are the same
nasdaq_data = nasdaq_data[
    (nasdaq_data["Open"] != nasdaq_data["Close"])
    & (nasdaq_data["Open"] != nasdaq_data["High"])
    & (nasdaq_data["Open"] != nasdaq_data["Low"])
]
nyse_data = nyse_data[
    (nyse_data["Open"] != nyse_data["Close"])
    & (nyse_data["Open"] != nyse_data["High"])
    & (nyse_data["Open"] != nyse_data["Low"])
]

# Create DataPreprocessor objects for both datasets
nasdaq_preprocessor = DataPreprocessor(nasdaq_data)
nyse_preprocessor = DataPreprocessor(nyse_data)

# Generate sequences for NASDAQ and NYSE data
sequence_length = 5
nasdaq_x, nasdaq_y = nasdaq_preprocessor.generate_sequences(sequence_length)
nyse_x, nyse_y = nyse_preprocessor.generate_sequences(sequence_length)

# Split the data into training and testing sets
test_size = int(0.2 * len(nasdaq_x))
nasdaq_x_train, nasdaq_x_test = nasdaq_x[:-test_size], nasdaq_x[-test_size:]
nasdaq_y_train, nasdaq_y_test = nasdaq_y[:-test_size], nasdaq_y[-test_size:]

test_size = int(0.2 * len(nyse_x))
nyse_x_train, nyse_x_test = nyse_x[:-test_size], nyse_x[-test_size:]
nyse_y_train, nyse_y_test = nyse_y[:-test_size], nyse_y[-test_size:]

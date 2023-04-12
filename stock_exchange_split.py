import pandas as pd
from sklearn.model_selection import train_test_split

# Load the preprocessed NASDAQ data
nasdaq_data = pd.read_csv('nasdaq_data.csv')
# Load the preprocessed NYSE data
nyse_data = pd.read_csv('nyse_data.csv')

# Filter out any rows where the open, close, high, and low values are the same
nasdaq_data = nasdaq_data[(nasdaq_data['Open'] != nasdaq_data['Close']) & (nasdaq_data['Open'] != nasdaq_data['High']) & (nasdaq_data['Open'] != nasdaq_data['Low'])]
nyse_data = nyse_data[(nyse_data['Open'] != nyse_data['Close']) & (nyse_data['Open'] != nyse_data['High']) & (nyse_data['Open'] != nyse_data['Low'])]

# Split the NASDAQ data into training and testing sets
nasdaq_train, nasdaq_test = train_test_split(nasdaq_data, test_size=0.2)

# Split the NYSE data into training and testing sets
nyse_train, nyse_test = train_test_split(nyse_data, test_size=0.2)

# Concatenate the training and testing data for both exchanges
train_data = pd.concat([nasdaq_train, nyse_train], ignore_index=True)
test_data = pd.concat([nasdaq_test, nyse_test], ignore_index=True)

# Separate the features and target variable for the training and testing data
X_train = train_data.drop(['Percent_Change'], axis=1)
y_train = train_data['Percent_Change']

X_test = test_data.drop(['Percent_Change'], axis=1)
y_test = test_data['Percent_Change']



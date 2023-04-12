import pandas as pd

# Load the stock market dataset
stock_data = pd.read_csv('indexData.csv')

# Split the dataset into two separate dataframes for the NASDAQ and NYSE indices
nasdaq_data = stock_data[stock_data['Index']=='IXIC']
nyse_data = stock_data[stock_data['Index']=='NYA']

# Drop the 'Index' column from both dataframes
nasdaq_data = nasdaq_data.drop('Index', axis=1)
nyse_data = nyse_data.drop('Index', axis=1)

# Perform data cleaning and preprocessing on both dataframes
# Remove duplicates
nasdaq_data = nasdaq_data.drop_duplicates()
nyse_data = nyse_data.drop_duplicates()

# Handle missing data
nasdaq_data = nasdaq_data.dropna()
nyse_data = nyse_data.dropna()

# Feature engineering
# Add features for percent change and total change per day from open to close
nasdaq_data['Percent_Change'] = (nasdaq_data['Close'] - nasdaq_data['Open']) / nasdaq_data['Open']
nasdaq_data['Total_Change'] = nasdaq_data['Close'] - nasdaq_data['Open']

nyse_data['Percent_Change'] = (nyse_data['Close'] - nyse_data['Open']) / nyse_data['Open']
nyse_data['Total_Change'] = nyse_data['Close'] - nyse_data['Open']

# Remove outliers
# Assuming that outliers have values outside of 3 standard deviations from the mean
# nasdaq_data = nasdaq_data[(nasdaq_data - nasdaq_data.mean()) / nasdaq_data.std() < 3]
# nyse_data = nyse_data[(nyse_data - nyse_data.mean()) / nyse_data.std() < 3]

# Normalize the data using Min-Max scaling
# nasdaq_data = (nasdaq_data - nasdaq_data.min()) / (nasdaq_data.max() - nasdaq_data.min())
# nyse_data = (nyse_data - nyse_data.min()) / (nyse_data.max() - nyse_data.min())

# Check for consistency and fix errors as necessary




# Save the preprocessed dataframes to separate CSV files
nasdaq_data.to_csv('nasdaq_data.csv', index=False)
nyse_data.to_csv('nyse_data.csv', index=False)

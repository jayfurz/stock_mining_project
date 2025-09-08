#!/usr/bin/env python3
"""
Stock Exchange Data Preprocessing Module

This module performs comprehensive preprocessing of raw stock market data for
NASDAQ and NYSE indices. It handles data cleaning, feature engineering, and
preparation for subsequent machine learning model training.

Developed for CSUN Data Mining Master's Course - Stock Price Prediction Project
Author: [Your Name]
Date: 2023
"""

import pandas as pd


def load_and_split_data(input_file="indexData.csv"):
    """
    Load raw stock market data and split by index type.
    
    Args:
        input_file (str): Path to the input CSV file containing stock data
        
    Returns:
        tuple: (nasdaq_data, nyse_data) - DataFrames for each index
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        KeyError: If required columns are missing from the data
    """
    try:
        # Load the stock market dataset
        stock_data = pd.read_csv(input_file)
        print(f"Loaded {len(stock_data)} records from {input_file}")
        
        # Validate required columns
        required_columns = ['Index', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in stock_data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        # Split the dataset into separate dataframes for NASDAQ and NYSE indices
        nasdaq_data = stock_data[stock_data["Index"] == "IXIC"].copy()
        nyse_data = stock_data[stock_data["Index"] == "NYA"].copy()
        
        print(f"NASDAQ records: {len(nasdaq_data)}, NYSE records: {len(nyse_data)}")
        
        # Drop the 'Index' column as it's no longer needed
        nasdaq_data = nasdaq_data.drop("Index", axis=1)
        nyse_data = nyse_data.drop("Index", axis=1)
        
        return nasdaq_data, nyse_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def clean_data(data, index_name):
    """
    Perform comprehensive data cleaning operations.
    
    Args:
        data (pd.DataFrame): Raw stock data for a specific index
        index_name (str): Name of the index for logging purposes
        
    Returns:
        pd.DataFrame: Cleaned data with duplicates and missing values removed
    """
    print(f"\nCleaning {index_name} data...")
    initial_count = len(data)
    
    # Remove duplicates
    data = data.drop_duplicates()
    after_duplicates = len(data)
    print(f"Removed {initial_count - after_duplicates} duplicate records")
    
    # Handle missing data
    data = data.dropna()
    final_count = len(data)
    print(f"Removed {after_duplicates - final_count} records with missing values")
    print(f"Final {index_name} dataset size: {final_count} records")
    
    return data


def engineer_features(data, index_name):
    """
    Perform feature engineering by adding calculated metrics.
    
    Creates new features:
    - Percent_Change: Daily percentage change from open to close
    - Total_Change: Absolute daily change from open to close
    
    Args:
        data (pd.DataFrame): Cleaned stock data
        index_name (str): Name of the index for logging
        
    Returns:
        pd.DataFrame: Data with engineered features added
    """
    print(f"\nEngineering features for {index_name}...")
    
    # Add percent change feature (relative change from open to close)
    data["Percent_Change"] = (data["Close"] - data["Open"]) / data["Open"]
    
    # Add total change feature (absolute change from open to close)
    data["Total_Change"] = data["Close"] - data["Open"]
    
    # Log feature statistics
    print(f"Percent Change - Mean: {data['Percent_Change'].mean():.4f}, "
          f"Std: {data['Percent_Change'].std():.4f}")
    print(f"Total Change - Mean: {data['Total_Change'].mean():.2f}, "
          f"Std: {data['Total_Change'].std():.2f}")
    
    return data


def save_processed_data(nasdaq_data, nyse_data):
    """
    Save preprocessed dataframes to CSV files.
    
    Args:
        nasdaq_data (pd.DataFrame): Processed NASDAQ data
        nyse_data (pd.DataFrame): Processed NYSE data
    """
    print("\nSaving processed data...")
    
    # Save to CSV files
    nasdaq_data.to_csv("nasdaq_data.csv", index=False)
    nyse_data.to_csv("nyse_data.csv", index=False)
    
    print(f"Saved nasdaq_data.csv with {len(nasdaq_data)} records")
    print(f"Saved nyse_data.csv with {len(nyse_data)} records")
    print("\nColumns in processed data:", list(nasdaq_data.columns))


def main():
    """
    Main preprocessing pipeline execution.
    
    Orchestrates the complete data preprocessing workflow:
    1. Load and split raw data by index
    2. Clean data (remove duplicates and missing values)
    3. Engineer new features
    4. Save processed data to CSV files
    """
    print("Starting stock market data preprocessing...")
    
    try:
        # Load and split data
        nasdaq_data, nyse_data = load_and_split_data()
        
        # Clean both datasets
        nasdaq_data = clean_data(nasdaq_data, "NASDAQ")
        nyse_data = clean_data(nyse_data, "NYSE")
        
        # Engineer features for both datasets
        nasdaq_data = engineer_features(nasdaq_data, "NASDAQ")
        nyse_data = engineer_features(nyse_data, "NYSE")
        
        # Save processed data
        save_processed_data(nasdaq_data, nyse_data)
        
        print("\n✓ Preprocessing completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Preprocessing failed: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stock Exchange Sequence Generation and Data Splitting Module

This module creates sequential data structures suitable for time series prediction
using GPT models. It generates sliding window sequences and splits data into
training and testing sets for NASDAQ and NYSE stock indices.

Developed for CSUN Data Mining Master's Course - Stock Price Prediction Project
Author: [Your Name]
Date: 2023
"""

import pandas as pd
import numpy as np


class DataPreprocessor:
    """
    A class for preprocessing stock market data into sequential formats
    suitable for time series prediction with machine learning models.
    
    This preprocessor creates sliding window sequences where each sequence
    contains multiple days of historical data to predict the next day's values.
    
    Attributes:
        data (pd.DataFrame): The stock market data to preprocess
    """
    
    def __init__(self, data):
        """
        Initialize the DataPreprocessor with stock market data.
        
        Args:
            data (pd.DataFrame): Stock market data with columns including
                               Date, Open, High, Low, Close, Volume, etc.
        """
        self.data = data
        print(f"DataPreprocessor initialized with {len(data)} records")
        print(f"Data columns: {list(data.columns)}")

    def generate_sequences(self, sequence_length=5):
        """
        Generate sequential data for time series prediction.
        
        Creates sliding window sequences where the first (sequence_length - 1) days
        serve as input features (X) and the last day serves as the target (y).
        
        Args:
            sequence_length (int): Total length of each sequence (default: 5)
                                 This means 4 days input -> 1 day prediction
        
        Returns:
            tuple: (x_data, y_data)
                - x_data (np.ndarray): Input sequences of shape (n_sequences, 4, n_features)
                - y_data (np.ndarray): Target values of shape (n_sequences, n_features)
        
        Example:
            If sequence_length=5:
            - Days 1-4 -> predict Day 5
            - Days 6-9 -> predict Day 10
            - etc. (non-overlapping windows)
        """
        x_data, y_data = [], []
        
        print(f"Generating sequences with length {sequence_length}...")
        
        # Create non-overlapping sequences
        for i in range(0, len(self.data) - sequence_length, sequence_length):
            # Input: first (sequence_length - 1) days
            x_sequence = self.data.iloc[i : i + sequence_length - 1].values
            # Target: the last day
            y_target = self.data.iloc[i + sequence_length - 1].values
            
            x_data.append(x_sequence)
            y_data.append(y_target)
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        print(f"Generated {len(x_data)} sequences")
        print(f"Input shape: {x_data.shape}, Target shape: {y_data.shape}")
        
        return x_data, y_data


def load_preprocessed_data():
    """
    Load preprocessed stock data from CSV files.
    
    Returns:
        tuple: (nasdaq_data, nyse_data) - Preprocessed DataFrames
        
    Raises:
        FileNotFoundError: If preprocessed data files don't exist
    """
    try:
        print("Loading preprocessed data...")
        nasdaq_data = pd.read_csv("nasdaq_data.csv")
        nyse_data = pd.read_csv("nyse_data.csv")
        
        print(f"Loaded NASDAQ data: {len(nasdaq_data)} records")
        print(f"Loaded NYSE data: {len(nyse_data)} records")
        
        return nasdaq_data, nyse_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run stock_exchange_preprocessing.py first!")
        raise


def filter_data(nasdaq_data, nyse_data):
    """
    Apply data filtering to remove problematic records.
    
    For NASDAQ: Uses only the second half of the dataset to focus on more recent data
    For NYSE: Removes records where OHLC values are identical (likely data errors)
    
    Args:
        nasdaq_data (pd.DataFrame): Raw NASDAQ data
        nyse_data (pd.DataFrame): Raw NYSE data
        
    Returns:
        tuple: (filtered_nasdaq, filtered_nyse) - Filtered DataFrames
    """
    print("\nApplying data filters...")
    
    # Filter NASDAQ: Use second half of data (more recent/relevant)
    nasdaq_filtered = nasdaq_data[int(len(nasdaq_data)/2):].copy()
    print(f"NASDAQ: Filtered to recent {len(nasdaq_filtered)} records")
    
    # Filter NYSE: Remove records where OHLC values are identical
    initial_nyse_count = len(nyse_data)
    nyse_filtered = nyse_data[
        (nyse_data["Open"] != nyse_data["Close"]) &
        (nyse_data["Open"] != nyse_data["High"]) &
        (nyse_data["Open"] != nyse_data["Low"])
    ].copy()
    
    removed_count = initial_nyse_count - len(nyse_filtered)
    print(f"NYSE: Removed {removed_count} records with identical OHLC values")
    print(f"NYSE: {len(nyse_filtered)} records remaining")
    
    return nasdaq_filtered, nyse_filtered


def create_train_test_split(x_data, y_data, test_ratio=0.2, dataset_name=""):
    """
    Split sequence data into training and testing sets.
    
    Uses chronological splitting where the most recent data becomes the test set,
    which is appropriate for time series forecasting evaluation.
    
    Args:
        x_data (np.ndarray): Input sequences
        y_data (np.ndarray): Target values  
        test_ratio (float): Proportion of data to use for testing (default: 0.2)
        dataset_name (str): Name for logging purposes
        
    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    test_size = int(test_ratio * len(x_data))
    
    # Chronological split: earlier data for training, later for testing
    x_train = x_data[:-test_size]
    x_test = x_data[-test_size:]
    y_train = y_data[:-test_size]
    y_test = y_data[-test_size:]
    
    print(f"\n{dataset_name} train/test split:")
    print(f"  Training: {len(x_train)} sequences")
    print(f"  Testing: {len(x_test)} sequences")
    print(f"  Test ratio: {len(x_test)/len(x_data):.2%}")
    
    return x_train, x_test, y_train, y_test


def main():
    """
    Main sequence generation and data splitting pipeline.
    
    Executes the complete workflow:
    1. Load preprocessed data
    2. Apply filtering
    3. Create sequence generators
    4. Generate sequences
    5. Create train/test splits
    
    The resulting variables are made available at module level for import.
    """
    print("Starting sequence generation and data splitting...")
    
    try:
        # Load preprocessed data
        nasdaq_data, nyse_data = load_preprocessed_data()
        
        # Apply filtering
        nasdaq_data, nyse_data = filter_data(nasdaq_data, nyse_data)
        
        # Create DataPreprocessor objects
        print("\nCreating preprocessors...")
        nasdaq_preprocessor = DataPreprocessor(nasdaq_data)
        nyse_preprocessor = DataPreprocessor(nyse_data)
        
        # Generate sequences
        sequence_length = 5
        print(f"\nGenerating {sequence_length}-day sequences...")
        
        global nasdaq_x, nasdaq_y, nyse_x, nyse_y
        nasdaq_x, nasdaq_y = nasdaq_preprocessor.generate_sequences(sequence_length)
        nyse_x, nyse_y = nyse_preprocessor.generate_sequences(sequence_length)
        
        # Create train/test splits
        print("\nCreating train/test splits...")
        
        global nasdaq_x_train, nasdaq_x_test, nasdaq_y_train, nasdaq_y_test
        global nyse_x_train, nyse_x_test, nyse_y_train, nyse_y_test
        
        nasdaq_x_train, nasdaq_x_test, nasdaq_y_train, nasdaq_y_test = create_train_test_split(
            nasdaq_x, nasdaq_y, dataset_name="NASDAQ"
        )
        
        nyse_x_train, nyse_x_test, nyse_y_train, nyse_y_test = create_train_test_split(
            nyse_x, nyse_y, dataset_name="NYSE"
        )
        
        print("\n✓ Sequence generation completed successfully!")
        print("\nGlobal variables created:")
        print(f"  nasdaq_x_train: {nasdaq_x_train.shape}")
        print(f"  nasdaq_x_test: {nasdaq_x_test.shape}")
        print(f"  nyse_x_train: {nyse_x_train.shape}")
        print(f"  nyse_x_test: {nyse_x_test.shape}")
        
    except Exception as e:
        print(f"\n✗ Sequence generation failed: {e}")
        raise


# Execute main pipeline and create global variables
if __name__ == "__main__":
    main()
else:
    # When imported, run the pipeline to create global variables
    main()
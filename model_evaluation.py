#!/usr/bin/env python3
"""
Stock Market Prediction Model Evaluation Module

This module provides comprehensive evaluation of GPT-based stock market prediction
models. It loads prediction results from JSON files and calculates various
performance metrics including regression statistics and prediction accuracy.

Developed for CSUN Data Mining Master's Course - Stock Price Prediction Project
Author: [Your Name]
Date: 2023
"""

import json
import os
import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def load_evaluation_data(data_file=None):
    """
    Load evaluation results from JSON file.
    
    If no file is specified, uses the most recent evaluation results
    from the dump directory.
    
    Args:
        data_file (str, optional): Path to specific evaluation results file.
                                  If None, uses most recent file.
    
    Returns:
        list: Loaded evaluation data containing prompts, responses, and actual values
        
    Raises:
        FileNotFoundError: If no evaluation files are found
        json.JSONDecodeError: If the file contains invalid JSON
    """
    if data_file is None:
        # Find the most recent evaluation file
        evaluation_files = glob.glob("./dump/evaluation_results_*.json")
        if not evaluation_files:
            raise FileNotFoundError("No evaluation result files found in ./dump/ directory")
        
        # Sort by modification time and get the most recent
        data_file = max(evaluation_files, key=os.path.getmtime)
        print(f"Using most recent evaluation file: {data_file}")
    else:
        print(f"Loading evaluation data from: {data_file}")
    
    try:
        with open(data_file, 'r') as file:
            data = json.load(file)
        print(f"Loaded {len(data)} evaluation records")
        return data
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {data_file}: {e}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Evaluation file not found: {data_file}")


def convert_value(val):
    """
    Convert string values to appropriate types (float for numbers, string for dates).
    
    Args:
        val (str): String value to convert
        
    Returns:
        float or str: Converted value - float if numeric, original string if not
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def parse_prediction_data(data):
    """
    Parse actual and predicted values from evaluation data.
    
    Extracts and converts the actual and predicted values from the JSON
    evaluation results into lists suitable for metric calculations.
    
    Args:
        data (list): Evaluation data loaded from JSON file
        
    Returns:
        tuple: (actual_values, predicted_values) - Lists of parsed values
               Each entry contains [date, open, high, low, close, volume, ...]
    """
    print("Parsing prediction data...")
    
    actual_values = []
    predicted_values = []
    
    for i, entry in enumerate(data):
        try:
            # Parse actual values (stored as JSON string)
            actual_str = entry['actual'].strip('[]')
            actual_list = [convert_value(val.strip().strip('"')) for val in actual_str.split(', ')]
            
            # Parse predicted values (stored as raw response)
            predicted_str = entry['response'].strip('[]')
            predicted_list = [convert_value(val.strip().strip('"')) for val in predicted_str.split(', ')]
            
            actual_values.append(actual_list)
            predicted_values.append(predicted_list)
            
        except Exception as e:
            print(f"Warning: Could not parse entry {i}: {e}")
            continue
    
    print(f"Successfully parsed {len(actual_values)} prediction pairs")
    
    if len(actual_values) > 0:
        print(f"Data structure: {len(actual_values[0])} features per prediction")
        feature_names = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Percent_Change", "Total_Change"]
        if len(actual_values[0]) <= len(feature_names):
            print(f"Features: {feature_names[:len(actual_values[0])]}")
    
    return actual_values, predicted_values


def check_dates(actual_values, predicted_values):
    """
    Check accuracy of date predictions.
    
    Compares the predicted dates (first element) with actual dates
    to assess temporal prediction accuracy.
    
    Args:
        actual_values (list): List of actual value arrays
        predicted_values (list): List of predicted value arrays
        
    Returns:
        int: Number of correct date predictions
    """
    print("Checking date prediction accuracy...")
    
    correct_dates = 0
    total_comparisons = 0
    
    for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
        if len(actual) > 0 and len(predicted) > 0:
            total_comparisons += 1
            if actual[0] == predicted[0]:
                correct_dates += 1
            elif i < 3:  # Show first few examples
                print(f"  Date mismatch {i+1}: predicted '{predicted[0]}' vs actual '{actual[0]}'")
    
    accuracy = correct_dates / total_comparisons if total_comparisons > 0 else 0
    print(f"Date accuracy: {correct_dates}/{total_comparisons} ({accuracy:.2%})")
    
    return correct_dates


def check_direction(actual_values, predicted_values):
    """
    Check accuracy of price direction predictions.
    
    Compares the direction (positive/negative) of price changes
    between predicted and actual values. Uses the Total_Change
    feature (typically at index 8) to determine direction.
    
    Args:
        actual_values (list): List of actual value arrays
        predicted_values (list): List of predicted value arrays
        
    Returns:
        int: Number of correct direction predictions
    """
    print("Checking price direction accuracy...")
    
    correct_direction = 0
    total_comparisons = 0
    direction_index = -1  # Use last element (Total_Change)
    
    for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values)):
        try:
            # Check if we have enough elements and they're numeric
            if (len(actual) > abs(direction_index) and len(predicted) > abs(direction_index)
                and isinstance(actual[direction_index], (int, float))
                and isinstance(predicted[direction_index], (int, float))):
                
                total_comparisons += 1
                actual_direction = np.sign(actual[direction_index])
                predicted_direction = np.sign(predicted[direction_index])
                
                if actual_direction == predicted_direction:
                    correct_direction += 1
                elif i < 3:  # Show first few examples
                    print(f"  Direction mismatch {i+1}: predicted {predicted_direction} vs actual {actual_direction}")
                    
        except (IndexError, TypeError, ValueError) as e:
            if i < 3:
                print(f"  Could not compare directions for entry {i+1}: {e}")
            continue
    
    accuracy = correct_direction / total_comparisons if total_comparisons > 0 else 0
    print(f"Direction accuracy: {correct_direction}/{total_comparisons} ({accuracy:.2%})")
    
    return correct_direction


def calculate_regression_metrics(actual_values, predicted_values):
    """
    Calculate comprehensive regression metrics for each stock attribute.
    
    Computes MAE, MSE, RMSE, and R² for each numerical feature
    (Open, High, Low, Close, Volume) excluding dates and derived features.
    
    Args:
        actual_values (list): List of actual value arrays
        predicted_values (list): List of predicted value arrays
        
    Returns:
        dict: Metrics dictionary with feature names as keys and
              metric dictionaries as values
    """
    print("\nCalculating regression metrics...")
    
    # Define feature names (skip Date at index 0)
    feature_names = ["Open", "High", "Low", "Close", "Volume"]
    metrics = {}
    
    for i, name in enumerate(feature_names):
        feature_index = i + 1  # Skip date at index 0
        
        try:
            # Extract values for this feature, filtering out non-numeric entries
            actual_col = []
            predicted_col = []
            
            for actual_row, predicted_row in zip(actual_values, predicted_values):
                if (len(actual_row) > feature_index and len(predicted_row) > feature_index
                    and isinstance(actual_row[feature_index], (int, float))
                    and isinstance(predicted_row[feature_index], (int, float))):
                    
                    actual_col.append(float(actual_row[feature_index]))
                    predicted_col.append(float(predicted_row[feature_index]))
            
            if len(actual_col) == 0:
                print(f"  Warning: No valid data for {name}")
                continue
            
            # Calculate metrics
            mae = mean_absolute_error(actual_col, predicted_col)
            mse = mean_squared_error(actual_col, predicted_col)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual_col, predicted_col)
            
            metrics[name] = {
                "MAE": mae, 
                "MSE": mse, 
                "RMSE": rmse, 
                "R²": r2,
                "n_samples": len(actual_col)
            }
            
            print(f"  {name}: {len(actual_col)} valid samples")
            
        except Exception as e:
            print(f"  Error calculating metrics for {name}: {e}")
            continue
    
    return metrics


def print_evaluation_summary(metrics, correct_dates, correct_direction, total_samples):
    """
    Print a comprehensive evaluation summary.
    
    Args:
        metrics (dict): Regression metrics for each feature
        correct_dates (int): Number of correct date predictions
        correct_direction (int): Number of correct direction predictions  
        total_samples (int): Total number of test samples
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 60)
    
    # Print regression metrics
    print("\nREGRESSION METRICS:")
    print("-" * 40)
    
    for name, values in metrics.items():
        print(f"\n{name}:")
        for metric, value in values.items():
            if metric == "n_samples":
                print(f"  Samples: {value}")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Print accuracy metrics
    print("\n\nPREDICTION ACCURACY:")
    print("-" * 40)
    
    date_accuracy = correct_dates / total_samples if total_samples > 0 else 0
    direction_accuracy = correct_direction / total_samples if total_samples > 0 else 0
    
    print(f"Date Accuracy: {correct_dates}/{total_samples} ({date_accuracy:.2%})")
    print(f"Direction Accuracy: {correct_direction}/{total_samples} ({direction_accuracy:.2%})")
    
    # Overall assessment
    print("\n\nOVERALL ASSESSMENT:")
    print("-" * 40)
    
    if metrics:
        avg_r2 = np.mean([m["R²"] for m in metrics.values() if "R²" in m])
        avg_mae = np.mean([m["MAE"] for m in metrics.values() if "MAE" in m])
        
        print(f"Average R² Score: {avg_r2:.4f}")
        print(f"Average MAE: {avg_mae:.4f}")
        
        # Performance rating
        if avg_r2 > 0.8:
            rating = "Excellent"
        elif avg_r2 > 0.6:
            rating = "Good"
        elif avg_r2 > 0.3:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
            
        print(f"Model Performance: {rating}")
    
    print("\n" + "=" * 60)


def main():
    """
    Main evaluation pipeline.
    
    Loads evaluation results, calculates metrics, and displays a comprehensive
    performance summary for the GPT-based stock prediction model.
    """
    print("Starting model evaluation...")
    
    try:
        # Load evaluation data (uses most recent file if none specified)
        data = load_evaluation_data()
        
        # Parse prediction data
        actual_values, predicted_values = parse_prediction_data(data)
        
        if len(actual_values) == 0:
            print("Error: No valid prediction data found")
            return
        
        # Calculate regression metrics
        metrics = calculate_regression_metrics(actual_values, predicted_values)
        
        # Check prediction accuracy
        correct_dates = check_dates(actual_values, predicted_values)
        correct_direction = check_direction(actual_values, predicted_values)
        
        # Print comprehensive summary
        print_evaluation_summary(metrics, correct_dates, correct_direction, len(actual_values))
        
        print("\n✓ Model evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
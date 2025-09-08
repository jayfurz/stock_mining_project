#!/usr/bin/env python3
"""
Stock Exchange GPT Fine-tuning and Prediction Module

This module implements the core functionality for fine-tuning OpenAI's GPT models
on stock market data and generating predictions. It converts numerical time series
data into JSON format for GPT consumption and provides comprehensive model
evaluation capabilities.

Developed for CSUN Data Mining Master's Course - Stock Price Prediction Project
Author: [Your Name]
Date: 2023

Security Note: This module requires OpenAI API key via environment variable
OPENAI_API_KEY for model fine-tuning and inference.
"""

import openai
import os
import numpy as np
import pandas as pd
import json
import time

# Import the data splitting module for training data
import stock_exchange_split as split

from datetime import datetime


def days_between(date1, date2):
    """
    Calculate the difference in days between two dates.
    
    This utility function is used in model evaluation to assess
    date prediction accuracy.
    
    Args:
        date1 (str): First date in YYYY-MM-DD format
        date2 (str): Second date in YYYY-MM-DD format
        
    Returns:
        int: Number of days between the two dates (date2 - date1)
        
    Raises:
        ValueError: If date strings are not in the expected format
    """
    date_format = "%Y-%m-%d"
    try:
        d1 = datetime.strptime(date1, date_format)
        d2 = datetime.strptime(date2, date_format)
        delta = d2 - d1
        return delta.days
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        raise


def prepare_finetuning_data(x_data, y_data, output_file, step=5):
    """
    Convert numerical sequence data into JSONL format for GPT fine-tuning.
    
    This function transforms the numerical stock market sequences into text-based
    JSON format that GPT models can process. Each sequence becomes a prompt-completion
    pair where the prompt contains the historical data and the completion contains
    the target prediction.
    
    Args:
        x_data (np.ndarray): Input sequences of shape (n_samples, sequence_length, n_features)
                            Contains historical stock data for multiple days
        y_data (np.ndarray): Target values of shape (n_samples, n_features)
                           Contains the stock data to be predicted
        output_file (str): Path to output JSONL file for fine-tuning data
        step (int): Sequence step size (currently unused, kept for compatibility)
        
    Returns:
        None: Saves formatted data directly to the specified file
        
    File Format:
        Each line in the output file contains a JSON object with:
        - "prompt": JSON string of historical data array
        - "completion": JSON string of target prediction array
        
    Example Usage:
        prepare_finetuning_data(split.nasdaq_x_train, split.nasdaq_y_train, 
                              "nasdaq_finetuning_data.jsonl")
        prepare_finetuning_data(split.nyse_x_train, split.nyse_y_train,
                              "nyse_finetuning_data.jsonl")
    """
    print(f"Preparing fine-tuning data for {len(x_data)} sequences...")
    
    finetuning_data = []

    for i in range(len(x_data)):
        # Convert numpy arrays to lists for JSON serialization
        x_chunk = x_data[i]
        y_chunk = y_data[i]

        # Create JSON strings for prompt and completion
        prompt = json.dumps(x_chunk.tolist())
        completion = json.dumps(y_chunk.tolist())

        finetuning_data.append({"prompt": prompt, "completion": completion})

    # Save to JSONL format (one JSON object per line)
    with open(output_file, "w") as f:
        for item in finetuning_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Saved {len(finetuning_data)} training examples to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")


def fine_tune_gpt_model(train_data_path):
    """
    Fine-tune a GPT model on stock market prediction data.
    
    This function uploads training data to OpenAI and initiates a fine-tuning job
    using the specified parameters. It monitors the job status until completion.
    
    Args:
        train_data_path (str): Path to JSONL file containing training data
                             (created by prepare_finetuning_data)
        
    Returns:
        str: Model ID of the fine-tuned model for use in predictions
        
    Raises:
        KeyError: If OPENAI_API_KEY environment variable is not set
        FileNotFoundError: If training data file doesn't exist
        openai.OpenAIError: If API calls fail
        
    Environment Variables Required:
        OPENAI_API_KEY: Your OpenAI API key for authentication
        
    Fine-tuning Parameters:
        - Base model: text-davinci-002 (GPT-3)
        - Epochs: 1 (single pass through training data)
        - Max tokens: 400 (sufficient for stock data sequences)
        - Learning rate: 0.0001 (conservative for financial data)
        - Batch size: 4 (balanced for API limits and training speed)
        
    Example Usage:
        model_id = fine_tune_gpt_model("nasdaq_finetuning_data.jsonl")
        print(f"Fine-tuned model ready: {model_id}")
    """
    print(f"Starting GPT fine-tuning process with {train_data_path}...")
    
    try:
        # Set OpenAI API key from environment
        openai.api_key = os.environ["OPENAI_API_KEY"]
        print("OpenAI API key loaded successfully")
    except KeyError:
        raise KeyError("OPENAI_API_KEY environment variable not set. "
                      "Please set it with your OpenAI API key.")

    try:
        # Upload the training dataset to OpenAI
        print("Uploading training dataset...")
        with open(train_data_path, 'rb') as f:
            dataset = openai.Dataset.create(file=f, purpose="fine-tuning")
        print(f"Dataset uploaded successfully. ID: {dataset.id}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Training data file not found: {train_data_path}")
    
    # Configure fine-tuning parameters
    base_model = "text-davinci-002"
    print(f"Fine-tuning {base_model} with the following parameters:")
    print(f"  Epochs: 1")
    print(f"  Max tokens: 400")
    print(f"  Learning rate: 0.0001")
    print(f"  Batch size: 4")

    # Create fine-tuning job
    fine_tuning_job = openai.FineTune.create(
        model=base_model,
        dataset_id=dataset.id,
        n_epochs=1,
        max_tokens=400,
        learning_rate=0.0001,
        batch_size=4,
    )
    
    print(f"Fine-tuning job created. ID: {fine_tuning_job.id}")
    print("Monitoring job status...")

    # Monitor fine-tuning progress
    while True:
        status = openai.FineTune.get(fine_tuning_job.id).status
        print(f"Status: {status}")

        if status == "succeeded":
            model_id = fine_tuning_job.model_id
            print(f"✓ Fine-tuning completed successfully!")
            print(f"  Model ID: {model_id}")
            return model_id
        elif status == "failed":
            print("✗ Fine-tuning failed")
            # Get more details about the failure
            job_details = openai.FineTune.get(fine_tuning_job.id)
            print(f"Error details: {job_details}")
            raise RuntimeError("Fine-tuning job failed")
        else:
            print("Fine-tuning in progress... checking again in 60 seconds")

        time.sleep(60)  # Check status every minute


def test_evaluate_data(x_test, y_test):
    """
    Display test data in prompt-completion format for debugging.
    
    This utility function shows how the test data appears to the GPT model,
    useful for debugging and understanding the data format.
    
    Args:
        x_test (np.ndarray): Test input sequences
        y_test (np.ndarray): Test target values
        
    Returns:
        None: Prints formatted data to console
    """
    print("Sample test data in GPT format:")
    print("-" * 50)
    
    for i, (x, y_true_list) in enumerate(zip(x_test, y_test)):
        if i >= 3:  # Show only first 3 examples
            break
            
        prompt = json.dumps(x.tolist())
        completion = y_true_list.tolist()
        
        print(f"Example {i+1}:")
        print(f"Prompt: {prompt}")
        print(f"Expected completion: {completion}")
        print("-" * 30)


def evaluate_model(model_id, x_test, y_test):
    """
    Evaluate a fine-tuned GPT model on stock market test data.
    
    This function generates predictions using the fine-tuned model and compares
    them against actual values. It calculates mean absolute error and saves
    detailed results for further analysis.
    
    Args:
        model_id (str): ID of the fine-tuned GPT model
        x_test (np.ndarray): Test input sequences
        y_test (np.ndarray): Test target values (ground truth)
        
    Returns:
        float: Mean absolute difference across all numerical predictions
               (excluding date predictions)
    
    Side Effects:
        - Prints prediction progress and comparisons
        - Saves detailed results to JSON file in dump/ directory
        - Filename format: evaluation_results_YYYYMMDD_HHMMSS.json
        
    Evaluation Process:
        1. Convert test sequences to JSON prompts
        2. Generate predictions using GPT model
        3. Parse and validate predictions
        4. Calculate absolute differences for numerical values
        5. Skip date comparison (index 0) in error calculation
        6. Save all prompts, predictions, and actual values
        
    Model Parameters:
        - Max tokens: 80 (sufficient for single day prediction)
        - Temperature: 0.5 (balance between determinism and variability)
        - N: 1 (single prediction per prompt)
        
    Example Usage:
        mae = evaluate_model("fine-tuned-model-id", x_test, y_test)
        print(f"Model accuracy (MAE): {mae:.4f}")
    """
    print(f"Evaluating model {model_id} on {len(x_test)} test samples...")
    
    # Set API key
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        raise KeyError("OPENAI_API_KEY environment variable not set")

    total_absolute_difference = 0
    total_values = 0
    results = []
    successful_predictions = 0

    for i, (x, y_true_list) in enumerate(zip(x_test, y_test)):
        print(f"\nProcessing test sample {i+1}/{len(x_test)}...")
        
        # Create prompt from input sequence
        prompt = json.dumps(x.tolist())
        print(f"Prompt: {prompt[:100]}...")  # Show first 100 chars
        
        try:
            # Generate prediction using fine-tuned model
            response = openai.Completion.create(
                engine=model_id,
                prompt=prompt,
                max_tokens=80,
                n=1,
                stop=None,
                temperature=0.5,
            )

            # Parse response and clean it
            response_text = response.choices[0].text.strip()
            
            # Ensure we get a complete JSON array
            if ']' in response_text:
                response_text = response_text.split(']')[0] + ']'
            
            # Parse prediction and ground truth
            try:
                y_pred_list = json.loads(response_text)
                y_true_list = y_true_list.tolist()
                
                print(f"Predicted: {y_pred_list}")
                print(f"Actual:    {y_true_list}")
                
                # Calculate errors for numerical values (skip date at index 0)
                for j, (y_pred, y_true) in enumerate(zip(y_pred_list, y_true_list)):
                    if j == 0:  # Skip date comparison for error calculation
                        continue
                    try:
                        absolute_difference = abs(float(y_pred) - float(y_true))
                        total_absolute_difference += absolute_difference
                        total_values += 1
                    except (ValueError, TypeError):
                        print(f"Warning: Could not compare values at index {j}")
                
                successful_predictions += 1
                
            except json.JSONDecodeError:
                print(f"Warning: Could not parse model response: {response_text}")
                y_pred_list = []
                y_true_list = y_true_list.tolist()

        except Exception as e:
            print(f"Error getting prediction: {e}")
            response_text = "ERROR"
            y_true_list = y_true_list.tolist()

        # Store results for detailed analysis
        results.append({
            "prompt": prompt, 
            "response": response_text, 
            "actual": json.dumps(y_true_list)
        })

    # Calculate final metrics
    if total_values > 0:
        mean_absolute_difference = total_absolute_difference / total_values
    else:
        mean_absolute_difference = float('inf')
        print("Warning: No valid predictions for error calculation")

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dump/evaluation_results_{timestamp}.json"
    
    # Ensure dump directory exists
    os.makedirs("dump", exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Total test samples: {len(x_test)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Success rate: {successful_predictions/len(x_test):.2%}")
    print(f"Mean Absolute Error: {mean_absolute_difference:.4f}")
    print(f"Detailed results saved to: {filename}")

    return mean_absolute_difference


def main():
    """
    Main pipeline for GPT model training and evaluation.
    
    This function orchestrates the complete workflow:
    1. Prepare fine-tuning data (optional - uncomment to run)
    2. Fine-tune GPT model (optional - uncomment to run)
    3. Evaluate model performance using existing trained model
    
    The function is designed to be flexible - you can uncomment different
    sections based on whether you want to train a new model or evaluate
    an existing one.
    
    Environment Variables:
        OPENAI_API_KEY: Required for all OpenAI API operations
        SECOND_TRAINED_MODEL: Pre-trained model ID for evaluation
                             (alternative to training new model)
    """
    print("Starting GPT-based stock market prediction pipeline...")
    
    try:
        # Step 1: Prepare fine-tuning data (uncomment to create new training data)
        print("\n=== Step 1: Data Preparation ===")
        print("Skipping data preparation (using existing files)")
        # Uncomment the following lines to create new training data:
        # nasdaq_train_data_path = "nasdaq_finetuning_data.jsonl"
        # prepare_finetuning_data(split.nasdaq_x_train, split.nasdaq_y_train, nasdaq_train_data_path)
        # print(f"Training data prepared: {nasdaq_train_data_path}")

        # Step 2: Fine-tune GPT model (uncomment to train new model)
        print("\n=== Step 2: Model Training ===")
        print("Skipping model training (using pre-trained model)")
        # Uncomment the following lines to train a new model:
        # model_id = fine_tune_gpt_model(nasdaq_train_data_path)
        # print(f"New model trained: {model_id}")
        
        # Use existing trained model
        try:
            model_id = os.environ["SECOND_TRAINED_MODEL"]
            print(f"Using pre-trained model: {model_id}")
        except KeyError:
            print("Warning: SECOND_TRAINED_MODEL not set. Please either:")
            print("1. Set SECOND_TRAINED_MODEL environment variable, or")
            print("2. Uncomment the training code above to train a new model")
            return

        # Step 3: Model evaluation
        print("\n=== Step 3: Model Evaluation ===")
        print("Sample test data format:")
        test_evaluate_data(split.nasdaq_x_test[:2], split.nasdaq_y_test[:2])
        
        print("\nEvaluating model performance...")
        mean_absolute_error = evaluate_model(model_id, split.nasdaq_x_test, split.nasdaq_y_test)
        
        print(f"\n=== Final Results ===")
        print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
        print(f"Model evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
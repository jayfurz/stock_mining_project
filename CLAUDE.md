# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stock market prediction project that forecasts stock prices based on historical data. The project uses machine learning, specifically fine-tuning GPT models, to predict future stock prices for NASDAQ and NYSE indices.

## Key Dependencies

- pandas - Data manipulation and analysis
- numpy - Numerical computations
- openai - GPT model integration and fine-tuning
- scikit-learn - Model evaluation metrics
- datetime - Date handling
- json - Data serialization

## Project Architecture

### Data Processing Pipeline

1. **stock_exchange_preprocessing.py**: Initial data preprocessing
   - Loads raw stock data from `indexData.csv`
   - Splits data into NASDAQ (IXIC) and NYSE (NYA) indices
   - Performs feature engineering (Percent_Change, Total_Change)
   - Exports cleaned data to `nasdaq_data.csv` and `nyse_data.csv`

2. **stock_exchange_split.py**: Sequence generation and train/test splitting
   - Creates DataPreprocessor class for sequence generation
   - Generates sequences of 5 days for training (4 days input, 1 day prediction)
   - Creates 80/20 train/test split for both datasets
   - Exports variables: nasdaq_x_train, nasdaq_x_test, nasdaq_y_train, nasdaq_y_test (same for NYSE)

3. **stock_exchange_gpt.py**: GPT model fine-tuning and prediction
   - Prepares data in JSONL format for GPT fine-tuning
   - Fine-tunes GPT models on stock sequences
   - Evaluates model predictions against test data
   - Saves evaluation results to JSON files in `dump/` directory

4. **model_evaluation.py**: Performance analysis
   - Loads evaluation results from JSON files
   - Calculates regression metrics (MAE, MSE, RMSE, R²) for each stock attribute
   - Checks prediction accuracy for dates and price direction

## Running the Pipeline

To run the complete pipeline:

```bash
# Step 1: Preprocess raw data
python stock_exchange_preprocessing.py

# Step 2: Generate sequences and split data
python stock_exchange_split.py

# Step 3: Fine-tune GPT model (requires OPENAI_API_KEY environment variable)
python stock_exchange_gpt.py

# Step 4: Evaluate model performance
python model_evaluation.py
```

## Data Files

- **Input**: `indexData.csv` - Raw stock market data with NASDAQ and NYSE indices
- **Intermediate**: `nasdaq_data.csv`, `nyse_data.csv` - Preprocessed index data
- **Fine-tuning**: `*_finetuning_data.jsonl` files - Formatted for GPT training
- **Results**: `dump/evaluation_results_*.json` - Model prediction results

## Environment Setup

Set OpenAI API key before running GPT fine-tuning:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
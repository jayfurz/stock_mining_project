# GPT-Based Stock Market Prediction System

A novel approach to stock market forecasting using OpenAI's GPT models for time series prediction. This project was developed as part of a Data Mining master's course at California State University, Northridge (CSUN), exploring the innovative application of large language models to financial prediction tasks.

## Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Model Performance](#model-performance)
- [File Structure](#file-structure)
- [Academic Context](#academic-context)
- [Limitations & Disclaimers](#limitations--disclaimers)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a unique approach to stock market prediction by fine-tuning OpenAI's GPT models on historical stock price sequences. Unlike traditional time series forecasting methods, this system leverages the pattern recognition capabilities of large language models to predict future stock prices for NASDAQ and NYSE indices.

### Key Innovation
- **GPT Fine-tuning for Financial Data**: Transforms numerical stock sequences into text format for GPT model consumption
- **Sequence-based Prediction**: Uses 4-day historical windows to predict the 5th day's trading values
- **Multi-attribute Forecasting**: Predicts Open, High, Low, Close prices, Volume, and calculated features

## Project Architecture

The system follows a modular pipeline architecture with four main components:

```
Raw Data → Preprocessing → Sequence Generation → GPT Fine-tuning → Evaluation
```

### Core Components

1. **Data Preprocessing** (`stock_exchange_preprocessing.py`)
   - Cleans and splits raw stock data by index (NASDAQ/NYSE)
   - Performs feature engineering (Percent_Change, Total_Change)
   - Handles missing data and removes duplicates

2. **Sequence Generation** (`stock_exchange_split.py`)
   - Creates sliding window sequences of 5-day periods
   - Splits data into training (80%) and testing (20%) sets
   - Prepares data for GPT fine-tuning format

3. **GPT Model Training** (`stock_exchange_gpt.py`)
   - Converts numerical sequences to JSON-formatted text
   - Fine-tunes GPT models on historical patterns
   - Implements prediction and evaluation workflows

4. **Performance Evaluation** (`model_evaluation.py`)
   - Calculates regression metrics (MAE, MSE, RMSE, R²)
   - Analyzes prediction accuracy for dates and price directions
   - Generates comprehensive performance reports

## Features

- **Multi-Index Support**: Handles both NASDAQ (IXIC) and NYSE (NYA) indices
- **Automated Pipeline**: Complete data processing and model training workflow
- **Comprehensive Evaluation**: Multiple metrics for model performance assessment
- **JSON-formatted Results**: Structured output for further analysis
- **Environment-based Configuration**: Secure API key management

## Installation

### Prerequisites
- Python 3.7 or higher
- OpenAI API account and API key

### Dependencies
Install required packages using pip:

```bash
pip install -r requirements.txt
```

### Environment Setup
Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

For model evaluation (optional):
```bash
export SECOND_TRAINED_MODEL="your-fine-tuned-model-id"
```

## Usage

### Complete Pipeline Execution

Run the complete pipeline in sequence:

```bash
# Step 1: Preprocess raw data
python stock_exchange_preprocessing.py

# Step 2: Generate sequences and split data
python stock_exchange_split.py

# Step 3: Fine-tune GPT model and generate predictions
python stock_exchange_gpt.py

# Step 4: Evaluate model performance
python model_evaluation.py
```

### Individual Component Usage

#### Data Preprocessing
```python
# Loads indexData.csv and creates nasdaq_data.csv, nyse_data.csv
python stock_exchange_preprocessing.py
```

#### Sequence Generation
```python
# Creates training sequences and train/test splits
python stock_exchange_split.py
```

#### Model Training
```python
from stock_exchange_gpt import prepare_finetuning_data, fine_tune_gpt_model
import stock_exchange_split as split

# Prepare training data
prepare_finetuning_data(split.nasdaq_x_train, split.nasdaq_y_train, "nasdaq_training.jsonl")

# Fine-tune model
model_id = fine_tune_gpt_model("nasdaq_training.jsonl")
```

## Data Pipeline

### Input Data Format
The system expects `indexData.csv` with columns:
- `Date`: Trading date (YYYY-MM-DD format)
- `Index`: Stock index identifier (IXIC for NASDAQ, NYA for NYSE)
- `Open`, `High`, `Low`, `Close`: Daily trading prices
- `Adj Close`: Adjusted closing price
- `Volume`: Trading volume

### Data Processing Steps

1. **Index Separation**: Splits combined data by index type
2. **Feature Engineering**: Adds Percent_Change and Total_Change metrics
3. **Data Cleaning**: Removes duplicates, handles missing values
4. **Sequence Creation**: Generates 5-day sliding windows
5. **Format Conversion**: Transforms to GPT-compatible JSON format

### Output Structure
- **Training Data**: JSONL files with prompt-completion pairs
- **Evaluation Results**: JSON files with predictions and actual values
- **Performance Metrics**: Calculated accuracy measures

## Model Performance

The system evaluates model performance using multiple metrics:

### Regression Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination

### Prediction Accuracy
- **Date Accuracy**: Percentage of correctly predicted trading dates
- **Direction Accuracy**: Percentage of correctly predicted price movement direction

### Sample Performance Output
```
Open:
  MAE: 12.45
  MSE: 234.56
  RMSE: 15.31
  R²: 0.89

Correct Dates: 85/100
Correct Direction: 72/100
```

## File Structure

```
stock_mining_project/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
├── stock_exchange_preprocessing.py        # Data preprocessing module
├── stock_exchange_split.py               # Sequence generation module
├── stock_exchange_gpt.py                 # GPT fine-tuning module
├── model_evaluation.py                   # Performance evaluation module
├── indexData.csv                         # Raw stock market data
├── nasdaq_data.csv                       # Processed NASDAQ data
├── nyse_data.csv                         # Processed NYSE data
├── nasdaq_finetuning_data.jsonl          # NASDAQ training data
├── nyse_finetuning_data.jsonl            # NYSE training data
└── dump/                                 # Evaluation results directory
    └── evaluation_results_*.json         # Model prediction results
```

## Academic Context

This project was developed as part of a **Data Mining master's course** at **California State University, Northridge (CSUN)**. The work represents an innovative exploration of applying Natural Language Processing techniques to financial time series forecasting.

### Research Contribution
- **Novel Approach**: First known application of GPT fine-tuning to stock price prediction
- **Methodological Innovation**: Transformation of numerical time series to text-based sequences
- **Empirical Validation**: Comprehensive evaluation on real market data

### Educational Objectives
- Explore unconventional applications of NLP models
- Understand the challenges of financial forecasting
- Develop skills in data pipeline construction and model evaluation

## Limitations & Disclaimers

### Technical Limitations
- **API Dependencies**: Requires OpenAI API access and associated costs
- **Model Constraints**: Limited by GPT model context window and token limits
- **Data Requirements**: Needs substantial historical data for effective training

### Financial Disclaimers
⚠️ **Important**: This project is for **academic and research purposes only**.
- **Not Financial Advice**: Predictions should not be used for actual trading decisions
- **No Performance Guarantee**: Past performance does not predict future results
- **Risk Warning**: Stock market investments carry inherent risks

### Research Limitations
- Limited to historical data patterns
- May not capture external market factors
- Performance varies with market conditions

## Contributing

This is an academic project, but contributions for educational improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add educational enhancement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

### Areas for Enhancement
- Additional evaluation metrics
- Support for more stock indices
- Improved data visualization
- Documentation improvements

## License

This project is available under the MIT License. See LICENSE file for details.

---

**Developed by**: [Your Name]  
**Institution**: California State University, Northridge  
**Course**: Data Mining (Master's Program)  
**Year**: 2023

For questions or academic discussions, please open an issue in the repository.
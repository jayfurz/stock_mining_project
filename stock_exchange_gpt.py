import openai
import os
import numpy as np
import pandas as pd
import json
import time

# Change the following line for notebook
import stock_exchange_split as split

def prepare_finetuning_data(x_data, y_data, output_file, step=5):
    finetuning_data = []

    for i in range(0, len(x_data), step):
        x_chunk = x_data[i:i + step]
        y_chunk = y_data[i:i + step]

        prompt = json.dumps(x_chunk.tolist())
        completion = json.dumps(y_chunk.tolist())

        finetuning_data.append({"prompt": prompt, "completion": completion})

    with open(output_file, "w") as f:
        for item in finetuning_data:
            f.write(json.dumps(item) + "\n")

# Replace 'split.nasdaq_x_train' and 'split.nyse_y_train' with the appropriate variables
prepare_finetuning_data(split.nasdaq_x_train, split.nasdaq_y_train, "nasdaq_finetuning_data.jsonl")
prepare_finetuning_data(split.nyse_x_train, split.nyse_y_train, "nyse_finetuning_data.jsonl")

def fine_tune_gpt_model():
    # Set your API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Upload the NASDAQ dataset
    with open("nasdaq_finetuning_data.jsonl") as f:
    nasdaq_dataset = openai.Dataset.create(file=f, purpose="fine-tuning")
    
    # Upload the formatted training data to GPT (not ready)
    # Set the model to fine-tune
    base_model = "text-davinci-002"

    # Fine-tune the model using the NASDAQ dataset
    fine_tuning_job = openai.FineTune.create(
        model=base_model,
        dataset_id=nasdaq_dataset.id,
        n_epochs=1, # Set the number of epochs for fine-tuning
        max_tokens=1024, # Set the maximum number of tokens per example
        learning_rate=0.0001, # Set the learning rate
        batch_size=4, # Set the batch size
    )

    # Check the status of the fine-tuning job
    while True:
        status = openai.FineTune.get(fine_tuning_job.id).status

        if status == "succeeded":
            print("Fine-tuning completed successfully")
            break
        elif status == "failed":
            print("Fine-tuning failed")
            break
        else:
            print("Fine-tuning in progress...")

        time.sleep(60) # Check the status every 60 seconds

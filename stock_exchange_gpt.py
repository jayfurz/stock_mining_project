import openai
import os
import numpy as np
import pandas as pd
import json
import time
import datetime

# Change the following line for notebook
import stock_exchange_split as split

def prepare_finetuning_data(x_data, y_data, output_file, step=5):
    """ This function takes x_data and y_data and puts it into an output jsonl file
    that is formatted for chatGPT fine-tuning. This one takes the stock data and splits it
    into prompt (x) and completion (y) according to a length of "step" days. It then goes up
    another "step" days and does it for the next five days.
    Example use of this function:
    Replace 'split.nasdaq_x_train' and 'split.nyse_y_train' with the appropriate variables
    prepare_finetuning_data(split.nasdaq_x_train, split.nasdaq_y_train, "nasdaq_finetuning_data.jsonl")
    prepare_finetuning_data(split.nyse_x_train, split.nyse_y_train, "nyse_finetuning_data.jsonl")"""
    
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


def fine_tune_gpt_model(train_data_path):
    """This code fine tunes the data using a jsonl file that was made from the previous
    function.
    Example call to Fine-tune the GPT model
    model_id = fine_tune_gpt_model(train_data_path)"""
    
    # Set your API key
    openai.api_key = os.environ["OPENAI_API_KEY"]

    # Upload the NASDAQ dataset
    with open(train_data_path) as f:
    nasdaq_dataset = openai.Dataset.create(file=f, purpose="fine-tuning")
    
    # Upload the formatted training data to GPT (not ready)
    # Set the model to fine-tune
    base_model = "text-davinci-002"

    # Fine-tune the model using the NASDAQ dataset
    fine_tuning_job = openai.FineTune.create(
        model=base_model,
        dataset_id=nasdaq_dataset.id,
        n_epochs=1, # Set the number of epochs for fine-tuning
        max_tokens=400, # Set the maximum number of tokens per example
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
    return fine_tuning_job.model_id


def evaluate_model(model_id, x_test, y_test):
    """This takes the finetuned model that was used before, and then uses the openai Completion
    to get responses from x_test prompts. Then it takes the response choices and compares it to y_test.
    The function then saves the prompts and responses to a json file.
    Example call:
    mean_absolute_difference = evaluate_model(model_id, x_test, y_test)
    print(f"Accuracy of the fine-tuned model: {mean_absolute_difference:.4f}")"""
    openai.api_key = os.environ["OPENAI_API_KEY"]

    total_absolute_difference = 0
    total_values = 0
    results = []

    for x, y_true_list in zip(x_test, y_test):
        prompt = json.dumps([x])
        response = openai.Completion.create(
            engine=model_id,
            prompt=prompt,
            max_tokens=100,  # Increase max_tokens to ensure the whole list is returned
            n=1,
            stop=None,
            temperature=0.5,
        )

        response_text = response.choices[0].text.strip()
        y_pred_list = json.loads(response_text)

        for y_pred, y_true in zip(y_pred_list, y_true_list):
            absolute_difference = abs(y_pred - y_true)
            total_absolute_difference += absolute_difference
            total_values += 1

        results.append({"prompt": prompt, "response": response_text})

    mean_absolute_difference = total_absolute_difference / total_values

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f)

    return mean_absolute_difference

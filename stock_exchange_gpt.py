import openai
import os
import numpy as np
import pandas as pd
import json

# if you are using a notebook, you do not need the following
import stock_exchange_split as split


def prepare_finetuning_data(data_obj, output_file, step=5):
    finetuning_data = []

    x_train_data = data_obj.x_train.values
    y_train_data = data_obj.y_train.values

    for i in range(0, len(x_train_data), step):
        x_data = x_train_data[i:i + step]
        y_data = y_train_data[i:i + step]

        prompt = json.dumps(x_data.tolist())
        completion = json.dumps(y_data.tolist())

        finetuning_data.append({"prompt": prompt, "completion": completion})

    with open(output_file, "w") as f:
        for item in finetuning_data:
            f.write(json.dumps(item) + "\n")

prepare_finetuning_data(split.nasdaq_train_test, "nasdaq_finetuning_data.jsonl")
prepare_finetuning_data(split.nyse_train_test, "nyse_finetuning_data.jsonl")


# Upload the formatted training data to GPT (not ready)
model_name = "ada"
openai.api_key = os.environ["OPENAI_API_KEY"]
#openai.FineTune.create(
#    engine=model_name,
#    prompt=formatted_train_data,
#    max_tokens=1024,
#    n=1,
#    stop=None,
#    temperature=0.5,
#)

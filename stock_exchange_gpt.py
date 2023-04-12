import openai
import os
import numpy as np
import pandas as pd

# if you are using a notebook, you do not need the following
import stock_exchange_split as split


def format_data_for_gpt(x_train, y_train, n=5):
    """
    Format the training data in the required GPT format.
    """
    formatted_data = []

    for i in range(len(x_train)):
        # Extract the previous n rows of data
        prev_data = x_train.iloc[max(0, i - n) : i].to_numpy().tolist()

        # Extract the current row's data
        curr_data = x_train.iloc[i].to_numpy().tolist()

        # Combine the previous n rows and current row's data
        input_data = np.array(prev_data + [curr_data]).flatten().tolist()

        # Convert the input data to a string
        input_str = ", ".join([str(x) for x in input_data])

        # Format the input string for GPT training
        formatted_str = f"{input_str} - {y_train.iloc[i]:.4f}"

        formatted_data.append(formatted_str)

    return formatted_data


# Format the training data for the GPT model (for without a notebook)
formatted_train_data = format_data_for_gpt(
    split.nasdaq_train_test.x_train, split.nasdaq_train_test.y_train
)

# Upload the formatted training data to the OpenAI API
model_name = "ada"
openai.api_key = os.environ["OPENAI_API_KEY"]
response = openai.Completion.create(
    engine=model_name,
    prompt=formatted_train_data,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)
print(response)

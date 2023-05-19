# stock_mining_project

This is the stock mining project by Team Stock Crash Bandacrew for CSUN Comp 541 class.

# stock_exchange_preprocessing.py

The dataset was first preprocessed by using a portion of the dataset (IXIC) and two features were added, percent change and absolute change. Here csv files were made of the preprocessed index in question.  

# stock_exchange_split.py

The dataset was then split into testing and training data for the GPT finetuning and training.

# stock_exchange_gpt.py

The datasets were then split into prompt and completion pairs, 4 days for prompt and 1 day for the completion for fine-tuning. These were saved into JSON files

# GPT Command line calls

Then the json file was sent to openai to train the personal da-vinci model using your own api key.

# model_evaluation.py

The model is then tested using the testing data (2016-2021) and the json file is recorded which shows the prompt, completion, and the actual data. Then the completions were compared with the actual data for evaluation, and the results were printed to the console.

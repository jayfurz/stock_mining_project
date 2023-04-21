import pandas as pd
import json
import os
import openai

ixic = pd.read_csv("^IXIC-2.csv")
ixic["Percent_Change"] = ( ixic["Close"] - ixic["Open"]) / ixic["Open"]
ixic["Total_Change"] = ixic["Close"] - ixic["Open"]
len(ixic)
ixic_short = ixic[-15:]
ixic_short.head()
ixic_short.to_csv('ixic_short.csv', index=False)
prompt_data = ixic_short.values
prompt_json = json.dumps(prompt_data.tolist())
with open("weekly_ixic_data.json", "w") as f:
    f.write(prompt_json)
openai.api_key = os.environ["OPENAI_API_KEY"]
model_id = os.environ["SECOND_TRAINED_MODEL"]
response = openai.Completion.create(
    engine=model_id,
    prompt=prompt_json,
    max_tokens=800,
    n=1,
    stop=None,
    temperature=0.5,
)
response_text = response.choices[0].text.strip()
results = []
results.append({"prompt":prompt_json, "response": response_text})
filename = "ixic_weekly_apr_18_23.json"
json_data = json.dumps(results, indent = 4)
with open(filename, "w") as f:
    f.write(json_data)

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import mplfinance as mpf
import pandas as pd
import os

with open('ixic_weekly_apr_18_23.json', 'r') as file:
    data = json.load(file)
print(data)  
def plot_candlestick(data, title):
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Change', 'ChangeValue', '???'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridaxis="both")
    fig, ax = mpf.plot(df, type='candle', title=title, style=s, ylabel='Price', figratio=(10, 5), tight_layout=True, returnfig=True)

    plt.show()

prompt_data = json.loads(data[0]["prompt"])
response_data = json.loads(data[0]["response"])
combined_data = prompt_data + response_data

index_2023_04_18 = [i for i, x in enumerate(combined_data) if x[0] == "2023-04-18"][0]

# Add a dotted line between 2023-04-18 and 2023-04-19
for i, row in enumerate(combined_data):
    if i == index_2023_04_18:
        combined_data[i].append(True)
    else:
        combined_data[i].append(False)

plot_candlestick(combined_data, "Stock Data with Dotted Line and Gridlines")

import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import mplfinance as mpf
import pandas as pd
import os

with open('ixic_tues_apr_18_23.json', 'r') as file:
    data = json.load(file)
    
def plot_candlestick(data):
    fig, ax = plt.subplots()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Candlestick Chart")

    for row in data:
        open_price, close_price = row[1], row[4]
        high, low = row[2], row[3]
        date = mdates.date2num(datetime.strptime(row[0], '%Y-%m-%d'))

        if close_price > open_price:
            color = 'g'
            ax.bar(date, close_price - open_price, bottom=open_price, color=color, width=0.3)
        else:
            color = 'r'
            ax.bar(date, open_price - close_price, bottom=close_price, color=color, width=0.3)

        ax.vlines(date, low, high, color=color)

    plt.show()

prompt_data = json.loads(data[0]["prompt"])
response_data = json.loads(data[0]["response"])
combined_data = prompt_data + response_data

plot_candlestick(combined_data)

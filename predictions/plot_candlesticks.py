# This python script was used to plot predicted candlesticks on the modern day IXIC
# The prompt for this test was 15 days of data, and the response was 10 days of data proceeding

import ast
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import mplfinance as mpf
import pandas as pd
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot_candlestick_interactive(response_data, actual_data, title):
    response_df = pd.DataFrame(response_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Change', 'ChangeValue'])
    response_df['Date'] = pd.to_datetime(response_df['Date'])
    
    actual_df = pd.DataFrame(actual_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Change', 'ChangeValue'])
    actual_df['Date'] = pd.to_datetime(actual_df['Date'])
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Candlestick(x=response_df['Date'],
                    open=response_df['Open'],
                    high=response_df['High'],
                    low=response_df['Low'],
                    close=response_df['Close'],
                    name="Predicted",
                    increasing_line_color='green',
                    decreasing_line_color='red'),
                    secondary_y=False)

    fig.add_trace(go.Candlestick(x=actual_df['Date'],
                    open=actual_df['Open'],
                    high=actual_df['High'],
                    low=actual_df['Low'],
                    close=actual_df['Close'],
                    name="Actual",
                    increasing_line_color='green',
                    decreasing_line_color='red',
                    visible="legendonly"),
                    secondary_y=False)
    
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, xaxis_title="Date", yaxis_title="Price")
    fig.show()

def plot_candlestick(data, title):
    df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Change', 'ChangeValue'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=":", gridaxis="both")
    fig, ax = mpf.plot(df, type='candle', title=title, style=s, ylabel='Price', figratio=(10, 5), tight_layout=True, returnfig=True)

    plt.show()

def april_18_prediction():
    with open('ixic_weekly_apr_18_23.json', 'r') as file:
        data = json.load(file)
    print(data)  
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

    plot_candlestick(combined_data, "Stock Data Prediction on the IXIC from April 18")

def evaluation_of_test_graph():
    with open('../dump/evaluation_results_20230418_134035.json') as file:
        data = json.load(file)

    response_data = []
    actual_data = []
    for datum in data:
        response_data.append(ast.literal_eval(datum["response"]))
        actual_data.append(ast.literal_eval(datum["actual"]))
    plot_candlestick_interactive(response_data, actual_data, "Model's predicted outcomes vs Actual outcomes from 2016-2021")
    plot_candlestick(response_data, "Model's predicted outcomes from 2016-2021")
    plot_candlestick(actual_data, "Actual outcomes from 2016-2021")
    
evaluation_of_test_graph()


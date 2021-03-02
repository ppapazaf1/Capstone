#import section
import plotly.graph_objs as go
import io
import pandas as pd
import numpy as np
import requests
import time
from datetime import date, datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def get_stocks():
    ''' 
    Get stock symbols
    
    Output
    ------
    symbols: List of stock labels
    
    '''    

    symbols = ['EEE','HTO','OPAP','EUROB','PRODEA','ETE','PPC','BELA','MYTIL','ELPE','TENERGY','MOH']

    return symbols

def clean_stock_data(data):
    ''' 
    Clean stock data
    
    Input
    ------
    data: Dataframe with stock data
    
    Output
    ------
    data: Dataframe with stock data after cleansing
    
    '''        
    # Clean Data
    data['Date']= pd.to_datetime(data['Trade Date'], format='%d/%m/%Y')
    #data = data.drop('Date', axis=1) 
    data = data.drop('Trade Date', axis=1) 
    data = data.drop('Unnamed: 9', axis=1) 
    data = data.drop('Prev. Close', axis=1) 
    print('\n\nDataset after data clean: ')
    print(data.head())
    
    return data


def retrieve_stock_data(symbol):
    ''' 
    Retrieve ALL columns for a reference stock
    
    Input
    ------
    symbol: selected stock symbol
    
    Output
    ------
    data: Dataframe with stock data 
    
    '''         

    # URL where data resides 
    url = 'https://www.naftemporiki.gr/finance/Data/getHistoryData.aspx?symbol='+symbol+'.ATH&type=csv'

    try:
        
        # Get Data
        s = requests.get(url).content
        data = pd.read_csv(io.StringIO(s.decode('utf-8')), delimiter=";", decimal=",", na_values=['Nan']) 
    
        # Initial Explore
        #print('\nRaw Dataset : ')
        #print(data.head()) # print starting rows
        #print(data.shape)    # summarize shape
    except requests.exceptions.RequestException as e:  # Catch Exception 
        raise SystemExit(e)
    
    return data


def plot_data(symbol):
    ''' 
    Create plot data
    
    Input
    ------
    symbol: selected stock symbol
    
    Output
    ------
    data: Dataframe with stock data 
    
    '''        
    
    df_raw = retrieve_stock_data(symbol)    

    data = clean_stock_data(df_raw)

    return data


	
def arima(df, n_pred = 5):
    ''' 
    Create forecasts by using ARIMA model
    
    Input
    ------
    df: Dataframe with stock data
    
    Output
    ------
    predicted_set: List of Predicted values
    history_set : List of Historical values
    
    '''      
    # number of values to be predicted
    split = int(df.shape[0]) 
    
    training_set = df.iloc[:split, 3:4].values
    
    
    # Create and fit ARIMA model   
    hist_set = [x for x in training_set]
    predicted_set = []
    history_set = [item for sublist in training_set for item in sublist]
    
    for time_point in range(n_pred):
        model_init = ARIMA(hist_set, order=(4,1,0)) #(1,1,0)
        #print(time_point)
        model = model_init.fit()
        forecast = model.forecast()
        pred_value = forecast[0]
        #print(pred_value)
        predicted_set.append(pred_value)
        hist_set.append([pred_value])
        history_set.append(pred_value)
    
    print('PREDICTED: ',predicted_set)
    
    return predicted_set,  history_set
    
		
def bollinger_bands(df, symbol, rolling_num = 20):
    '''
    Bollinger Bands is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively)
    away from a simple moving average (SMA) of a stock's price, but which can be adjusted to user preferences.
    Bollinger Bands = mean (+- std) ** 2
    
    Input
    ------
    df: Dataframe
    symbol: Stock symbol
    start_d: Start Date ('yyyy-mm-dd')
    end_d: End Date ('yyyy-mm-dd')
    rolling_num: no of past days to be used
    
    Output
    ------  
    df_rol: Dataframe with bollinger bands related values
    '''

    rol = df['Close'].rolling(rolling_num).mean()
    rol_std = df['Close'].rolling(rolling_num).std()
    rol_std_up = rol + 2 * rol_std
    rol_std_down = rol - 2 * rol_std

    df_rol = pd.DataFrame()
    df_rol['Price'] = df['Close']
    df_rol['STD Above'] = rol_std_up
    df_rol['STD Below'] = rol_std_down
    
    return df_rol


def daily_returns(df):
    '''
    Daily Returns measures the change in a stock's price as a percentage of the previous day's closing price.
    A stock with lower positive and negative daily returns is typically less risky than a stock with higher daily returns.
    
    Input
    ------
    df: Dataframe
    symbol: Stock symbol
    start_d: Start Date ('yyyy-mm-dd')
    end_d: End Date ('yyyy-mm-dd')
    
    Output
    ------  
    daily_ret: Dataframe with daily returns
    '''    
    daily_ret = (df['Open']/df['Open'].shift(1))-1
    daily_ret.iloc[0,:] = 0
    daily_ret.dropna(inplace=True)
    
    return daily_ret

def sharpe_ratio(daily_ret):
    '''
    Daily Returns measures the change in a stock's price as a percentage of the previous day's closing price.
    A stock with lower positive and negative daily returns is typically less risky than a stock with higher daily returns.
    
    Input
    ------
    daily_ret: Dataframe with daily returns
    
    Output
    ------  
    sharpe_ratio: Sharpe Ratio

    '''    
    # Cummulative return
    #cum_ret = (data['Open'][-1:].values/data['Open'][:1].values)-1
    #print('Cummulative return:',cum_ret)

    # Average Daily Return
    avg_daily_ret = daily_ret.mean()
    #print('Average Daily Return:', avg_daily_ret)
    
    # Standard Daily Return
    std_daily_ret = daily_ret.std()
    
    # Contant definition
    _daily_rf_ = 0.0002

    # Daily Sampling for year
    k = np.sqrt(252)

    # Sharpe Ratio
    sharpe_ratio = k * (avg_daily_ret - _daily_rf_) / std_daily_ret
    print('\nSharp Ratio:',sharpe_ratio)
    return sharpe_ratio


def return_figures(symbol, pred_days):
    '''
    Creates  plotly visualizations

    Input
    ------
    symbol: selected stock symbol
    pred_days: the days forward to create forecasts
    
    Output
    ------
    list (dict): list containing the  plotly visualizations

    '''

    data = plot_data(symbol)

    data_bol = bollinger_bands(data, symbol)

    pred, history = arima(data, int(pred_days))

    daily_ret = daily_returns(data)
    
    sharp_ratio = round(sharpe_ratio(daily_ret),3)
    # Standard Deviation of Daily Returns  -- Risk
    #std_daily_ret = daily_ret.std()

    #print('\nCummulative return:',cum_ret, '\nAverage Daily Return:', avg_daily_ret, '\nRisk (Standard Deviation of Daily Returns):', std_daily_ret)
    

  # first chart

    graph_one = []

    graph_one.append(
      go.Scatter(
      y = data_bol['Price'].tolist(),
      )  
    )
    graph_one.append(
      go.Scatter(
      y = data_bol['STD Above'].tolist(),
      )  
    )
    graph_one.append(
      go.Scatter(
      y = data_bol['STD Below'].tolist(),
      )  
    )        

    layout_one = dict(title = 'Bollinger Bands',
                xaxis = dict(title = 'Time'),
                #autotick=False, tick0=1990, dtick=25),
                yaxis = dict(title = 'Price'),
                )  
    
    # second chart  
    graph_two = []

    graph_two.append(
      go.Scatter(
      #x = data['Date'].tolist(),
      y = pred,
      )      
    )


    layout_two = dict(title = 'Stock Price Prediction for ' + symbol,
                xaxis = dict(title = 'Time',),
                yaxis = dict(title = 'Price'))
  

    
    # Stock Graph

    graph_three = []


    graph_three.append(
      go.Scatter(
      x = data['Date'].tolist(),
      y = data['Close'].tolist(),
      )
    )

    layout_three = dict(title = 'Stock Prices for ' + symbol,
                xaxis = dict(title = 'Date',),
                yaxis = dict(title = 'Price'))
    
    # append all charts to the figures list
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))

    return figures, sharp_ratio    

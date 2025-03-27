import configparser
import pandas as pd
#from binance.client import Client

#config = configparser.RawConfigParser()
#config.read('config/Config.properties')
#API_KEY = config.get('BINANCE', 'binance.apikey')
#API_SECRET = config.get('BINANCE', 'binance.secret')

#client = Client(API_KEY, API_SECRET, {"timeout": 20})

def get_data_frame(symbol, interval, lookback, interval_type):
#    frame = pd.DataFrame(client.get_historical_klines(symbol, interval, lookback + ' ' + interval_type + ' ' + 'ago UTC'))
    frame = {}
    if (frame.empty):
        return frame
    frame = frame.iloc[:, :6]
    frame.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    frame['Time'] = pd.to_datetime(frame['Time'], unit='ms')
    frame = frame.set_index('Time')
    frame['Open'] = frame.Open.astype(float)
    frame['High'] = frame.High.astype(float)
    frame['Low'] = frame.Low.astype(float)
    frame['Close'] = frame.Close.astype(float)
    frame['Volume'] = frame.Volume.astype(float)
    return frame

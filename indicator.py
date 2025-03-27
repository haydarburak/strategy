from enum import Enum
import pandas_ta as ta
import pandas as pd

class IndicatorEnum(Enum):
    EMA = 1
    RSI = 2
    MACD = 3
    STOCH = 4

def ma(Data, lookback, what, where):
    for i in range(len(Data)):
        try:
            Data[i, where] = (Data[i - lookback + 1:i + 1, what].mean())

        except IndexError:
            pass

    return Data

# Commodity Channel Index
def cci(Data, lookback, where, constant):
    # Calculating Typical Price
    Data[:, where] = (Data[:, 1] + Data[:, 2] + Data[:, 3]) / 3

    # Calculating the Absolute Mean Deviation
    specimen = Data[:, where]
    MAD_Data = pd.Series(specimen)

    for i in range(len(Data)):
        Data[i, where + 1] = MAD_Data[i - lookback:i].mad()

    # Calculating Mean of Typical Price
    Data = ma(Data, lookback, where, where + 2)

    # CCI
    for i in range(len(Data)):
        Data[i, where + 3] = (Data[i, where] - Data[i, where + 2]) / (constant * Data[i, where + 1])

    return Data


def add_indicator(frame, indicator_enum, period):
    if len(frame) > 200:
        if indicator_enum == IndicatorEnum.EMA:
            frame['EMA' + str(period)] = frame['Close'].ewm(span=period, adjust=False).mean()

        elif indicator_enum == IndicatorEnum.RSI:
            rsi = ta.momentum.rsi(frame['Close'], window=period)
            if rsi is not None:
                frame['RSI' + str(period)] = rsi

        elif indicator_enum == IndicatorEnum.MACD:
            macd = ta.macd(frame['Close'], period, 100, 9)
            if macd is not None:
                frame['MACD'] = macd.get(f"MACD_{period}_100_9", None)
                frame['MACD_S'] = macd.get(f"MACDs_{period}_100_9", None)
        elif indicator_enum == IndicatorEnum.STOCH:
            # Veri kontrolü
            if frame[['High', 'Low', 'Close']].isnull().any().any():
                print("Error: High, Low, or Close contains NaN values.")
                return frame

            # STOCH hesaplama
            try:
                stoch = ta.stoch(frame['High'], frame['Low'], frame['Close'], period, 3, 3)
                if stoch is not None and f"STOCHk_{period}_3_3" in stoch.columns:
                    frame['STOCH_K'] = stoch[f"STOCHk_{period}_3_3"]
                    frame['STOCH_D'] = stoch[f"STOCHd_{period}_3_3"]
                else:
                    print("Warning: Stochastic calculation returned None or invalid columns.")
            except Exception as e:
                print(f"Error using pandas-ta: {e} ")

    return frame

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_rsi_divergence(df, min_bars=30):
    df = df.copy()

    df['RSI'] = df['RSI14']

    df['Bearish_Divergence'] = np.nan
    df['Bullish_Divergence'] = np.nan

    # Previous highs and lows
    past_high = df.iloc[len(df) - min_bars -1:len(df)-1]['Close'].max()
    past_low = df.iloc[len(df) - min_bars -1:len(df)-1]['Close'].min()

    past_rsi_high = df.iloc[len(df) - min_bars -1:len(df)-1]['RSI'].max()
    past_rsi_low = df.iloc[len(df) - min_bars -1:len(df)-1]['RSI'].min()

    # Current price and RSI
    curr_price = df.iloc[-1]['Close']
    curr_rsi = df.iloc[-1]['RSI']

    # Bearish divergence (price higher high, RSI lower high)
    if past_rsi_high > 70 and curr_price > past_high and curr_rsi < past_rsi_high:
        df.at[df.index[-1], 'Bearish_Divergence'] = curr_price
    else:
        df.at[df.index[-1], 'Bearish_Divergence'] = 0

    # Bullish divergence (price lower low, RSI higher low)
    if past_rsi_low < 30 and curr_price < past_low and curr_rsi > past_rsi_low:
        df.at[df.index[-1], 'Bullish_Divergence'] = curr_price
    else:
        df.at[df.index[-1], 'Bullish_Divergence'] = 0

#    plot_rsi_divergence(df)

    return df

def plot_rsi_divergence(df):
    """
    Plots RSI divergence on price and RSI charts.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['EMA20'], label='EMA20', color='orange')
    ax1.plot(df.index, df['EMA50'], label='EMA50', color='purple')
    ax1.plot(df.index, df['EMA100'], label='EMA100', color='green')
    ax1.plot(df.index, df['EMA200'], label='EMA200', color='red')

    ax1.scatter(df.index, df['Bearish_Divergence'], color='red', marker='v', s=100, label='Bearish Divergence')
    ax1.scatter(df.index, df['Bullish_Divergence'], color='green', marker='^', s=100, label='Bullish Divergence')
    ax1.set_title('Stock Price with RSI Divergences')
    ax1.legend()

    ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
    ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
    ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
    ax2.set_title('Relative Strength Index (RSI)')
    ax2.legend()

    plt.show()


# Örnek Kullanım:
if __name__ == "__main__":
    # Rastgele fiyat verisi üretelim (örnek amaçlı)
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    prices = np.cumsum(np.random.randn(100)) + 100  # Fiyat serisi

    df = pd.DataFrame({'Date': dates, 'Close': prices})
    df.set_index('Date', inplace=True)

    # RSI divergence'ları bul
    df = find_rsi_divergence(df)

    # Grafiği çiz
    plot_rsi_divergence(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Proper pivot-based divergence detection (regular + hidden)
# ---------------------------------------------------------------------------

def _find_pivot_highs(series: np.ndarray, left: int, right: int):
    """
    Return indices of confirmed swing highs.
    A bar at index i is a swing high if series[i] is the highest value
    in the window [i-left … i+right].
    """
    pivots = []
    for i in range(left, len(series) - right):
        window = series[i - left: i + right + 1]
        if series[i] == window.max() and series[i] > series[i - 1]:
            pivots.append(i)
    return pivots


def _find_pivot_lows(series: np.ndarray, left: int, right: int):
    """
    Return indices of confirmed swing lows.
    A bar at index i is a swing low if series[i] is the lowest value
    in the window [i-left … i+right].
    """
    pivots = []
    for i in range(left, len(series) - right):
        window = series[i - left: i + right + 1]
        if series[i] == window.min() and series[i] < series[i - 1]:
            pivots.append(i)
    return pivots


def find_rsi_divergence_v2(df, left: int = 5, right: int = 5,
                            rsi_ob: float = 70.0, rsi_os: float = 30.0,
                            rsi_min_spread: float = 5.0,
                            hidden_bull_rsi_max: float = 55.0,
                            hidden_bear_rsi_min: float = 45.0):
    """
    Proper pivot-based RSI divergence detector.

    Detects four divergence types:
      - Bearish          : price higher high  + RSI lower high  (reversal warning)
      - Bullish          : price lower low    + RSI higher low  (reversal warning)
      - Hidden Bearish   : price lower high   + RSI higher high (trend continuation short)
      - Hidden Bullish   : price higher low   + RSI lower low   (trend continuation long)

    Both price AND RSI are compared at the same pivot bar, so the comparison
    is structurally valid — unlike comparing separate window max/min values.

    Quality filters (prevent RSI-neutral noise):
      rsi_min_spread     : minimum RSI difference between the two compared pivots.
                           Eliminates micro-oscillation false positives (default 5).
      hidden_bull_rsi_max: hidden bullish only fires when RSI at p2 is BELOW this
                           threshold — signal must still be in "weak / recovering"
                           territory, not already overbought (default 55).
      hidden_bear_rsi_min: hidden bearish only fires when RSI at p2 is ABOVE this
                           threshold — signal must be in "recovering but failing"
                           territory, not already oversold (default 45).

    Confirmed pivots only (right bars must have already closed), so no
    signals fire on live unclosed candles.

    Parameters
    ----------
    df       : DataFrame with columns Close, High, Low, RSI14
    left     : bars to the left of the pivot required to confirm it
    right    : bars to the right of the pivot required to confirm it
    rsi_ob   : RSI overbought threshold for regular bearish divergence
    rsi_os   : RSI oversold threshold for regular bullish divergence

    Returns
    -------
    df with four new signal columns (float, 0 = no signal, >0 = close price at p2):
        Bearish_Divergence_V2, Bullish_Divergence_V2,
        Hidden_Bearish_Divergence, Hidden_Bullish_Divergence

    Plus a metadata dict stored on df.attrs['divergence_meta']:
        {
          col_name: {
            'p1_price': float, 'p1_rsi': float,
            'p2_price': float, 'p2_rsi': float,
            'price_label': str,   # e.g. 'Higher High'
            'rsi_label':   str,   # e.g. 'Lower High'
            'reason':      str,   # human-readable one-liner
          }
        }
    Only the most recently detected pivot pair per column is stored.
    """
    df = df.copy()
    df['Bearish_Divergence_V2']    = 0.0
    df['Bullish_Divergence_V2']    = 0.0
    df['Hidden_Bearish_Divergence'] = 0.0
    df['Hidden_Bullish_Divergence'] = 0.0
    meta: dict = {}          # populated below, stored in df.attrs
    df.attrs['divergence_meta'] = meta

    if 'RSI14' not in df.columns or len(df) < left + right + 2:
        return df

    rsi   = df['RSI14'].values
    high  = df['High'].values
    low   = df['Low'].values
    close = df['Close'].values

    pivot_highs = _find_pivot_highs(high,  left, right)
    pivot_lows  = _find_pivot_lows(low,    left, right)

    # --- Bearish & Hidden Bearish (compare consecutive pivot highs) ---
    for i in range(1, len(pivot_highs)):
        p1, p2 = pivot_highs[i - 1], pivot_highs[i]

        rsi_spread = abs(rsi[p2] - rsi[p1])
        if rsi_spread < rsi_min_spread:
            # RSI difference too small — likely noise, not a real divergence
            continue

        price_hh = high[p2] > high[p1]   # price: higher high
        price_lh = high[p2] < high[p1]   # price: lower high
        rsi_lh   = rsi[p2]  < rsi[p1]    # RSI:   lower high
        rsi_hh   = rsi[p2]  > rsi[p1]    # RSI:   higher high

        # Regular bearish: price HH + RSI LH — must start from overbought territory
        if price_hh and rsi_lh and rsi[p1] >= rsi_ob:
            col = 'Bearish_Divergence_V2'
            df.iat[p2, df.columns.get_loc(col)] = close[p2]
            meta[col] = {
                'p1_price': round(float(high[p1]), 4),
                'p1_rsi':   round(float(rsi[p1]),  2),
                'p2_price': round(float(high[p2]), 4),
                'p2_rsi':   round(float(rsi[p2]),  2),
                'price_label': 'Higher High',
                'rsi_label':   'Lower High',
                'reason': (
                    f"Price made a Higher High ({high[p1]:.2f} → {high[p2]:.2f}) "
                    f"but RSI made a Lower High ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"from overbought — momentum failing at the top."
                ),
            }

        # Hidden bearish: price LH + RSI HH — p2 RSI must still be above midline
        if price_lh and rsi_hh and rsi[p2] >= hidden_bear_rsi_min:
            col = 'Hidden_Bearish_Divergence'
            df.iat[p2, df.columns.get_loc(col)] = close[p2]
            meta[col] = {
                'p1_price': round(float(high[p1]), 4),
                'p1_rsi':   round(float(rsi[p1]),  2),
                'p2_price': round(float(high[p2]), 4),
                'p2_rsi':   round(float(rsi[p2]),  2),
                'price_label': 'Lower High',
                'rsi_label':   'Higher High',
                'reason': (
                    f"Price made a Lower High ({high[p1]:.2f} → {high[p2]:.2f}) "
                    f"but RSI made a Higher High ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"— bounce is weakening while RSI recovered, downtrend continues."
                ),
            }

    # --- Bullish & Hidden Bullish (compare consecutive pivot lows) ---
    for i in range(1, len(pivot_lows)):
        p1, p2 = pivot_lows[i - 1], pivot_lows[i]

        rsi_spread = abs(rsi[p2] - rsi[p1])
        if rsi_spread < rsi_min_spread:
            continue

        price_ll = low[p2] < low[p1]    # price: lower low
        price_hl = low[p2] > low[p1]    # price: higher low
        rsi_hl   = rsi[p2] > rsi[p1]   # RSI:   higher low
        rsi_ll   = rsi[p2] < rsi[p1]   # RSI:   lower low

        # Regular bullish: price LL + RSI HL — must start from oversold territory
        if price_ll and rsi_hl and rsi[p1] <= rsi_os:
            col = 'Bullish_Divergence_V2'
            df.iat[p2, df.columns.get_loc(col)] = close[p2]
            meta[col] = {
                'p1_price': round(float(low[p1]), 4),
                'p1_rsi':   round(float(rsi[p1]), 2),
                'p2_price': round(float(low[p2]), 4),
                'p2_rsi':   round(float(rsi[p2]), 2),
                'price_label': 'Lower Low',
                'rsi_label':   'Higher Low',
                'reason': (
                    f"Price made a Lower Low ({low[p1]:.2f} → {low[p2]:.2f}) "
                    f"but RSI made a Higher Low ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"from oversold — sellers exhausted, reversal likely."
                ),
            }

        # Hidden bullish: price HL + RSI LL — p2 RSI must still be below midline
        # (i.e. price is making higher lows in an uptrend while RSI pulled back
        #  below 55 → the dip is healthy, trend continues up)
        if price_hl and rsi_ll and rsi[p2] <= hidden_bull_rsi_max:
            col = 'Hidden_Bullish_Divergence'
            df.iat[p2, df.columns.get_loc(col)] = close[p2]
            meta[col] = {
                'p1_price': round(float(low[p1]), 4),
                'p1_rsi':   round(float(rsi[p1]), 2),
                'p2_price': round(float(low[p2]), 4),
                'p2_rsi':   round(float(rsi[p2]), 2),
                'price_label': 'Higher Low',
                'rsi_label':   'Lower Low',
                'reason': (
                    f"Price made a Higher Low ({low[p1]:.2f} → {low[p2]:.2f}) "
                    f"but RSI made a Lower Low ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"— dip is shallower while RSI resets, uptrend continues."
                ),
            }

    return df

# ---------------------------------------------------------------------------
# Original (legacy) divergence — kept unchanged
# ---------------------------------------------------------------------------

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

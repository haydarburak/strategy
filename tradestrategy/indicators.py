import pandas as pd
import pandas_ta as ta

REQUIRED_COLUMNS = ['EMA20', 'EMA50', 'EMA100', 'EMA200', 'STOCH_K', 'STOCH_D', 'MACD', 'MACD_S']
_EMA_PERIODS = [20, 50, 100, 200]


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach all required indicators to df.
    Expects OHLCV columns (Open, High, Low, Close, Volume).
    Returns a new DataFrame with NaN rows dropped.
    Raises ValueError if fewer than 200 rows are provided.
    """
    if len(df) < 200:
        raise ValueError(f"Need at least 200 rows; got {len(df)}.")

    df = df.copy()
    df = _add_emas(df)
    df = _add_stoch(df)
    df = _add_macd(df)
    df = _add_rsi(df)
    return df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=False)


def has_required_columns(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in REQUIRED_COLUMNS)


# ── private helpers ────────────────────────────────────────────────────────────

def _add_emas(df: pd.DataFrame) -> pd.DataFrame:
    for p in _EMA_PERIODS:
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    return df


def _add_stoch(df: pd.DataFrame, k: int = 5, d: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    result = ta.stoch(df['High'], df['Low'], df['Close'], k=k, d=d, smooth_k=smooth_k)
    if result is not None:
        df['STOCH_K'] = result[f'STOCHk_{k}_{d}_{smooth_k}']
        df['STOCH_D'] = result[f'STOCHd_{k}_{d}_{smooth_k}']
    return df


def _add_macd(df: pd.DataFrame, fast: int = 50, slow: int = 100, signal: int = 9) -> pd.DataFrame:
    result = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
    if result is not None:
        df['MACD'] = result[f'MACD_{fast}_{slow}_{signal}']
        df['MACD_S'] = result[f'MACDs_{fast}_{slow}_{signal}']
    return df


def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df[f'RSI{period}'] = ta.rsi(df['Close'], length=period)
    return df

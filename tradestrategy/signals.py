"""
Signal generator — combines common entry filters with the four patterns.

Usage:
    import pandas as pd
    from tradestrategy import add_all, generate_signals

    df = pd.read_csv('AAPL.csv', parse_dates=['Date'], index_col='Date')
    df = add_all(df)
    signals = generate_signals(df)
    for s in signals:
        print(s)
"""

from typing import List

import pandas as pd

from . import indicators
from .patterns import (
    Direction, Signal,
    two_candle_reversal,
    double_tail_piercing,
    body_piercing,
    single_candle_reversal,
)

# ── thresholds ─────────────────────────────────────────────────────────────────
_STOCH_OVERSOLD   = 40
_STOCH_OVERBOUGHT = 65

# candle indices
_A = -1   # approve
_R = -2   # reversal
_I = -3   # initial

_EMAS = ['EMA20', 'EMA50', 'EMA100', 'EMA200']

_ALL_PATTERNS = [
    two_candle_reversal,
    double_tail_piercing,
    body_piercing,
    single_candle_reversal,
]


# ── common entry filters ───────────────────────────────────────────────────────

def _ema_bull(df: pd.DataFrame) -> bool:
    """EMA20 > EMA50 > EMA100 > EMA200 at the approve candle."""
    return all(
        df[_EMAS[i]].iloc[_A] > df[_EMAS[i + 1]].iloc[_A]
        for i in range(len(_EMAS) - 1)
    )


def _ema_bear(df: pd.DataFrame) -> bool:
    """EMA20 < EMA50 < EMA100 < EMA200 at the approve candle."""
    return all(
        df[_EMAS[i]].iloc[_A] < df[_EMAS[i + 1]].iloc[_A]
        for i in range(len(_EMAS) - 1)
    )


def _stoch_turning_up(df: pd.DataFrame) -> bool:
    """Stochastic is oversold and K has crossed above D."""
    k, d = df['STOCH_K'].iloc[_A], df['STOCH_D'].iloc[_A]
    return (k < _STOCH_OVERSOLD or d < _STOCH_OVERSOLD) and k > d


def _stoch_overbought(df: pd.DataFrame) -> bool:
    k, d = df['STOCH_K'].iloc[_A], df['STOCH_D'].iloc[_A]
    return k > _STOCH_OVERBOUGHT or d > _STOCH_OVERBOUGHT


def _macd_bullish(df: pd.DataFrame) -> bool:
    """MACD > Signal on at least one of the three candles."""
    return any(df['MACD'].iloc[i] > df['MACD_S'].iloc[i] for i in [_A, _R, _I])


def _macd_bearish(df: pd.DataFrame) -> bool:
    """MACD < Signal on at least one of the three candles."""
    return any(df['MACD'].iloc[i] < df['MACD_S'].iloc[i] for i in [_A, _R, _I])


def _candle_structure_long(df: pd.DataFrame) -> bool:
    """
    Long candle structure:
    - Initial candle is bearish (sets up the pullback)
    - Approve candle makes a higher low than reversal (momentum turning)
    - Approve close breaks above the reversal candle's high (breakout)
    """
    return (
        df['Close'].iloc[_I]  < df['Open'].iloc[_I]        # initial: bearish
        and df['Low'].iloc[_A]   > df['Low'].iloc[_R]      # higher low
        and df['Close'].iloc[_A] > df['High'].iloc[_R]     # breakout above reversal high
    )


def _candle_structure_short(df: pd.DataFrame) -> bool:
    """
    Short candle structure:
    - Reversal candle's low stays above the approve candle's close (gap / rejection)
    - Reversal made the higher high of the pair
    """
    return (
        df['Low'].iloc[_R]  > df['Close'].iloc[_A]         # reversal low above approve close
        and df['High'].iloc[_R] > df['High'].iloc[_A]      # reversal made the higher high
    )


def _long_conditions_met(df: pd.DataFrame) -> bool:
    return (
        _ema_bull(df)
        and _stoch_turning_up(df)
        and _macd_bullish(df)
        and _candle_structure_long(df)
    )


def _short_conditions_met(df: pd.DataFrame) -> bool:
    return (
        _ema_bear(df)
        and _stoch_overbought(df)
        and _macd_bearish(df)
        and _candle_structure_short(df)
    )


# ── public API ─────────────────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> List[Signal]:
    """
    Run all four patterns on `df` and return every triggered Signal.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain OHLCV columns and indicator columns produced by
        ``indicators.add_all()``.  At least 3 rows are required.

    Returns
    -------
    list[Signal]
        Each Signal carries .pattern, .direction, and .triggered_ema.
        An empty list means no pattern fired on the current bar.

    Raises
    ------
    ValueError
        If the required indicator columns are missing.
    """
    if not indicators.has_required_columns(df):
        raise ValueError(
            "Missing indicator columns. Call indicators.add_all(df) before generate_signals()."
        )
    if len(df) < 3:
        return []

    signals: List[Signal] = []

    for direction, common_ok in (
        (Direction.LONG,  _long_conditions_met(df)),
        (Direction.SHORT, _short_conditions_met(df)),
    ):
        if not common_ok:
            continue
        for pattern_fn in _ALL_PATTERNS:
            sig = pattern_fn(df, direction)
            if sig is not None:
                signals.append(sig)

    return signals

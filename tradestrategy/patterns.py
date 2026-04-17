"""
Four candlestick reversal patterns, each returning an Optional[Signal].

Candle layout (relative to the last row of the DataFrame):
  _I = -3  initial   — sets up the move
  _R = -2  reversal  — the key reversal candle
  _A = -1  approve   — confirms the reversal

Each pattern is tested against EMA20, EMA50, EMA100, EMA200 in order;
the first EMA that satisfies all conditions triggers the signal.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

_I = -3   # initial candle
_R = -2   # reversal candle
_A = -1   # approve (confirmation) candle

_EMAS = ['EMA20', 'EMA50', 'EMA100', 'EMA200']


class Direction(str, Enum):
    LONG  = 'LONG'
    SHORT = 'SHORT'


@dataclass(frozen=True)
class Signal:
    pattern:       str
    direction:     Direction
    triggered_ema: str


# ── candlestick shape helpers ──────────────────────────────────────────────────

def _candle(df: pd.DataFrame, idx: int):
    o = df['Open'].iloc[idx]
    h = df['High'].iloc[idx]
    l = df['Low'].iloc[idx]
    c = df['Close'].iloc[idx]
    body        = abs(c - o)
    upper_wick  = h - max(o, c)
    lower_wick  = min(o, c) - l
    return o, h, l, c, body, upper_wick, lower_wick


def is_hammer(df: pd.DataFrame, idx: int) -> bool:
    """Small body at the top, lower wick ≥ 2× body, upper wick ≤ 0.5× body."""
    _, _, _, _, body, upper_wick, lower_wick = _candle(df, idx)
    if body == 0:
        return False
    return lower_wick >= 2 * body and upper_wick <= 0.5 * body


def is_inverted_hammer(df: pd.DataFrame, idx: int) -> bool:
    """Small body at the bottom, upper wick ≥ 2× body, lower wick ≤ 0.5× body."""
    _, _, _, _, body, upper_wick, lower_wick = _candle(df, idx)
    if body == 0:
        return False
    return upper_wick >= 2 * body and lower_wick <= 0.5 * body


def is_gravestone_doji(df: pd.DataFrame, idx: int) -> bool:
    """Near-zero body, long upper wick, almost no lower wick."""
    _, h, l, _, body, upper_wick, lower_wick = _candle(df, idx)
    candle_range = h - l
    if candle_range == 0:
        return False
    return (body / candle_range) < 0.05 and upper_wick > lower_wick * 3


# ── Pattern 1 — Two-Candle Reversal (İki Mum Çubuklu Dönüş) ──────────────────
#
# Long:  initial is bearish → reversal wicks below EMA but body is above →
#        reversal made the lower low of the pair.
#
# Short: initial is bullish → reversal wicks above EMA but body is below →
#        reversal made the higher high of the pair.

def _p1_long(df: pd.DataFrame, ema: str) -> bool:
    return (
        df['Open'].iloc[_I]  > df['Close'].iloc[_I]    # initial: bearish
        and df['Low'].iloc[_R]   < df[ema].iloc[_R]    # reversal: low wicks below EMA
        and df['Open'].iloc[_R]  > df[ema].iloc[_R]    # reversal: opens above EMA
        and df['Close'].iloc[_R] > df[ema].iloc[_R]    # reversal: closes above EMA
        and df['Low'].iloc[_I]   > df['Low'].iloc[_R]  # reversal made the lower low
    )


def _p1_short(df: pd.DataFrame, ema: str) -> bool:
    return (
        df['Close'].iloc[_I]  > df['Open'].iloc[_I]    # initial: bullish
        and df['Close'].iloc[_R] < df[ema].iloc[_R]    # reversal: closes below EMA
        and df['Open'].iloc[_R]  < df[ema].iloc[_R]    # reversal: opens below EMA
        and df['High'].iloc[_R]  > df[ema].iloc[_R]    # reversal: wicks above EMA
        and df['High'].iloc[_R]  > df['High'].iloc[_I] # reversal made the higher high
    )


def two_candle_reversal(df: pd.DataFrame, direction: Direction) -> Optional[Signal]:
    check = _p1_long if direction == Direction.LONG else _p1_short
    for ema in _EMAS:
        if check(df, ema):
            return Signal('TWO_CANDLE_REVERSAL', direction, ema)
    return None


# ── Pattern 2 — Double Tail Piercing (Çift Kuyruk Deliş) ─────────────────────
#
# Long:  two consecutive candles both wick below the EMA with bodies above it;
#        the initial candle has the deeper low.
#
# Short: two consecutive candles both wick above the EMA with bodies below it.

def _p2_long(df: pd.DataFrame, ema: str) -> bool:
    return (
        df['Low'].iloc[_I]   < df['Low'].iloc[_R]      # initial has the deeper low
        and df['Low'].iloc[_R]   < df[ema].iloc[_R]    # reversal wicks below EMA
        and df['Open'].iloc[_R]  > df[ema].iloc[_R]    # reversal body above EMA
        and df['Close'].iloc[_R] > df[ema].iloc[_R]
        and df['Low'].iloc[_I]   < df[ema].iloc[_I]    # initial also wicks below EMA
        and df['Open'].iloc[_I]  > df[ema].iloc[_I]    # initial body above EMA
        and df['Close'].iloc[_I] > df[ema].iloc[_I]
    )


def _p2_short(df: pd.DataFrame, ema: str) -> bool:
    return (
        df['Close'].iloc[_I] > df['Open'].iloc[_I]     # initial candle is bullish …
        and df['Open'].iloc[_I]  < df[ema].iloc[_I]    # … but body is entirely below EMA
        and df['Close'].iloc[_I] < df[ema].iloc[_I]
        and df['Open'].iloc[_R]  < df[ema].iloc[_R]    # reversal body also below EMA
        and df['Close'].iloc[_R] < df[ema].iloc[_R]
        and df['High'].iloc[_I]  > df[ema].iloc[_I]    # initial wicks above EMA
        and df['High'].iloc[_R]  > df[ema].iloc[_R]    # reversal wicks above EMA
    )


def double_tail_piercing(df: pd.DataFrame, direction: Direction) -> Optional[Signal]:
    check = _p2_long if direction == Direction.LONG else _p2_short
    for ema in _EMAS:
        if check(df, ema):
            return Signal('DOUBLE_TAIL_PIERCING', direction, ema)
    return None


# ── Pattern 3 — Body Piercing (Gövde Deliş) ───────────────────────────────────
#
# Long:  two candles both open above the EMA and wick below it, but their bodies
#        cross through the EMA (close < EMA) — EMA acts as magnet/resistance being
#        tested from above; reversal made the lower low.
#
# Short: initial bullish body crosses the EMA upward (open < EMA, close > EMA);
#        reversal is bearish and crosses it downward (open > EMA, close < EMA).

def _p3_long(df: pd.DataFrame, ema: str) -> bool:
    return (
        df['Low'].iloc[_I]   > df['Low'].iloc[_R]      # reversal made the lower low
        and df['Low'].iloc[_R]   < df[ema].iloc[_R]    # reversal wicks below EMA
        and df['Open'].iloc[_R]  > df[ema].iloc[_R]    # reversal opens above EMA
        and df['Close'].iloc[_R] < df[ema].iloc[_R]    # reversal body pierces EMA downward
        and df['Low'].iloc[_I]   < df[ema].iloc[_I]    # initial also wicks below EMA
        and df['Open'].iloc[_I]  > df[ema].iloc[_I]    # initial opens above EMA
        and df['Close'].iloc[_I] < df[ema].iloc[_I]    # initial body pierces EMA downward
    )


def _p3_short(df: pd.DataFrame, ema: str) -> bool:
    return (
        df['Close'].iloc[_I] > df['Open'].iloc[_I]     # initial: bullish
        and df['Close'].iloc[_R] < df['Open'].iloc[_R] # reversal: bearish
        and df['Open'].iloc[_I]  < df[ema].iloc[_I]    # initial opens below EMA …
        and df['Close'].iloc[_I] > df[ema].iloc[_I]    # … closes above (upward pierce)
        and df['Open'].iloc[_R]  > df[ema].iloc[_R]    # reversal opens above EMA …
        and df['Close'].iloc[_R] < df[ema].iloc[_R]    # … closes below (downward pierce)
    )


def body_piercing(df: pd.DataFrame, direction: Direction) -> Optional[Signal]:
    check = _p3_long if direction == Direction.LONG else _p3_short
    for ema in _EMAS:
        if check(df, ema):
            return Signal('BODY_PIERCING', direction, ema)
    return None


# ── Pattern 4 — Single Candle Reversal (Tek Mum Dönüş) ───────────────────────
#
# Long:  reversal candle is a Hammer that wicks below the EMA and closes above;
#        confirmation candle closes higher than the reversal.
#
# Short: reversal candle is an Inverted Hammer or Gravestone Doji that wicks
#        above the EMA while body is below.

def _p4_long(df: pd.DataFrame, ema: str) -> bool:
    return (
        df['Low'].iloc[_I]    > df['Low'].iloc[_R]      # reversal made the lower low
        and df['Low'].iloc[_R]    < df[ema].iloc[_R]    # reversal wicks below EMA
        and df['Open'].iloc[_R]   > df[ema].iloc[_R]    # reversal opens above EMA
        and df['Close'].iloc[_R]  > df[ema].iloc[_R]    # reversal closes above EMA
        and df['Close'].iloc[_A]  > df['Close'].iloc[_R]# confirmation closes higher
        and is_hammer(df, _R)                            # reversal is a hammer
    )


def _p4_short(df: pd.DataFrame, ema: str) -> bool:
    return (
        (is_inverted_hammer(df, _R) or is_gravestone_doji(df, _R))
        and df['High'].iloc[_R]  > df[ema].iloc[_R]     # reversal wicks above EMA
        and df['Open'].iloc[_R]  < df[ema].iloc[_R]     # reversal opens below EMA
        and df['Close'].iloc[_R] < df[ema].iloc[_R]     # reversal closes below EMA
    )


def single_candle_reversal(df: pd.DataFrame, direction: Direction) -> Optional[Signal]:
    check = _p4_long if direction == Direction.LONG else _p4_short
    for ema in _EMAS:
        if check(df, ema):
            return Signal('SINGLE_CANDLE_REVERSAL', direction, ema)
    return None

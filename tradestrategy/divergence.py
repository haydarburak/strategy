"""
Pivot-based RSI divergence detector.

Four divergence types
---------------------
  bearish        : price Higher High  + RSI Lower High   (reversal warning ↓)
  bullish        : price Lower Low    + RSI Higher Low   (reversal warning ↑)
  hidden_bearish : price Lower High   + RSI Higher High  (downtrend continuation)
  hidden_bullish : price Higher Low   + RSI Lower Low    (uptrend continuation)

Quality filters prevent RSI-noise false positives:
  • rsi_min_spread  — minimum RSI difference between compared pivots (default 5)
  • rsi_ob / rsi_os — regular divergences must start from overbought/oversold zones
  • hidden_bull_rsi_max / hidden_bear_rsi_min — hidden divergences stay near midline

Usage
-----
    from tradestrategy.divergence import find_divergences

    df  = add_all(raw_ohlcv)            # must have RSI14 column
    sigs = find_divergences(df)         # list[DivergenceSignal], most recent last
    if sigs:
        latest = sigs[-1]
        print(latest.label, latest.reason)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

# ── signal type metadata ───────────────────────────────────────────────────────

_DIV_DEFS = {
    'Bearish_Divergence_V2':    ('bearish',        'Bearish Divergence'),
    'Bullish_Divergence_V2':    ('bullish',        'Bullish Divergence'),
    'Hidden_Bearish_Divergence': ('hidden_bearish', 'Hidden Bearish Divergence'),
    'Hidden_Bullish_Divergence': ('hidden_bullish', 'Hidden Bullish Divergence'),
}


@dataclass(frozen=True)
class DivergenceSignal:
    div_type:  str         # 'bearish' | 'bullish' | 'hidden_bearish' | 'hidden_bullish'
    label:     str         # human-readable name
    bar_index: int         # integer position in df where p2 pivot was detected
    price:     float       # close price at bar_index
    reason:    str         # one-line explanation
    meta:      Dict = field(default_factory=dict, compare=False, hash=False)
    # meta keys: p1_price, p1_rsi, p2_price, p2_rsi, price_label, rsi_label


# ── pivot helpers ──────────────────────────────────────────────────────────────

def _pivot_highs(arr: np.ndarray, left: int, right: int) -> List[int]:
    """Integer positions of confirmed swing highs."""
    out = []
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == window.max() and arr[i] > arr[i - 1]:
            out.append(i)
    return out


def _pivot_lows(arr: np.ndarray, left: int, right: int) -> List[int]:
    """Integer positions of confirmed swing lows."""
    out = []
    for i in range(left, len(arr) - right):
        window = arr[i - left: i + right + 1]
        if arr[i] == window.min() and arr[i] < arr[i - 1]:
            out.append(i)
    return out


# ── core detector ──────────────────────────────────────────────────────────────

def find_divergences(
    df: pd.DataFrame,
    left:               int   = 5,
    right:              int   = 5,
    rsi_ob:             float = 70.0,
    rsi_os:             float = 30.0,
    rsi_min_spread:     float = 5.0,
    hidden_bull_rsi_max: float = 55.0,
    hidden_bear_rsi_min: float = 45.0,
) -> List[DivergenceSignal]:
    """
    Detect all RSI divergences in `df` and return them sorted by bar_index
    (oldest first, most recent last).

    Parameters
    ----------
    df               : DataFrame produced by ``indicators.add_all()``.
                       Must contain columns: High, Low, Close, RSI14.
    left / right     : confirmation bars on each side of a pivot.
    rsi_ob / rsi_os  : overbought / oversold thresholds for regular divergences.
    rsi_min_spread   : minimum RSI spread between pivot pair (noise filter).
    hidden_bull_rsi_max / hidden_bear_rsi_min : midline filters for hidden types.

    Returns
    -------
    List[DivergenceSignal] sorted ascending by bar_index.
    An empty list means no divergence found.
    """
    if 'RSI14' not in df.columns or len(df) < left + right + 2:
        return []

    rsi   = df['RSI14'].values
    high  = df['High'].values
    low   = df['Low'].values
    close = df['Close'].values

    ph = _pivot_highs(high, left, right)
    pl = _pivot_lows(low,  left, right)

    signals: List[DivergenceSignal] = []

    # ── bearish & hidden bearish (compare consecutive swing highs) ─────────────
    for i in range(1, len(ph)):
        p1, p2 = ph[i - 1], ph[i]
        if abs(rsi[p2] - rsi[p1]) < rsi_min_spread:
            continue

        price_hh = high[p2] > high[p1]
        price_lh = high[p2] < high[p1]
        rsi_lh   = rsi[p2]  < rsi[p1]
        rsi_hh   = rsi[p2]  > rsi[p1]

        if price_hh and rsi_lh and rsi[p1] >= rsi_ob:
            signals.append(DivergenceSignal(
                div_type  = 'bearish',
                label     = 'Bearish Divergence',
                bar_index = p2,
                price     = float(close[p2]),
                reason    = (
                    f"Price Higher High ({high[p1]:.2f} → {high[p2]:.2f}) "
                    f"but RSI Lower High ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"from overbought — momentum failing at the top."
                ),
                meta = {
                    'p1_price': round(float(high[p1]), 4), 'p1_rsi': round(float(rsi[p1]), 2),
                    'p2_price': round(float(high[p2]), 4), 'p2_rsi': round(float(rsi[p2]), 2),
                    'price_label': 'Higher High', 'rsi_label': 'Lower High',
                },
            ))

        if price_lh and rsi_hh and rsi[p2] >= hidden_bear_rsi_min:
            signals.append(DivergenceSignal(
                div_type  = 'hidden_bearish',
                label     = 'Hidden Bearish Divergence',
                bar_index = p2,
                price     = float(close[p2]),
                reason    = (
                    f"Price Lower High ({high[p1]:.2f} → {high[p2]:.2f}) "
                    f"but RSI Higher High ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"— bounce weakening while RSI recovered, downtrend continues."
                ),
                meta = {
                    'p1_price': round(float(high[p1]), 4), 'p1_rsi': round(float(rsi[p1]), 2),
                    'p2_price': round(float(high[p2]), 4), 'p2_rsi': round(float(rsi[p2]), 2),
                    'price_label': 'Lower High', 'rsi_label': 'Higher High',
                },
            ))

    # ── bullish & hidden bullish (compare consecutive swing lows) ──────────────
    for i in range(1, len(pl)):
        p1, p2 = pl[i - 1], pl[i]
        if abs(rsi[p2] - rsi[p1]) < rsi_min_spread:
            continue

        price_ll = low[p2] < low[p1]
        price_hl = low[p2] > low[p1]
        rsi_hl   = rsi[p2] > rsi[p1]
        rsi_ll   = rsi[p2] < rsi[p1]

        if price_ll and rsi_hl and rsi[p1] <= rsi_os:
            signals.append(DivergenceSignal(
                div_type  = 'bullish',
                label     = 'Bullish Divergence',
                bar_index = p2,
                price     = float(close[p2]),
                reason    = (
                    f"Price Lower Low ({low[p1]:.2f} → {low[p2]:.2f}) "
                    f"but RSI Higher Low ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"from oversold — sellers exhausted, reversal likely."
                ),
                meta = {
                    'p1_price': round(float(low[p1]), 4), 'p1_rsi': round(float(rsi[p1]), 2),
                    'p2_price': round(float(low[p2]), 4), 'p2_rsi': round(float(rsi[p2]), 2),
                    'price_label': 'Lower Low', 'rsi_label': 'Higher Low',
                },
            ))

        if price_hl and rsi_ll and rsi[p2] <= hidden_bull_rsi_max:
            signals.append(DivergenceSignal(
                div_type  = 'hidden_bullish',
                label     = 'Hidden Bullish Divergence',
                bar_index = p2,
                price     = float(close[p2]),
                reason    = (
                    f"Price Higher Low ({low[p1]:.2f} → {low[p2]:.2f}) "
                    f"but RSI Lower Low ({rsi[p1]:.1f} → {rsi[p2]:.1f}) "
                    f"— dip shallower while RSI resets, uptrend continues."
                ),
                meta = {
                    'p1_price': round(float(low[p1]), 4), 'p1_rsi': round(float(rsi[p1]), 2),
                    'p2_price': round(float(low[p2]), 4), 'p2_rsi': round(float(rsi[p2]), 2),
                    'price_label': 'Higher Low', 'rsi_label': 'Lower Low',
                },
            ))

    return sorted(signals, key=lambda s: s.bar_index)


def most_recent(df: pd.DataFrame, **kwargs) -> DivergenceSignal | None:
    """Convenience wrapper — return only the single most recent divergence."""
    sigs = find_divergences(df, **kwargs)
    return sigs[-1] if sigs else None

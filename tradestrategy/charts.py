"""
Chart generation — candlestick + EMA stack + Stochastic + MACD histogram.
Produces the same 3-pane layout as creategraphics.py in the original strategy.
"""

from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .patterns import Direction, Signal

_TAIL_BARS = 50

# candle indices (relative to end of df)
_I = -3   # initial
_R = -2   # reversal
_A = -1   # approve

_EMA_PALETTE = {
    'EMA20':  '#ff9800',
    'EMA50':  '#26a69a',
    'EMA100': '#ef5350',
    'EMA200': '#42a5f5',
}


def create_signal_chart(
    df: pd.DataFrame,
    symbol: str,
    exchange: str,
    signal: Signal,
) -> go.Figure:
    """
    Build a 3-pane Plotly chart for a triggered signal.

    Pane 1 — Candlestick + EMA20/50/100/200.
              Three key candles are highlighted:
                • blue   circle   → initial  (-3)
                • red    diamond  → reversal (-2)
                • green  triangle → approve  (-1)

    Pane 2 — Stochastic K / D with 20 / 80 reference lines.
    Pane 3 — MACD line, Signal line, histogram.
    """
    tail     = df.tail(_TAIL_BARS).copy()
    is_long  = signal.direction == Direction.LONG
    full_sym = f'{exchange}:{symbol}'

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=('Price + EMA', 'Stochastic (5,3,3)', 'MACD (50,100,9)'),
    )

    # ── Pane 1: candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=tail.index,
        open=tail['Open'], high=tail['High'],
        low=tail['Low'],   close=tail['Close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
    ), row=1, col=1)

    for ema_col, color in _EMA_PALETTE.items():
        if ema_col in tail.columns:
            fig.add_trace(go.Scatter(
                x=tail.index, y=tail[ema_col],
                mode='lines', name=ema_col,
                line=dict(color=color, width=1.3),
            ), row=1, col=1)

    # highlight the 3 key candles
    marker_defs = [
        (_I, 'Initial',  '#1565c0', 'circle'),
        (_R, 'Reversal', '#c62828', 'diamond'),
        (_A, 'Approve',  '#2e7d32', 'triangle-up' if is_long else 'triangle-down'),
    ]
    for idx, label, color, shape in marker_defs:
        y_pos = (tail['Low'].iloc[idx]  * 0.997 if is_long
                 else tail['High'].iloc[idx] * 1.003)
        fig.add_trace(go.Scatter(
            x=[tail.index[idx]], y=[y_pos],
            mode='markers',
            marker=dict(size=11, color=color, symbol=shape,
                        line=dict(color='white', width=1)),
            name=label,
        ), row=1, col=1)

    # ── Pane 2: Stochastic ─────────────────────────────────────────────────────
    if 'STOCH_K' in tail.columns and 'STOCH_D' in tail.columns:
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['STOCH_K'],
            mode='lines', name='Stoch K',
            line=dict(color='#9c27b0', width=1.3),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['STOCH_D'],
            mode='lines', name='Stoch D',
            line=dict(color='#ff9800', width=1.3),
        ), row=2, col=1)
        for level in (20, 80):
            fig.add_hline(y=level, line_dash='dot',
                          line_color='rgba(255,255,255,0.3)', row=2, col=1)

    # ── Pane 3: MACD ───────────────────────────────────────────────────────────
    if 'MACD' in tail.columns and 'MACD_S' in tail.columns:
        histogram = tail['MACD'] - tail['MACD_S']
        fig.add_trace(go.Bar(
            x=tail.index, y=histogram,
            name='Histogram',
            marker_color=histogram.apply(
                lambda v: '#26a69a' if v >= 0 else '#ef5350'
            ),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['MACD'],
            mode='lines', name='MACD',
            line=dict(color='#42a5f5', width=1.3),
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['MACD_S'],
            mode='lines', name='Signal',
            line=dict(color='#ff9800', width=1.3),
        ), row=3, col=1)

    direction_label = '📈 LONG' if is_long else '📉 SHORT'
    fig.update_layout(
        title=dict(
            text=(f'{direction_label}  ·  {signal.pattern}  '
                  f'[{signal.triggered_ema}]  —  {full_sym}'),
            font=dict(size=14),
        ),
        height=880,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=55, r=40, t=65, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=10)),
    )
    return fig


def figure_to_png(fig: go.Figure, width: int = 1280, height: int = 880) -> BytesIO:
    """Render a Plotly figure to a PNG BytesIO buffer (requires kaleido)."""
    buf = BytesIO(pio.to_image(fig, format='png', width=width, height=height))
    buf.seek(0)
    return buf

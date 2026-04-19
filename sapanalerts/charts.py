"""
Chart generation — Sapan Strateji sinyal grafiği.

3 panel:
  1. Candlestick + EMA stack + Entry / SL / TP seviyeleri + bölge dolguları
  2. Stochastic (5,3,3)
  3. MACD (50,100,9)

Plotly ile üretilir, kaleido ile PNG'ye dönüştürülür.
"""

from io import BytesIO

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

_TAIL_BARS = 60   # sinyalden önce gösterilecek mum sayısı
_POST_BARS = 8    # sinyalden sonra gösterilecek mum sayısı

_EMA_PALETTE = {
    'ema20':  '#ffa726',   # turuncu
    'ema50':  '#26c6da',   # cyan
    'ema100': '#ef5350',   # kırmızı
    'ema200': '#ab47bc',   # mor
}

_GREEN  = '#26a69a'
_RED    = '#ef5350'
_ENTRY  = '#ffd54f'   # sarı
_TP     = '#66bb6a'   # yeşil
_SL     = '#ef5350'   # kırmızı


def create_signal_chart(
    df: pd.DataFrame,
    symbol: str,
    exchange: str,
    direction: str,     # 'LONG' veya 'SHORT'
    sig_type: str,
    sig_idx: int,       # df içindeki sinyal satırı konumu
    entry: float,
    sl: float,
    tp: float,
) -> go.Figure:
    """
    Sapan strateji sinyali için Plotly grafiği üretir.

    df       : add_indicators + detect_signals uygulanmış DataFrame
    sig_idx  : df.iloc[sig_idx] sinyalin bulunduğu satır
    entry    : giriş fiyatı (stop-order)
    sl       : stop-loss fiyatı
    tp       : take-profit fiyatı
    """
    start = max(0, sig_idx - _TAIL_BARS)
    end   = min(len(df), sig_idx + _POST_BARS + 1)
    tail  = df.iloc[start:end].copy()

    is_long   = direction == 'LONG'
    full_sym  = f'{exchange}:{symbol}'
    risk      = abs(entry - sl)
    risk_pct  = risk / entry * 100
    dir_label = '📈 LONG' if is_long else '📉 SHORT'

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.56, 0.22, 0.22],
        subplot_titles=('Fiyat + EMA + Sinyal Seviyeleri',
                        'Stochastic (5,3,3)',
                        'MACD (50,100,9)'),
    )

    # ── Panel 1: Candlestick ─────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=tail.index,
        open=tail['Open'], high=tail['High'],
        low=tail['Low'],   close=tail['Close'],
        name='Fiyat',
        increasing_line_color=_GREEN, decreasing_line_color=_RED,
        increasing_fillcolor=_GREEN,  decreasing_fillcolor=_RED,
    ), row=1, col=1)

    # EMAs
    for col, color in _EMA_PALETTE.items():
        if col in tail.columns:
            fig.add_trace(go.Scatter(
                x=tail.index, y=tail[col],
                mode='lines',
                name=col.upper(),
                line=dict(color=color, width=1.2),
                opacity=0.85,
            ), row=1, col=1)

    # Sinyal mumu dikey işaretçi
    sig_date = df.index[sig_idx]
    x_sig = str(sig_date) if not isinstance(sig_date, (int, float)) else sig_date
    for row_n in (1, 2, 3):
        fig.add_vline(
            x=x_sig,
            line_dash='dash', line_color='rgba(255,213,79,0.5)',
            line_width=1.5, row=row_n, col=1,
        )

    # Risk bölgesi (entry → SL)  – hafif kırmızı dolgu
    fig.add_hrect(
        y0=min(entry, sl), y1=max(entry, sl),
        fillcolor='rgba(239,83,80,0.10)', line_width=0,
        row=1, col=1,
    )
    # Ödül bölgesi (entry → TP)  – hafif yeşil dolgu
    fig.add_hrect(
        y0=min(entry, tp), y1=max(entry, tp),
        fillcolor='rgba(102,187,106,0.10)', line_width=0,
        row=1, col=1,
    )

    # Entry / SL / TP yatay çizgileri + etiketler
    for price, color, label in (
        (entry, _ENTRY, f'Entry {entry:.4g}'),
        (sl,    _SL,    f'SL {sl:.4g}'),
        (tp,    _TP,    f'TP {tp:.4g}'),
    ):
        fig.add_hline(
            y=price, line_dash='dash', line_color=color,
            line_width=1.8,
            annotation_text=f'<b>{label}</b>',
            annotation_position='right',
            annotation_font_color=color,
            annotation_font_size=11,
            row=1, col=1,
        )

    # ── Panel 2: Stochastic ──────────────────────────────────────────────────
    if 'stoch_k_val' in tail.columns:
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['stoch_k_val'],
            mode='lines', name='%K',
            line=dict(color='#ce93d8', width=1.3),
        ), row=2, col=1)
    if 'stoch_d_val' in tail.columns:
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['stoch_d_val'],
            mode='lines', name='%D',
            line=dict(color='#ffa726', width=1.3),
        ), row=2, col=1)
    for level in (30, 70):
        fig.add_hline(y=level, line_dash='dot',
                      line_color='rgba(255,255,255,0.25)', row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # ── Panel 3: MACD ────────────────────────────────────────────────────────
    if 'macd_hist' in tail.columns:
        colors = tail['macd_hist'].apply(
            lambda v: _GREEN if v >= 0 else _RED
        )
        fig.add_trace(go.Bar(
            x=tail.index, y=tail['macd_hist'],
            name='Histogram',
            marker_color=colors,
            opacity=0.7,
        ), row=3, col=1)
    if 'macd_line' in tail.columns:
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['macd_line'],
            mode='lines', name='MACD',
            line=dict(color='#42a5f5', width=1.3),
        ), row=3, col=1)
    if 'macd_sig' in tail.columns:
        fig.add_trace(go.Scatter(
            x=tail.index, y=tail['macd_sig'],
            mode='lines', name='Sinyal',
            line=dict(color='#ffa726', width=1.3),
        ), row=3, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=(f'{dir_label}  ·  {sig_type}  —  {full_sym}   '
                  f'<span style="font-size:12px;color:#aaa">'
                  f'Entry {entry:.4g}  |  SL {sl:.4g}  |  TP {tp:.4g}  '
                  f'|  1R={risk:.4g} ({risk_pct:.1f}%)</span>'),
            font=dict(size=14),
        ),
        height=900,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=120, t=70, b=30),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='right', x=1, font=dict(size=10),
        ),
    )
    return fig


def figure_to_png(fig: go.Figure, width: int = 1280, height: int = 900) -> BytesIO:
    """Plotly figürü PNG BytesIO buffer'ına dönüştürür (kaleido gerektirir)."""
    buf = BytesIO(pio.to_image(fig, format='png', width=width, height=height))
    buf.seek(0)
    return buf

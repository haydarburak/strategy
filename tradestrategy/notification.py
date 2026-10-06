"""
Telegram notification module.

Required environment variables
-------------------------------
  TELEGRAM_BOT_TOKEN   — bot API token from @BotFather
  TELEGRAM_BOT_CHAT_ID — target chat / channel ID.
                         Comma-separate multiple recipients: "12345,67890".

Both must be set for any message to be sent.  If either is missing, every
call silently succeeds (returns False) and logs a warning once per process.
"""

import logging
import os
from io import BytesIO
from typing import Optional

import requests

from .patterns import Direction, Signal
from .divergence import DivergenceSignal

logger = logging.getLogger(__name__)

# warn at most once per process if credentials are absent
_warned_missing = False


def _credentials() -> tuple[str, list[str]] | tuple[None, None]:
    global _warned_missing
    token    = os.getenv('TELEGRAM_BOT_TOKEN')
    raw_ids  = os.getenv('TELEGRAM_BOT_CHAT_ID', '')
    chat_ids = [c.strip() for c in raw_ids.split(',') if c.strip()]
    if not token or not chat_ids:
        if not _warned_missing:
            logger.warning(
                'TELEGRAM_BOT_TOKEN / TELEGRAM_BOT_CHAT_ID not set '
                '— Telegram notifications disabled.'
            )
            _warned_missing = True
        return None, None
    return token, chat_ids


def _post_message(token: str, chat_ids: list[str], text: str) -> bool:
    ok = True
    for chat_id in chat_ids:
        try:
            r = requests.post(
                f'https://api.telegram.org/bot{token}/sendMessage',
                json={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'},
                timeout=15,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f'Telegram sendMessage failed ({chat_id}): {e}')
            ok = False
    return ok


def _post_photo(token: str, chat_ids: list[str], caption: str, png: BytesIO) -> bool:
    ok = True
    for chat_id in chat_ids:
        try:
            png.seek(0)   # rewind the buffer for each recipient
            r = requests.post(
                f'https://api.telegram.org/bot{token}/sendPhoto',
                data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                files={'photo': ('chart.png', png, 'image/png')},
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f'Telegram sendPhoto failed ({chat_id}): {e}')
            ok = False
    return ok


# ── public API ─────────────────────────────────────────────────────────────────

def send_signal(
    symbol: str,
    exchange: str,
    signal: Signal,
    df,                        # pd.DataFrame — needs last 3 rows with Close
    interval: str,
    chart_png: Optional[BytesIO] = None,
) -> bool:
    """
    Send a trade signal alert.

    If `chart_png` is provided (BytesIO of a PNG image) the alert is sent as
    a photo with the message as caption; otherwise as plain text.

    Parameters
    ----------
    symbol      : ticker, e.g. 'AAPL'
    exchange    : e.g. 'NASDAQ'
    signal      : Signal(pattern, direction, triggered_ema)
    df          : DataFrame with at least 3 rows; must have 'Close' column
    interval    : timeframe string shown in the TradingView link, e.g. 'D'
    chart_png   : pre-rendered PNG buffer (from charts.figure_to_png)

    Returns
    -------
    True on HTTP 200, False otherwise.
    """
    token, chat_ids = _credentials()
    if token is None:
        return False

    direction_label = '📈 LONG' if signal.direction == Direction.LONG else '📉 SHORT'
    tv_link = f'https://www.tradingview.com/chart/?symbol={exchange}:{symbol}&interval={interval}'

    text = (
        f'🔔 <b>SIGNAL — {direction_label}</b>\n'
        f'Symbol   : <code>{exchange}:{symbol}</code>\n'
        f'Pattern  : <code>{signal.pattern}</code>\n'
        f'EMA      : <code>{signal.triggered_ema}</code>\n'
        f'─────────────────────\n'
        f'Initial  close : <code>{df["Close"].iloc[-3]:.4f}</code>\n'
        f'Reversal close : <code>{df["Close"].iloc[-2]:.4f}</code>\n'
        f'Approve  close : <code>{df["Close"].iloc[-1]:.4f}</code>\n'
        f'─────────────────────\n'
        f'<a href="{tv_link}">📊 Open on TradingView</a>'
    )

    if chart_png is not None:
        return _post_photo(token, chat_ids, text, chart_png)
    return _post_message(token, chat_ids, text)


def send_divergence_alert(
    symbol: str,
    exchange: str,
    signal: DivergenceSignal,
    interval: str,
    chart_png=None,
    pivot_date: Optional[str] = None,
) -> bool:
    """
    Send a single divergence alert.

    Parameters
    ----------
    symbol     : ticker, e.g. 'AAPL'
    exchange   : e.g. 'NASDAQ'
    signal     : DivergenceSignal from divergence.find_divergences()
    interval   : timeframe string shown in the TradingView link, e.g. 'D'
    chart_png  : optional BytesIO of pre-rendered PNG chart
    pivot_date : date of the p2 pivot bar, e.g. '2026-09-23'

    Returns
    -------
    True on HTTP 200, False otherwise.
    """
    token, chat_ids = _credentials()
    if token is None:
        return False

    type_emoji = {
        'bearish':        '🔴',
        'bullish':        '🟢',
        'hidden_bearish': '🟠',
        'hidden_bullish': '🔵',
    }.get(signal.div_type, '⚪')

    tv_link = f'https://www.tradingview.com/chart/?symbol={exchange}:{symbol}&interval={interval}'
    rsi_meta = (
        f"RSI: {signal.meta.get('p1_rsi', '?')} → {signal.meta.get('p2_rsi', '?')}"
        if signal.meta else ''
    )

    text = (
        f'{type_emoji} <b>{signal.label}</b>\n'
        f'Symbol : <code>{exchange}:{symbol}</code>\n'
        f'Close  : <code>{signal.price:.4f}</code>\n'
        + (f'Pivot  : <code>{pivot_date}</code> (confirmed on last close)\n' if pivot_date else '')
        + f'{rsi_meta}\n'
        f'─────────────────────\n'
        f'{signal.reason}\n'
        f'─────────────────────\n'
        f'<a href="{tv_link}">📊 Open on TradingView</a>'
    )

    if chart_png is not None:
        return _post_photo(token, chat_ids, text, chart_png)
    return _post_message(token, chat_ids, text)


def send_divergence_batch(alerts: list[dict],
                          title: str = 'RSI Divergence Scan — Summary') -> bool:
    """
    Send a summary message listing all divergences found in the current scan.

    Each entry in `alerts` must have keys:
        exchange, symbol, label, reason, interval  (optional: pivot_date)

    Long summaries are split into several messages so each stays under
    Telegram's 4096-character limit.

    Returns True if every message was sent successfully.
    """
    if not alerts:
        return False

    token, chat_ids = _credentials()
    if token is None:
        return False

    header  = f'📋 <b>{title}</b>\n'
    entries = []
    for a in alerts:
        tv_link = (
            f'https://www.tradingview.com/chart/'
            f'?symbol={a["exchange"]}:{a["symbol"]}&interval={a["interval"]}'
        )
        pivot = f' (pivot {a["pivot_date"]})' if a.get('pivot_date') else ''
        entries.append(
            f'• <code>{a["exchange"]}:{a["symbol"]}</code> — {a["label"]}{pivot}\n'
            f'  {a["reason"]}\n'
            f'  <a href="{tv_link}">chart</a>'
        )

    ok = True
    for chunk in _chunk_lines(entries, header):
        ok &= _post_message(token, chat_ids, chunk)
    return ok


_TELEGRAM_SAFE_LEN = 3800   # below the 4096 limit, with room for HTML markup


def _chunk_lines(entries: list[str], header: str) -> list[str]:
    """Join entries under `header`, starting a new message before the limit."""
    chunks, current = [], header
    for entry in entries:
        if len(current) + len(entry) + 1 > _TELEGRAM_SAFE_LEN and current != header:
            chunks.append(current)
            current = header
        current += '\n' + entry
    chunks.append(current)
    return chunks


def send_index_status(
    index_symbol: str,
    index_exchange: str,
    direction: Optional[bool],
    interval: str = 'D',
) -> bool:
    """
    Notify the index bias determined for an exchange before scanning its stocks.
    Sends only when direction is not None (skipped exchanges are silent).
    """
    if direction is None:
        return False          # neutral → no notification

    token, chat_ids = _credentials()
    if token is None:
        return False

    bias  = '🟢 LONG bias' if direction else '🔴 SHORT bias'
    tv_link = (f'https://www.tradingview.com/chart/'
               f'?symbol={index_exchange}:{index_symbol}&interval={interval}')

    text = (
        f'📊 <b>Index Update</b>\n'
        f'<code>{index_exchange}:{index_symbol}</code> → {bias}\n'
        f'<a href="{tv_link}">View chart</a>'
    )
    return _post_message(token, chat_ids, text)

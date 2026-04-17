"""
Telegram notification module.

Required environment variables
-------------------------------
  TELEGRAM_BOT_TOKEN   — bot API token from @BotFather
  TELEGRAM_BOT_CHAT_ID — target chat / channel ID

Both must be set for any message to be sent.  If either is missing, every
call silently succeeds (returns False) and logs a warning once per process.
"""

import logging
import os
from io import BytesIO
from typing import Optional

import requests

from .patterns import Direction, Signal

logger = logging.getLogger(__name__)

# warn at most once per process if credentials are absent
_warned_missing = False


def _credentials() -> tuple[str, str] | tuple[None, None]:
    global _warned_missing
    token   = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_BOT_CHAT_ID')
    if not token or not chat_id:
        if not _warned_missing:
            logger.warning(
                'TELEGRAM_BOT_TOKEN / TELEGRAM_BOT_CHAT_ID not set '
                '— Telegram notifications disabled.'
            )
            _warned_missing = True
        return None, None
    return token, chat_id


def _post_message(token: str, chat_id: str, text: str) -> bool:
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            json={'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'},
            timeout=15,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f'Telegram sendMessage failed: {e}')
        return False


def _post_photo(token: str, chat_id: str, caption: str, png: BytesIO) -> bool:
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendPhoto',
            data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
            files={'photo': ('chart.png', png, 'image/png')},
            timeout=30,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error(f'Telegram sendPhoto failed: {e}')
        return False


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
    token, chat_id = _credentials()
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
        return _post_photo(token, chat_id, text, chart_png)
    return _post_message(token, chat_id, text)


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

    token, chat_id = _credentials()
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
    return _post_message(token, chat_id, text)

"""
Telegram notification — Sapan Strateji alert modülü.

Ortam değişkenleri (mevcut strategy repo ile aynı):
  TELEGRAM_BOT_TOKEN   — BotFather'dan alınan token
  TELEGRAM_BOT_CHAT_ID — hedef chat/channel/group ID.
                         Birden fazla alıcı için virgülle ayır: "12345,67890".
"""

import logging
import os
from io import BytesIO
from typing import Optional

import requests

logger = logging.getLogger(__name__)
_warned_missing = False


def _credentials() -> tuple[str, list[str]] | tuple[None, None]:
    global _warned_missing
    token    = os.getenv('TELEGRAM_BOT_TOKEN')
    raw_ids  = os.getenv('TELEGRAM_BOT_CHAT_ID', '')
    chat_ids = [c.strip() for c in raw_ids.split(',') if c.strip()]
    if not token or not chat_ids:
        if not _warned_missing:
            logger.warning('TELEGRAM_BOT_TOKEN / TELEGRAM_BOT_CHAT_ID not set — bildirimler devre dışı.')
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
            logger.error(f'Telegram sendMessage hatası ({chat_id}): {e}')
            ok = False
    return ok


def _post_photo(token: str, chat_ids: list[str], caption: str, png: BytesIO) -> bool:
    ok = True
    for chat_id in chat_ids:
        try:
            png.seek(0)   # her alıcı için akışı başa sar
            r = requests.post(
                f'https://api.telegram.org/bot{token}/sendPhoto',
                data={'chat_id': chat_id, 'caption': caption, 'parse_mode': 'HTML'},
                files={'photo': ('chart.png', png, 'image/png')},
                timeout=30,
            )
            r.raise_for_status()
        except Exception as e:
            logger.error(f'Telegram sendPhoto hatası ({chat_id}): {e}')
            ok = False
    return ok


# ── public API ──────────────────────────────────────────────────────────────

def send_sapan_signal(
    symbol: str,
    exchange: str,
    direction: str,           # 'LONG' veya 'SHORT'
    sig_type: str,
    sig_date: str,
    entry: float,
    sl: float,
    tp: float,
    index_symbol: str,
    index_trend: Optional[bool],
    status: Optional[str] = None,
    chart_png: Optional[BytesIO] = None,
) -> bool:
    """
    Sapan stratejisi sinyalini Telegram'a gönderir.
    chart_png verilirse foto olarak, verilmezse metin olarak gönderir.
    """
    token, chat_ids = _credentials()
    if token is None:
        return False

    arrow      = '📈 LONG'  if direction == 'LONG'  else '📉 SHORT'
    risk       = abs(entry - sl)
    risk_pct   = risk / entry * 100
    reward     = abs(tp - entry)
    tv_link    = (f'https://www.tradingview.com/chart/'
                  f'?symbol={exchange}:{symbol}&interval=D')

    if index_trend is True:
        trend_txt = f'✅ Uptrend ({index_symbol})'
    elif index_trend is False:
        trend_txt = f'🔽 Downtrend ({index_symbol})'
    else:
        trend_txt = f'❓ Bilinmiyor ({index_symbol})'

    text = (
        f'🔔 <b>SAPAN STRATEJİSİ — {arrow}</b>\n'
        f'Symbol  : <code>{exchange}:{symbol}</code>\n'
        f'Tip     : <code>{sig_type}</code>\n'
        f'Tarih   : <code>{sig_date}</code>  (teyit mumu)\n'
        + (f'Durum   : <code>{status}</code>\n' if status else '') +
        f'─────────────────────\n'
        f'💰 Entry    : <code>{entry:.4g}</code>  (stop-order)\n'
        f'🛑 Stop Loss: <code>{sl:.4g}</code>\n'
        f'🎯 Take Profit: <code>{tp:.4g}</code>\n'
        f'─────────────────────\n'
        f'1R = {risk:.4g}  ({risk_pct:.1f}%)   R:R = 1:2\n'
        f'Endeks: {trend_txt}\n'
        f'─────────────────────\n'
        f'<a href="{tv_link}">📊 TradingView\'de aç</a>'
    )

    if chart_png is not None:
        return _post_photo(token, chat_ids, text, chart_png)
    return _post_message(token, chat_ids, text)


def send_index_status(
    index_symbol: str,
    exchange_label: str,
    direction: Optional[bool],
) -> bool:
    """Piyasa taraması öncesi endeks trend durumunu bildirir."""
    if direction is None:
        return False
    token, chat_ids = _credentials()
    if token is None:
        return False

    bias    = '🟢 LONG yönü' if direction else '🔴 SHORT yönü'
    tv_link = f'https://www.tradingview.com/chart/?symbol={index_symbol}&interval=D'
    text = (
        f'📊 <b>Endeks Güncelleme — {exchange_label}</b>\n'
        f'<code>{index_symbol}</code> → {bias}\n'
        f'<a href="{tv_link}">Grafiği görüntüle</a>'
    )
    return _post_message(token, chat_ids, text)


def send_scan_summary(
    date_str: str,
    market_results: list[dict],
    total_signals: int,
) -> bool:
    """
    Tüm piyasaların tarama özet mesajını gönderir.
    market_results: [{'label': 'BIST100', 'signals': 2, 'scanned': 88}, ...]
    """
    token, chat_ids = _credentials()
    if token is None:
        return False

    lines = [f'✅ <b>Sapan Strateji Taraması Tamamlandı</b>\n📅 {date_str}\n']
    for m in market_results:
        lines.append(f"  • {m['label']}: {m['signals']} sinyal / {m['scanned']} hisse")
    lines.append(f'\n📊 Toplam sinyal: <b>{total_signals}</b>')

    return _post_message(token, chat_ids, '\n'.join(lines))


def send_no_signal_message(date_str: str, market_results: list[dict]) -> bool:
    """
    Hiç sinyal bulunamadığında bilgi mesajı gönderir.
    Taranan hisse sayıları da yazılır; 0 görünen piyasa veri sorununa işaret eder.
    """
    token, chat_ids = _credentials()
    if token is None:
        return False

    lines = [f'📊 <b>Sapan Strateji</b> — {date_str}\n',
             'Son seansta <b>yeni sinyal bulunamadı</b>.']
    for m in market_results:
        warn = '  ⚠️' if m['scanned'] == 0 else ''
        lines.append(f"  • {m['label']}: {m['scanned']} hisse tarandı{warn}")
    return _post_message(token, chat_ids, '\n'.join(lines))

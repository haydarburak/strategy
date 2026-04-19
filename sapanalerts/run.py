"""
Sapan Strateji — Ana çalıştırma scripti.

GitHub Actions (sapan_daily.yml) veya elle çalıştırılır.

Ortam değişkenleri:
  TELEGRAM_BOT_TOKEN   — BotFather token
  TELEGRAM_BOT_CHAT_ID — hedef chat ID
  LOOKBACK_DAYS        — kaç gün geriye (default: 3)
  MARKETS              — taranacak piyasalar, virgülle ayrılmış
                         (default: nasdaq,nyse,bist,xetr)
"""

import datetime
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-7s  %(name)s — %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('sapanalerts')

from . import charts, notification
from .scanner import scan_market, MARKET_DEFS


def main() -> int:
    lookback = int(os.environ.get('LOOKBACK_DAYS', '3'))
    markets  = [m.strip().lower()
                for m in os.environ.get('MARKETS', 'nasdaq,nyse,bist,xetr').split(',')]

    today_str = datetime.date.today().strftime('%d %b %Y')

    print(f'\n{"="*60}')
    print(f'  SAPAN STRATEJİSİ — CANLI TARAMA')
    print(f'  Tarih    : {today_str}')
    print(f'  Piyasalar: {", ".join(markets)}')
    print(f'  Lookback : {lookback} gün')
    print(f'{"="*60}')

    unknown = [m for m in markets if m not in MARKET_DEFS]
    if unknown:
        logger.error('Bilinmeyen piyasa(lar): %s', ', '.join(unknown))
        return 1

    all_alerts    = []
    market_results = []

    for mkey in markets:
        alerts, minfo = scan_market(mkey, lookback_days=lookback)

        # Endeks durumunu Telegram'a bildir
        notification.send_index_status(
            index_symbol=minfo['index_symbol'],
            exchange_label=minfo['label'],
            direction=minfo.get('index_direction'),
        )

        # Her sinyal için chart üret ve gönder
        for alert in alerts:
            try:
                fig = charts.create_signal_chart(
                    df=alert['df'],
                    symbol=alert['symbol'],
                    exchange=alert['exchange'],
                    direction=alert['direction'],
                    sig_type=alert['sig_type'],
                    sig_idx=alert['sig_pos'],
                    entry=alert['entry'],
                    sl=alert['sl'],
                    tp=alert['tp'],
                )
                png = charts.figure_to_png(fig)
            except Exception as e:
                logger.warning('Chart üretilemedi (%s): %s', alert['symbol'], e)
                png = None

            notification.send_sapan_signal(
                symbol=alert['symbol'],
                exchange=alert['exchange'],
                direction=alert['direction'],
                sig_type=alert['sig_type'],
                sig_date=alert['sig_date'].strftime('%d %b %Y'),
                entry=alert['entry'],
                sl=alert['sl'],
                tp=alert['tp'],
                index_symbol=alert['index'],
                index_trend=alert['idx_uptrend'],
                chart_png=png,
            )
            print(f'  📤 {alert["symbol"]} ({alert["direction"]}) gönderildi.')

        all_alerts.extend(alerts)
        market_results.append(minfo)

    # ── Özet bildirimi ───────────────────────────────────────────────────────
    if not all_alerts:
        notification.send_no_signal_message(
            date_str=today_str,
            markets=[MARKET_DEFS[m][2] for m in markets],
        )
        print('\n  Sinyal bulunamadı — bilgi mesajı gönderildi.')
    else:
        notification.send_scan_summary(
            date_str=today_str,
            market_results=market_results,
            total_signals=len(all_alerts),
        )

    print(f'\n  Tamamlandı. Toplam {len(all_alerts)} sinyal.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""
Multi-exchange reversal pattern scanner — BIST, XETR, NYSE, NASDAQ.

Symbol universe  : TradingView screener (same source as main.py).
                   NYSE/NASDAQ are filtered to S&P 500 constituents only.
Index gating     : Each exchange is pre-checked against its parent index
                   (XU100, DAX, SPX, NDX). If the index EMA stack is bullish
                   only LONG signals fire; bearish → SHORT only; neutral → skip.

Output
------
  results/signals_<DATE>.csv    — one row per triggered signal
  results/errors_<DATE>.txt     — symbols that could not be processed

Usage
-----
  python run_sp500.py
  python run_sp500.py --bars 700           # fetch more history
  python run_sp500.py --exchange BIST      # scan one exchange only
"""

import argparse
import os
import sys
import time
from datetime import date
from io import StringIO
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm
from tradingview_screener import Query, Column
from tvDatafeed import TvDatafeed, Interval
from websocket import WebSocketTimeoutException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tradestrategy import add_all, generate_signals
from tradestrategy.patterns import Direction, Signal
from tradestrategy import charts, notification, persistence

# ── configuration ──────────────────────────────────────────────────────────────

DEFAULT_BARS      = 500
MAX_RETRIES       = 3
RETRY_BACKOFF_SEC = 1
RESULTS_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

_TV = TvDatafeed()

# Same index mapping as main.py TradingConfig.MARKET_INDICES
MARKET_INDICES = {
    'BIST':   {'symbol': 'XU100', 'exchange': 'BIST'},
    'XETR':   {'symbol': 'DAX',   'exchange': 'XETR'},
    'NYSE':   {'symbol': 'SPX',   'exchange': 'SP'},
    'NASDAQ': {'symbol': 'NDX',   'exchange': 'NASDAQ'},
}

_EMA_COLS = ['EMA20', 'EMA50', 'EMA100', 'EMA200']

# ── symbol universe (same logic as getdata_stock.py) ──────────────────────────

_SP500_URL     = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
_HTTP_HEADERS  = {'User-Agent': 'Mozilla/5.0'}

_SP500_FALLBACK = [
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM','V','UNH',
    'XOM','JNJ','PG','MA','HD','CVX','MRK','ABBV','AVGO','PEP','KO',
    'LLY','COST','BAC','TMO','WMT','ADBE','CRM','ACN','MCD','NKE','NEE',
    'DHR','CSCO','QCOM','ABT','ORCL','TXN','HON','PM','AMGN','UPS',
    'INTC','IBM','GS','CAT','BLK','RTX','NOW',
]


def _fetch_sp500_set() -> set:
    try:
        r = requests.get(_SP500_URL, headers=_HTTP_HEADERS, timeout=20)
        r.raise_for_status()
        symbols = pd.read_html(StringIO(r.text))[0]['Symbol'].tolist()
        return set(symbols)
    except Exception as e:
        print(f"  Wikipedia fetch failed ({e}), using fallback S&P 500 list.")
        return set(_SP500_FALLBACK)


def get_stock_symbols() -> dict[str, list[str]]:
    """
    Fetch all tradeable stocks across BIST, XETR, NYSE, NASDAQ via the
    TradingView screener — identical to getdata_stock.get_stock_symbols().
    NYSE/NASDAQ results are filtered down to S&P 500 constituents.
    Returns: {'BIST': [...], 'XETR': [...], 'NYSE': [...], 'NASDAQ': [...]}
    """
    print("Fetching symbol universe from TradingView screener …")
    sp500 = _fetch_sp500_set()

    q = (
        Query()
        .limit(50000)
        .select()
        .where(
            Column('type') == 'stock',
            Column('exchange').isin(['BIST', 'XETR', 'NYSE', 'NASDAQ']),
        )
        .set_markets('america', 'germany', 'turkey')
    )
    _, raw = q.get_scanner_data()

    def _keep(row) -> bool:
        exchange, symbol = row['ticker'].split(':')
        if exchange in ('NYSE', 'NASDAQ'):
            return symbol in sp500
        return True  # keep all BIST / XETR

    filtered = raw[raw.apply(_keep, axis=1)].copy()
    filtered['exchange']     = filtered['ticker'].str.split(':').str[0]
    filtered['symbol_clean'] = filtered['ticker'].str.split(':').str[1]

    result = filtered.groupby('exchange')['symbol_clean'].apply(list).to_dict()

    for exch, syms in result.items():
        print(f"  {exch}: {len(syms)} symbols")

    return result


# ── index EMA gating (same logic as main.py analsys()) ────────────────────────

def _compute_emas(df: pd.DataFrame) -> pd.DataFrame:
    for p in [20, 50, 100, 200]:
        df[f'EMA{p}'] = df['Close'].ewm(span=p, adjust=False).mean()
    return df


def get_index_direction(exchange: str, n_bars: int) -> Optional[bool]:
    """
    Fetch the parent index for `exchange`, compute EMA alignment on the
    most recent bar, and return:
      True  — EMA20 > EMA50 > EMA100 > EMA200  (bullish → LONG signals only)
      False — EMA20 < EMA50 < EMA100 < EMA200  (bearish → SHORT signals only)
      None  — neither aligned (neutral → skip exchange)
    """
    cfg = MARKET_INDICES[exchange]
    raw = fetch_ohlcv(cfg['symbol'], cfg['exchange'], n_bars)

    if raw.empty or len(raw) < 200:
        print(f"  [{exchange}] Index data unavailable — skipping exchange.")
        return None

    df = _compute_emas(raw.copy()).dropna(subset=_EMA_COLS)
    last = df.iloc[-1]

    long_aligned  = all(last[_EMA_COLS[i]] > last[_EMA_COLS[i + 1]] for i in range(3))
    short_aligned = all(last[_EMA_COLS[i]] < last[_EMA_COLS[i + 1]] for i in range(3))

    if long_aligned:
        return True
    if short_aligned:
        return False
    return None


# ── OHLCV fetching with retry ──────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, exchange: str, n_bars: int) -> pd.DataFrame:
    """Fetch daily OHLCV from TradingView with retry. Returns empty DF on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _TV.get_hist(symbol, exchange, interval=Interval.in_daily, n_bars=n_bars)
            if raw is not None and not raw.empty:
                raw = raw.rename(columns={
                    'open': 'Open', 'high': 'High',
                    'low':  'Low',  'close': 'Close', 'volume': 'Volume',
                })
                raw = raw[raw['High'] != raw['Low']]
                return raw.dropna()
        except (TimeoutError, WebSocketTimeoutException):
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC)
        except Exception:
            break
    return pd.DataFrame()


# ── signal payload helpers ─────────────────────────────────────────────────────

def _price_data(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    return {
        'open':           float(last['Open']),
        'high':           float(last['High']),
        'low':            float(last['Low']),
        'close':          float(last['Close']),
        'volume':         float(last['Volume']) if 'Volume' in df.columns else 0.0,
        'initial_close':  float(df['Close'].iloc[-3]),
        'reversal_close': float(df['Close'].iloc[-2]),
    }


def _indicator_data(df: pd.DataFrame, sig: Signal) -> dict:
    last = df.iloc[-1]
    return {
        'triggered_ema': sig.triggered_ema,
        'ema20':         float(last['EMA20']),
        'ema50':         float(last['EMA50']),
        'ema100':        float(last['EMA100']),
        'ema200':        float(last['EMA200']),
        'stoch_k':       float(last['STOCH_K']),
        'stoch_d':       float(last['STOCH_D']),
        'macd':          float(last['MACD']),
        'macd_signal':   float(last['MACD_S']),
        'rsi':           float(last['RSI14']) if 'RSI14' in df.columns else 0.0,
    }


# ── main scan ──────────────────────────────────────────────────────────────────

def run(n_bars: int, filter_exchange: Optional[str]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    db    = persistence.get_db()        # Firebase singleton (no-op if not configured)
    today = date.today().isoformat()
    results_path = os.path.join(RESULTS_DIR, f'signals_{today}.csv')
    errors_path  = os.path.join(RESULTS_DIR, f'errors_{today}.txt')

    # ── 1. build symbol universe ───────────────────────────────────────────────
    symbols_by_exchange = get_stock_symbols()

    if filter_exchange:
        key = filter_exchange.upper()
        if key not in symbols_by_exchange:
            print(f"Exchange '{key}' not found. Available: {list(symbols_by_exchange)}")
            return
        symbols_by_exchange = {key: symbols_by_exchange[key]}

    total_symbols = sum(len(v) for v in symbols_by_exchange.values())
    print(f"\nTotal symbols to scan: {total_symbols}\n")

    all_signals: list[dict] = []
    errors:      list[str]  = []

    # ── 2. per-exchange: check index, then scan stocks ─────────────────────────
    for exchange, symbols in symbols_by_exchange.items():

        # ── index gating ──────────────────────────────────────────────────────
        print(f"\n{'─'*55}")
        print(f"[{exchange}] Checking index: "
              f"{MARKET_INDICES[exchange]['symbol']} ({MARKET_INDICES[exchange]['exchange']})")

        index_direction = get_index_direction(exchange, n_bars)

        # ── persist and notify index bias ──────────────────────────────────────
        idx_sym  = MARKET_INDICES[exchange]['symbol']
        idx_exch = MARKET_INDICES[exchange]['exchange']
        db.save_index_status(idx_sym, idx_exch, index_direction, '1D')
        notification.send_index_status(idx_sym, idx_exch, index_direction)

        if index_direction is None:
            print(f"[{exchange}] Index is NEUTRAL — exchange skipped.")
            errors.append(f"{exchange}  [index neutral — all {len(symbols)} symbols skipped]")
            continue

        allowed = Direction.LONG if index_direction else Direction.SHORT
        print(f"[{exchange}] Index bias: {allowed.value} — scanning {len(symbols)} symbols …")

        # ── symbol scan ───────────────────────────────────────────────────────
        for symbol in tqdm(symbols, desc=exchange, unit='sym'):
            df_raw = fetch_ohlcv(symbol, exchange, n_bars)

            if df_raw.empty:
                errors.append(f"{exchange}:{symbol}  [no data]")
                continue

            try:
                df = add_all(df_raw)
            except ValueError as e:
                errors.append(f"{exchange}:{symbol}  [{e}]")
                continue
            except Exception as e:
                errors.append(f"{exchange}:{symbol}  [indicator error: {e}]")
                continue

            try:
                # generate_signals checks both directions; we keep only
                # the direction approved by the index
                sigs = [s for s in generate_signals(df) if s.direction == allowed]
            except Exception as e:
                errors.append(f"{exchange}:{symbol}  [signal error: {e}]")
                continue

            last = df.iloc[-1]
            for sig in sigs:
                # ── build chart ────────────────────────────────────────────────
                fig = charts.create_signal_chart(df, symbol, exchange, sig)
                try:
                    png = charts.figure_to_png(fig)
                except Exception as e:
                    print(f"  ⚠️  Chart render failed for {symbol}: {e}")
                    png = None

                # ── Telegram alert (with chart image) ─────────────────────────
                notification.send_signal(symbol, exchange, sig, df, 'D', png)

                # ── Firebase persistence ───────────────────────────────────────
                doc_id = db.save_signal(
                    symbol    = symbol,
                    exchange  = exchange,
                    signal    = sig,
                    interval  = '1D',
                    price_data = _price_data(df),
                    indicators = _indicator_data(df, sig),
                )
                if doc_id:
                    print(f"  🔥 Firebase: {doc_id}")

                all_signals.append({
                    'date':          today,
                    'exchange':      exchange,
                    'symbol':        symbol,
                    'direction':     sig.direction.value,
                    'pattern':       sig.pattern,
                    'triggered_ema': sig.triggered_ema,
                    'close':         round(float(last['Close']), 4),
                    'ema20':         round(float(last['EMA20']), 4),
                    'ema50':         round(float(last['EMA50']), 4),
                    'ema100':        round(float(last['EMA100']), 4),
                    'ema200':        round(float(last['EMA200']), 4),
                    'stoch_k':       round(float(last['STOCH_K']), 2),
                    'stoch_d':       round(float(last['STOCH_D']), 2),
                    'macd':          round(float(last['MACD']), 4),
                    'macd_signal':   round(float(last['MACD_S']), 4),
                    'firebase_id':   doc_id or '',
                    'tv_link':       (f'https://www.tradingview.com/chart/'
                                     f'?symbol={exchange}:{symbol}&interval=D'),
                })

    # ── 3. save & report ──────────────────────────────────────────────────────

    print(f"\n{'═'*55}")

    if all_signals:
        pd.DataFrame(all_signals).to_csv(results_path, index=False)
        print(f"✅  {len(all_signals)} signal(s) saved → {results_path}")
    else:
        print("No signals triggered on the current bar.")

    if errors:
        with open(errors_path, 'w') as f:
            f.write('\n'.join(errors))
        print(f"⚠️   {len(errors)} error(s) logged → {errors_path}")

    if all_signals:
        df_res = pd.DataFrame(all_signals)
        print("\n── Signal Summary ──────────────────────────────────────")
        print(df_res.groupby(['exchange', 'direction', 'pattern']).size().to_string())
        print("\n── Triggered Signals ───────────────────────────────────")
        cols = ['exchange', 'symbol', 'direction', 'pattern', 'triggered_ema', 'close']
        print(df_res[cols].to_string(index=False))


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Multi-exchange reversal pattern scanner (BIST, XETR, NYSE, NASDAQ)'
    )
    parser.add_argument(
        '--bars', type=int, default=DEFAULT_BARS,
        help=f'Daily bars to fetch per symbol (default {DEFAULT_BARS})',
    )
    parser.add_argument(
        '--exchange', type=str, default=None,
        help='Scan one exchange only: BIST | XETR | NYSE | NASDAQ',
    )
    args = parser.parse_args()

    run(n_bars=args.bars, filter_exchange=args.exchange)

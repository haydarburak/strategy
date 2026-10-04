"""
Sapan Strateji — piyasa tarayıcı.

Dört piyasa:
  NASDAQ  — NASDAQ 100 hisseleri   | trend filtresi: SPY (S&P 500)
  NYSE    — NYSE'deki S&P 500 hisseleri | trend filtresi: SPY (S&P 500)
  BIST    — BIST 100 hisseleri     | trend filtresi: XU100.IS
  XETR    — DAX 40 hisseleri       — trend filtresi: ^GDAXI

Long pozisyon: endeks uptrend (EMA20>50>100>200)
Short pozisyon: endeks downtrend (EMA20<50<100<200)
"""

import contextlib
import datetime
import io
import logging
import traceback
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from .sapan_strategy import (
    add_indicators,
    detect_signals,
    BIST_CONFIG_BASE,
    _wiki_tables,
    fetch_sp500_tickers,
    fetch_nasdaq100_tickers,
    fetch_dax_tickers,
)

logger = logging.getLogger(__name__)


def fetch_bist_tickers() -> list[str]:
    """
    Borsa Istanbul'daki tüm şirketlerin ticker listesini Wikipedia'dan çeker.
    Kaynak: https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Borsa_Istanbul
    Tablo 1: Company / Symbol / Notes sütunları
    Geçersiz semboller (çoklu, [edit] satırları, fon isimleri) filtrelenir.
    """
    try:
        tables = _wiki_tables(
            'https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Borsa_Istanbul'
        )
        df = tables[1]   # Company | Symbol | Notes
        # [edit] başlık satırları ve virgüllü çoklu semboller çıkar
        df = df[~df['Symbol'].str.contains(r'\[edit\]|,', na=True)]
        # Sadece 2–6 harf büyük ASCII semboller (gerçek hisse kodları)
        df = df[df['Symbol'].str.match(r'^[A-Z]{2,6}$', na=False)]
        tickers = (df['Symbol'] + '.IS').tolist()
        print(f'  BIST: {len(tickers)} hisse Wikipedia\'dan alındı.')
        return tickers
    except Exception as e:
        logger.warning('BIST listesi Wikipedia\'dan alınamadı: %s', e)
        return []


# ── Strateji konfigürasyonu (live) ──────────────────────────────────────────
LIVE_CFG = {
    **BIST_CONFIG_BASE,
    'interval': '1d',
}

# ── Piyasa tanımları ─────────────────────────────────────────────────────────
#   key → (tickers_fn_or_list, index_symbol, exchange_label, tv_exchange)
MARKET_DEFS = {
    'nasdaq': (fetch_nasdaq100_tickers, 'SPY',      'NASDAQ100',  'NASDAQ'),
    'nyse':   (None,                    'SPY',      'NYSE/SP500', 'NYSE'),
    'bist':   (fetch_bist_tickers,      'XU100.IS', 'BIST',       'BIST'),
    'xetr':   (fetch_dax_tickers,       '^GDAXI',   'DAX40',      'XETR'),
}

# Piyasa kapanışı (yerel saat) — kapanmamış günlük bar teyit mumu sayılmasın diye
MARKET_CLOSE = {
    'nasdaq': ('America/New_York', datetime.time(16, 0)),
    'nyse':   ('America/New_York', datetime.time(16, 0)),
    'bist':   ('Europe/Istanbul',  datetime.time(18, 10)),
    'xetr':   ('Europe/Berlin',    datetime.time(17, 35)),
}
_CLOSE_BUFFER = datetime.timedelta(minutes=20)


def _drop_unclosed_bar(df: pd.DataFrame, market_key: str) -> pd.DataFrame:
    """Seans henüz kapanmadıysa bugünün (eksik) barını çıkarır."""
    if df is None or df.empty:
        return df
    tz_name, close_t = MARKET_CLOSE[market_key]
    now_local = datetime.datetime.now(ZoneInfo(tz_name))
    close_dt  = datetime.datetime.combine(now_local.date(), close_t,
                                          tzinfo=now_local.tzinfo)
    if df.index[-1].date() >= now_local.date() and now_local < close_dt + _CLOSE_BUFFER:
        return df.iloc[:-1]
    return df


def _signal_status(df: pd.DataFrame, pos: int, direction: str,
                   entry: float, sl: float, tp: float) -> str:
    """Sinyal barından sonraki barlara bakarak stop emrinin durumunu döner."""
    is_long   = direction == 'LONG'
    triggered = False
    for _, bar in df.iloc[pos + 1:].iterrows():
        if not triggered:
            if (bar['High'] >= entry) if is_long else (bar['Low'] <= entry):
                triggered = True
            elif (bar['Low'] <= sl) if is_long else (bar['High'] >= sl):
                return 'GEÇERSİZ (tetiklenmeden SL seviyesine gitti)'
            else:
                continue
        # Aynı barda SL ve TP birlikte görülürse muhafazakâr davranıp SL say
        if (bar['Low'] <= sl) if is_long else (bar['High'] >= sl):
            return 'STOP OLDU'
        if (bar['High'] >= tp) if is_long else (bar['Low'] <= tp):
            return 'HEDEF GELDİ'
    return 'GİRİŞ TETİKLENDİ' if triggered else 'BEKLEMEDE'


def _get_tickers(market_key: str) -> list[str]:
    """NASDAQ ve NYSE ayrıştırması ile ticker listesi döner."""
    if market_key == 'nyse':
        sp500   = set(fetch_sp500_tickers())
        nasdaq  = set(fetch_nasdaq100_tickers())
        return sorted(sp500 - nasdaq)      # NYSE = SP500 - NASDAQ100
    src = MARKET_DEFS[market_key][0]
    return src() if callable(src) else list(src)


def _load_index_trend(index_symbol: str, market_key: str,
                      days_back: int = 600) -> tuple[pd.DataFrame, bool | None] | None:
    """Endeks trend verisi indir (EMA stack → uptrend / downtrend flag)."""
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
    end   = (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    with contextlib.redirect_stdout(io.StringIO()):
        idx = yf.download(index_symbol, start=start, end=end,
                          interval='1d', auto_adjust=True, progress=False)
    if idx is None or idx.empty:
        logger.warning('Endeks verisi alınamadı: %s', index_symbol)
        return None

    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.get_level_values(0)
    idx = idx[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Close'])
    idx.index = pd.to_datetime(idx.index).tz_localize(None)
    idx = _drop_unclosed_bar(idx, market_key)
    if idx.empty:
        logger.warning('Endeks verisi alınamadı: %s', index_symbol)
        return None

    for p in (20, 50, 100, 200):
        idx[f'ema{p}'] = idx['Close'].ewm(span=p, adjust=False).mean()

    idx['idx_uptrend']   = ((idx['ema20'] > idx['ema50']) &
                            (idx['ema50'] > idx['ema100']) &
                            (idx['ema100'] > idx['ema200']))
    idx['idx_downtrend'] = ((idx['ema20'] < idx['ema50']) &
                            (idx['ema50'] < idx['ema100']) &
                            (idx['ema100'] < idx['ema200']))

    # Son bar trend durumu
    latest = idx.iloc[-1]
    if latest['idx_uptrend']:
        direction = True
    elif latest['idx_downtrend']:
        direction = False
    else:
        direction = None

    logger.info('%s endeksi: %s', index_symbol,
                'UPTREND' if direction is True else
                'DOWNTREND' if direction is False else 'NÖTR')

    return idx[['idx_uptrend', 'idx_downtrend']], direction


def _download_stock(ticker: str, days_back: int = 500) -> pd.DataFrame | None:
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=days_back)).strftime('%Y-%m-%d')
    end   = (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

    with contextlib.redirect_stdout(io.StringIO()):
        df = yf.download(ticker, start=start, end=end,
                         interval='1d', auto_adjust=True, progress=False)

    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna(subset=['Close'])


def scan_market(
    market_key: str,
    lookback_days: int = 1,
) -> tuple[list[dict], dict]:
    """
    Bir piyasayı Sapan Stratejisi ile tarar.

    Döner: (alerts, market_info)
      alerts      — bulunan sinyal listesi (her biri dict)
      market_info — {'label', 'scanned', 'signals', 'index_direction'}
    """
    _, index_sym, label, tv_exch = MARKET_DEFS[market_key]

    print(f'\n{"="*60}')
    print(f'  {label}  —  endeks: {index_sym}')
    print(f'{"="*60}')

    tickers = _get_tickers(market_key)
    if not tickers:
        logger.warning('%s için ticker listesi alınamadı.', label)
        return [], {'label': label, 'scanned': 0, 'signals': 0,
                    'index_direction': None, 'index_symbol': index_sym, 'tv_exchange': tv_exch}

    print(f'  {len(tickers)} hisse taranıyor...')

    # Endeks trend yükle
    idx_result = _load_index_trend(index_sym, market_key)
    if idx_result is None:
        idx_trend, idx_direction = None, None
    else:
        idx_trend, idx_direction = idx_result

    # Sinyal penceresi duvar saatinden değil, piyasanın son kapanmış
    # seansından hesaplanır — cron gecikip gece yarısını (UTC) geçse de
    # son seansın sinyalleri kaçmaz.
    #   lookback_days=1 → yalnızca son kapanmış seans
    #   lookback_days=N → son N seans
    lookback_days = max(1, lookback_days)
    session_cutoff = None
    if idx_trend is not None:
        sessions = idx_trend.index.unique()
        session_cutoff = sessions[-min(lookback_days, len(sessions))]
        logger.info('%s: son seans %s, sinyal penceresi %s itibarıyla',
                    label, sessions[-1].date(), session_cutoff.date())

    alerts = []
    n_scanned = 0

    for i, ticker in enumerate(tickers, 1):
        try:
            df = _drop_unclosed_bar(_download_stock(ticker), market_key)
            if df is None or len(df) < 220:
                continue

            if session_cutoff is not None:
                cutoff = session_cutoff
            else:   # endeks yoksa hissenin kendi seanslarını kullan
                cutoff = df.index[-min(lookback_days, len(df))]

            n_scanned += 1
            cfg = {**LIVE_CFG, 'symbol': ticker}

            if idx_trend is not None:
                df = df.join(idx_trend, how='left')
                df['idx_uptrend']   = df['idx_uptrend'].ffill().fillna(False)
                df['idx_downtrend'] = df['idx_downtrend'].ffill().fillna(False)

            df = add_indicators(df, cfg)
            with contextlib.redirect_stdout(io.StringIO()):
                df = detect_signals(df, cfg)

            recent = df[(df['signal'] != 0) & (df.index >= cutoff)]
            if recent.empty:
                continue

            for sig_date, sig_row in recent.iterrows():
                sig_pos   = df.index.get_loc(sig_date)
                direction = 'LONG' if int(sig_row['signal']) == 1 else 'SHORT'

                idx_up = None
                if 'idx_uptrend' in df.columns:
                    idx_up = bool(df['idx_uptrend'].iloc[sig_pos])

                entry  = float(sig_row['entry_price'])
                sl     = float(sig_row['stop_loss'])
                tp     = float(sig_row['take_profit'])
                status = _signal_status(df, sig_pos, direction, entry, sl, tp)

                alerts.append({
                    'symbol':       ticker,
                    'exchange':     tv_exch,
                    'market':       label,
                    'index':        index_sym,
                    'sig_date':     sig_date,
                    'sig_pos':      sig_pos,
                    'direction':    direction,
                    'sig_type':     str(sig_row.get('signal_type', '')),
                    'entry':        entry,
                    'sl':           sl,
                    'tp':           tp,
                    'status':       status,
                    'idx_uptrend':  idx_up,
                    'df':           df,
                })
                print(f'  ✅ [{i:>4}/{len(tickers)}] {ticker:<12} '
                      f'{direction:<5} {sig_row.get("signal_type","")} '
                      f'{sig_date.strftime("%Y-%m-%d")}  {status}')

        except Exception as e:
            logger.debug('%s hatası: %s', ticker, e)

    print(f'  → {len(alerts)} sinyal, {n_scanned} hisse tarandı.\n')

    return alerts, {
        'label':           label,
        'scanned':         n_scanned,
        'signals':         len(alerts),
        'index_direction': idx_direction,
        'index_symbol':    index_sym,
        'tv_exchange':     tv_exch,
    }

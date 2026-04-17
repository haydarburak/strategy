"""
Firebase Firestore persistence for trading signals and index status.

Required environment variables (pick one)
------------------------------------------
  FIREBASE_CREDENTIALS      — path to a service-account JSON file
  FIREBASE_CREDENTIALS_JSON — raw JSON string  OR  path to JSON file
                               (supports GitHub Actions / CI secrets)

If neither is set, all save calls are no-ops that log a warning once.

Firestore collections
----------------------
  trading_signals    — one doc per triggered pattern signal
  index_status       — upserted doc per (exchange, symbol, interval)
  daily_statistics   — aggregated daily counters per exchange / pattern
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import credentials, firestore

from .patterns import Direction, Signal
from .divergence import DivergenceSignal

logger = logging.getLogger(__name__)

_warned_missing = False


# ── credential loading ─────────────────────────────────────────────────────────

def _load_credentials() -> Optional[credentials.Certificate]:
    global _warned_missing

    # 1. explicit file path
    path = os.getenv('FIREBASE_CREDENTIALS')
    if path and os.path.isfile(path):
        return credentials.Certificate(path)

    # 2. JSON string or path in FIREBASE_CREDENTIALS_JSON
    raw = os.getenv('FIREBASE_CREDENTIALS_JSON')
    if raw:
        try:
            data = json.loads(raw)
            return credentials.Certificate(data)
        except (json.JSONDecodeError, ValueError):
            # treat as file path
            if os.path.isfile(raw):
                with open(raw, encoding='utf-8') as f:
                    return credentials.Certificate(json.load(f))

    if not _warned_missing:
        logger.warning(
            'No Firebase credentials found '
            '(FIREBASE_CREDENTIALS / FIREBASE_CREDENTIALS_JSON not set) '
            '— Firestore persistence disabled.'
        )
        _warned_missing = True
    return None


# ── FirestoreDB ────────────────────────────────────────────────────────────────

class FirestoreDB:
    """
    Thin wrapper around the Firestore client.
    Initialised once; all public methods are safe to call even when
    Firebase is not configured (they become no-ops).
    """

    def __init__(self) -> None:
        self.db = None
        self._connect()

    def _connect(self) -> None:
        try:
            if firebase_admin._apps:
                app = firebase_admin.get_app()
            else:
                cred = _load_credentials()
                if cred is None:
                    return
                app = firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info('Firebase Firestore connected.')
        except Exception as e:
            logger.error(f'Firebase init failed: {e}')

    @property
    def connected(self) -> bool:
        return self.db is not None

    # ── public methods ─────────────────────────────────────────────────────────

    def save_signal(
        self,
        symbol: str,
        exchange: str,
        signal: Signal,
        interval: str,
        price_data: Dict[str, float],
        indicators: Dict[str, Any],
    ) -> Optional[str]:
        """
        Persist a triggered pattern signal.

        Parameters
        ----------
        symbol      : ticker, e.g. 'AAPL'
        exchange    : e.g. 'NASDAQ'
        signal      : Signal(pattern, direction, triggered_ema)
        interval    : timeframe string, e.g. '1D'
        price_data  : {'open', 'high', 'low', 'close', 'volume',
                       'initial_close', 'reversal_close'}
        indicators  : {'ema20', 'ema50', 'ema100', 'ema200',
                       'stoch_k', 'stoch_d', 'macd', 'macd_signal',
                       'rsi', 'triggered_ema'}

        Returns
        -------
        Firestore document ID on success, None on failure / not connected.
        """
        if not self.connected:
            return None
        try:
            now = datetime.now(timezone.utc)
            doc = {
                'timestamp':            now,
                'symbol':               symbol,
                'exchange':             exchange,
                'full_symbol':          f'{exchange}:{symbol}',
                'pattern':              signal.pattern,
                'direction':            signal.direction.value,
                'is_long':              signal.direction == Direction.LONG,
                'triggered_ema':        signal.triggered_ema,
                'interval':             interval,
                'price_data':           price_data,
                'technical_indicators': indicators,
                'status':               'ACTIVE',
                'created_at':           now,
                'chart_url': (
                    f'https://www.tradingview.com/chart/'
                    f'?symbol={exchange}:{symbol}&interval={interval}'
                ),
            }
            _, ref = self.db.collection('trading_signals').add(doc)
            self._bump_daily_stats(exchange, signal, interval)
            logger.info(f'Signal saved [{ref.id}]: {exchange}:{symbol} {signal.direction.value} {signal.pattern}')
            return ref.id
        except Exception as e:
            logger.error(f'save_signal failed for {exchange}:{symbol}: {e}')
            return None

    def save_index_status(
        self,
        index_symbol: str,
        index_exchange: str,
        direction: Optional[bool],
        interval: str,
    ) -> Optional[str]:
        """
        Upsert the current bias of a market index.

        direction : True → LONG, False → SHORT, None → NEUTRAL
        Returns the document ID on success.
        """
        if not self.connected:
            return None
        try:
            status_map = {True: 'LONG', False: 'SHORT', None: 'NEUTRAL'}
            status = status_map[direction]
            doc_id = f'{index_exchange}_{index_symbol}_{interval}'
            now    = datetime.now(timezone.utc)
            ref    = self.db.collection('index_status').document(doc_id)

            payload: Dict[str, Any] = {
                'timestamp':      now,
                'index_symbol':   index_symbol,
                'index_exchange': index_exchange,
                'full_symbol':    f'{index_exchange}:{index_symbol}',
                'status':         status,
                'direction':      direction,
                'interval':       interval,
                'updated_at':     now,
                'chart_url': (
                    f'https://www.tradingview.com/chart/'
                    f'?symbol={index_exchange}:{index_symbol}&interval={interval}'
                ),
            }
            snap = ref.get()
            if snap.exists:
                ref.update(payload)
            else:
                payload['created_at'] = now
                ref.set(payload)

            logger.info(f'Index status saved: {doc_id} → {status}')
            return doc_id
        except Exception as e:
            logger.error(f'save_index_status failed for {index_exchange}:{index_symbol}: {e}')
            return None

    def save_divergence_signal(
        self,
        symbol: str,
        exchange: str,
        signal: DivergenceSignal,
        interval: str,
        price_data: Dict[str, float],
        pivot_timestamp=None,
    ) -> Optional[str]:
        """
        Persist a detected RSI divergence.

        Parameters
        ----------
        symbol          : ticker, e.g. 'AAPL'
        exchange        : e.g. 'NASDAQ'
        signal          : DivergenceSignal from divergence.find_divergences()
        interval        : timeframe string, e.g. '1D'
        price_data      : {'open', 'high', 'low', 'close', 'volume', 'rsi'}
        pivot_timestamp : datetime of the p2 pivot bar (optional)

        Returns
        -------
        Firestore document ID on success, None on failure / not connected.
        """
        if not self.connected:
            return None
        try:
            now = datetime.now(timezone.utc)
            doc: Dict[str, Any] = {
                'timestamp':      now,
                'symbol':         symbol,
                'exchange':       exchange,
                'full_symbol':    f'{exchange}:{symbol}',
                'divergence_type': signal.div_type,
                'label':          signal.label,
                'reason':         signal.reason,
                'interval':       interval,
                'price_data':     price_data,
                'meta':           signal.meta,
                'status':         'ACTIVE',
                'created_at':     now,
                'chart_url': (
                    f'https://www.tradingview.com/chart/'
                    f'?symbol={exchange}:{symbol}&interval={interval}'
                ),
            }
            if pivot_timestamp is not None:
                doc['pivot_timestamp'] = pivot_timestamp

            _, ref = self.db.collection('divergence_signals').add(doc)
            self._bump_divergence_stats(exchange, signal.div_type, interval)
            logger.info(
                f'Divergence saved [{ref.id}]: {exchange}:{symbol} {signal.div_type}'
            )
            return ref.id
        except Exception as e:
            logger.error(f'save_divergence_signal failed for {exchange}:{symbol}: {e}')
            return None

    # ── internal helpers ───────────────────────────────────────────────────────

    def _bump_divergence_stats(
        self, exchange: str, div_type: str, interval: str
    ) -> None:
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            ref   = self.db.collection('divergence_statistics').document(today)
            snap  = ref.get()

            data: Dict[str, Any] = snap.to_dict() if snap.exists else {
                'date':      today,
                'total':     0,
                'types':     {},
                'exchanges': {},
                'intervals': {},
            }

            data['total'] = data.get('total', 0) + 1
            for bucket_key, value in [
                ('types',     div_type),
                ('exchanges', exchange),
                ('intervals', interval),
            ]:
                bucket = data.get(bucket_key, {})
                bucket[value] = bucket.get(value, 0) + 1
                data[bucket_key] = bucket

            data['updated_at'] = datetime.now(timezone.utc)
            ref.set(data)
        except Exception as e:
            logger.error(f'_bump_divergence_stats failed: {e}')

    def _bump_daily_stats(
        self, exchange: str, signal: Signal, interval: str
    ) -> None:
        """Increment per-day counters for signals, patterns, exchanges, intervals."""
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            ref   = self.db.collection('daily_statistics').document(today)
            snap  = ref.get()

            data: Dict[str, Any] = snap.to_dict() if snap.exists else {
                'date':      today,
                'total':     0,
                'long':      0,
                'short':     0,
                'patterns':  {},
                'exchanges': {},
                'intervals': {},
            }

            data['total'] = data.get('total', 0) + 1
            if signal.direction == Direction.LONG:
                data['long'] = data.get('long', 0) + 1
            else:
                data['short'] = data.get('short', 0) + 1

            for bucket_key, value in [
                ('patterns',  signal.pattern),
                ('exchanges', exchange),
                ('intervals', interval),
            ]:
                bucket = data.get(bucket_key, {})
                bucket[value] = bucket.get(value, 0) + 1
                data[bucket_key] = bucket

            data['updated_at'] = datetime.now(timezone.utc)
            ref.set(data)
        except Exception as e:
            logger.error(f'_bump_daily_stats failed: {e}')


# ── singleton ──────────────────────────────────────────────────────────────────

_instance: Optional[FirestoreDB] = None


def get_db() -> FirestoreDB:
    """Return the process-wide FirestoreDB singleton (lazy-initialised)."""
    global _instance
    if _instance is None:
        _instance = FirestoreDB()
    return _instance

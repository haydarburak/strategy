import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseDB:
    """Firebase Firestore integration for trading alarms"""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """
        Initialize Firebase connection
        
        Args:
            credentials_path: Path to Firebase service account key JSON file
                             If None, will look for FIREBASE_CREDENTIALS environment variable
        """
        self.db = None
        self.app = None
        self._initialize_firebase(credentials_path)
    
    def _initialize_firebase(self, credentials_path: Optional[str] = None):
        """Initialize Firebase Admin SDK"""
        try:
            # Check if Firebase is already initialized
            if not firebase_admin._apps:
                if credentials_path:
                    cred = credentials.Certificate(credentials_path)
                else:
                    # Try to get credentials from environment variable
                    cred_path = os.getenv('FIREBASE_CREDENTIALS')
                    if cred_path:
                        cred = credentials.Certificate(cred_path)
                    else:
                        # Try to get credentials from GitHub Secrets
                        github_credentials = os.getenv('FIREBASE_CREDENTIALS_JSON')

                        try:
                            with open(github_credentials, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                cred = credentials.Certificate(data)
                                print("📄 JSON dosyası başarıyla yüklendi.")
                        except Exception as e:
                            print("❌ JSON dosyası okunamadı:", e)

                self.app = firebase_admin.initialize_app(cred)
            else:
                self.app = firebase_admin.get_app()
            
            self.db = firestore.client()
            print("✅ Firebase initialized successfully")
            
        except Exception as e:
            print(f"❌ Error initializing Firebase: {e}")
            self.db = None  # Set to None so we can handle gracefully
    
    def is_connected(self) -> bool:
        """Check if Firebase is properly connected"""
        return self.db is not None
    
    def save_trading_signal(self, 
                           symbol: str,
                           strategy_name: str,
                           is_long: bool,
                           interval: str,
                           price_data: Dict[str, float],
                           technical_indicators: Dict[str, Any] = None) -> Optional[str]:
        """
        Save trading signal to Firestore
        
        Args:
            symbol: Trading symbol (e.g., 'NASDAQ:AAPL')
            strategy_name: Name of the strategy that triggered
            is_long: True for long position, False for short
            interval: Timeframe (e.g., '1H', '4H', '1D')
            price_data: Dictionary with OHLCV data
            technical_indicators: Technical indicators data (EMA, MACD, etc.)
            
        Returns:
            str: Document ID of the saved signal, None if failed
        """
        if not self.is_connected():
            print("❌ Firebase not connected. Signal not saved.")
            return None
            
        try:
            # Parse symbol to extract exchange and ticker
            exchange, ticker = self._parse_symbol(symbol)
            
            # Create signal document
            signal_data = {
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'exchange': exchange,
                'ticker': ticker,
                'strategy_name': strategy_name,
                'direction': 'LONG' if is_long else 'SHORT',
                'is_long': is_long,
                'interval': interval,
                'price_data': price_data or {},
                'technical_indicators': technical_indicators or {},
                'status': 'ACTIVE',  # ACTIVE, EXECUTED, EXPIRED, CANCELLED
                'created_at': datetime.now(timezone.utc),
                'chart_url': f"https://www.tradingview.com/chart/?symbol={symbol}&interval={interval}"
            }
            
            # Add to trading_signals collection
            doc_ref = self.db.collection('trading_signals').add(signal_data)
            signal_id = doc_ref[1].id
            
            # Update statistics
            self._update_daily_stats(exchange, strategy_name, is_long, interval)
            
            print(f"✅ Trading signal saved: {signal_id}")
            return signal_id
            
        except Exception as e:
            print(f"❌ Error saving trading signal: {e}")
            return None
    
    def save_divergence_signal(self,
                              symbol: str,
                              divergence_type: str,
                              interval: str,
                              price_data: Dict[str, float] = None,
                              pivot_timestamp: Optional[datetime] = None,
                              divergence_meta: Optional[Dict] = None,
                              ttl_hours: int = 48) -> Optional[str]:
        """
        Save RSI divergence signal to Firestore.

        Before saving, any existing ACTIVE signal for the same
        symbol + divergence_type + interval whose pivot_timestamp differs
        from the new one is expired automatically.  This prevents stale
        historical pivots from staying "ACTIVE" across multiple bot runs.

        Args:
            symbol:          Trading symbol  (e.g. 'NASDAQ:GOOGL')
            divergence_type: One of 'bullish', 'bearish', 'hidden_bullish', 'hidden_bearish'
            interval:        Timeframe string (e.g. '1H', '4H', '1D')
            price_data:      OHLCV + RSI dict at the pivot bar
            pivot_timestamp: The actual datetime of the pivot bar (not now).
                             Pass this so duplicate runs don't re-save the same pivot.
            ttl_hours:       Automatically expire signals older than this many hours
                             (default 48 h — keeps signals visible for ~2 days).

        Returns:
            str: Document ID of the saved signal, or None if skipped / failed.
        """
        if not self.is_connected():
            print("❌ Firebase not connected. Divergence signal not saved.")
            return None

        try:
            exchange, ticker = self._parse_symbol(symbol)
            now = datetime.now(timezone.utc)

            # ------------------------------------------------------------------
            # 1. Expire any ACTIVE signals for this symbol+type+interval that
            #    are either (a) too old or (b) reference a different pivot bar.
            # ------------------------------------------------------------------
            collection = self.db.collection('divergence_signals')
            stale_query = (collection
                           .where('symbol', '==', symbol)
                           .where('divergence_type', '==', divergence_type)
                           .where('interval', '==', interval)
                           .where('status', '==', 'ACTIVE'))

            for stale_doc in stale_query.stream():
                stale_data = stale_doc.to_dict()
                stale_ts = stale_data.get('pivot_timestamp') or stale_data.get('created_at')

                # Expire if older than TTL
                age_hours = (now - stale_ts).total_seconds() / 3600 if stale_ts else ttl_hours + 1
                is_too_old = age_hours > ttl_hours

                # Expire if it belongs to a different pivot bar
                is_different_pivot = (
                    pivot_timestamp is not None
                    and stale_data.get('pivot_timestamp') is not None
                    and abs((pivot_timestamp - stale_data['pivot_timestamp']).total_seconds()) > 3600
                )

                if is_too_old or is_different_pivot:
                    stale_doc.reference.update({'status': 'EXPIRED', 'expired_at': now})
                    print(f"⏰ Expired stale {divergence_type} signal for {symbol} (age {age_hours:.1f}h)")
                elif stale_data.get('pivot_timestamp') == pivot_timestamp:
                    # Exact same pivot already saved — skip duplicate
                    print(f"⏭️  Skipping duplicate {divergence_type} pivot for {symbol}")
                    return stale_doc.id

            # ------------------------------------------------------------------
            # 2. Save the new signal.
            # ------------------------------------------------------------------
            divergence_data = {
                'timestamp':       now,
                'pivot_timestamp': pivot_timestamp or now,
                'symbol':          symbol,
                'exchange':        exchange,
                'ticker':          ticker,
                'divergence_type': divergence_type,
                'interval':        interval,
                'price_data':      price_data or {},
                # Pivot comparison: p1 & p2 values + human-readable reason
                'divergence_meta': divergence_meta or {},
                'status':          'ACTIVE',
                'created_at':      now,
                'expires_at':      datetime.fromtimestamp(
                                       now.timestamp() + ttl_hours * 3600, tz=timezone.utc
                                   ),
                'chart_url': (
                    f"https://www.tradingview.com/chart/"
                    f"?symbol={symbol}&interval={interval}"
                ),
            }

            doc_ref = self.db.collection('divergence_signals').add(divergence_data)
            signal_id = doc_ref[1].id

            self._update_divergence_stats(exchange, divergence_type, interval)

            print(f"✅ Divergence signal saved: {signal_id}")
            return signal_id

        except Exception as e:
            print(f"❌ Error saving divergence signal: {e}")
            return None
    
    def get_recent_signals(self, 
                          limit: int = 50,
                          signal_type: str = 'trading',  # 'trading' or 'divergence'
                          hours_back: int = 24) -> List[Dict]:
        """
        Retrieve recent signals
        
        Args:
            limit: Maximum number of signals to return
            signal_type: Type of signals to retrieve
            hours_back: How many hours back to look
            
        Returns:
            List of signal documents
        """
        if not self.is_connected():
            return []
            
        try:
            collection_name = f"{signal_type}_signals"
            query = self.db.collection(collection_name)
            
            # Filter by time
            time_filter = datetime.now(timezone.utc).replace(
                hour=max(0, datetime.now(timezone.utc).hour - hours_back)
            )
            query = query.where('timestamp', '>=', time_filter)
            
            # Order by timestamp descending and limit
            query = query.order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit)
            
            docs = query.stream()
            signals = []
            
            for doc in docs:
                signal_data = doc.to_dict()
                signal_data['id'] = doc.id
                signals.append(signal_data)
            
            return signals
            
        except Exception as e:
            print(f"❌ Error retrieving signals: {e}")
            return []
    
    def get_dashboard_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get dashboard statistics for web visualization
        
        Args:
            days: Number of days to include
            
        Returns:
            Dictionary with dashboard data
        """
        if not self.is_connected():
            return {}
            
        try:
            # Get recent signals for quick stats
            recent_trading = self.get_recent_signals(100, 'trading', days * 24)
            recent_divergence = self.get_recent_signals(100, 'divergence', days * 24)
            
            # Calculate basic statistics
            total_signals = len(recent_trading)
            total_divergence = len(recent_divergence)
            
            long_signals = sum(1 for s in recent_trading if s.get('is_long', False))
            short_signals = total_signals - long_signals
            
            # Exchange distribution
            exchange_dist = {}
            for signal in recent_trading:
                exchange = signal.get('exchange', 'Unknown')
                exchange_dist[exchange] = exchange_dist.get(exchange, 0) + 1
            
            # Strategy distribution
            strategy_dist = {}
            for signal in recent_trading:
                strategy = signal.get('strategy_name', 'Unknown')
                strategy_dist[strategy] = strategy_dist.get(strategy, 0) + 1
            
            return {
                'summary': {
                    'total_trading_signals': total_signals,
                    'total_divergence_signals': total_divergence,
                    'long_signals': long_signals,
                    'short_signals': short_signals,
                    'long_percentage': (long_signals / total_signals * 100) if total_signals > 0 else 0
                },
                'exchange_distribution': exchange_dist,
                'strategy_distribution': strategy_dist,
                'period_days': days,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error getting dashboard stats: {e}")
            return {}
    
    def _parse_symbol(self, symbol: str) -> tuple:
        """Parse symbol string to extract exchange and ticker"""
        if ':' in symbol:
            exchange, ticker = symbol.split(':', 1)
            return exchange, ticker
        else:
            return 'UNKNOWN', symbol
    
    def _update_daily_stats(self, exchange: str, strategy: str, is_long: bool, interval: str):
        """Update daily statistics"""
        if not self.is_connected():
            return
            
        try:
            today = datetime.now(timezone.utc).date()
            doc_id = f"{today.isoformat()}"
            
            doc_ref = self.db.collection('daily_statistics').document(doc_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
            else:
                data = {
                    'date': datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
                    'total_signals': 0,
                    'long_signals': 0,
                    'short_signals': 0,
                    'strategies': {},
                    'exchanges': {},
                    'intervals': {}
                }
            
            # Update counters
            data['total_signals'] = data.get('total_signals', 0) + 1
            
            if is_long:
                data['long_signals'] = data.get('long_signals', 0) + 1
            else:
                data['short_signals'] = data.get('short_signals', 0) + 1
            
            # Update nested counters
            strategies = data.get('strategies', {})
            strategies[strategy] = strategies.get(strategy, 0) + 1
            data['strategies'] = strategies
            
            exchanges = data.get('exchanges', {})
            exchanges[exchange] = exchanges.get(exchange, 0) + 1
            data['exchanges'] = exchanges
            
            intervals = data.get('intervals', {})
            intervals[interval] = intervals.get(interval, 0) + 1
            data['intervals'] = intervals
            
            data['updated_at'] = datetime.now(timezone.utc)
            
            doc_ref.set(data)
            
        except Exception as e:
            print(f"❌ Error updating daily statistics: {e}")
    
    def _update_divergence_stats(self, exchange: str, divergence_type: str, interval: str):
        """Update divergence statistics"""
        if not self.is_connected():
            return
            
        try:
            today = datetime.now(timezone.utc).date()
            doc_id = f"{today.isoformat()}-divergence"
            
            doc_ref = self.db.collection('divergence_statistics').document(doc_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
            else:
                data = {
                    'date': datetime.combine(today, datetime.min.time()).replace(tzinfo=timezone.utc),
                    'total_divergence': 0,
                    'bullish_divergence': 0,
                    'bearish_divergence': 0,
                    'exchanges': {},
                    'intervals': {}
                }
            
            # Update counters
            data['total_divergence'] = data.get('total_divergence', 0) + 1
            
            if divergence_type.lower() == 'bullish':
                data['bullish_divergence'] = data.get('bullish_divergence', 0) + 1
            else:
                data['bearish_divergence'] = data.get('bearish_divergence', 0) + 1
            
            # Update nested counters
            exchanges = data.get('exchanges', {})
            exchanges[exchange] = exchanges.get(exchange, 0) + 1
            data['exchanges'] = exchanges
            
            intervals = data.get('intervals', {})
            intervals[interval] = intervals.get(interval, 0) + 1
            data['intervals'] = intervals
            
            data['updated_at'] = datetime.now(timezone.utc)
            
            doc_ref.set(data)
            
        except Exception as e:
            print(f"❌ Error updating divergence statistics: {e}")
    
    def save_index_status(self, 
                         index_symbol: str,
                         index_exchange: str,
                         index_long: Optional[bool],
                         interval: str) -> Optional[str]:
        """
        Save or update index status in Firestore
        
        Args:
            index_symbol: Index symbol (e.g., 'SPX')
            index_exchange: Index exchange (e.g., 'OANDA')
            index_long: True for LONG, False for SHORT, None for NOTR
            interval: Timeframe (e.g., '1H', '4H', '1D')
            
        Returns:
            str: Document ID of the saved/updated index status, None if failed
        """
        if not self.is_connected():
            print("❌ Firebase not connected. Index status not saved.")
            return None
            
        try:
            # Determine status based on index_long value
            if index_long is True:
                status = "LONG"
            elif index_long is False:
                status = "SHORT"
            else:  # index_long is None
                status = "NOTR"
            
            # Create a unique document ID based on symbol, exchange, and interval
            doc_id = f"{index_exchange}_{index_symbol}_{interval}"
            
            # Check if document already exists
            doc_ref = self.db.collection('index_status').document(doc_id)
            existing_doc = doc_ref.get()
            
            current_time = datetime.now(timezone.utc)
            
            if existing_doc.exists:
                # Update existing document
                update_data = {
                    'timestamp': current_time,
                    'status': status,
                    'index_long': index_long,
                    'updated_at': current_time,
                    'chart_url': f"https://www.tradingview.com/chart/?symbol={index_exchange}:{index_symbol}&interval={interval}"
                }
                
                doc_ref.update(update_data)
                print(f"🔄 Index status updated: {doc_id} - {index_symbol} ({index_exchange}) - {status}")
                return doc_id
            else:
                # Create new document with specific ID
                index_data = {
                    'timestamp': current_time,
                    'index_symbol': index_symbol,
                    'index_exchange': index_exchange,
                    'full_symbol': f"{index_exchange}:{index_symbol}",
                    'status': status,
                    'index_long': index_long,
                    'interval': interval,
                    'created_at': current_time,
                    'updated_at': current_time,
                    'chart_url': f"https://www.tradingview.com/chart/?symbol={index_exchange}:{index_symbol}&interval={interval}"
                }
                
                doc_ref.set(index_data)
                print(f"✅ Index status created: {doc_id} - {index_symbol} ({index_exchange}) - {status}")
                return doc_id
            
        except Exception as e:
            print(f"❌ Error saving index status: {e}")
            return None
    
    def get_all_watchlist_stocks(self) -> list:
        """
        Read all users' watchlists from Firestore and return the union of all stocks.

        Returns:
            List of unique stock symbols e.g. ["NASDAQ:GOOGL", "BIST:KRONT"]
        """
        if not self.is_connected():
            print("❌ Firebase not connected. Cannot fetch watchlists.")
            return []

        try:
            docs = self.db.collection('watchlists').stream()
            all_stocks = set()
            user_count = 0

            for doc in docs:
                data = doc.to_dict()
                stocks = data.get('stocks', [])
                all_stocks.update(stocks)
                user_count += 1

            stock_list = list(all_stocks)
            print(f"📋 Loaded {len(stock_list)} unique stocks from {user_count} user watchlist(s)")
            return stock_list

        except Exception as e:
            print(f"❌ Error fetching watchlists: {e}")
            return []

    def close(self):
        """Clean up Firebase connection"""
        if self.app:
            firebase_admin.delete_app(self.app)
            print("🔌 Firebase connection closed")


# Global instance for easy access
_firebase_instance = None

def get_firebase_db() -> FirebaseDB:
    """Get singleton Firebase instance"""
    global _firebase_instance
    if _firebase_instance is None:
        _firebase_instance = FirebaseDB()
    return _firebase_instance

def close_firebase():
    """Close Firebase connection"""
    global _firebase_instance
    if _firebase_instance:
        _firebase_instance.close()
        _firebase_instance = None


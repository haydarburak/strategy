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
                              divergence_type: str,  # 'Bullish' or 'Bearish'
                              interval: str,
                              price_data: Dict[str, float] = None) -> Optional[str]:
        """
        Save RSI divergence signal to Firestore
        
        Args:
            symbol: Trading symbol
            divergence_type: 'Bullish' or 'Bearish'
            interval: Timeframe
            price_data: Price data at divergence point
            
        Returns:
            str: Document ID of the saved divergence signal, None if failed
        """
        if not self.is_connected():
            print("❌ Firebase not connected. Divergence signal not saved.")
            return None
            
        try:
            exchange, ticker = self._parse_symbol(symbol)
            
            divergence_data = {
                'timestamp': datetime.now(timezone.utc),
                'symbol': symbol,
                'exchange': exchange,
                'ticker': ticker,
                'divergence_type': divergence_type,
                'interval': interval,
                'price_data': price_data or {},
                'status': 'ACTIVE',
                'created_at': datetime.now(timezone.utc),
                'chart_url': f"https://www.tradingview.com/chart/?symbol={symbol}&interval={interval}"
            }
            
            # Add to divergence_signals collection
            doc_ref = self.db.collection('divergence_signals').add(divergence_data)
            signal_id = doc_ref[1].id
            
            # Update divergence statistics
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


#!/usr/bin/env python3
"""
Update Price Script

This script fetches unique exchange-symbol pairs from the Firebase 'portfolio' collection,
retrieves their current prices using TradingView data feed, and stores the price information
in the 'stock_prices' collection.

Usage:
    python updateprice.py
"""

import time
from datetime import datetime, timezone
from typing import Set, Tuple, Dict, Any, Optional
from firebase_db import get_firebase_db
from tvDatafeed import TvDatafeed
from websocket import WebSocketTimeoutException


class PriceUpdater:
    """Handles fetching and updating stock prices from TradingView to Firebase"""
    
    def __init__(self):
        """Initialize the price updater with Firebase and TradingView connections"""
        self.firebase_db = get_firebase_db()
        self.tv = TvDatafeed()
        self.max_retries = 3
        self.backoff_factor = 2
    
    def get_unique_portfolio_symbols(self) -> Set[Tuple[str, str]]:
        """
        Get unique exchange-symbol pairs from the portfolio collection
        
        Returns:
            Set of tuples containing (exchange, symbol) pairs
        """
        if not self.firebase_db.is_connected():
            print("❌ Firebase not connected. Cannot fetch portfolio symbols.")
            return set()
        
        try:
            print("📊 Fetching portfolio symbols from Firebase...")
            
            # Get all documents from portfolio collection
            portfolio_ref = self.firebase_db.db.collection('portfolio')
            docs = portfolio_ref.stream()
            
            unique_symbols = set()
            
            for doc in docs:
                data = doc.to_dict()
                exchange = data.get('exchange')
                symbol = data.get('symbol')
                
                if exchange and symbol:
                    unique_symbols.add((exchange, symbol))
                    print(f"  Found: {exchange}:{symbol}")
                else:
                    print(f"  ⚠️ Skipping document {doc.id}: missing exchange or symbol")
            
            print(f"✅ Found {len(unique_symbols)} unique exchange-symbol pairs")
            return unique_symbols
        
        except Exception as e:
            print(f"❌ Error fetching portfolio symbols: {e}")
            return set()
    
    def get_current_price(self, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        """
        Get current price for a symbol from TradingView
        
        Args:
            symbol: The stock symbol (e.g., 'AAPL')
            exchange: The exchange (e.g., 'NASDAQ')
        
        Returns:
            Dictionary with price information or None if failed
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                print(f"  📈 Fetching price for {exchange}:{symbol} (attempt {attempt}/{self.max_retries})")
                
                # Get the latest data (just 1 bar to get current price)
                data = self.tv.get_hist(symbol, exchange, n_bars=1)
                
                if data is not None and not data.empty:
                    # Get the latest row
                    latest = data.iloc[-1]
                    
                    price_info = {
                        'symbol': symbol,
                        'exchange': exchange,
                        'price': float(latest['close']),
                        'open': float(latest['open']),
                        'high': float(latest['high']),
                        'low': float(latest['low']),
                        'volume': float(latest['volume']) if 'volume' in latest else 0.0,
                        'currency': self._get_currency_for_exchange(exchange),
                        'updated_at': datetime.now(timezone.utc)
                    }
                    
                    print(f"  ✅ Price fetched: {price_info['price']} {price_info['currency']}")
                    return price_info
                else:
                    print(f"  ⚠️ No data returned for {exchange}:{symbol}")
            
            except (TimeoutError, WebSocketTimeoutException) as e:
                print(f"  ⏱️ Timeout for {exchange}:{symbol} on attempt {attempt}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor * attempt)
            
            except Exception as e:
                print(f"  ❌ Error fetching price for {exchange}:{symbol}: {e}")
                if attempt < self.max_retries:
                    time.sleep(self.backoff_factor)
        
        print(f"  ❌ Failed to fetch price for {exchange}:{symbol} after {self.max_retries} attempts")
        return None
    
    def _get_currency_for_exchange(self, exchange: str) -> str:
        """
        Get the currency for a given exchange
        
        Args:
            exchange: Exchange name
        
        Returns:
            Currency code
        """
        currency_map = {
            'NASDAQ': 'USD',
            'NYSE': 'USD',
            'BIST': 'TRY',
            'XETR': 'EUR',
            'LSE': 'GBP',
            'TSE': 'JPY',
            'ASX': 'AUD',
            'TSX': 'CAD',
            'NSE': 'INR',
            'HKEX': 'HKD'
        }
        return currency_map.get(exchange.upper(), 'USD')
    
    def save_price_to_firebase(self, price_info: Dict[str, Any]) -> bool:
        """
        Save price information to Firebase stock_prices collection
        
        Args:
            price_info: Dictionary containing price information
        
        Returns:
            True if saved successfully, False otherwise
        """
        if not self.firebase_db.is_connected():
            print("❌ Firebase not connected. Cannot save price.")
            return False
        
        try:
            # Create document ID based on exchange and symbol
            doc_id = f"{price_info['exchange']}_{price_info['symbol']}"
            
            # Save to stock_prices collection
            doc_ref = self.firebase_db.db.collection('stock_prices').document(doc_id)
            doc_ref.set(price_info)
            
            print(f"  💾 Price saved to Firebase: {doc_id}")
            return True
        
        except Exception as e:
            print(f"  ❌ Error saving price to Firebase: {e}")
            return False
    
    def update_all_prices(self) -> Dict[str, int]:
        """
        Update prices for all unique symbols in the portfolio
        
        Returns:
            Dictionary with update statistics
        """
        print("🚀 Starting price update process...")
        
        # Get unique symbols from portfolio
        symbols = self.get_unique_portfolio_symbols()
        
        if not symbols:
            print("❌ No symbols found in portfolio. Exiting.")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        stats = {'total': len(symbols), 'success': 0, 'failed': 0}
        
        print(f"📊 Updating prices for {stats['total']} symbols...")
        
        for i, (exchange, symbol) in enumerate(symbols, 1):
            print(f"\n[{i}/{stats['total']}] Processing {exchange}:{symbol}")
            
            # Get current price from TradingView
            price_info = self.get_current_price(symbol, exchange)
            
            if price_info:
                # Save to Firebase
                if self.save_price_to_firebase(price_info):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            else:
                stats['failed'] += 1
            
            # Add small delay between requests to avoid rate limiting
            if i < stats['total']:
                time.sleep(1)
        
        # Print summary
        print(f"\n📊 Price update completed!")
        print(f"  Total symbols: {stats['total']}")
        print(f"  ✅ Successfully updated: {stats['success']}")
        print(f"  ❌ Failed: {stats['failed']}")
        print(f"  📈 Success rate: {stats['success']/stats['total']*100:.1f}%")
        
        return stats


def main():
    """Main function to run the price update process"""
    print("💰 Stock Price Updater Started")
    print("=" * 50)
    
    try:
        updater = PriceUpdater()
        stats = updater.update_all_prices()
        
        # Exit with error code if no prices were updated successfully
        if stats['success'] == 0 and stats['total'] > 0:
            print("\n❌ No prices were updated successfully!")
            exit(1)
        
        print("\n✅ Price update process completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()


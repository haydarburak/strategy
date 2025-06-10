import os
from datetime import datetime

import getdata
import getdata_stock
import indicator
import creategraphics
import divergence
from candlestick import candlestick
from tqdm import tqdm
from notification import sendtotelegram
from tvDatafeed import Interval
# from binance.client import Client  # Uncomment when needed

# Configuration Constants
class TradingConfig:
    """Trading configuration and constants"""
    EMA_PERIODS = [20, 50, 100, 200]
    EMA_COLUMNS = ['EMA20', 'EMA50', 'EMA100', 'EMA200']
    
    # Candle position indices (negative indexing from current)
    CANDLE_INDICES = {'approve': -1, 'reversal': -2, 'initial': -3}
    
    # Stochastic thresholds
    STOCH_OVERSOLD = 40
    STOCH_OVERBOUGHT = 65
    
    # Analysis periods tracking
    ANALYZED_PERIODS = []
    
    # Market close hours for 4H timeframe
    FOUR_HOUR_CLOSE_HOURS = [1, 5, 9, 18, 21]
    
    # Required technical indicators
    REQUIRED_COLUMNS = ['STOCH_K', 'STOCH_D', 'MACD', 'MACD_S', 'EMA20', 'EMA50', 'EMA100', 'EMA200']
    
    # Market indices mapping
    MARKET_INDICES = {
        'BIST': {'symbol': 'XU100', 'exchange': 'BIST'},
        'XETR': {'symbol': 'DAX', 'exchange': 'XETR'},
        'NYSE': {'symbol': 'SPX', 'exchange': 'SP'},
        'NASDAQ': {'symbol': 'NDX', 'exchange': 'NASDAQ'}
    }
    
    # Portfolio holdings for divergence analysis
    HOLDING_STOCKS = [
        "NASDAQ:GOOGL", "NASDAQ:AMZN", "NASDAQ:NVDA", "NASDAQ:AMD", "NASDAQ:PYPL",
        "NYSE:MMM", "NYSE:BABA", "XETR:BAS", "NYSE:BTI", "NYSE:CVX", "NYSE:ENB",
        "XETR:MBG", "NASDAQ:MSFT", "NYSE:O", "NYSE:TSM", "NYSE:TM", "NYSE:VZ",
        "XETR:VIB3", "XETR:ASME", "NASDAQ:ACLS", "NASDAQ:MTCH", "XETR:PFE",
        "NYSE:SLB", "TRADEGATE:4OQ1", "XETR:BAYN", "NYSE:EL", "NASDAQ:INTC",
        "TRADEGATE:M3P", "NASDAQ:NWL", "NASDAQ:PSEC", "NYSE:VFC", "XETR:VOW3",
        "TRADEGATE:IU2", "NYSE:T", "NYSE:MO", "XETR:ITB", "XETR:UNVB",
        "NYSE:ORCL", "NYSE:SPOT", "NASDAQ:PLTR", "NASDAQ:META", "TRADEGATE:CMC",
        "XETR:DB1"
    ]


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def check_conditions(self, df, ema_length, common_condition, candle_indices, is_long: bool) -> bool:
        """Check if strategy conditions are met"""
        raise NotImplementedError("Subclasses must implement check_conditions")
    
    def get_long_conditions(self, df, ema, indices):
        """Define long entry conditions"""
        raise NotImplementedError("Subclasses must implement get_long_conditions")
    
    def get_short_conditions(self, df, ema, indices):
        """Define short entry conditions"""
        raise NotImplementedError("Subclasses must implement get_short_conditions")


class TwoCandelReversalStrategy(TradingStrategy):
    """İki Mum Çubuklu Dönüş Strategy"""
    
    def __init__(self):
        super().__init__("IKI MUM CUBUKLU DONUS")
    
    def get_long_conditions(self, df, ema, indices):
        return (
            df['Open'].iloc[indices['initial']] > df['Close'].iloc[indices['initial']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Low'].iloc[indices['initial']] > df['Low'].iloc[indices['reversal']]
        )
    
    def get_short_conditions(self, df, ema, indices):
        return (
            df['Close'].iloc[indices['initial']] > df['Open'].iloc[indices['initial']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['High'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['High'].iloc[indices['reversal']] > df['High'].iloc[indices['initial']]
        )
    
    def check_conditions(self, df, ema_length, common_condition, candle_indices, is_long: bool) -> bool:
        """Check if two-candle reversal conditions are met"""
        for ema in ema_length:
            if is_long:
                condition_met = common_condition and self.get_long_conditions(df, ema, candle_indices)
            else:
                condition_met = common_condition and self.get_short_conditions(df, ema, candle_indices)
            
            if condition_met:
                return True
        return False


class DoubleTailPiercingStrategy(TradingStrategy):
    """Çift Kuyruk Delis Strategy"""
    
    def __init__(self):
        super().__init__("CIFT KUYRUK DELIS")
    
    def get_long_conditions(self, df, ema, indices):
        return (
            df['Low'].iloc[indices['initial']] < df['Low'].iloc[indices['reversal']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Low'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] > df[ema].iloc[indices['initial']]
        )
    
    def get_short_conditions(self, df, ema, indices):
        return (
            df['Close'].iloc[indices['initial']] > df['Open'].iloc[indices['initial']] and
            df['Open'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['High'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['High'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']]
        )
    
    def check_conditions(self, df, ema_length, common_condition, candle_indices, is_long: bool) -> bool:
        """Check if double tail piercing conditions are met"""
        for ema in ema_length:
            if is_long:
                condition_met = common_condition and self.get_long_conditions(df, ema, candle_indices)
            else:
                condition_met = common_condition and self.get_short_conditions(df, ema, candle_indices)
            
            if condition_met:
                return True
        return False


class BodyPiercingStrategy(TradingStrategy):
    """Gövde Delis Strategy"""
    
    def __init__(self):
        super().__init__("GOVDE DELIS")
    
    def get_long_conditions(self, df, ema, indices):
        return (
            df['Low'].iloc[indices['initial']] > df['Low'].iloc[indices['reversal']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Low'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] < df[ema].iloc[indices['initial']]
        )
    
    def get_short_conditions(self, df, ema, indices):
        return (
            df['Close'].iloc[indices['initial']] > df['Open'].iloc[indices['initial']] and
            df['Close'].iloc[indices['reversal']] < df['Open'].iloc[indices['reversal']] and
            df['Close'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']]
        )
    
    def check_conditions(self, df, ema_length, common_condition, candle_indices, is_long: bool) -> bool:
        """Check if body piercing conditions are met"""
        for ema in ema_length:
            if is_long:
                condition_met = common_condition and self.get_long_conditions(df, ema, candle_indices)
            else:
                condition_met = common_condition and self.get_short_conditions(df, ema, candle_indices)
            
            if condition_met:
                return True
        return False


class SingleCandleReversalStrategy(TradingStrategy):
    """Tek Mum Dönüş Strategy"""
    
    def __init__(self):
        super().__init__("SINGLE CANDLE REVERSAL")
    
    def get_long_conditions(self, df, ema, indices):
        return (
            df['Low'].iloc[indices['initial']] > df['Low'].iloc[indices['reversal']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['approve']] > df['Close'].iloc[indices['reversal']] and
            df['HAMMER'].iloc[indices['reversal']]
        )
    
    def get_short_conditions(self, df, ema, indices):
        return (
            (df['INVERTED_HAMMER'].iloc[indices['reversal']] or df['GRAVESTONE_DOJI'].iloc[indices['reversal']]) and
            df['High'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']]
        )
    
    def check_conditions(self, df, ema_length, common_condition, candle_indices, is_long: bool) -> bool:
        """Check if single candle reversal conditions are met"""
        for ema in ema_length:
            if is_long:
                condition_met = common_condition and self.get_long_conditions(df, ema, candle_indices)
            else:
                condition_met = common_condition and self.get_short_conditions(df, ema, candle_indices)
            
            if condition_met:
                return True
        return False

def get_crypto_symbols(client, exclude_keywords=None, base_currency='USDT'):
    exclude_keywords = exclude_keywords or ['UP', 'DOWN', 'BEAR', 'BULL']
    info = client.get_exchange_info()
    symbols = [x['symbol'] for x in info['symbols']]
    return [
        symbol for symbol in symbols
        if all(exclude not in symbol for exclude in exclude_keywords) and symbol.endswith(base_currency)
    ]

def get_symbols(type_, target_symbols=None):
#    if type_ == 'crypto':
#        if not client:
#            raise ValueError("Client is required for fetching crypto symbols.")
#        return get_crypto_symbols(client)

    if type_ == 'stock':
        return getdata_stock.get_stock_symbols(target_symbols)

    raise ValueError(f"Unsupported symbol type: {type_}")


def evaluate_conditions_new(ema_length, df, conditions, candle_indices):
    for ema in ema_length:
        if all(condition(df, ema, candle_indices) for condition in conditions):
            return True
    return False

def evaluate_conditions(condition_name, symbol, interval, ema_length, df, conditions, candle_indices, long=True):
    for ema in ema_length:
        if all(condition(df, ema, candle_indices) for condition in conditions):
            alarm_name = f"{'LONG' if long else 'SHORT'} {condition_name}"
            print(f'Condition Triggered: {alarm_name}')
            if df.get('symbol') is not None:
                exchange_and_symbol = df['symbol'].iloc[0]
            else:
                exchange_and_symbol = symbol

            try:
                fig = creategraphics.create_graphics(df, long)
                sendtotelegram.send_telegram(
                    exchange_and_symbol,
                    alarm_name,
                    df['Close'].iloc[candle_indices['initial']],
                    df['Close'].iloc[candle_indices['reversal']],
                    df['Close'].iloc[candle_indices['approve']],
                    interval,
                    fig
                )
            except KeyError as e:
                print(f"KeyError: {e} - One of the keys in 'candle_indices' is missing or incorrect.")
            except IndexError as e:
                print(f"IndexError: {e} - One of the indices is out of bounds.")
            except TypeError as e:
                print(f"TypeError: {e} - Invalid type for index. Check 'candle_indices' values.")
            except Exception as e:
                print(f"Unexpected error: {e}")
            break

def check_conditions_and_send_alarm(
    interval, symbol, ema_length, common_condition, df, reversal_candle_index,
    initial_candle_index, approve_candle_index, conditions, alarm_name, long
):
    for ema in ema_length:
        if all(condition(df, ema, reversal_candle_index, initial_candle_index, approve_candle_index) for condition in conditions):
            print(f'Condition Met: {alarm_name}')
            print(symbol)

            if isinstance(approve_candle_index, int):
                candle_indices = {
                    'initial': initial_candle_index,
                    'reversal': reversal_candle_index,
                    'approve': approve_candle_index
                }

                fig = creategraphics.create_graphics(df, long)

                sendtotelegram.send_telegram(
                    symbol,
                    alarm_name,
                    df['Close'].iloc[candle_indices['initial']],
                    df['Close'].iloc[candle_indices['reversal']],
                    df['Close'].iloc[candle_indices['approve']],
                    interval,
                    fig
                )

            else:
                print("approve_candle_index is NOT an integer")
            break

def iki_mum_cubuklu_donus(ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):
    """
    İki mum çubuklu dönüş alarmı - Using new strategy class.
    """
    strategy = TwoCandelReversalStrategy()
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    return strategy.check_conditions(df, ema_length, common_condition, candle_indices, long)

def cift_kuyruk_delis(interval, symbol, ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):
    """
    Çift kuyruk dönüş alarmı - Using new strategy class.
    """
    strategy = DoubleTailPiercingStrategy()
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    
    if strategy.check_conditions(df, ema_length, common_condition, candle_indices, long):
        conditions = [strategy.get_long_conditions if long else strategy.get_short_conditions]
        evaluate_conditions(strategy.name, symbol, interval, ema_length, df, conditions, candle_indices, long)

def govde_delis(interval, symbol, ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):
    """
    Gövde delis alarmı - Using new strategy class.
    """
    strategy = BodyPiercingStrategy()
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    
    if strategy.check_conditions(df, ema_length, common_condition, candle_indices, long):
        conditions = [strategy.get_long_conditions if long else strategy.get_short_conditions]
        evaluate_conditions(strategy.name, symbol, interval, ema_length, df, conditions, candle_indices, long)

def bir_mum_cubunundan_olusan_donus_noktalari(interval, symbol, ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):
    """
    Tek mum dönüş noktaları alarmı - Using new strategy class.
    """
    strategy = SingleCandleReversalStrategy()
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    
    if strategy.check_conditions(df, ema_length, common_condition, candle_indices, long):
        conditions = [strategy.get_long_conditions if long else strategy.get_short_conditions]
        evaluate_conditions(strategy.name, symbol, interval, ema_length, df, conditions, candle_indices, long)

def analsys(type, interval, kline_interval, interval_str, lookback, relevant):
    if type == 'crypto' and isinstance(relevant, list):
        for r in tqdm(relevant):
            df = getdata.get_data_frame(r, kline_interval, lookback, interval_str)
            do_analysis(r, df, interval, None)

    if type == 'stock' and isinstance(relevant, dict):
        rsi_divergence_message = ""
        message = ""

        for exchange, symbols in relevant.items():
            # Get index info from configuration
            if exchange in TradingConfig.MARKET_INDICES:
                index_info = TradingConfig.MARKET_INDICES[exchange]
                index_symbol = index_info['symbol']
                index_exchange = index_info['exchange']
            else:
                print(f"Exchange {exchange} not found in MARKET_INDICES configuration. Skipping.")
                continue

            index_df = getdata_stock.get_data_frame(index_symbol, index_exchange, kline_interval, lookback)
            index_df = add_indicators(index_df)
            approve_idx, reversal_idx, initial_idx = TradingConfig.CANDLE_INDICES.values()
            long_ema_check = all(index_df[TradingConfig.EMA_COLUMNS[i]].iloc[approve_idx] > index_df[TradingConfig.EMA_COLUMNS[i + 1]].iloc[approve_idx]
                            for i in range(len(TradingConfig.EMA_COLUMNS) - 1))
            short_ema_check = all(index_df[TradingConfig.EMA_COLUMNS[i]].iloc[approve_idx] < index_df[TradingConfig.EMA_COLUMNS[i + 1]].iloc[approve_idx]
                            for i in range(len(TradingConfig.EMA_COLUMNS) - 1))

            if long_ema_check:
                index_long = True
            elif short_ema_check:
                index_long = False
            else:
                index_long = None
            if index_long != None:
                print('Index: ' + index_symbol + ' Index Exchange: ' + index_exchange + ' is started')

                for symbol in tqdm(symbols):
                    df = getdata_stock.get_data_frame(symbol, exchange, kline_interval, lookback)
                    do_analysis(symbol, df, interval, index_long)
            else:
                print('Index: ' + index_symbol + ' Index Exchange: ' + index_exchange + ' is neither Long nor Short. Skipped')
                sendtotelegram.send_message_telegram(
                    index_symbol + '-' + index_exchange,
                    'INDEX SKIPPED'
                )

        # Use portfolio holdings from configuration
        holding_stocks = TradingConfig.HOLDING_STOCKS

        print("Divergence analsys is starting")
        for symbol in tqdm(holding_stocks):
            print("SYMBOL: " + symbol)
            df = getdata_stock.get_data_frame(symbol, exchange, kline_interval, lookback)

            df = df.dropna()
            df = add_indicators(df)
            if not validate_columns(df):
                return

            df = add_candlestick_patterns(df)

            if df is not None:
                exchange_and_symbol = df.get('symbol', [symbol]).iloc[0] if 'symbol' in df else symbol

                df = divergence.find_rsi_divergence(df)
                if df.iloc[-1]['Bearish_Divergence'] > 0:
                    message += f"SYMBOL: {df.iloc[-1]['symbol']}\nBearish Divergence\nLink: https://www.tradingview.com/chart/?symbol={df.iloc[-1]['symbol']}&interval={interval}"
                if df.iloc[-1]['Bullish_Divergence'] > 0:
                    message += f"SYMBOL: {df.iloc[-1]['symbol']}\nBullish Divergence\nLink: https://www.tradingview.com/chart/?symbol={df.iloc[-1]['symbol']}&interval={interval}"

                if message:
                    rsi_divergence_message += f"\n{message}"

                message = ""

        if rsi_divergence_message:
            sendtotelegram.send_message_telegram(rsi_divergence_message, 'RSI DIVERGENCE FOR STOCKS')

    print('finished')

def do_analysis(symbol, df, interval, index_long):
    if df.empty:
        print("DataFrame is empty. Exiting analysis.")
        return

    exchange_and_symbol = df.get('symbol', [symbol]).iloc[0] if 'symbol' in df else symbol

    df = df.dropna()
    df = add_indicators(df)
    if not validate_columns(df):
        return

    df = add_candlestick_patterns(df)

    long_condition, short_condition = get_trade_conditions(df, TradingConfig.CANDLE_INDICES, TradingConfig.EMA_COLUMNS)

    if long_condition and index_long:
        execute_positions(df, interval, symbol, TradingConfig.EMA_COLUMNS, TradingConfig.CANDLE_INDICES, is_long=True)
    elif short_condition and not index_long:
        execute_positions(df, interval, symbol, TradingConfig.EMA_COLUMNS, TradingConfig.CANDLE_INDICES, is_long=False)

    return df

def add_indicators(df):
    indicators_to_add = [
        (indicator.IndicatorEnum.EMA, 20),
        (indicator.IndicatorEnum.EMA, 50),
        (indicator.IndicatorEnum.EMA, 100),
        (indicator.IndicatorEnum.EMA, 200),
        (indicator.IndicatorEnum.STOCH, 5),
        (indicator.IndicatorEnum.MACD, 50),
        (indicator.IndicatorEnum.RSI, 14)
    ]
    for ind, param in indicators_to_add:
        df = indicator.add_indicator(df, ind, param)
    return df

def validate_columns(df):
    required_columns = TradingConfig.REQUIRED_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False
    return True

def add_candlestick_patterns(df):
    ohlc = ['Open', 'High', 'Low', 'Close']
    patterns = {
        'HAMMER': candlestick.hammer,
        'INVERTED_HAMMER': candlestick.inverted_hammer,
        'GRAVESTONE_DOJI': candlestick.gravestone_doji
    }
    for target, pattern_func in patterns.items():
        if target in df.columns:
            df.drop(columns=[target], inplace=True)
        df = pattern_func(df, ohlc=ohlc, target=target)
    return df

def get_trade_conditions(df, indices, ema_columns):
    approve_idx, reversal_idx, initial_idx = indices.values()
    long_ema_check = all(df[TradingConfig.EMA_COLUMNS[i]].iloc[approve_idx] > df[TradingConfig.EMA_COLUMNS[i + 1]].iloc[approve_idx]
                    for i in range(len(TradingConfig.EMA_COLUMNS) - 1))
    short_ema_check = all(df[TradingConfig.EMA_COLUMNS[i]].iloc[approve_idx] < df[TradingConfig.EMA_COLUMNS[i + 1]].iloc[approve_idx]
                    for i in range(len(TradingConfig.EMA_COLUMNS) - 1))

    long_condition = (
        (df['STOCH_K'].iloc[approve_idx] < TradingConfig.STOCH_OVERSOLD or df['STOCH_D'].iloc[approve_idx] < TradingConfig.STOCH_OVERSOLD) and
        df['STOCH_K'].iloc[approve_idx] > df['STOCH_D'].iloc[approve_idx] and
        any(df['MACD'].iloc[idx] > df['MACD_S'].iloc[idx] for idx in [approve_idx, reversal_idx, initial_idx]) and
        long_ema_check and
        df['Close'].iloc[initial_idx] < df['Open'].iloc[initial_idx] and
        df['Low'].iloc[approve_idx] > df['Low'].iloc[reversal_idx] and
        df['Close'].iloc[approve_idx] > df['High'].iloc[reversal_idx]
    )

    short_condition = (
        (df['STOCH_K'].iloc[approve_idx] > TradingConfig.STOCH_OVERBOUGHT or df['STOCH_D'].iloc[approve_idx] > TradingConfig.STOCH_OVERBOUGHT) and
        any(df['MACD'].iloc[idx] < df['MACD_S'].iloc[idx] for idx in [approve_idx, reversal_idx, initial_idx]) and
        short_ema_check and
        df['Low'].iloc[reversal_idx] > df['Close'].iloc[approve_idx] and
        df['High'].iloc[reversal_idx] > df['High'].iloc[approve_idx]
    )

    return long_condition, short_condition

def execute_positions(df, interval, symbol, ema_columns, indices, is_long):
    reversal_idx, initial_idx, approve_idx = indices['reversal'], indices['initial'], indices['approve']
    common_condition = (
        df['Low'].iloc[reversal_idx] > df['Close'].iloc[approve_idx] if not is_long else
        df['Low'].iloc[approve_idx] > df['Low'].iloc[reversal_idx]
    )

    candle_indices = {
        'initial': initial_idx,
        'reversal': reversal_idx,
        'approve': approve_idx
    }

    strategy = TwoCandelReversalStrategy()
    if strategy.check_conditions(df, ema_columns, common_condition, candle_indices, is_long):
        sendtotelegram.send_trade_signal(df, symbol, interval, candle_indices, is_long=is_long, strategy_name=strategy.name)

    cift_kuyruk_delis(interval, symbol, ema_columns, common_condition, df, reversal_idx, initial_idx, approve_idx, is_long)
    govde_delis(interval, symbol, ema_columns, common_condition, df, reversal_idx, initial_idx, approve_idx, is_long)
    bir_mum_cubunundan_olusan_donus_noktalari(interval, symbol, ema_columns, common_condition, df, reversal_idx, initial_idx, approve_idx, is_long)

def is_day_closed():
    return datetime.now().hour == 1

def is_four_hour_closed():
    return datetime.now().hour in TradingConfig.FOUR_HOUR_CLOSE_HOURS

# Use configuration for analyzed periods tracking
# Note: This should be handled by TradingConfig.ANALYZED_PERIODS in the future

def run_analysis(type):
    print(f"Starting analysis for {type}")
    fifteen_minute_analsys_key = '15M' + str(datetime.now().day) + type
    one_day_analsys_key = '1D' + str(datetime.now().day) + type
    one_hour_analsys_key = '1H' + str(datetime.now().hour) + str(datetime.now().day)
    four_hour_analsys_key = '4H' + str(datetime.now().hour) + str(datetime.now().day) + type

    if is_day_closed() or is_four_hour_closed() or not one_hour_analsys_key in TradingConfig.ANALYZED_PERIODS:
        symbollist = get_symbols(type)
        klines = {}

        if 1==1 or (not one_day_analsys_key in TradingConfig.ANALYZED_PERIODS and is_day_closed()):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {type} - {one_day_analsys_key} Started")

            if type == 'stock':
                analsys(type, '1D', Interval.in_daily, 'day', 500, symbollist)
#            elif type == 'crypto':
#                analsys(type, '1D', Client.KLINE_INTERVAL_1DAY, 'day', '500', symbollist)

            TradingConfig.ANALYZED_PERIODS.append(one_day_analsys_key)

        if not one_hour_analsys_key in TradingConfig.ANALYZED_PERIODS:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {type} - {one_hour_analsys_key} Started")

#            if type == 'crypto':
#                analsys(type, '1H', Client.KLINE_INTERVAL_1HOUR, 'hour', '500', symbollist)
#            elif type == 'stock':
#                analsys(type, '1H', Interval.in_1_hour, 'hour', 500, symbollist)

            TradingConfig.ANALYZED_PERIODS.append(one_hour_analsys_key)

        if not four_hour_analsys_key in TradingConfig.ANALYZED_PERIODS and is_four_hour_closed():
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {type} - {four_hour_analsys_key} Started")

#            if type == 'crypto':
#                analsys(type, '4H', Client.KLINE_INTERVAL_4HOUR, 'hour', '500', symbollist)
#            elif type == 'stock':
#                analsys(type, '4H', Interval.in_4_hour, 'hour', 500, symbollist)

            TradingConfig.ANALYZED_PERIODS.append(four_hour_analsys_key)

if __name__ == "__main__":
    for type in ['stock']:
        run_analysis(type)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finished")

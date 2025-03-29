import os

import getdata
import getdata_stock
import indicator
import creategraphics
import divergence
#from binance.client import Client
from candlestick import candlestick
from tqdm import tqdm
from notification import sendtotelegram
from datetime import datetime
from tvDatafeed import Interval

#client = Client()

ema_columns = ['EMA20', 'EMA50', 'EMA100', 'EMA200']
CANDLE_INDICES = {'approve': -1, 'reversal': -2, 'initial': -3}

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

def iki_mum_cubuklu_donus(interval, symbol, ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):
    """
    İki mum çubuklu dönüş alarmı.
    """
    def long_conditions(df, ema, indices):
        return (
            common_condition and
            df['Open'].iloc[indices['initial']] > df['Close'].iloc[indices['initial']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Low'].iloc[indices['initial']] > df['Low'].iloc[indices['reversal']]
        )

    def short_conditions(df, ema, indices):
        return (
            common_condition and
            df['Close'].iloc[indices['initial']] > df['Open'].iloc[indices['initial']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['High'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['High'].iloc[indices['reversal']] > df['High'].iloc[indices['initial']]
        )

    conditions = [long_conditions if long else short_conditions]
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    evaluate_conditions("IKI MUM CUBUKLU DONUS", symbol, interval, ema_length, df, conditions, candle_indices, long)

def cift_kuyruk_delis(interval, symbol, ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):
    """
    Çift kuyruk dönüş alarmı.
    """
    def long_conditions(df, ema, indices):
        return (
            common_condition and
            df['Low'].iloc[indices['initial']] < df['Low'].iloc[indices['reversal']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Low'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] > df[ema].iloc[indices['initial']]
        )

    def short_conditions(df, ema, indices):
        return (
            common_condition and
            df['Close'].iloc[indices['initial']] > df['Open'].iloc[indices['initial']] and
            df['Open'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['High'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['High'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']]
        )

    conditions = [long_conditions if long else short_conditions]
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    evaluate_conditions("CIFT KUYRUK DELIS", symbol, interval, ema_length, df, conditions, candle_indices, long)

def govde_delis(interval, symbol, ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }

    def long_conditions(df, ema, indices):
        return (common_condition and
            df['Low'].iloc[indices['initial']] > df['Low'].iloc[indices['reversal']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Low'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] < df[ema].iloc[indices['initial']]
        )

        alarm_name = "LONG GOVDE DELIS"

    def short_conditions(df, ema, indices):
        return (common_condition and
            df['Close'].iloc[indices['initial']] > df['Open'].iloc[indices['initial']] and
            df['Close'].iloc[indices['reversal']] < df['Open'].iloc[indices['reversal']] and
            df['Close'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['initial']] < df[ema].iloc[indices['initial']] and
            df['Close'].iloc[indices['initial']] > df[ema].iloc[indices['initial']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']]
        )
        alarm_name = "SHORT GOVDE DELIS"

    conditions = [long_conditions if long else short_conditions]
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    evaluate_conditions("GOVDE DELIS", symbol, interval, ema_length, df, conditions, candle_indices, long)

def bir_mum_cubunundan_olusan_donus_noktalari(interval, symbol, ema_length, common_condition, df, reversal_candle_index, initial_candle_index, approve_candle_index, long=True):

    def long_conditions(df, ema, indices):
        return (common_condition and
            df['Low'].iloc[indices['initial']] > df['Low'].iloc[indices['reversal']] and
            df['Low'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['approve']] > df['Close'].iloc[indices['reversal']] and
            df['HAMMER'].iloc[indices['reversal']]
        )

    def short_conditions(df, ema, indices):
        return (common_condition and
            (df['INVERTED_HAMMER'].iloc[indices['reversal']] or df['GRAVESTONE_DOJI'].iloc[indices['reversal']]) and
            df['High'].iloc[indices['reversal']] > df[ema].iloc[indices['reversal']] and
            df['Open'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']] and
            df['Close'].iloc[indices['reversal']] < df[ema].iloc[indices['reversal']]
        )

    conditions = [long_conditions if long else short_conditions]
    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }
    evaluate_conditions("GOVDE DELIS", symbol, interval, ema_length, df, conditions, candle_indices, long)

def analsys(type, interval, kline_interval, interval_str, lookback, relevant):
    if type == 'crypto' and isinstance(relevant, list):
        for r in tqdm(relevant):
            df = getdata.get_data_frame(r, kline_interval, lookback, interval_str)
            do_analysis(r, df, interval, None)

    if type == 'stock' and isinstance(relevant, dict):
        rsi_divergence_message = ""

        for exchange, symbols in relevant.items():

            if exchange == 'BIST':
                index_symbol = 'XU100'
                index_exchange = 'BIST'
            elif exchange == 'XETR':
                index_symbol = 'DAX'
                index_exchange = 'XETR'
            elif exchange == 'NYSE':
                index_symbol = 'SPX'
                index_exchange = 'SP'
            elif exchange == 'NASDAQ':
                index_symbol = 'NDX'
                index_exchange = 'NASDAQ'

            index_df = getdata_stock.get_data_frame(index_symbol, index_exchange, kline_interval, lookback)
            index_df = add_indicators(index_df)
            approve_idx, reversal_idx, initial_idx = CANDLE_INDICES.values()
            long_ema_check = all(index_df[ema_columns[i]].iloc[approve_idx] > index_df[ema_columns[i + 1]].iloc[approve_idx]
                            for i in range(len(ema_columns) - 1))
            short_ema_check = all(index_df[ema_columns[i]].iloc[approve_idx] < index_df[ema_columns[i + 1]].iloc[approve_idx]
                            for i in range(len(ema_columns) - 1))

            if long_ema_check:
                index_long = True
            elif short_ema_check:
                index_long = False
            else:
                index_long = None
            if index_long != None:
                for symbol in tqdm(symbols):
                    print("Stock: " + symbol + " Exchange:" + exchange)
                    df = getdata_stock.get_data_frame(symbol, exchange, kline_interval, lookback)
                    message = do_analysis(symbol, df, interval, index_long)
                    if message:
                        rsi_divergence_message += f"{rsi_divergence_message}\n{message}"
            else:
                print('Index: '+ index_symbol + ' Index Exchange: ' + index_exchange + ' is neither Long nor Short. Skipped')
                sendtotelegram.send_message_telegram(
                    index_symbol + '-' + index_exchange,
                    'INDEX SKIPPED'
                )

    if rsi_divergence_message:
        sendtotelegram.send_message_telegram(rsi_divergence_message, 'RSI DIVERGENCE FOR STOCKS')

    print('finished')

def do_analysis(symbol, df, interval, index_long):
    rsi_divergence_message = ""

    if df.empty:
        print("DataFrame is empty. Exiting analysis.")
        return

    exchange_and_symbol = df.get('symbol', [symbol]).iloc[0] if 'symbol' in df else symbol

    df = df.dropna()
    df = add_indicators(df)
    if not validate_columns(df):
        return

    df = add_candlestick_patterns(df)

    HOLDING_STOCKS = os.getenv("HOLDING_STOCKS", "")
    stock_list = HOLDING_STOCKS.split(",") if HOLDING_STOCKS else []

    if exchange_and_symbol in stock_list:
        df = divergence.find_rsi_divergence(df)
        if df.iloc[-1]['Bearish_Divergence'] > 0:
            rsi_divergence_message += f"SYMBOL: {df.iloc[-1]['symbol']}\nBearish Divergence\nLink: https://www.tradingview.com/chart/?symbol={df.iloc[-1]['symbol']}&interval={interval}"
        if df.iloc[-1]['Bullish_Divergence'] > 0:
            rsi_divergence_message += f"SYMBOL: {df.iloc[-1]['symbol']}\nBullish Divergence\nLink: https://www.tradingview.com/chart/?symbol={df.iloc[-1]['symbol']}&interval={interval}"

    long_condition, short_condition = get_trade_conditions(df, CANDLE_INDICES, ema_columns)

    if long_condition and index_long:
        execute_long_positions(df, interval, symbol, ema_columns, CANDLE_INDICES)
    elif short_condition and not index_long:
        execute_short_positions(df, interval, symbol, ema_columns, CANDLE_INDICES)

    return rsi_divergence_message

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
    required_columns = ['STOCH_K', 'STOCH_D', 'MACD', 'MACD_S', 'EMA20', 'EMA50', 'EMA100', 'EMA200']
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
        df = pattern_func(df, ohlc=ohlc, target=target)
    return df

def get_trade_conditions(df, indices, ema_columns):
    approve_idx, reversal_idx, initial_idx = indices.values()
    long_ema_check = all(df[ema_columns[i]].iloc[approve_idx] > df[ema_columns[i + 1]].iloc[approve_idx]
                    for i in range(len(ema_columns) - 1))
    short_ema_check = all(df[ema_columns[i]].iloc[approve_idx] < df[ema_columns[i + 1]].iloc[approve_idx]
                    for i in range(len(ema_columns) - 1))

    long_condition = (
        (df['STOCH_K'].iloc[approve_idx] < 40 or df['STOCH_D'].iloc[approve_idx] < 40) and
        df['STOCH_K'].iloc[approve_idx] > df['STOCH_D'].iloc[approve_idx] and
        any(df['MACD'].iloc[idx] > df['MACD_S'].iloc[idx] for idx in [approve_idx, reversal_idx, initial_idx]) and
        long_ema_check and
        df['Close'].iloc[initial_idx] < df['Open'].iloc[initial_idx] and
        df['Low'].iloc[approve_idx] > df['Low'].iloc[reversal_idx] and
        df['Close'].iloc[approve_idx] > df['High'].iloc[reversal_idx]
    )

    short_condition = (
        (df['STOCH_K'].iloc[approve_idx] > 65 or df['STOCH_D'].iloc[approve_idx] > 65) and
        any(df['MACD'].iloc[idx] < df['MACD_S'].iloc[idx] for idx in [approve_idx, reversal_idx, initial_idx]) and
        short_ema_check and
        df['Low'].iloc[reversal_idx] > df['Close'].iloc[approve_idx] and
        df['High'].iloc[reversal_idx] > df['High'].iloc[approve_idx]
    )

    return long_condition, short_condition


def execute_long_positions(df, interval, symbol, ema_columns, indices):
    execute_positions(df, interval, symbol, ema_columns, indices, is_long=True)

def execute_short_positions(df, interval, symbol, ema_columns, indices):
    execute_positions(df, interval, symbol, ema_columns, indices, is_long=False)

def execute_positions(df, interval, symbol, ema_columns, indices, is_long):
    reversal_idx, initial_idx, approve_idx = indices['reversal'], indices['initial'], indices['approve']
    common_condition = (
        df['Low'].iloc[reversal_idx] > df['Close'].iloc[approve_idx] if not is_long else
        df['Low'].iloc[approve_idx] > df['Low'].iloc[reversal_idx]
    )

    iki_mum_cubuklu_donus(interval, symbol, ema_columns, common_condition, df, reversal_idx, initial_idx, approve_idx, is_long)
    cift_kuyruk_delis(interval, symbol, ema_columns, common_condition, df, reversal_idx, initial_idx, approve_idx)
    govde_delis(interval, symbol, ema_columns, common_condition, df, reversal_idx, initial_idx, approve_idx)
    bir_mum_cubunundan_olusan_donus_noktalari(interval, symbol, ema_columns, common_condition, df, reversal_idx, initial_idx, approve_idx, is_long)

def is_day_closed():
    return datetime.now().hour == 1

def is_four_hour_closed():
    four_hour_close_hours = [1, 5, 9, 18, 21]
    return datetime.now().hour in four_hour_close_hours

analsysed_periods = []

def run_analysis(type):
    print(f"Starting analysis for {type}")
    fifteen_minute_analsys_key = '15M' + str(datetime.now().day) + type
    one_day_analsys_key = '1D' + str(datetime.now().day) + type
    one_hour_analsys_key = '1H' + str(datetime.now().hour) + str(datetime.now().day)
    four_hour_analsys_key = '4H' + str(datetime.now().hour) + str(datetime.now().day) + type

    if is_day_closed() or is_four_hour_closed() or not one_hour_analsys_key in analsysed_periods:
        symbollist = get_symbols(type)
        klines = {}

        if 1==1 or (not one_day_analsys_key in analsysed_periods and is_day_closed()):
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {type} - {one_day_analsys_key} Started")

            if type == 'stock':
                analsys(type, '1D', Interval.in_daily, 'day', 500   , symbollist)
#            elif type == 'crypto':
#                analsys(type, '1D', Client.KLINE_INTERVAL_1DAY, 'day', '500', symbollist)

            analsysed_periods.append(one_day_analsys_key)

        if not one_hour_analsys_key in analsysed_periods:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {type} - {one_hour_analsys_key} Started")

#            if type == 'crypto':
#                analsys(type, '1H', Client.KLINE_INTERVAL_1HOUR, 'hour', '500', symbollist)
#            elif type == 'stock':
#                analsys(type, '1H', Interval.in_1_hour, 'hour', 500, symbollist)

            analsysed_periods.append(one_hour_analsys_key)

        if not four_hour_analsys_key in analsysed_periods and is_four_hour_closed():
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {type} - {four_hour_analsys_key} Started")

#            if type == 'crypto':
#                analsys(type, '4H', Client.KLINE_INTERVAL_4HOUR, 'hour', '500', symbollist)
#            elif type == 'stock':
#                analsys(type, '4H', Interval.in_4_hour, 'hour', 500, symbollist)

            analsysed_periods.append(four_hour_analsys_key)

if __name__ == "__main__":
    for type in ['stock']:
        run_analysis(type)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finished")

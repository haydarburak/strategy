import time
import pandas as pd
from tvDatafeed import TvDatafeed
from websocket import WebSocketTimeoutException
from tradingview_screener import Query, Column

tv = TvDatafeed()

def get_sp500_symbols():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)  # Sayfadaki tüm tabloları alır
    sp500_table = tables[0]  # İlk tablo genellikle S&P 500 şirketleridir
    return sp500_table['Symbol'].tolist()  # Sembolleri çek

def get_stock_symbols(target_symbols=None):
    target_symbols = target_symbols or set()

    #q = Query().limit(50000).select() \
    #.where(Column('type') == 'stock', Column('exchange').isin(['BIST'])) \
    #.set_markets('turkey')

    #q = Query().limit(50000).select() \
    #.where(Column('type') == 'stock', Column('exchange').isin(['XETR'])) \
    #.set_markets('germany')

    #q = Query().limit(50000).select() \
    #.where(Column('type') == 'stock', Column('exchange').isin(['NYSE', 'NASDAQ'])) \
    #.set_markets('america')

    q = Query().limit(50000).select() \
    .where(Column('type') == 'stock', Column('exchange').isin(['BIST', 'XETR', 'NYSE', 'NASDAQ'])) \
    .set_markets('america', 'germany', 'turkey')

    sp500_symbols = get_sp500_symbols()

    n_rows, symbols = q.get_scanner_data()

    def filter_sp500(row):
        # Ticker'i parçala ve borsa ile sembol bilgilerini al
        exchange, symbol = row['ticker'].split(':')
        # Eğer NYSE veya NASDAQ ise ve sembol S&P 500'de varsa, True döndür
        if exchange in ['NYSE', 'NASDAQ'] and symbol in sp500_symbols:
            return True
        # Diğer borsalar (ör. BIST, XETR) için olduğu gibi bırak
        elif exchange not in ['NYSE', 'NASDAQ']:
            return True
        return False

    filtered_symbols = symbols[symbols.apply(filter_sp500, axis=1)]

    #        symbols = get_all_symbols(market=market)

    filtered_symbols['category'] = filtered_symbols['ticker'].str.split(':').str[0]
    filtered_symbols['symbol_clean'] = filtered_symbols['ticker'].str.split(':').str[1]

    return filtered_symbols.groupby('category')['symbol_clean'].apply(list).to_dict()


def get_data_frame(symbol, exchange, interval, lookback, max_retries=20, backoff_factor=2):
    """
    Fetch historical data for a given symbol, with retry logic for handling timeouts.

    :param symbol: The stock/asset symbol to fetch data for.
    :param exchange: The exchange where the symbol is listed.
    :param interval: The time interval for the data (e.g., '1D').
    :param lookback: Number of bars to look back.
    :param max_retries: Maximum number of retries for fetching data.
    :param backoff_factor: Time (in seconds) to wait before retrying.
    :return: A DataFrame with historical data or an empty DataFrame on failure.
    """
    def fetch_data():
        """Helper function to fetch data from the tvDatafeed API."""
        return tv.get_hist(symbol, exchange, n_bars=lookback)

    for attempt in range(1, max_retries + 1):
        try:
            # Attempt to fetch historical data
            historical_data = fetch_data()

            if historical_data is not None:
                # Process and return the DataFrame if successful
                historical_data = historical_data.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                return clean_dataframe(historical_data)
            else:
                print(f"Attempt {attempt}: No data returned for {symbol} on {exchange}. Retrying...")

        except (TimeoutError, WebSocketTimeoutException):
            print(f"Attempt {attempt}: Timeout occurred for {symbol}. Retrying in {backoff_factor} seconds...")
            time.sleep(backoff_factor)

    # Log final failure
    print(f"Failed to retrieve data for {symbol} on {exchange} after {max_retries} attempts.")

    # Return an empty DataFrame with the expected structure
    return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


def clean_dataframe(df):
    df.dropna(inplace=True)
    df = df[df['High'] != df['Low']]
    return df

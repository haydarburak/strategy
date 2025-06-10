import backtrader as bt
import pandas as pd

from tqdm import tqdm

import indicator
import main
import getdata_stock

all_profit_losses = []

class PandasData(bt.feeds.PandasData):
    # Eğer ekstra kolonlar varsa, Backtrader'a bildirmelisin.
    # Örneğin EMA20 gibi kolonları ekle:
    lines = ('ema20', 'ema50', 'ema100', 'ema200', 'stoch_k', 'stoch_d', 'macd', 'macd_s', 'rsi14')

    # kolon isimlerini dataframe’deki isimlerle eşle
    params = (
        ('ema20', 'EMA20'),
        ('ema50', 'EMA50'),
        ('ema100', 'EMA100'),
        ('ema200', 'EMA200'),
        ('stoch_k', 'STOCH_K'),
        ('stoch_d', 'STOCH_D'),
        ('macd', 'MACD'),
        ('macd_s', 'MACD_S'),
        ('rsi14', 'RSI14'),
    )

class IkiMumCubukluDonusStrategy(bt.Strategy):
    params = (
        ('ema_period', 21),
    )

    def __init__(self):
        self.stockdata = self.datas[0]
        self.stockindex = self.datas[1]
        self.entry_price = None  # Pozisyona giriş fiyatını tutmak için
        self.stop_loss_price = None
        self.total_profit = 0.0
        self.total_profit_loss = 0.0

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"AL emri gerçekleşti: {self.stockdata.datetime.date(0)} - Fiyat: {order.executed.price:.2f} - Miktar: {order.executed.size}")
                self.entry_price = order.executed.price
            elif order.issell():
                print(f"SAT emri gerçekleşti: {self.stockdata.datetime.date(0)} - Fiyat: {order.executed.price:.2f} - Miktar: {order.executed.size}")
                if self.entry_price is not None:
                    profit = order.executed.price - self.entry_price
                    self.total_profit += profit
                    print(f"İşlem kapandı: Brüt Kar/Zarar: {profit:.2f}, Toplam Kar/Zarar: {self.total_profit:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"Emir iptal edildi veya reddedildi: {order.Status[order.status]}")

    def stop(self):
        print(f"\nBaşlangıç Bakiyesi: {self.broker.startingcash}")
        print(f"Bitiş Bakiyesi: {self.broker.getvalue()}")
        print(f"Toplam Kâr/Zarar: {self.broker.getvalue() - self.broker.startingcash}")
        self.total_profit_loss += self.broker.getvalue() - self.broker.startingcash
        all_profit_losses.append(self.total_profit_loss)
        print(f"Toplam Genel Kâr/Zarar: {sum(all_profit_losses)}")

    def next(self):
        prev_low = self.stockdata.low[-1]
        prev_close = self.stockdata.close[-1]
        current_close = self.stockdata.close[0]

        # En az 3 mum olmalı
        if len(self.stockdata) < 3:
            return

        # Kapanmış 3 mumun indeksleri (bugün=0, dün=-1, evvelsi gün=-2)
        approve_idx = -1
        reversal_idx = -2
        initial_idx = -3
        # EMA sıralama kontrolü
        long_ema_check = (self.stockdata.ema20[approve_idx] > self.stockdata.ema50[approve_idx] > self.stockdata.ema100[approve_idx] > self.stockdata.ema200[approve_idx])
        long_ema_check_index = (self.stockindex.ema20[approve_idx] > self.stockindex.ema50[approve_idx] > self.stockindex.ema100[approve_idx] > self.stockindex.ema200[approve_idx])
        short_ema_check = (self.stockdata.ema20[approve_idx] < self.stockdata.ema50[approve_idx] < self.stockdata.ema100[approve_idx] < self.stockdata.ema200[approve_idx])
        short_ema_check_index = (self.stockindex.ema20[approve_idx] < self.stockindex.ema50[approve_idx] < self.stockindex.ema100[approve_idx] < self.stockindex.ema200[approve_idx])

        long_condition = (
            (self.stockdata.stoch_k[approve_idx] <= 35 or self.stockdata.stoch_d[approve_idx] <= 35) and
            self.stockdata.stoch_k[approve_idx] > self.stockdata.stoch_d[approve_idx] and
            any(self.stockdata.macd[i] > self.stockdata.macd_s[i] for i in [approve_idx, reversal_idx, initial_idx]) and
            long_ema_check and
            self.stockdata.close[initial_idx] < self.stockdata.open[initial_idx] and
            self.stockdata.low[approve_idx] > self.stockdata.low[reversal_idx] and
            self.stockdata.close[approve_idx] > self.stockdata.high[reversal_idx]
        )

        short_condition = (
            (self.stockdata.stoch_k[approve_idx] > 65 or self.stockdata.stoch_d[approve_idx] > 65) and
            any(self.stockdata.macd[i] < self.stockdata.macd_s[i] for i in [approve_idx, reversal_idx, initial_idx]) and
            short_ema_check and
            self.stockdata.low[reversal_idx] > self.stockdata.close[approve_idx] and
            self.stockdata.high[reversal_idx] > self.stockdata.high[approve_idx]
        )

        common_long_condition = (
            self.stockdata.low[approve_idx] > self.stockdata.low[reversal_idx]
        )

        common_short_condition = (
            self.stockdata.low[reversal_idx] > self.stockdata.close[approve_idx]
        )

        ema_columns = ['ema20', 'ema50', 'ema100', 'ema200']
        buy_signal = False
        short_signal = False
        for ema in ema_columns:
            ema_series = getattr(self.stockdata, ema)

            condition = (
                self.stockdata.open[initial_idx] > self.stockdata.close[initial_idx] and
                self.stockdata.low[reversal_idx] < ema_series[reversal_idx] and
                self.stockdata.open[reversal_idx] > ema_series[reversal_idx] and
                self.stockdata.close[reversal_idx] > ema_series[reversal_idx] and
                self.stockdata.low[initial_idx] > self.stockdata.low[reversal_idx]
            )

            short_condition_1 = (
                self.stockdata.close[initial_idx] > self.stockdata.open[initial_idx] and
                self.stockdata.close[reversal_idx] < ema_series[reversal_idx] and
                self.stockdata.open[reversal_idx] < ema_series[reversal_idx] and
                self.stockdata.high[reversal_idx] > ema_series[reversal_idx] and
                self.stockdata.high[reversal_idx] > self.stockdata.high[initial_idx]
            )

            condition1 = (
                self.stockdata.low[initial_idx] < self.stockdata.low[reversal_idx] and
                self.stockdata.low[reversal_idx] < ema_series[reversal_idx] and
                self.stockdata.open[reversal_idx] > ema_series[reversal_idx] and
                self.stockdata.close[reversal_idx] > ema_series[reversal_idx] and
                self.stockdata.low[initial_idx] < ema_series[initial_idx] and
                self.stockdata.open[initial_idx] > ema_series[initial_idx] and
                self.stockdata.close[initial_idx] > ema_series[initial_idx]
            )

            short_condition_2 = (
                self.stockdata.close[initial_idx] > self.stockdata.open[initial_idx] and
                self.stockdata.open[initial_idx] < ema_series[initial_idx] and
                self.stockdata.close[initial_idx] < ema_series[initial_idx] and
                self.stockdata.open[reversal_idx] < ema_series[reversal_idx] and
                self.stockdata.close[reversal_idx] < ema_series[reversal_idx] and
                self.stockdata.high[initial_idx] > ema_series[initial_idx] and
                self.stockdata.high[reversal_idx] > ema_series[reversal_idx]
            )

            condition2 = (
                    self.stockdata.low[initial_idx] > self.stockdata.low[reversal_idx] and
                    self.stockdata.low[reversal_idx] < ema_series[reversal_idx] and
                    self.stockdata.open[reversal_idx] > ema_series[reversal_idx] and
                    self.stockdata.close[reversal_idx] < ema_series[reversal_idx] and
                    self.stockdata.low[initial_idx] < ema_series[initial_idx] and
                    self.stockdata.open[initial_idx] > ema_series[initial_idx] and
                    self.stockdata.close[initial_idx] < ema_series[initial_idx]
            )

            short_condition_3 = (
                    self.stockdata.close[initial_idx] > self.stockdata.open[initial_idx] and
                    self.stockdata.close[reversal_idx] < self.stockdata.open[reversal_idx] and
                    self.stockdata.close[initial_idx] > ema_series[initial_idx] and
                    self.stockdata.close[reversal_idx] < ema_series[reversal_idx] and
                    self.stockdata.open[initial_idx] < ema_series[initial_idx] and
                    self.stockdata.close[initial_idx] > ema_series[initial_idx] and
                    self.stockdata.open[reversal_idx] > ema_series[reversal_idx] and
                    self.stockdata.close[reversal_idx] < ema_series[reversal_idx]
            )

            if condition or condition1 or condition2:
                buy_signal = True
                break

            if short_condition_1 or short_condition_2 or short_condition_3:
                short_signal = True
                break

        if not self.position:
            if long_ema_check_index and common_long_condition and long_condition and buy_signal:
                if current_close > self.stockdata.high[-1]:
                    cash = self.broker.getcash()
                    size = int((cash * 0.1) / current_close)
                    if size > 0:
                        self.buy(size=size)
                        self.entry_price = current_close
                        self.stop_loss_price = self.stockdata.low[-2]
                        risk = (self.entry_price - self.stop_loss_price) * size
                        target_profit = risk * 2
                        self.take_profit_price = self.entry_price + (target_profit/size)
                        print(f"AL emri verildi: {self.stockdata.datetime.date(0)} - Fiyat: {current_close}")
                        print(f"condition: {condition} - condition1: {condition1} - condition2: {condition2}")
                        print(f"STOP LOSS PRICE: {self.stop_loss_price} - entry price: {self.entry_price} - take profit price: {self.take_profit_price}")

#            elif short_ema_check_index and common_short_condition and short_condition and short_signal:
#                if current_close < self.stockdata.high[-1]:
#                    cash = self.broker.getcash()
#                    size = int((cash * 0.1) / current_close)
#                    if size > 0:
#                        self.sell(size=size)
#                        self.entry_price = current_close

#                        self.stop_loss_price = self.stockdata.high[-2]
#                        risk = (self.stop_loss_price - self.entry_price) * size
#                        target_profit = risk * 2
#                        self.take_profit_price = self.entry_price - (target_profit/size)
#                        print(f"SHORT emri verildi: {self.stockdata.datetime.date(0)} - Fiyat: {current_close}")
#                        print(f"condition: {short_condition_1} - condition1: {short_condition_2} - condition2: {short_condition_3}")
#                        print(f"STOP LOSS PRICE: {self.stop_loss_price} - entry price: {self.entry_price} - take profit price: {self.take_profit_price}")

        elif self.position:
            long_sell_condition1 = current_close >= self.take_profit_price
            long_sell_condition2 = current_close < self.stop_loss_price
            short_sell_condition1 = current_close <= self.take_profit_price
            short_sell_condition2 = current_close > self.stop_loss_price

            if self.position.size > 0 and (long_sell_condition1 or long_sell_condition2):
                self.close()
                if long_sell_condition1:
                    print(f"LONG SAT isleminde KAR alindi: {self.stockdata.datetime.date(0)} - Fiyat: {current_close}")
                elif long_sell_condition2:
                    print(f"LONG SAT isleminde ZARAR edildi: {self.stockdata.datetime.date(0)} - Fiyat: {current_close}")
            elif self.position.size < 0 and (short_sell_condition1 or short_sell_condition2):
                self.close()
                if short_sell_condition1:
                    print(f"SHORT SAT isleminde KAR alindi: {self.stockdata.datetime.date(0)} - Fiyat: {current_close}")
                elif short_sell_condition2:
                    print(f"SHORT SAT isleminde ZARAR edildi: {self.stockdata.datetime.date(0)} - Fiyat: {current_close}")

def write_backtest_results_to_file(exchange, symbol, sharpe, drawdown, returns, trades):
    try:
        with open("backtest_sonuclari.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n{'=' * 50}\n")
            f.write(f"{symbol} - {exchange} - Backtest Sonuçları\n")
            f.write(f"{'=' * 50}\n")

            # Sharpe Ratio
            try:
                sharpe_ratio = sharpe.get('sharperatio', None)
                if sharpe_ratio is not None:
                    if isinstance(sharpe_ratio, float):
                        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
                    else:
                        f.write(f"Sharpe Ratio: {sharpe_ratio}\n")
                else:
                    f.write("Sharpe Ratio: Hesaplanamadı\n")
            except Exception as e:
                f.write(f"Sharpe Ratio hatası: {e}\n")

            # Max Drawdown
            try:
                f.write(f"Max Drawdown: {drawdown.max.drawdown:.2f}%\n")
            except Exception as e:
                f.write(f"Max Drawdown hatası: {e}\n")

            # Getiri
            try:
                f.write(f"Toplam Getiri: {returns.get('rtot', 0) * 100:.2f}%\n")
                f.write(f"Yıllık Getiri: {returns.get('rannual', 0) * 100:.2f}%\n")
            except Exception as e:
                f.write(f"Getiri hatası: {e}\n")

            # Trade verisi
            try:
                total = trades.total.total if hasattr(trades, 'total') and hasattr(trades.total, 'total') else 0
                won = trades.won.total if hasattr(trades, 'won') and hasattr(trades.won, 'total') else 0
                lost = trades.lost.total if hasattr(trades, 'lost') and hasattr(trades.lost, 'total') else 0

                f.write(f"Toplam İşlem Sayısı: {total}\n")
                f.write(f"Karlı İşlem Sayısı: {won}\n")
                f.write(f"Zararlı İşlem Sayısı: {lost}\n")

                if total > 0:
                    success_rate = (won / total) * 100
                    f.write(f"Başarı Yüzdesi: {success_rate:.2f}%\n")
                else:
                    f.write("İşlem yapılmadı, başarı yüzdesi hesaplanamadı.\n")

            except Exception as e:
                f.write(f"Trade analizi hatası: {e}\n")

    except Exception as e:
        print(f"Dosya yazım hatası: {e}")

if __name__ == '__main__':
    relevant = getdata_stock.get_stock_symbols(None)
    assert isinstance(relevant, dict)

    for exchange, symbols in relevant.items():
        if exchange == 'BIST':
            index_symbol = 'XU100'
            index_exchange = 'BIST'
#        elif exchange == 'XETR':
#            index_symbol = 'DAX'
#            index_exchange = 'XETR'
#        elif exchange == 'NYSE':
#            index_symbol = 'SPX'
#            index_exchange = 'SP'
#        elif exchange == 'NASDAQ':
#            index_symbol = 'NDX'
#            index_exchange = 'NASDAQ'
#        if exchange == 'LSE':
#            index_symbol = 'UKX'
#            index_exchange = 'FTSE'
        else:
            continue  # Bilinmeyen exchange, atla

        try:
            df_index = getdata_stock.get_data_frame(index_symbol, index_exchange, '1D', 10000)
        except Exception as e:
            print(f"{index_symbol} verisi alınamadı: {e}")
            continue

        # Toplam sonuçlar
        total_trades = 0
        total_wins = 0
        total_losses = 0

        for symbol in tqdm(symbols, desc=f"{exchange} hisseleri işleniyor"):
            try:
                df = getdata_stock.get_data_frame(symbol, exchange, '1D', 10000)
                if df is None or df.empty:
                    continue

                df.columns = [col.capitalize() for col in df.columns]

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
                    df_index = indicator.add_indicator(df_index, ind, param)

                df = main.add_candlestick_patterns(df)
                df_index = main.add_candlestick_patterns(df_index)

                if not main.validate_columns(df):
                    continue
                print(f"{symbol} backtest basliyor")

                data = PandasData(dataname=df)
                dataindex = PandasData(dataname=df_index)

                cerebro = bt.Cerebro()
                cerebro.broker.setcash(100000)
                cerebro.adddata(data)
                cerebro.adddata(dataindex)
                cerebro.addstrategy(IkiMumCubukluDonusStrategy)

                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, riskfreerate=0.0)
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

                results = cerebro.run()
                strategy = results[0]

                sharpe = strategy.analyzers.sharpe.get_analysis()
                drawdown = strategy.analyzers.drawdown.get_analysis()
                returns = strategy.analyzers.returns.get_analysis()
                trades = strategy.analyzers.trades.get_analysis()

                write_backtest_results_to_file(exchange, symbol, sharpe, drawdown, returns, trades)

                # Trade analizini kontrol et
                total = trades.get('total', {}).get('total', 0)
                won = trades.get('won', {}).get('total', 0)
                lost = trades.get('lost', {}).get('total', 0)

                total_trades += total
                total_wins += won
                total_losses += lost

            except Exception as e:
                print(f"{symbol} için hata oluştu: {e}")
                continue

        try:
            with open("exchange_summary_results.txt", "a", encoding="utf-8") as f:
                f.write(f"\n{'#' * 60}\n")
                f.write(f"{exchange} Borsası Toplam Sonuçları:\n")
                f.write(f"Toplam İşlem Sayısı: {total_trades}\n")
                f.write(f"Toplam Karlı İşlem: {total_wins}\n")
                f.write(f"Toplam Zararlı İşlem: {total_losses}\n")
                if total_trades > 0:
                    success_rate = (total_wins / total_trades) * 100
                    f.write(f"Başarı Yüzdesi: {success_rate:.2f}%\n")
                else:
                    f.write("Hiç işlem yapılmamış.\n")
                f.write(f"{'#' * 60}\n")
        except Exception as e:
            print(f"Sonuç dosyasına yazarken hata oluştu: {e}")
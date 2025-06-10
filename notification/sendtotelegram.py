import os

import requests
import plotly.io as pio
from io import BytesIO
import creategraphics
from notification import sendtotelegram

def send_message_telegram(symbol, alarm_name):
    message = (
        f"ALARM!! SYMBOL: {symbol}\n"
        f"Alarm Name: {alarm_name}"
    )
    send_to_telegram(message, None)

def send_trade_signal(df, symbol, interval, candle_indices, is_long, strategy_name):
    alarm_name = f"{'LONG' if is_long else 'SHORT'} {strategy_name}"
    exchange_and_symbol = df['symbol'].iloc[0] if df.get('symbol') is not None else symbol

    try:
        fig = creategraphics.create_graphics(df, is_long)
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


def send_telegram(symbol, alarm_name, initial_candle_close, reversal_candle_close, approve_candle_close, interval, photo):
    message = (
        f"ALARM!! SYMBOL: {symbol}\n"
        f"Alarm Name: {alarm_name}\n"
        f"Initial Candle Close Price: {initial_candle_close}\n"
        f"Reversal Candle Close Price: {reversal_candle_close}\n"
        f"Approve Candle Close Price: {approve_candle_close}\n"
        f"Link: https://www.tradingview.com/chart/?symbol={symbol}&interval={interval}"
    )
    send_to_telegram(message, photo)

def send_to_telegram(message, photo):

    apiToken = os.getenv("TELEGRAM_BOT_TOKEN")
    chatID = os.getenv("TELEGRAM_BOT_CHAT_ID")
    apiURL = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    apiSendPhotoURL = f'https://api.telegram.org/bot{apiToken}/sendPhoto'
    try:
        if photo is not None:
            img_bytes = pio.to_image(photo, format='png')
            files = {'photo': ('plot.png', BytesIO(img_bytes), 'image/png')}
            data = {'chat_id': chatID, 'caption': message}

            response = requests.post(apiSendPhotoURL, data=data, files=files)
        else:
            response = requests.post(apiURL, json={'chat_id': chatID, 'text': message})

        print(response.text)
    except Exception as e:
        print(e)

#send_to_telegram("selam guys!!")

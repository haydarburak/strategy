import os

import requests
import plotly.io as pio
from io import BytesIO

def send_message_telegram(symbol, alarm_name):
    message = (
        f"ALARM!! SYMBOL: {symbol}\n"
        f"Alarm Name: {alarm_name}"
    )
    send_to_telegram(message, None)

def send_telegram(symbol, alarm_name, initial_candle_close, reversal_candle_close, approve_candle_close, interval, photo):
    """
    Sends an alarm message to Telegram with the provided details.

    Args:
        symbol (str): Trading symbol.
        alarm_name (str): Name of the alarm.
        initial_candle_close (float): Closing price of the initial candle.
        reversal_candle_close (float): Closing price of the reversal candle.
        approve_candle_close (float): Closing price of the approve candle.
        interval (str): Chart interval for TradingView link.

    Returns:
        None
    """
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

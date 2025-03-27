import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido

def create_graphics(df, long):
    approve_candle_index = -1
    reversal_candle_index = -2
    initial_candle_index = -3

    candle_indices = {
        'initial': initial_candle_index,
        'reversal': reversal_candle_index,
        'approve': approve_candle_index
    }

    #    if long:
    #    high_price = df.tail(15)['High'].max()
    #    low_price = df.tail(4)['Low'].min()

    #    diff = high_price - low_price


    #else:
    #   high_price = df.tail(4)['High'].max()
    #   low_price = df.tail(15)['Low'].min()

    #   diff = high_price - low_price


    #print('Hight Price: ' + str(high_price))
    #print('Low Price: ' + str(low_price))

    # Fibonacci seviyeleri
    #retracement_levels = {
    #    '0.0%': low_price,
    #    '23.6%': low_price + (diff * 0.236),
    #    '38.2%': low_price + (diff * 0.382),
    #    '50.0%': low_price + (diff * 0.5),
    #    '61.8%': low_price + (diff * 0.618),
    #    '78.6%': low_price + (diff * 0.786),
    #    '100.0%': low_price + (diff * 1)
    #}

    # Fibonacci Extension seviyeleri
    #extension_levels = {
    #    '161.8%': low_price + (diff * 1.618),
    #    '261.8%': low_price + (diff * 2.618),
    #    '423.6%': low_price + (diff * 4.236)
    #}

    #print('retracement_levels: ')
    #print(retracement_levels)
    #print('extension_levels: ')
    #print(extension_levels)

    # Grafik oluşturma
    fig = make_subplots(
        rows=3, cols=1,  # 3 satır, 1 sütunlu grid (Price, Stochastic ve MACD)
        shared_xaxes=True,  # X ekseni ortak
        vertical_spacing=0.1,  # Alt ve üst grafik arasındaki boşluk
        row_heights=[0.5, 0.25, 0.25],  # Üst grafiğin daha büyük olmasını sağla
        subplot_titles=('Price Chart', 'Stochastic Indicators', 'MACD Indicators')
    )

    df_last = df.tail(50)  # Son 50 veriyi alıyoruz

    # 1. Satırda: Candlestick grafiği ve EMA'lar
    fig.add_trace(go.Candlestick(
        x=df_last.index,
        open=df_last['Open'],  # Açılış fiyatı
        high=df_last['High'],  # Yüksek fiyat
        low=df_last['Low'],  # Düşük fiyat
        close=df_last['Close'],  # Kapanış fiyatı
        name='Candlestick'
    ), row=1, col=1)

    # Fibonacci seviyelerini grafiğe ekleme
    #for level, price in retracement_levels.items():
    #    fig.add_hline(y=price, line_dash="dash", annotation_text=level, annotation_position="right", row=1, col=1)

    #for level, price in extension_levels.items():
    #    fig.add_hline(y=price, line_dash="dash", annotation_text=level, annotation_position="right", row=1, col=1)

    # EMA'lar (EMA20, EMA50, EMA100, EMA200) ilk grafikte ekleniyor
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last['EMA20'], mode='lines', name='EMA20',
                             line=dict(color='orange')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last['EMA50'], mode='lines', name='EMA50',
                             line=dict(color='green')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last['EMA100'], mode='lines', name='EMA100',
                             line=dict(color='red')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last['EMA200'], mode='lines', name='EMA200',
                             line=dict(color='blue')),
                  row=1, col=1)

    # Alarm noktaları (Initial, Reversal, Approve)
    fig.add_trace(go.Scatter(
        x=[df_last.index[candle_indices['initial']]],
        y=[df_last['Close'].iloc[candle_indices['initial']]],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Initial Candle'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[df_last.index[candle_indices['reversal']]],
        y=[df_last['Close'].iloc[candle_indices['reversal']]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Reversal Candle'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[df_last.index[candle_indices['approve']]],
        y=[df_last['Close'].iloc[candle_indices['approve']]],
        mode='markers',
        marker=dict(size=10, color='green'),
        name='Approve Candle'
    ), row=1, col=1)

    # 2. Satırda: Stochastic göstergeleri
    fig.add_trace(
        go.Scatter(x=df_last.index, y=df_last['STOCH_K'], mode='lines', name='STOCH_K',
                   line=dict(color='purple')), row=2,
        col=1)
    fig.add_trace(
        go.Scatter(x=df_last.index, y=df_last['STOCH_D'], mode='lines', name='STOCH_D',
                   line=dict(color='orange')), row=2,
        col=1)

    fig.add_hline(y=30, row=2, col=1)
    fig.add_hline(y=70, row=2, col=1)

    # 3. Satırda: MACD göstergeleri
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last['MACD'], mode='lines', name='MACD',
                             line=dict(color='blue')),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df_last.index, y=df_last['MACD_S'], mode='lines', name='MACD_S',
                             line=dict(color='green')),
                  row=3, col=1)

    symbol = df_last['symbol'].iloc[0] if not df_last.empty and 'symbol' in df_last.columns else "Unknown"

    # Grafik düzeni
    fig.update_layout(
        title = f"Alarm SYMBOL: {symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        height=900,  # Yüksekliği artırarak alt grafiklere yer açıyoruz
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    return fig
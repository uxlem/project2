from vnstock import Vnstock
import talib as ta 
import mplfinance as mpf
import pandas as pd
import numpy as np

def test(symbol = 'AAA', start = '2023-01-01', end = '2023-10-01'):
    if symbol is not None:
        stock = Vnstock().stock(symbol)
        df = stock.quote.history(start = start, end = end, interval = '1D')
        df['time'] = pd.DatetimeIndex(df['time'])
        df.set_index('time', inplace=True)

        cci = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        profit = 0
        in_position = False
        buy_price = 0

        for i in range(len(df)):
            if cci.iloc[i-1] < -100 and cci.iloc[i] > -100 and not in_position:
                buy_price = df['close'].iloc[i]
                in_position = True
                print(f"Buy at {buy_price} on {df.index[i]}")
            elif cci.iloc[i-1] > 100 and cci.iloc[i] < 100 and in_position:
                sell_price = df['close'].iloc[i]
                profit += sell_price - buy_price
                in_position = False
                print(f"Sell at {sell_price} on {df.index[i]}")

        return profit
    else:
        return None

print(test())
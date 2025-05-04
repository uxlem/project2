import talib as ta 
import pandas as pd
import numpy as np

file_path = "stock_data/A32.csv"
df = pd.read_csv(file_path, index_col='time')
# df['time'] = pd.DatetimeIndex(df['time'])
# df.set_index('time', inplace = True)

def find_low_index(series, current_index, lookback = 20):
    if current_index < lookback:
        return None
    window = series.iloc[current_index-lookback:current_index]
    if window.empty or window.isnull().all():
        return None
    return window.idxmin()

def find_high_index(series, current_index, lookback = 20):
    if current_index < lookback:
        return None
    window = series.iloc[current_index-lookback:current_index]
    if window.empty or window.isnull().all():
        return None
    return window.idxmax()

# simple strategy using CCI, MACD and RSI indices
def strategyOne(df):
    # CCI(14)
    cci = ta.CCI(df['high'], df['low'], df['close'])
    # MACD(12, 26, 9)
    macd, macdsignal, macdhist = ta.MACD(df['close'])
    # RSI(14)
    rsi = ta.RSI(df['close'])

    PnL = []
    stocks_bought = 0
    buy_price = 0

    for i in range(len(df)):
        rsi_bearish = False
        rsi_bullish = False
        macd_bearish = False
        macd_bullish = False
        
        
        prev_low_idx = find_low_index(df['low'], i, 20)
        prev_high_idx = find_high_index(df['high'], i, 20)
        # print(f"{prev_low_idx}, {prev_high_idx}")
        prev_low_rsi_idx = find_low_index(rsi, i, 20)
        prev_high_rsi_idx = find_high_index(rsi, i, 20)   

        prev_low_macd_idx = find_low_index(macd, i, 26)
        prev_high_macd_idx = find_high_index(macd, i, 26)


        # Bullish Divergence
        if (prev_low_idx is not None) and (prev_low_rsi_idx is not None) and (prev_low_macd_idx is not None):
            try:
                prev_low_pos = prev_low_idx if isinstance(prev_low_idx, int) else df.index.get_loc(prev_low_idx)
                prev_low_rsi_pos = prev_low_rsi_idx if isinstance(prev_low_rsi_idx, int) else df.index.get_loc(prev_low_rsi_idx)
                prev_low_macd_pos = prev_low_macd_idx if isinstance(prev_low_macd_idx, int) else df.index.get_loc(prev_low_macd_idx)
                if df['low'].iloc[i] < df['low'].iloc[prev_low_pos]:
                    if rsi.iloc[i] > rsi.iloc[prev_low_rsi_pos]:
                        rsi_bullish = True
                    if macd.iloc[i] > macd.iloc[prev_low_macd_pos]:  
                        macd_bullish = True
            except (TypeError, KeyError) as e:
                print(f"Error in bullish divergence at index {i}: {e}")

        # Bearish Divergence
        if (prev_high_idx is not None) and (prev_high_rsi_idx is not None) and (prev_high_macd_idx is not None):
            try:
                prev_high_pos = prev_high_idx if isinstance(prev_high_idx, int) else df.index.get_loc(prev_high_idx)
                prev_high_rsi_pos = prev_high_rsi_idx if isinstance(prev_high_rsi_idx, int) else df.index.get_loc(prev_high_rsi_idx)
                prev_high_macd_pos = prev_high_macd_idx if isinstance(prev_high_macd_idx, int) else df.index.get_loc(prev_high_macd_idx)
                if df['high'].iloc[i] > df['high'].iloc[prev_high_pos]:
                    if rsi.iloc[i] < rsi.iloc[prev_high_rsi_pos]:
                        rsi_bearish = True
                    if macd.iloc[i] < macd.iloc[prev_high_macd_pos]: 
                        macd_bearish = True
            except (TypeError, KeyError) as e:
                print(f"Error in bearish divergence at index {i}: {e}")
        
        
        if ((cci.iloc[i-1] < -100 and cci.iloc[i] > -100) 
            or (rsi.iloc[i-1] < 30 and rsi.iloc[i] > 30) 
            or (macdhist.iloc[i-1] < 0 and macdhist.iloc[i] > 0 and (rsi.iloc[i-1] < 45 or rsi.iloc[i-1] > 55))
            or rsi_bullish == True or macd_bullish == True):
            stocks_bought += 1
            buy_price += df['close'].iloc[i]
            # print(f"Mua 1 phiếu giá {buy_price}")
        elif (((cci.iloc[i-1] > 100 and cci.iloc[i] < 100)
                or (rsi.iloc[i-1] > 70 and rsi.iloc[i] < 70)
                or (macdhist.iloc[i-1] > 0 and macdhist.iloc[i] < 0 and (rsi.iloc[i-1] < 45 or rsi.iloc[i-1] > 55))
                or rsi_bearish or macd_bearish)
                    and stocks_bought > 0):
            
            sell_price = df['close'].iloc[i]*stocks_bought
            PnL.append((sell_price - buy_price)*100/buy_price)
            # print(f"Bán {stocks_bought} tổng giá {sell_price}")
            buy_price = 0
            stocks_bought = 0
    
    return pd.Series(PnL)
    
print(df)
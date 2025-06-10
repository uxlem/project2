import pandas as pd
import numpy as np
import talib as ta
from scipy.signal import argrelextrema
import bisect

# Sử dụng (Slow) Stochastic
def stoch_signals(df, extrema_order = 10, lookback_window = 45, max_idx_dist = 4, wait = 3, inplace = True):

    if not inplace:
        df = df.copy()

    # Tính Stochastic, n = 14
    df['slowk'], df['slowd'] = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=14)
    # Tạo cột
    df['stoch_buy_signals'] = np.nan
    df['stoch_sell_signals'] = np.nan
    
    # Xét tín hiệu thông thường - Crossovers

    col_close = df.columns.get_loc('close')
    col_slowk = df.columns.get_loc('slowk')
    col_slowd = df.columns.get_loc('slowd')

    for i in range(26, len(df)-2):
        # Mua: %K cắt lên trên %D && %K và %D cùng từ dưới 20 (vùng quá bán) cắt lên
        if 20 >= df.iat[i, col_slowk] >= df.iat[i, col_slowd] and \
            df.iat[i-1, col_slowk] <= df.iat[i-1, col_slowd] and \
                (df.iat[i+1, col_slowk] > 20 or df.iat[i+2, col_slowk] > 20) and \
                (df.iat[i+1, col_slowd] > 20 or df.iat[i+2, col_slowd] > 20):
            df.iat[i+1, df.columns.get_loc('stoch_buy_signals')] = df.iat[i+1, col_close]
            print(f"mua {i+1}")
        # Bán: %K cắt xuống dưới %D && %K và %D cùng từ trên 80 (vùng quá mua) cắt xuống
        if 80 <= df.iat[i, col_slowk] <= df.iat[i, col_slowd] and \
                df.iat[i - 1, col_slowk] >= df.iat[i - 1, col_slowd] and \
                (df.iat[i + 1, col_slowk] < 80 or df.iat[i + 2, col_slowk] < 80) and \
                (df.iat[i + 1, col_slowd] < 80 or df.iat[i + 2, col_slowd] < 80):
            df.iat[i+1, df.columns.get_loc('stoch_sell_signals')] = df.iat[i+1, col_close]
            print(f"bán {i+1}")
    if not inplace:
        return df
    return None

def strategy(df):
    # initial_cash = 10000
    PnL = []
    stocks_bought = 0
    total_buy_price = 0

    stoch_signals(df)

    df.to_csv('strat_v4_df.csv')

    for i in range(36, len(df)):
        buy_signals = {
            'stoch': pd.notna(df.iat[i, df.columns.get_loc('stoch_buy_signals')]),
        }

        sell_signals = {
            'stoch': pd.notna(df.iat[i, df.columns.get_loc('stoch_sell_signals')]),
        }

        buy_vote = sum(buy_signals.values())
        sell_vote = sum(sell_signals.values())
        if (buy_vote > 0 or sell_vote > 0):
            print(f"{buy_vote} / {sell_vote}")
        if buy_vote > 0:
            print(f"buy {i}")
            stocks_bought += 1
            total_buy_price += df['close'].iloc[i]
        elif sell_vote > 0 and stocks_bought > 0:
            print(f"sell {i}")
            sell_price = df['close'].iloc[i] * stocks_bought
            PnL.append((sell_price - total_buy_price) * 100 / total_buy_price)
            # print(f"Bán {stocks_bought} tổng giá {sell_price}")
            total_buy_price = 0
            stocks_bought = 0

    return pd.Series(PnL)

if __name__ == "__main__":
    sample_df = pd.read_csv("precalc_data/AAA.csv", index_col="time")

    results = strategy(sample_df)

    profits = results[results > 0].dropna()
    breakeven = results[results == 0].dropna()
    loss = results[results < 0].dropna()

    for i in profits, breakeven, loss, results:
        if len(i) > 0:
            print(f"count = {len(i)} = {len(i) / len(results) * 100:.1f}%, mean = {i.values.mean():.2f}, "
                  f"max = {i.values.max():.2f}, min = {i.values.min():.2f}")

import pandas as pd

# file_path = "stock_data/A32.csv"
# df = pd.read_csv(file_path, index_col='time')


# df['time'] = pd.DatetimeIndex(df['time'])
# df.set_index('time', inplace = True)

def find_low_index(series, current_index, lookback=20):
    if current_index < lookback:
        return None
    window = series.iloc[current_index - lookback:current_index]
    if window.empty or window.isnull().all():
        return None
    return window.idxmin()


def find_high_index(series, current_index, lookback=20):
    if current_index < lookback:
        return None
    window = series.iloc[current_index - lookback:current_index]
    if window.empty or window.isnull().all():
        return None
    return window.idxmax()


# simple strategy using CCI, MACD and RSI indices
def strategy_one(df):
    # CCI(14)
    cci = df['cci']
    # MACD(12, 26, 9)
    macd, macdsignal, macdhist = df['macd'], df['macdsignal'], df['macdhist']
    # RSI(14)
    rsi = df['rsi']

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
                prev_low_rsi_pos = prev_low_rsi_idx if isinstance(prev_low_rsi_idx, int) else df.index.get_loc(
                    prev_low_rsi_idx)
                prev_low_macd_pos = prev_low_macd_idx if isinstance(prev_low_macd_idx, int) else df.index.get_loc(
                    prev_low_macd_idx)
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
                prev_high_rsi_pos = prev_high_rsi_idx if isinstance(prev_high_rsi_idx, int) else df.index.get_loc(
                    prev_high_rsi_idx)
                prev_high_macd_pos = prev_high_macd_idx if isinstance(prev_high_macd_idx, int) else df.index.get_loc(
                    prev_high_macd_idx)
                if df['high'].iloc[i] > df['high'].iloc[prev_high_pos]:
                    if rsi.iloc[i] < rsi.iloc[prev_high_rsi_pos]:
                        rsi_bearish = True
                    if macd.iloc[i] < macd.iloc[prev_high_macd_pos]:
                        macd_bearish = True
            except (TypeError, KeyError) as e:
                print(f"Error in bearish divergence at index {i}: {e}")

        if ((cci.iloc[i - 1] < -100 < cci.iloc[i])
                or (rsi.iloc[i - 1] < 30 < rsi.iloc[i])
                or (macdhist.iloc[i - 1] < 0 < macdhist.iloc[i] and (rsi.iloc[i - 1] < 45 or rsi.iloc[i - 1] > 55))
                or rsi_bullish or macd_bullish):
            stocks_bought += 1
            buy_price += df['close'].iloc[i]
            # print(f"Mua 1 phiếu giá {buy_price}")
        elif (((cci.iloc[i - 1] > 100 > cci.iloc[i])
               or (rsi.iloc[i - 1] > 70 > rsi.iloc[i])
               or (macdhist.iloc[i - 1] > 0 > macdhist.iloc[i] and (rsi.iloc[i - 1] < 45 or rsi.iloc[i - 1] > 55))
               or rsi_bearish or macd_bearish)
              and stocks_bought > 0):

            sell_price = df['close'].iloc[i] * stocks_bought
            PnL.append((sell_price - buy_price) * 100 / buy_price)
            # print(f"Bán {stocks_bought} tổng giá {sell_price}")
            buy_price = 0
            stocks_bought = 0

    return pd.Series(PnL)


def strategy_two(df):
    
    # initial_cash = 10000
    PnL = []
    stocks_bought = 0
    total_buy_price = 0

    high, low = df.columns.get_loc('high'), df.columns.get_loc('low')
    cci, rsi, macdhist = df.columns.get_loc('cci'), df.columns.get_loc('rsi'), df.columns.get_loc('macdhist')

    def mini(col):
        return df.iloc[i-20:i-1, col].min()

    def maxi(col):
        return df.iloc[i-20:i-1, col].max()

    def bullish(col=rsi):
        # Phân kỳ tăng (bullish) - Xét đáy
        if df.iat[i, col] < df.iat[i - 1, col] and df.iat[i, low] < df.iat[i - 1, low] and mini(col) is not None:
            if df.iat[i, col] > mini(col) and df.iat[i, low] < mini(low):
                return True
            if df.iat[i, col] < mini(col) and df.iat[i, low] > mini(low):
                return True
        return False

    def bearish(col=rsi):
        # Phân kỳ giảm (bullish) - Xét đỉnh
        if df.iat[i, col] > df.iat[i - 1, col] and df.iat[i, high] > df.iat[i - 1, high] and maxi(col) is not None:
            if df.iat[i, col] > maxi(col) and df.iat[i, high] < maxi(high):
                return True
            if df.iat[i, col] < maxi(col) and df.iat[i, high] > maxi(high):
                return True

        return False
    
    for i in range(36, len(df)):
        yesterday = {
            'cci' : df.iat[i-1, cci],
            'rsi' : df.iat[i-1, rsi],
            'macdhist' : df.iat[i-1, macdhist]
        }

        today = {
            'cci' : df.iat[i, cci],
            'rsi' : df.iat[i, rsi],
            'macdhist' : df.iat[i, macdhist]
        }



        buy_signals = {
            'cci': True if today['cci'] > -100 > yesterday['cci'] else False,
            'rsi': True if today['rsi'] > 30 > yesterday['rsi'] else False,
            'macdhist': True if today['macdhist'] > 0 > yesterday['macdhist']
            and not (45 < today['rsi'] < 55) else False,
            'bullish_cci': bullish(cci),
            'bullish_rsi': bullish(rsi),
            'bullish_macdhist': bullish(macdhist),
        }

        sell_signals = {
            'cci': True if today['cci'] < 100 < yesterday['cci'] else False,
            'rsi': True if today['rsi'] < 70 < yesterday['rsi'] else False,
            'macdhist': True if today['macdhist'] < 0 < yesterday['macdhist']
            and not (45 < today['rsi'] < 55) else False,
            'bearish_cci': bearish(cci),
            'bearish_rsi': bearish(rsi),
            'bearish_macdhist': bearish(macdhist),
        }

        buy_vote = sum(buy_signals.values())
        sell_vote = sum(sell_signals.values())
        
        if buy_vote > 0:
            stocks_bought += 1
            total_buy_price += df['close'].iloc[i]
        elif sell_vote > 0 and stocks_bought > 0:
            sell_price = df['close'].iloc[i] * stocks_bought
            PnL.append((sell_price - total_buy_price) * 100 / total_buy_price)
            # print(f"Bán {stocks_bought} tổng giá {sell_price}")
            total_buy_price = 0
            stocks_bought = 0

        
    return pd.Series(PnL)

df = pd.read_csv('precalc_data/AAA.csv', index_col='time')
strategy_two(df)



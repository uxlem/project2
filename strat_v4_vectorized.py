import pandas as pd
import numpy as np
import talib as ta
from scipy.signal import argrelextrema
import bisect
import logging
import result_reader
import os

# os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="strat_v4vec.log",
    filemode="w",
    level=logging.INFO,  # Can also use DEBUG, WARNING, etc.
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_and_filter_from_csv(filepath: str):
    df = pd.read_csv(filepath, index_col="time", parse_dates=True)
    df = df[(df["close"] > 0) & (df["volume"] > 0)].copy()
    return df


def stoch_crossover(df: pd.DataFrame, inplace: bool = True):
    """
    Tính tín hiệu mua bán bằng Stochastic chậm cơ bản (crossover).

    Mua: `%K` cắt lên trên `%D`, đồng thời `%K` và `%D` cùng từ dưới 20 (vùng quá bán) cắt lên

    Bán: `%K` cắt xuống dưới `%D`, đồng thời `%K` và `%D` cùng từ trên 80 (vùng quá mua) cắt xuống

    Args:
        df (pd.DataFrame): _description_
        inplace (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    if not inplace:
        df = df.copy()

    # Tính Stochastic, n = 14
    df["slowk"], df["slowd"] = ta.STOCH(
        df["high"].to_numpy(dtype="float64"),
        df["low"].to_numpy(dtype="float64"),
        df["close"].to_numpy(dtype="float64"),
        fastk_period=14,
    )
    # Tạo cột
    df["stoch_crossover_buy_signals"] = np.nan
    df["stoch_crossover_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    slowk = df["slowk"].values
    slowd = df["slowd"].values
    close = df["close"].values

    # Buy: %K crosses above %D, both from below 20, and both move above 20 in next 2 bars
    cond_buy = (
        (slowk[26:-2] >= slowd[26:-2])
        & (slowk[26:-2] <= 20)
        & (slowd[26:-2] <= 20)
        & (slowk[25:-3] <= slowd[25:-3])
        & ((slowk[27:-1] > 20) | (slowk[28:] > 20))
        & ((slowd[27:-1] > 20) | (slowd[28:] > 20))
    )
    buy_idx = np.where(cond_buy)[0] + 27  # +27 aligns with i+1 in your loop

    # Sell: %K crosses below %D, both from above 80, and both move below 80 in next 2 bars
    cond_sell = (
        (slowk[26:-2] <= slowd[26:-2])
        & (slowk[26:-2] >= 80)
        & (slowd[26:-2] >= 80)
        & (slowk[25:-3] >= slowd[25:-3])
        & ((slowk[27:-1] < 80) | (slowk[28:] < 80))
        & ((slowd[27:-1] < 80) | (slowd[28:] < 80))
    )
    sell_idx = np.where(cond_sell)[0] + 27  # +27 aligns with i+1 in your loop

    df.loc[df.index[buy_idx], "stoch_crossover_buy_signals"] = close[buy_idx]
    df.loc[df.index[sell_idx], "stoch_crossover_sell_signals"] = close[sell_idx]

    if not inplace:
        return df
    return None


def rsi_crossover(df: pd.DataFrame, inplace: bool = True):
    if not inplace:
        df = df.copy()

    # Tính RSI, n = 14
    df["rsi"] = ta.RSI(df["close"].to_numpy(dtype="float64"))
    # Tạo cột
    df["rsi_crossover_buy_signals"] = np.nan
    df["rsi_crossover_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    close = df["close"].values
    rsi = df["rsi"].values

    buy_idx = np.where((rsi[:-1] < 30) & (rsi[1:] >= 30))[0] + 1
    df.loc[df.index[buy_idx], "rsi_crossover_buy_signals"] = close[buy_idx]

    sell_idx = np.where((rsi[:-1] > 70) & (rsi[1:] <= 70))[0] + 1
    df.loc[df.index[sell_idx], "rsi_crossover_sell_signals"] = close[sell_idx]

    if not inplace:
        return df
    return None


def macd_crossover(df: pd.DataFrame, inplace: bool = True):
    if not inplace:
        df = df.copy()

    # Tính CCI(14)
    df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(
        df["close"].to_numpy("float64")
    )
    df["rsi"] = ta.RSI(df["close"].to_numpy("float64"))
    # Tạo cột
    df["macd_crossover_buy_signals"] = np.nan
    df["macd_crossover_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    close = df["close"].values
    hist = df["macdhist"].values
    rsi = df["rsi"].values

    cond_rsi = (rsi[1:] <= 45) | (rsi[1:] >= 55)
    cond_buy = (hist[:-1] < 0) & (hist[1:] >= 0) & cond_rsi
    buy_idx = np.where(cond_buy)[0] + 1

    cond_sell = (hist[:-1] > 0) & (hist[1:] <= 0) & cond_rsi
    sell_idx = np.where(cond_sell)[0] + 1

    df.loc[df.index[buy_idx], "macd_crossover_buy_signals"] = close[buy_idx]
    df.loc[df.index[sell_idx], "macd_crossover_sell_signals"] = close[sell_idx]

    if not inplace:
        return df
    return None


def cci_crossover(df: pd.DataFrame, inplace: bool = True):
    if not inplace:
        df = df.copy()

    # Tính CCI(14)
    df["cci"] = ta.CCI(
        df["high"].to_numpy(dtype="float64"),
        df["low"].to_numpy(dtype="float64"),
        df["close"].to_numpy(dtype="float64"),
    )
    # Tạo cột
    df["cci_crossover_buy_signals"] = np.nan
    df["cci_crossover_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    close = df["close"].values
    cci = df["cci"].values

    buy_idx = np.where((cci[:-1] < -100) & (cci[1:] >= -100))[0] + 1
    df.loc[df.index[buy_idx], "cci_crossover_buy_signals"] = close[buy_idx]

    sell_idx = np.where((cci[:-1] > 100) & (cci[1:] <= 100))[0] + 1
    df.loc[df.index[sell_idx], "cci_crossover_sell_signals"] = close[sell_idx]

    if not inplace:
        return df
    return None


def divergence_signals(
    df: pd.DataFrame,
    indicator: str = "macd",
    extrema_order: int = 10,
    lookback_window: int = 45,
    max_idx_dist: int = 4,
    wait: int = 3,
    inplace: bool = True,
):
    if not inplace:
        df = df.copy()
    logging.info("Tìm các điểm phân kỳ")
    if indicator == "macd":
        logging.info("Indicator: MACD(12, 26, 9)")
        df[f"{indicator}"], df["macdsignal"], df["macdhist"] = ta.MACD(
            df["close"].to_numpy(dtype="float64")
        )
    elif indicator == "cci":
        logging.info("Indicator: CCI(14)")
        df[f"{indicator}"] = ta.CCI(
            df["high"].to_numpy(dtype="float64"),
            df["low"].to_numpy(dtype="float64"),
            df["close"].to_numpy(dtype="float64"),
        )
    elif indicator == "rsi":
        logging.info("Indicator: RSI(14)")
        df[f"{indicator}"] = ta.RSI(df["close"].to_numpy(dtype="float64"))
    elif indicator == "stoch":
        logging.info("Indicator: Stoch(14)")
        df[f"{indicator}"], df["slowd"] = ta.STOCH(
            df["high"].to_numpy(dtype="float64"),
            df["low"].to_numpy(dtype="float64"),
            df["close"].to_numpy(dtype="float64"),
            fastk_period=14,
        )

    # Tạo cột
    df[f"{indicator}_divergence_buy_signals"] = np.nan
    df[f"{indicator}_divergence_sell_signals"] = np.nan

    df["price_low"] = np.nan
    df[f"{indicator}_low"] = np.nan

    df["price_high"] = np.nan
    df[f"{indicator}_high"] = np.nan

    # Lưu index của các cặp đáy/đỉnh của giá và chỉ số để vẽ đồ thị (mplfinance)
    df["p1"] = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")
    df["p2"] = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")
    df["i1"] = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")
    df["i2"] = pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns]")

    # Tạo các mảng NumPy để tính toán nhanh hơn (không nhiều)
    low = df["low"].values
    high = df["high"].values
    close = df["close"].values
    ind = df[indicator].values

    buy_signal_indices = []
    buy_signal_values = []
    p1_list = []
    p2_list = []
    i1_list = []
    i2_list = []

    # Tính cực tiểu
    price_low_idxs = argrelextrema(low, np.less, order=extrema_order)[0]
    ind_low_idxs = argrelextrema(ind, np.less, order=extrema_order)[0]

    # Tính cực đại
    price_high_idxs = argrelextrema(high, np.greater, order=extrema_order)[0]
    ind_high_idxs = argrelextrema(ind, np.greater, order=extrema_order)[0]

    # Chép giá trị cực tiểu
    df.loc[df.index[price_low_idxs], "price_low"] = low[price_low_idxs]
    df.loc[df.index[ind_low_idxs], f"{indicator}_low"] = ind[ind_low_idxs]

    # Chép giá trị cực đại
    df.loc[df.index[price_high_idxs], "price_high"] = high[price_high_idxs]
    df.loc[df.index[ind_high_idxs], f"{indicator}_high"] = ind[ind_high_idxs]

    # Xét từng đáy của chỉ số
    logging.info("Phân kỳ tăng - Bullish Divergence:")
    for i in range(1, len(ind_low_idxs)):
        current_ind_low_idx = ind_low_idxs[i]
        previous_ind_low_idx = ind_low_idxs[i - 1]
        if current_ind_low_idx - previous_ind_low_idx <= lookback_window:
            # Tìm đáy giá gần đáy của chỉ số
            ip = bisect.bisect_right(price_low_idxs, current_ind_low_idx + max_idx_dist)
            if ip <= 0:
                continue
            price2_idx = price_low_idxs[ip - 1]
            ip = bisect.bisect_right(
                price_low_idxs, previous_ind_low_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price1_idx = price_low_idxs[ip - 1]
            if (
                current_ind_low_idx - price1_idx <= lookback_window
                and abs(current_ind_low_idx - price2_idx) <= max_idx_dist
                and abs(previous_ind_low_idx - price1_idx) <= max_idx_dist
            ):
                pre_mh = ind[previous_ind_low_idx]
                cur_mh = ind[current_ind_low_idx]
                price1 = low[price1_idx]
                price2 = low[price2_idx]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh < cur_mh else "hidden"
                    logging.info(f"\tType: {divergence_type}")
                    logging.info(
                        f"\t\tPrice lows: \t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    logging.info(
                        f"\t\t{indicator} lows:\t{pre_mh:.2f} at index {previous_ind_low_idx},\t{cur_mh:.2f} at index {current_ind_low_idx}"
                    )
                    if current_ind_low_idx + wait < len(df):
                        buy_signal_indices.append(current_ind_low_idx + wait)
                        buy_signal_values.append(close[current_ind_low_idx + wait])
                        logging.info(
                            f"\t\tAdded buy signals at index {current_ind_low_idx + wait}"
                        )
                        p1_list.append(price1_idx)
                        p2_list.append(price2_idx)
                        i1_list.append(previous_ind_low_idx)
                        i2_list.append(current_ind_low_idx)

    df.loc[df.index[buy_signal_indices], f"{indicator}_divergence_buy_signals"] = (
        buy_signal_values
    )
    df.loc[df.index[buy_signal_indices], "p1"] = df.index[p1_list].tolist()
    df.loc[df.index[buy_signal_indices], "p2"] = df.index[p2_list].tolist()
    df.loc[df.index[buy_signal_indices], "i1"] = df.index[i1_list].tolist()
    df.loc[df.index[buy_signal_indices], "i2"] = df.index[i2_list].tolist()

    sell_signal_indices = []
    sell_signal_values = []
    p1_list = []
    p2_list = []
    i1_list = []
    i2_list = []

    # Xét từng đỉnh của chỉ số
    logging.info("Phân kỳ giảm - Bearish Divergence")
    for i in range(1, len(ind_high_idxs)):
        current_ind_high_idx = ind_high_idxs[i]
        previous_ind_high_idx = ind_high_idxs[i - 1]
        if current_ind_high_idx - previous_ind_high_idx <= lookback_window:
            # Tìm đỉnh giá gần đỉnh của chỉ số
            ip = bisect.bisect_right(
                price_high_idxs, current_ind_high_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price2_idx = price_high_idxs[ip - 1]
            ip = bisect.bisect_right(
                price_high_idxs, previous_ind_high_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price1_idx = price_high_idxs[ip - 1]
            if (
                current_ind_high_idx - price1_idx <= lookback_window
                and abs(current_ind_high_idx - price2_idx) <= max_idx_dist
                and abs(previous_ind_high_idx - price1_idx) <= max_idx_dist
            ):
                pre_mh = ind[previous_ind_high_idx]
                cur_mh = ind[current_ind_high_idx]
                price1 = high[price1_idx]
                price2 = high[price2_idx]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh > cur_mh else "hidden"
                    logging.info(f"\tType: {divergence_type}")
                    logging.info(
                        f"\t\tPrice highs:\t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    logging.info(
                        f"\t\t{indicator} highs:\t{pre_mh:.2f} at index {previous_ind_high_idx},\t{cur_mh:.2f} at index {current_ind_high_idx}"
                    )

                    if current_ind_high_idx + wait < len(df):
                        sell_signal_indices.append(current_ind_high_idx + wait)
                        sell_signal_values.append(close[current_ind_high_idx + wait])
                        logging.info(
                            f"\t\tAdded sell signals at index {current_ind_high_idx + wait}"
                        )
                        p1_list.append(price1_idx)
                        p2_list.append(price2_idx)
                        i1_list.append(previous_ind_high_idx)
                        i2_list.append(current_ind_high_idx)

    df.loc[df.index[sell_signal_indices], f"{indicator}_divergence_sell_signals"] = (
        sell_signal_values
    )
    df.loc[df.index[sell_signal_indices], "p1"] = df.index[p1_list].tolist()
    df.loc[df.index[sell_signal_indices], "p2"] = df.index[p2_list].tolist()
    df.loc[df.index[sell_signal_indices], "i1"] = df.index[i1_list].tolist()
    df.loc[df.index[sell_signal_indices], "i2"] = df.index[i2_list].tolist()

    if not inplace:
        return df
    return None


def macd_divergence(df):
    return divergence_signals(df, indicator="macd")


def rsi_divergence(df):
    return divergence_signals(df, indicator="rsi")


def cci_divergence(df):
    return divergence_signals(df, indicator="cci")


def stoch_divergence(df):
    return divergence_signals(df, indicator="stoch")


def bns1(
    df: pd.DataFrame, filepath: str, signals: list[str] = ["rsi_crossover"]
) -> list:
    symbol = os.path.splitext(os.path.basename(filepath))[0]
    logging.info(f"Mua bán mã CK {symbol} với danh sách tín hiệu {signals}")
    PnL = []
    stocks_bought = 0
    total_buy_price = 0

    df["buy_mark"] = np.nan
    df["sell_mark"] = np.nan

    all_funcs = {
        "rsi_crossover": rsi_crossover,
        "macd_crossover": macd_crossover,
        "cci_crossover": cci_crossover,
        "stoch_crossover": stoch_crossover,
        "macd_divergence": macd_divergence,
        "rsi_divergence": rsi_divergence,
        "cci_divergence": cci_divergence,
        "stoch_divergence": stoch_divergence,
    }

    buy_signals_columns = []
    sell_signals_columns = []

    for signal in signals:
        all_funcs[signal](df)
        buy_signals_columns.append(df.columns.get_loc(f"{signal}_buy_signals"))
        sell_signals_columns.append(df.columns.get_loc(f"{signal}_sell_signals"))

    for i in range(36, len(df)):
        buy_signals = {
            df.columns[col]: pd.notna(df.iat[i, col]) for col in buy_signals_columns
        }
        sell_signals = {
            df.columns[col]: pd.notna(df.iat[i, col]) for col in sell_signals_columns
        }
        buy_vote = sum(buy_signals.values())
        sell_vote = sum(sell_signals.values())
        if buy_vote > 0 or sell_vote > 0:
            # logging.info(f"{buy_vote} / {sell_vote}")
            pass
        if buy_vote > 0:
            logging.info(
                f"{symbol}: {i}: {df.index[i]}: Mua 1 phiếu giá {df['close'].iloc[i]}"
            )
            stocks_bought += 1
            total_buy_price += df["close"].iloc[i]
            df.loc[df.index[i], "buy_mark"] = df.loc[df.index[i], "close"]
        elif sell_vote > 0 and stocks_bought > 0:
            logging.info(
                f"{symbol}: {i}: {df.index[i]}: Bán {stocks_bought} phiếu, mỗi phiếu giá {df['close'].iloc[i]}"
            )
            sell_price = df["close"].iloc[i] * stocks_bought
            # if total_buy_price == 0:
            #     print(f"Divide by zero in file {filepath}")
            PnL.append((sell_price - total_buy_price) * 100 / total_buy_price)
            logging.info(
                f"{symbol}: {i}: PnL = {((sell_price - total_buy_price) * 100 / total_buy_price):.2f}%"
            )
            total_buy_price = 0
            stocks_bought = 0
            df.loc[df.index[i], "sell_mark"] = df.loc[df.index[i], "close"]

    return PnL


def b1s1(
    df: pd.DataFrame,
    filepath: str,
    signals: list[str] = ["rsi_crossover"],
    vote_required=1,
    tpsl: tuple[float, float] = None
) -> dict:
    symbol = os.path.splitext(os.path.basename(filepath))[0]
    logging.info(f"Mua bán xen kẽ mã CK {symbol} với danh sách tín hiệu {signals}")
    PnL = []
    stocks_bought = 0
    total_buy_price = 0
    _return = 1.0

    df["buy_mark"] = np.nan
    df["sell_mark"] = np.nan

    all_funcs = {
        "rsi_crossover": rsi_crossover,
        "macd_crossover": macd_crossover,
        "cci_crossover": cci_crossover,
        "stoch_crossover": stoch_crossover,
        "macd_divergence": macd_divergence,
        "rsi_divergence": rsi_divergence,
        "cci_divergence": cci_divergence,
        "stoch_divergence": stoch_divergence,
    }

    buy_signals_columns = []
    sell_signals_columns = []

    for signal in signals:
        all_funcs[signal](df)
        buy_signals_columns.append(df.columns.get_loc(f"{signal}_buy_signals"))
        sell_signals_columns.append(df.columns.get_loc(f"{signal}_sell_signals"))

    first_buy = True
    first_buy_date = df.index[0]
    last_sell_date = df.index[0]

    for i in range(36, len(df)):
        buy_signals = {
            df.columns[col]: pd.notna(df.iat[i, col]) for col in buy_signals_columns
        }
        sell_signals = {
            df.columns[col]: pd.notna(df.iat[i, col]) for col in sell_signals_columns
        }

        buy_vote = sum(buy_signals.values())
        sell_vote = sum(sell_signals.values())
        if buy_vote >= vote_required or sell_vote >= vote_required:
            # logging.info(f"{buy_vote} / {sell_vote}")
            pass
        if buy_vote >= vote_required and stocks_bought == 0:
            logging.info(
                f"{symbol}: {i}: {df.index[i]}: Mua 1 phiếu giá {df['close'].iloc[i]}"
            )
            stocks_bought = 1
            total_buy_price = df["close"].iloc[i]
            df.loc[df.index[i], "buy_mark"] = df.loc[df.index[i], "close"]
            if first_buy:
                first_buy_date = df.index[i]
                first_buy = False
        elif sell_vote >= vote_required and stocks_bought > 0:
            logging.info(
                f"{symbol}: {i}: {df.index[i]}: Bán {stocks_bought} phiếu, mỗi phiếu giá {df['close'].iloc[i]}"
            )
            sell_price = df["close"].iloc[i] * stocks_bought
            # if total_buy_price == 0:
            #     print(f"Divide by zero in file {filepath}")
            PnL.append((sell_price - total_buy_price) * 100 / total_buy_price)
            logging.info(
                f"{symbol}: {i}: PnL = {((sell_price - total_buy_price) * 100 / total_buy_price):.2f}%"
            )
            _return *= sell_price / total_buy_price
            total_buy_price = 0
            stocks_bought = 0
            df.loc[df.index[i], "sell_mark"] = df.loc[df.index[i], "close"]
            last_sell_date = df.index[i]
        elif tpsl is not None and stocks_bought > 0 and \
            ((df["close"].iloc[i]-total_buy_price)/total_buy_price >= tpsl[0] or\
            (df["close"].iloc[i]-total_buy_price)/total_buy_price <= -tpsl[1]):
            logging.info(
                f"{symbol}: {i}: {df.index[i]}: Bán {stocks_bought} phiếu, mỗi phiếu giá {df['close'].iloc[i]}, lí do: TPSL"
            )
            sell_price = df["close"].iloc[i] * stocks_bought
            PnL.append((sell_price - total_buy_price) * 100 / total_buy_price)
            logging.info(
                f"{symbol}: {i}: PnL = {((sell_price - total_buy_price) * 100 / total_buy_price):.2f}%"
            )
            _return *= sell_price / total_buy_price
            total_buy_price = 0
            stocks_bought = 0
            df.loc[df.index[i], "sell_mark"] = df.loc[df.index[i], "close"]
            last_sell_date = df.index[i]
    return {
        "PnL": PnL,
        "return": _return,
        "return_per_trade": _return ** (1 / len(PnL)) if len(PnL) > 0 else 0,
        "first_buy_date": first_buy_date,
        "last_sell_date": last_sell_date,
    }


if __name__ == "__main__":
    filepath = "stock_data/VNM.csv"
    sample_df = pd.read_csv(filepath, index_col="time")
    sample_df = sample_df[(sample_df["volume"] > 0) & (sample_df["close"] > 0)].copy()
    results = b1s1(sample_df, filepath, ["macd_divergence"], 1, (0.5, 0.25))
    result_reader.read_PnL_results(results["PnL"])
    print(f"{'Cumulative return:':<15}{(results['return']-1) * 100:.2f}%")
    print(f"Return per trade: {results['return_per_trade']:.2f}")
    print(f"{results['first_buy_date']}, {results['last_sell_date']}")

    results = b1s1(sample_df, filepath, ["macd_crossover"], 1)
    result_reader.read_PnL_results(results["PnL"])
    print(f"{'Cumulative return:':<15}{(results['return']-1) * 100:.2f}%")
    print(f"Return per trade: {results['return_per_trade']:.2f}")
    print(f"{results['first_buy_date']}, {results['last_sell_date']}")

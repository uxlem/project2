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
    filename="strat_v4.log",
    filemode='w',
    level=logging.INFO,  # Can also use DEBUG, WARNING, etc.
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

    col_close = df.columns.get_loc("close")
    col_slowk = df.columns.get_loc("slowk")
    col_slowd = df.columns.get_loc("slowd")

    for i in range(26, len(df) - 2):
        # Mua: %K cắt lên trên %D && %K và %D cùng từ dưới 20 (vùng quá bán) cắt lên
        if (
            20 >= df.iat[i, col_slowk] >= df.iat[i, col_slowd]
            and df.iat[i - 1, col_slowk] <= df.iat[i - 1, col_slowd]
            and (df.iat[i + 1, col_slowk] > 20 or df.iat[i + 2, col_slowk] > 20)
            and (df.iat[i + 1, col_slowd] > 20 or df.iat[i + 2, col_slowd] > 20)
        ):
            df.iat[i + 1, df.columns.get_loc("stoch_crossover_buy_signals")] = df.iat[
                i + 1, col_close
            ]
        # Bán: %K cắt xuống dưới %D && %K và %D cùng từ trên 80 (vùng quá mua) cắt xuống
        if (
            80 <= df.iat[i, col_slowk] <= df.iat[i, col_slowd]
            and df.iat[i - 1, col_slowk] >= df.iat[i - 1, col_slowd]
            and (df.iat[i + 1, col_slowk] < 80 or df.iat[i + 2, col_slowk] < 80)
            and (df.iat[i + 1, col_slowd] < 80 or df.iat[i + 2, col_slowd] < 80)
        ):
            df.iat[i + 1, df.columns.get_loc("stoch_crossover_sell_signals")] = df.iat[
                i + 1, col_close
            ]

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

    col_close = df.columns.get_loc("close")
    col_rsi = df.columns.get_loc("rsi")

    for i in range(26, len(df) - 2):
        if df.iat[i, col_rsi] < 30 <= df.iat[i + 1, col_rsi]:
            df.iat[i + 1, df.columns.get_loc("rsi_crossover_buy_signals")] = df.iat[
                i + 1, col_close
            ]

        if df.iat[i, col_rsi] > 70 >= df.iat[i + 1, col_rsi]:
            df.iat[i + 1, df.columns.get_loc("rsi_crossover_sell_signals")] = df.iat[
                i + 1, col_close
            ]

    if not inplace:
        return df
    return None


def macd_crossover(df: pd.DataFrame, inplace: bool = True):
    if not inplace:
        df = df.copy()

    # Tính CCI(14)
    df["macd"], df["macdsignal"], df["macdhist"] = ta.MACD(df["close"].to_numpy("float64"))
    df["rsi"] = ta.RSI(df["close"].to_numpy("float64"))
    # Tạo cột
    df["macd_crossover_buy_signals"] = np.nan
    df["macd_crossover_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    col_close = df.columns.get_loc("close")
    col_hist = df.columns.get_loc("macdhist")
    col_rsi = df.columns.get_loc("rsi")

    for i in range(26, len(df) - 2):
        if df.iat[i, col_hist] < 0 <= df.iat[i + 1, col_hist] and not (
            45 < df.iat[i+1, col_rsi] < 55
        ):
            df.iat[i + 1, df.columns.get_loc("macd_crossover_buy_signals")] = df.iat[
                i + 1, col_close
            ]
            # logging.info(f"mua {i+1}")

        if df.iat[i, col_hist] > 0 >= df.iat[i + 1, col_hist] and not (
            45 < df.iat[i+1, col_rsi] < 55
        ):
            df.iat[i + 1, df.columns.get_loc("macd_crossover_sell_signals")] = df.iat[
                i + 1, col_close
            ]
            # logging.info(f"bán {i+1}")
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

    col_close = df.columns.get_loc("close")
    col_cci = df.columns.get_loc("cci")

    for i in range(26, len(df) - 2):
        if df.iat[i, col_cci] < -100 <= df.iat[i + 1, col_cci]:
            df.iat[i + 1, df.columns.get_loc("cci_crossover_buy_signals")] = df.iat[
                i + 1, col_close
            ]
            # logging.info(f"mua {i+1}")

        if df.iat[i, col_cci] > 100 >= df.iat[i + 1, col_cci]:
            df.iat[i + 1, df.columns.get_loc("cci_crossover_sell_signals")] = df.iat[
                i + 1, col_close
            ]
            # logging.info(f"bán {i+1}")
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
        df[f"{indicator}"], _, _ = ta.MACD(df["close"].to_numpy(dtype="float64"))
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
        df[f"{indicator}"], _ = ta.STOCH(
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
    df["p1"] = np.nan
    df["p2"] = np.nan
    df["i1"] = np.nan
    df["i2"] = np.nan

    # Tính cực tiểu
    price_low_idxs = argrelextrema(df["low"].values, np.less, order=extrema_order)[0]
    ind_low_idxs = argrelextrema(df[indicator].values, np.less, order=extrema_order)[0]

    # Tính cực đại
    price_high_idxs = argrelextrema(df["high"].values, np.greater, order=extrema_order)[
        0
    ]
    ind_high_idxs = argrelextrema(
        df[indicator].values, np.greater, order=extrema_order
    )[0]

    # Chép giá trị cực tiểu
    df.iloc[price_low_idxs, df.columns.get_loc("price_low")] = df.iloc[
        price_low_idxs, df.columns.get_loc("low")
    ].values
    df.iloc[ind_low_idxs, df.columns.get_loc(f"{indicator}_low")] = df.iloc[
        ind_low_idxs, df.columns.get_loc(indicator)
    ].values

    # Chép giá trị cực đại
    df.iloc[price_high_idxs, df.columns.get_loc("price_high")] = df.iloc[
        price_high_idxs, df.columns.get_loc("high")
    ].values
    df.iloc[ind_high_idxs, df.columns.get_loc(f"{indicator}_high")] = df.iloc[
        ind_high_idxs, df.columns.get_loc(indicator)
    ].values

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
                pre_mh = df.iat[
                    previous_ind_low_idx, df.columns.get_loc(f"{indicator}_low")
                ]
                cur_mh = df.iat[
                    current_ind_low_idx, df.columns.get_loc(f"{indicator}_low")
                ]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_low")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_low")]
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
                        df.iat[
                            current_ind_low_idx + wait,
                            df.columns.get_loc(f"{indicator}_divergence_buy_signals"),
                        ] = df.iat[current_ind_low_idx + wait, df.columns.get_loc("close")]
                        logging.info(
                            f"\t\tAdded buy signals at index {current_ind_low_idx + wait}"
                        )
                        df.iat[current_ind_low_idx + wait, df.columns.get_loc("p1")] = (
                            price1_idx
                        )
                        df.iat[current_ind_low_idx + wait, df.columns.get_loc("p2")] = (
                            price2_idx
                        )
                        df.iat[current_ind_low_idx + wait, df.columns.get_loc("i1")] = (
                            previous_ind_low_idx
                        )
                        df.iat[current_ind_low_idx + wait, df.columns.get_loc("i2")] = (
                            current_ind_low_idx
                        )

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
                pre_mh = df.iat[
                    previous_ind_high_idx, df.columns.get_loc(f"{indicator}_high")
                ]
                cur_mh = df.iat[
                    current_ind_high_idx, df.columns.get_loc(f"{indicator}_high")
                ]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_high")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_high")]
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
                        df.iat[
                            current_ind_high_idx + wait,
                            df.columns.get_loc(f"{indicator}_divergence_sell_signals"),
                        ] = df.iat[current_ind_high_idx + wait, df.columns.get_loc("close")]
                        logging.info(
                            f"\t\tAdded sell signals at index {current_ind_high_idx + wait}"
                        )
                        df.iat[
                            current_ind_high_idx + wait, df.columns.get_loc("i1")
                        ] = previous_ind_high_idx
                        df.iat[
                            current_ind_high_idx + wait, df.columns.get_loc("p1")
                        ] = price1_idx
                        df.iat[
                            current_ind_high_idx + wait, df.columns.get_loc("i2")
                        ] = current_ind_high_idx
                        df.iat[
                            current_ind_high_idx + wait, df.columns.get_loc("p2")
                        ] = price2_idx

    df.to_csv("divergence.csv")
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

def strategy(df: pd.DataFrame, filepath:str, signals: list[str] = ["rsi_crossover"]) -> list:
    
    symbol = os.path.splitext(os.path.basename(filepath))[0]
    logging.info(f"Mua bán mã CK {symbol} với danh sách tín hiệu {signals}")
    PnL = []
    stocks_bought = 0
    total_buy_price = 0
    
    df = df[(df['volume'] > 0) & (df['close'] > 0)].copy()
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
            logging.info(f"{symbol}: {i}: {df.index[i]}: Mua 1 phiếu giá {df['close'].iloc[i]}")
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
    
    # df.to_csv("strat_v4_df_mark.csv")
    return PnL


if __name__ == "__main__":
    filepath = "stock_data/AAA.csv"
    sample_df = pd.read_csv(filepath, index_col="time")

    results = strategy(df=sample_df, signals=["macd_divergence"], filepath=filepath) 

    result_reader.read_PnL_results(results)

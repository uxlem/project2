import pandas as pd
import numpy as np
import talib as ta
from scipy.signal import argrelextrema
import bisect


def stoch_crossover(df: pd.DataFrame, inplace: bool = True):
    """Tính tín hiệu mua bán bằng Stochastic crossover cơ bản.
    
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
        df["high"], df["low"], df["close"], fastk_period=14
    )
    # Tạo cột
    df["stoch_buy_signals"] = np.nan
    df["stoch_sell_signals"] = np.nan

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
            df.iat[i + 1, df.columns.get_loc("stoch_buy_signals")] = df.iat[
                i + 1, col_close
            ]
            print(f"mua {i + 1}")
        # Bán: %K cắt xuống dưới %D && %K và %D cùng từ trên 80 (vùng quá mua) cắt xuống
        if (
            80 <= df.iat[i, col_slowk] <= df.iat[i, col_slowd]
            and df.iat[i - 1, col_slowk] >= df.iat[i - 1, col_slowd]
            and (df.iat[i + 1, col_slowk] < 80 or df.iat[i + 2, col_slowk] < 80)
            and (df.iat[i + 1, col_slowd] < 80 or df.iat[i + 2, col_slowd] < 80)
        ):
            df.iat[i + 1, df.columns.get_loc("stoch_sell_signals")] = df.iat[
                i + 1, col_close
            ]
            print(f"bán {i + 1}")
    if not inplace:
        return df
    return None


def rsi_crossover(df: pd.DataFrame, inplace: bool = True):
    if not inplace:
        df = df.copy()

    # Tính RSI, n = 14
    df["rsi"] = ta.RSI(df["close"])
    # Tạo cột
    df["rsi_buy_signals"] = np.nan
    df["rsi_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    col_close = df.columns.get_loc("close")
    col_rsi = df.columns.get_loc("rsi")

    for i in range(26, len(df) - 2):
        # Mua: %K cắt lên trên %D && %K và %D cùng từ dưới 20 (vùng quá bán) cắt lên
        if df.iat[i, col_rsi] < 30 <= df.iat[i + 1, col_rsi]:
            df.iat[i + 1, df.columns.get_loc("rsi_buy_signals")] = df.iat[
                i + 1, col_close
            ]
            print(f"mua {i + 1}")
        # Bán: %K cắt xuống dưới %D && %K và %D cùng từ trên 80 (vùng quá mua) cắt xuống
        if df.iat[i, col_rsi] > 70 >= df.iat[i + 1, col_rsi]:
            df.iat[i + 1, df.columns.get_loc("rsi_sell_signals")] = df.iat[
                i + 1, col_close
            ]
            print(f"bán {i + 1}")
    if not inplace:
        return df
    return None


def macd_crossover(df: pd.DataFrame, inplace: bool = True):
    if not inplace:
        df = df.copy()

    # Tính CCI(14)
    df["macd"], _, _ = ta.MACD(df["close"])
    df["rsi"] = ta.RSI(df["close"])
    # Tạo cột
    df["macd_buy_signals"] = np.nan
    df["macd_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    col_close = df.columns.get_loc("close")
    col_hist = df.columns.get_loc("macdhist")
    col_rsi = df.columns.get_loc("rsi")

    for i in range(26, len(df) - 2):
        if df.iat[i, col_hist] < 0 <= df.iat[i + 1, col_hist] and not (
            45 < df.iat[i, col_rsi] < 55
        ):
            df.iat[i + 1, df.columns.get_loc("macd_buy_signals")] = df.iat[
                i + 1, col_close
            ]
            # print(f"mua {i+1}")

        if df.iat[i, col_hist] > 0 >= df.iat[i + 1, col_hist] and not (
            45 < df.iat[i, col_rsi] < 55
        ):
            df.iat[i + 1, df.columns.get_loc("macd_sell_signals")] = df.iat[
                i + 1, col_close
            ]
            # print(f"bán {i+1}")
    if not inplace:
        return df
    return None


def cci_crossover(df: pd.DataFrame, inplace: bool = True):
    if not inplace:
        df = df.copy()

    # Tính CCI(14)
    df["cci"] = ta.CCI(df["high"], df["low"], df["close"])
    # Tạo cột
    df["cci_buy_signals"] = np.nan
    df["cci_sell_signals"] = np.nan

    # Xét tín hiệu thông thường - Crossovers

    col_close = df.columns.get_loc("close")
    col_cci = df.columns.get_loc("cci")

    for i in range(26, len(df) - 2):
        if df.iat[i, col_cci] < -100 <= df.iat[i + 1, col_cci]:
            df.iat[i + 1, df.columns.get_loc("cci_buy_signals")] = df.iat[
                i + 1, col_close
            ]
            # print(f"mua {i+1}")

        if df.iat[i, col_cci] > 100 >= df.iat[i + 1, col_cci]:
            df.iat[i + 1, df.columns.get_loc("cci_sell_signals")] = df.iat[
                i + 1, col_close
            ]
            # print(f"bán {i+1}")
    if not inplace:
        return df
    return None


def stoch_divergence_signals(
    df, extrema_order=10, lookback_window=45, max_idx_dist=4, wait=3, inplace=True
):
    if not inplace:
        df = df.copy()

    # Tính Stochastic, n = 14
    df["slowk"], df["slowd"] = ta.STOCH(
        df["high"], df["low"], df["close"], fastk_period=14
    )
    # Tạo cột
    df["stoch_buy_signals"] = np.nan
    df["stoch_sell_signals"] = np.nan

    df["price_low"] = np.nan
    df["slowk_low"] = np.nan

    df["price_high"] = np.nan
    df["slowk_high"] = np.nan

    # Lưu index của các cặp đáy/đỉnh của giá và chỉ số
    df["p1"] = np.nan
    df["p2"] = np.nan
    df["i1"] = np.nan
    df["i2"] = np.nan

    # Tính cực tiểu
    price_low_idxs = argrelextrema(df["low"].values, np.less, order=extrema_order)[0]
    slowk_low_idxs = argrelextrema(df["slowk"].values, np.less, order=extrema_order)[0]

    # Tính cực đại
    price_high_idxs = argrelextrema(df["high"].values, np.greater, order=extrema_order)[
        0
    ]
    slowk_high_idxs = argrelextrema(
        df["slowk"].values, np.greater, order=extrema_order
    )[0]

    # Chép giá trị cực tiểu
    df.iloc[price_low_idxs, df.columns.get_loc("price_low")] = df.iloc[
        price_low_idxs, df.columns.get_loc("low")
    ].values
    df.iloc[slowk_low_idxs, df.columns.get_loc("slowk_low")] = df.iloc[
        slowk_low_idxs, df.columns.get_loc("slowk")
    ].values

    # Chép giá trị cực đại
    df.iloc[price_high_idxs, df.columns.get_loc("price_high")] = df.iloc[
        price_high_idxs, df.columns.get_loc("high")
    ].values
    df.iloc[slowk_high_idxs, df.columns.get_loc("slowk_high")] = df.iloc[
        slowk_high_idxs, df.columns.get_loc("slowk")
    ].values

    # Xét từng đáy của chỉ số
    print("BULLISH DIVERGENCE")
    for i in range(1, len(slowk_low_idxs)):
        current_slowk_low_idx = slowk_low_idxs[i]
        previous_slowk_low_idx = slowk_low_idxs[i - 1]
        if current_slowk_low_idx - previous_slowk_low_idx <= lookback_window:
            # Tìm đáy giá gần đáy của chỉ số
            ip = bisect.bisect_right(
                price_low_idxs, current_slowk_low_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price2_idx = price_low_idxs[ip - 1]
            ip = bisect.bisect_right(
                price_low_idxs, previous_slowk_low_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price1_idx = price_low_idxs[ip - 1]
            if (
                current_slowk_low_idx - price1_idx <= lookback_window
                and abs(current_slowk_low_idx - price2_idx) <= max_idx_dist
                and abs(previous_slowk_low_idx - price1_idx) <= max_idx_dist
            ):
                pre_mh = df.iat[previous_slowk_low_idx, df.columns.get_loc("slowk_low")]
                cur_mh = df.iat[current_slowk_low_idx, df.columns.get_loc("slowk_low")]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_low")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_low")]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh < cur_mh else "hidden"
                    print(f"Type: {divergence_type}")
                    print(
                        f"\tPrice lows: \t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    print(
                        f"\tslowk lows:\t{pre_mh:.2f} at index {previous_slowk_low_idx},\t{cur_mh:.2f} at index {current_slowk_low_idx}"
                    )
                    if current_slowk_low_idx + wait < len(df):
                        df.iat[
                            current_slowk_low_idx + wait,
                            df.columns.get_loc("stoch_buy_signals"),
                        ] = df.iat[i, df.columns.get_loc("close")]
                        print(
                            f"\tAdded sell signals at index {current_slowk_low_idx + wait}"
                        )
                        df.iat[
                            current_slowk_low_idx + wait, df.columns.get_loc("p1")
                        ] = price1_idx
                        df.iat[
                            current_slowk_low_idx + wait, df.columns.get_loc("p2")
                        ] = price2_idx
                        df.iat[
                            current_slowk_low_idx + wait, df.columns.get_loc("i1")
                        ] = previous_slowk_low_idx
                        df.iat[
                            current_slowk_low_idx + wait, df.columns.get_loc("i2")
                        ] = current_slowk_low_idx

    # Xét từng đỉnh của chỉ số
    print("BEARISH DIVERGENCE")
    for i in range(1, len(slowk_high_idxs)):
        current_slowk_high_idx = slowk_high_idxs[i]
        previous_slowk_high_idx = slowk_high_idxs[i - 1]
        if current_slowk_high_idx - previous_slowk_high_idx <= lookback_window:
            # Tìm đỉnh giá gần đỉnh của chỉ số
            ip = bisect.bisect_right(
                price_high_idxs, current_slowk_high_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price2_idx = price_high_idxs[ip - 1]
            ip = bisect.bisect_right(
                price_high_idxs, previous_slowk_high_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price1_idx = price_high_idxs[ip - 1]
            if (
                current_slowk_high_idx - price1_idx <= lookback_window
                and abs(current_slowk_high_idx - price2_idx) <= max_idx_dist
                and abs(previous_slowk_high_idx - price1_idx) <= max_idx_dist
            ):
                pre_mh = df.iat[
                    previous_slowk_high_idx, df.columns.get_loc("slowk_high")
                ]
                cur_mh = df.iat[
                    current_slowk_high_idx, df.columns.get_loc("slowk_high")
                ]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_high")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_high")]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh > cur_mh else "hidden"
                    print(f"Type: {divergence_type}")
                    print(
                        f"\tPrice highs:\t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    print(
                        f"\tslowk highs:\t{pre_mh:.2f} at index {previous_slowk_high_idx},\t{cur_mh:.2f} at index {current_slowk_high_idx}"
                    )

                    if current_slowk_high_idx + wait < len(df):
                        df.iat[
                            current_slowk_high_idx + wait,
                            df.columns.get_loc("stoch_sell_signals"),
                        ] = df.iat[i, df.columns.get_loc("close")]
                        print(
                            f"\tAdded sell signals at index {current_slowk_high_idx + wait}"
                        )
                        df.iat[
                            current_slowk_high_idx + wait, df.columns.get_loc("i1")
                        ] = previous_slowk_high_idx
                        df.iat[
                            current_slowk_high_idx + wait, df.columns.get_loc("p1")
                        ] = price1_idx
                        df.iat[
                            current_slowk_high_idx + wait, df.columns.get_loc("i2")
                        ] = current_slowk_high_idx
                        df.iat[
                            current_slowk_high_idx + wait, df.columns.get_loc("p2")
                        ] = price2_idx

    if not inplace:
        return df
    return None


def rsi_divergence_signals(
    df, extrema_order=10, lookback_window=45, max_idx_dist=4, wait=3, inplace=True
):
    if not inplace:
        df = df.copy()

    df["rsi"] = ta.RSI(df["close"])
    # Tạo cột
    df["rsi_div_buy_signals"] = np.nan
    df["rsi_div_sell_signals"] = np.nan

    df["price_low"] = np.nan
    df["rsi_low"] = np.nan

    df["price_high"] = np.nan
    df["rsi_high"] = np.nan

    # Lưu index của các cặp đáy/đỉnh của giá và chỉ số
    df["p1"] = np.nan
    df["p2"] = np.nan
    df["i1"] = np.nan
    df["i2"] = np.nan

    # Tính cực tiểu
    price_low_idxs = argrelextrema(df["low"].values, np.less, order=extrema_order)[0]
    rsi_low_idxs = argrelextrema(df["rsi"].values, np.less, order=extrema_order)[0]

    # Tính cực đại
    price_high_idxs = argrelextrema(df["high"].values, np.greater, order=extrema_order)[
        0
    ]
    rsi_high_idxs = argrelextrema(df["rsi"].values, np.greater, order=extrema_order)[0]

    # Chép giá trị cực tiểu
    df.iloc[price_low_idxs, df.columns.get_loc("price_low")] = df.iloc[
        price_low_idxs, df.columns.get_loc("low")
    ].values
    df.iloc[rsi_low_idxs, df.columns.get_loc("rsi_low")] = df.iloc[
        rsi_low_idxs, df.columns.get_loc("rsi")
    ].values

    # Chép giá trị cực đại
    df.iloc[price_high_idxs, df.columns.get_loc("price_high")] = df.iloc[
        price_high_idxs, df.columns.get_loc("high")
    ].values
    df.iloc[rsi_high_idxs, df.columns.get_loc("rsi_high")] = df.iloc[
        rsi_high_idxs, df.columns.get_loc("rsi")
    ].values

    # Xét từng đáy của chỉ số
    print("BULLISH DIVERGENCE")
    for i in range(1, len(rsi_low_idxs)):
        current_rsi_low_idx = rsi_low_idxs[i]
        previous_rsi_low_idx = rsi_low_idxs[i - 1]
        if current_rsi_low_idx - previous_rsi_low_idx <= lookback_window:
            # Tìm đáy giá gần đáy của chỉ số
            ip = bisect.bisect_right(price_low_idxs, current_rsi_low_idx + max_idx_dist)
            if ip <= 0:
                continue
            price2_idx = price_low_idxs[ip - 1]
            ip = bisect.bisect_right(
                price_low_idxs, previous_rsi_low_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price1_idx = price_low_idxs[ip - 1]
            if (
                current_rsi_low_idx - price1_idx <= lookback_window
                and abs(current_rsi_low_idx - price2_idx) <= max_idx_dist
                and abs(previous_rsi_low_idx - price1_idx) <= max_idx_dist
            ):
                pre_mh = df.iat[previous_rsi_low_idx, df.columns.get_loc("rsi_low")]
                cur_mh = df.iat[current_rsi_low_idx, df.columns.get_loc("rsi_low")]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_low")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_low")]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh < cur_mh else "hidden"
                    print(f"Type: {divergence_type}")
                    print(
                        f"\tPrice lows: \t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    print(
                        f"\trsi lows:\t{pre_mh:.2f} at index {previous_rsi_low_idx},\t{cur_mh:.2f} at index {current_rsi_low_idx}"
                    )
                    if current_rsi_low_idx + wait < len(df):
                        df.iat[
                            current_rsi_low_idx + wait,
                            df.columns.get_loc("rsi_div_buy_signals"),
                        ] = df.iat[i, df.columns.get_loc("close")]
                        print(
                            f"\tAdded sell signals at index {current_rsi_low_idx + wait}"
                        )
                        df.iat[current_rsi_low_idx + wait, df.columns.get_loc("p1")] = (
                            price1_idx
                        )
                        df.iat[current_rsi_low_idx + wait, df.columns.get_loc("p2")] = (
                            price2_idx
                        )
                        df.iat[current_rsi_low_idx + wait, df.columns.get_loc("i1")] = (
                            previous_rsi_low_idx
                        )
                        df.iat[current_rsi_low_idx + wait, df.columns.get_loc("i2")] = (
                            current_rsi_low_idx
                        )

    # Xét từng đỉnh của chỉ số
    print("BEARISH DIVERGENCE")
    for i in range(1, len(rsi_high_idxs)):
        current_rsi_high_idx = rsi_high_idxs[i]
        previous_rsi_high_idx = rsi_high_idxs[i - 1]
        if current_rsi_high_idx - previous_rsi_high_idx <= lookback_window:
            # Tìm đỉnh giá gần đỉnh của chỉ số
            ip = bisect.bisect_right(
                price_high_idxs, current_rsi_high_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price2_idx = price_high_idxs[ip - 1]
            ip = bisect.bisect_right(
                price_high_idxs, previous_rsi_high_idx + max_idx_dist
            )
            if ip <= 0:
                continue
            price1_idx = price_high_idxs[ip - 1]
            if (
                current_rsi_high_idx - price1_idx <= lookback_window
                and abs(current_rsi_high_idx - price2_idx) <= max_idx_dist
                and abs(previous_rsi_high_idx - price1_idx) <= max_idx_dist
            ):
                pre_mh = df.iat[previous_rsi_high_idx, df.columns.get_loc("rsi_high")]
                cur_mh = df.iat[current_rsi_high_idx, df.columns.get_loc("rsi_high")]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_high")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_high")]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh > cur_mh else "hidden"
                    print(f"Type: {divergence_type}")
                    print(
                        f"\tPrice highs:\t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    print(
                        f"\trsi highs:\t{pre_mh:.2f} at index {previous_rsi_high_idx},\t{cur_mh:.2f} at index {current_rsi_high_idx}"
                    )

                    if current_rsi_high_idx + wait < len(df):
                        df.iat[
                            current_rsi_high_idx + wait,
                            df.columns.get_loc("rsi_div_sell_signals"),
                        ] = df.iat[i, df.columns.get_loc("close")]
                        print(
                            f"\tAdded sell signals at index {current_rsi_high_idx + wait}"
                        )
                        df.iat[
                            current_rsi_high_idx + wait, df.columns.get_loc("i1")
                        ] = previous_rsi_high_idx
                        df.iat[
                            current_rsi_high_idx + wait, df.columns.get_loc("p1")
                        ] = price1_idx
                        df.iat[
                            current_rsi_high_idx + wait, df.columns.get_loc("i2")
                        ] = current_rsi_high_idx
                        df.iat[
                            current_rsi_high_idx + wait, df.columns.get_loc("p2")
                        ] = price2_idx

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

    if indicator == "macd":
        print("Indicator: MACD(12, 26, 9)")
        df["ind"], _, _ = ta.MACD(df["close"])
    elif indicator == "cci":
        print("Indicator: CCI(14)")
        df["ind"] = ta.CCI(df["high"], df["low"], df["close"])
    elif indicator == "rsi":
        print("Indicator: RSI(14)")
        df["ind"] = ta.RSI(df["close"])
    elif indicator == "stoch":
        print("Indicator: Stoch(14)")
        df["ind"], _ = ta.STOCH(df["high"], df["low"], df["close"], fastk_period=14)

    # Tạo cột
    df["ind_div_buy_signals"] = np.nan
    df["ind_div_sell_signals"] = np.nan

    df["price_low"] = np.nan
    df["ind_low"] = np.nan

    df["price_high"] = np.nan
    df["ind_high"] = np.nan

    # Lưu index của các cặp đáy/đỉnh của giá và chỉ số
    df["p1"] = np.nan
    df["p2"] = np.nan
    df["i1"] = np.nan
    df["i2"] = np.nan

    # Tính cực tiểu
    price_low_idxs = argrelextrema(df["low"].values, np.less, order=extrema_order)[0]
    ind_low_idxs = argrelextrema(df["ind"].values, np.less, order=extrema_order)[0]

    # Tính cực đại
    price_high_idxs = argrelextrema(df["high"].values, np.greater, order=extrema_order)[0]
    ind_high_idxs = argrelextrema(df["ind"].values, np.greater, order=extrema_order)[0]

    # Chép giá trị cực tiểu
    df.iloc[price_low_idxs, df.columns.get_loc("price_low")] = df.iloc[
        price_low_idxs, df.columns.get_loc("low")
    ].values
    df.iloc[ind_low_idxs, df.columns.get_loc("ind_low")] = df.iloc[
        ind_low_idxs, df.columns.get_loc("ind")
    ].values

    # Chép giá trị cực đại
    df.iloc[price_high_idxs, df.columns.get_loc("price_high")] = df.iloc[
        price_high_idxs, df.columns.get_loc("high")
    ].values
    df.iloc[ind_high_idxs, df.columns.get_loc("ind_high")] = df.iloc[
        ind_high_idxs, df.columns.get_loc("ind")
    ].values

    # Xét từng đáy của chỉ số
    print("BULLISH DIVERGENCE")
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
                pre_mh = df.iat[previous_ind_low_idx, df.columns.get_loc("ind_low")]
                cur_mh = df.iat[current_ind_low_idx, df.columns.get_loc("ind_low")]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_low")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_low")]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh < cur_mh else "hidden"
                    print(f"Type: {divergence_type}")
                    print(
                        f"\tPrice lows: \t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    print(
                        f"\tind lows:\t{pre_mh:.2f} at index {previous_ind_low_idx},\t{cur_mh:.2f} at index {current_ind_low_idx}"
                    )
                    if current_ind_low_idx + wait < len(df):
                        df.iat[
                            current_ind_low_idx + wait,
                            df.columns.get_loc("ind_div_buy_signals"),
                        ] = df.iat[i, df.columns.get_loc("close")]
                        print(
                            f"\tAdded sell signals at index {current_ind_low_idx + wait}"
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
    print("BEARISH DIVERGENCE")
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
                pre_mh = df.iat[previous_ind_high_idx, df.columns.get_loc("ind_high")]
                cur_mh = df.iat[current_ind_high_idx, df.columns.get_loc("ind_high")]
                price1 = df.iat[price1_idx, df.columns.get_loc("price_high")]
                price2 = df.iat[price2_idx, df.columns.get_loc("price_high")]
                if (pre_mh - cur_mh) * (price1 - price2) < 0:
                    divergence_type = "regular" if pre_mh > cur_mh else "hidden"
                    print(f"Type: {divergence_type}")
                    print(
                        f"\tPrice highs:\t{price1} at index {price1_idx},\t{price2} at index {price2_idx}"
                    )
                    print(
                        f"\tind highs:\t{pre_mh:.2f} at index {previous_ind_high_idx},\t{cur_mh:.2f} at index {current_ind_high_idx}"
                    )

                    if current_ind_high_idx + wait < len(df):
                        df.iat[
                            current_ind_high_idx + wait,
                            df.columns.get_loc("ind_div_sell_signals"),
                        ] = df.iat[i, df.columns.get_loc("close")]
                        print(
                            f"\tAdded sell signals at index {current_ind_high_idx + wait}"
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

    if not inplace:
        return df
    return None


def strategy(df: pd.DataFrame):
    # initial_cash = 10000
    PnL = []
    stocks_bought = 0
    total_buy_price = 0
    df["buy_mark"] = np.nan
    df["sell_mark"] = np.nan

    # rsi_signals(df)
    # rsi_divergence_signals(df)
    # divergence_signals(df, 'macd')
    # stoch_signals(df)
    # stoch_divergence_signals(df)
    macd_crossover(df)
    # cci_signals(df)

    for i in range(36, len(df)):
        buy_signals = {
            # 'stoch': pd.notna(df.iat[i, df.columns.get_loc('stoch_buy_signals')]),
            # "rsi": pd.notna(df.iat[i, df.columns.get_loc('rsi_buy_signals')]),
            # "rsi_div": pd.notna(df.iat[i, df.columns.get_loc('rsi_div_buy_signals')]),
            "macd": pd.notna(df.iat[i, df.columns.get_loc("macd_buy_signals")]),
            # "macd_div": pd.notna(df.iat[i, df.columns.get_loc('ind_div_buy_signals')]),
            # "cci": pd.notna(df.iat[i, df.columns.get_loc('cci_buy_signals')]),
        }

        sell_signals = {
            # 'stoch': pd.notna(df.iat[i, df.columns.get_loc('stoch_sell_signals')]),
            # "rsi": pd.notna(df.iat[i, df.columns.get_loc('rsi_sell_signals')]),
            # "rsi_div": pd.notna(df.iat[i, df.columns.get_loc('rsi_div_sell_signals')]),
            "macd": pd.notna(df.iat[i, df.columns.get_loc("macd_sell_signals")]),
            # "macd_div": pd.notna(df.iat[i, df.columns.get_loc('ind_div_sell_signals')]),
            # "cci": pd.notna(df.iat[i, df.columns.get_loc('cci_sell_signals')]),
        }

        buy_vote = sum(buy_signals.values())
        sell_vote = sum(sell_signals.values())
        if buy_vote > 0 or sell_vote > 0:
            # print(f"{buy_vote} / {sell_vote}")
            pass
        if buy_vote > 0:
            print(f"{i}: {df.index[i]}: Mua 1 phiếu giá {df['close'].iloc[i]}")
            stocks_bought += 1
            total_buy_price += df["close"].iloc[i]
            df.loc[df.index[i], "buy_mark"] = df.loc[df.index[i], "close"]
        elif sell_vote > 0 and stocks_bought > 0:
            print(
                f"{i}: {df.index[i]}: Bán {stocks_bought} phiếu, mỗi phiếu giá {df['close'].iloc[i]}"
            )
            sell_price = df["close"].iloc[i] * stocks_bought
            PnL.append((sell_price - total_buy_price) * 100 / total_buy_price)
            print(
                f"PnL = {((sell_price - total_buy_price) * 100 / total_buy_price):.2f}%"
            )
            total_buy_price = 0
            stocks_bought = 0
            df.loc[df.index[i], "sell_mark"] = df.loc[df.index[i], "close"]

    df.to_csv("strat_v4_df_mark.csv")
    return pd.Series(PnL)


if __name__ == "__main__":
    sample_df = pd.read_csv("precalc_data/BBC.csv", index_col="time")

    results = strategy(sample_df)

    profits = results[results > 0].dropna()
    breakeven = results[results == 0].dropna()
    loss = results[results < 0].dropna()

    for i in profits, breakeven, loss, results:
        if len(i) > 0:
            print(
                f"count = {len(i)} = {len(i) / len(results) * 100:.1f}%, mean = {i.values.mean():.2f}, "
                f"max = {i.values.max():.2f}, min = {i.values.min():.2f}"
            )

import mplfinance as mpf
import pandas as pd
import strat_v4_vectorized as strat
import matplotlib.pyplot as plt
import result_reader
from pathlib import Path

divergence_linestyle = {
    "linestyle": "-",
    "linewidth": 1.5,
    "marker": "o",
    "markersize": 4,
}

trade_signal_linestyle = {
    "alpha": 0.8,
    "linestyle": "--",
    "linewidth": 1.5,
}

buy_signal_linestyle = {**trade_signal_linestyle, "color": "blue"}

sell_signal_linestyle = {**trade_signal_linestyle, "color": "red"}

def plot_divergence_onesignal(df: pd.DataFrame, signal: str, plot_title=""):
    
    plot_title = Path(plot_title).stem
    plot_title = f"{plot_title} - {df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}"
    # Định nghĩa các plot phụ (addplot)
    apd = [
        mpf.make_addplot(
            df["buy_mark"],
            panel=0,
            color="blue",
            type="scatter",
            markersize=200,
            alpha=0.7,
            label="Điểm mua"
        ),
        mpf.make_addplot(
            df["sell_mark"],
            panel=0,
            color="red",
            type="scatter",
            markersize=200,
            alpha=0.7,
            label="Điểm bán"
        ),
    ]

    if signal == "macd_divergence":
        apd.extend(
            [
                mpf.make_addplot(df["macd"], panel=1, color="blue", label="MACD"),
                mpf.make_addplot(
                    df["macdsignal"],
                    panel=1,
                    color="orange",
                    label="Signal",
                    secondary_y=False,
                ),
                mpf.make_addplot(
                    df["macdhist"],
                    type="bar",
                    panel=1,
                    color="gray",
                    alpha=0.5,
                    secondary_y=False,
                ),
            ]
        )
    elif signal == "rsi_divergence":
        apd.extend([mpf.make_addplot(df["rsi"], panel=1, color="blue", label="RSI")])
    elif signal == "cci_divergence":
        apd.extend([mpf.make_addplot(df["rsi"], panel=1, color="blue", label="CCI")])
    elif signal == "stoch_divergence":
        apd.extend(
            [
                mpf.make_addplot(df["slowk"], panel=1, color="blue", label="%K"),
                mpf.make_addplot(
                    df["slowd"], panel=1, color="orange", label="%D", secondary_y=False
                ),
            ]
        )

    # Vẽ biểu đồ và lấy Figure, Axes
    fig, axes = mpf.plot(
        df,
        style="starsandstripes",
        type="candle",
        addplot=apd,
        volume=False,
        panel_ratios=(2, 1),
        figsize=(16, 6),
        figscale=2,
        returnfig=True,
        title=plot_title
    )

    # Xác định các Axes cần vẽ
    ax_price = axes[0]
    ax_macd = axes[2]

    # --- LOGIC VẼ ĐƯỜNG PHÂN KỲ ---
    first_buy = True
    first_sell = True

    # 1. Vẽ phân kỳ tăng (Bullish Divergence)
    buy_divergences = df[df[f"{signal}_buy_signals"].notna() & df["p1"].notna()]
    for _, row in buy_divergences.iterrows():
        # Lấy các chỉ số (index) từ DataFrame. Chuyển thành int.
        p1_idx, p2_idx = df.index.get_loc(row["p1"]), df.index.get_loc(row["p2"])
        i1_idx, i2_idx = df.index.get_loc(row["i1"]), df.index.get_loc(row["i2"])
        print(p1_idx, p2_idx, i1_idx, i2_idx)
        # Vẽ đường phân kỳ trên panel giá (nối các đáy giá)
        price_y = [df["low"].iloc[p1_idx], df["low"].iloc[p2_idx]]
        price_x = [p1_idx, p2_idx]
        ax_price.plot(price_x, price_y, color="lime", **divergence_linestyle, label = "Phân kỳ tăng" if first_buy else "")
        first_buy = False
        # Vẽ đường phân kỳ trên panel MACD (nối các đáy MACD)
        # Bạn đã yêu cầu macdhist_low, nên ta sẽ dùng cột 'macdhist'
        indicator_y = [df["macd"].iloc[i1_idx], df["macd"].iloc[i2_idx]]
        indicator_x = [i1_idx, i2_idx]
        ax_macd.plot(indicator_x, indicator_y, color="lime", **divergence_linestyle)

    # 2. Vẽ phân kỳ giảm (Bearish Divergence)
    sell_divergences = df[df[f"{signal}_sell_signals"].notna() & df["p1"].notna()]
    for _, row in sell_divergences.iterrows():
        # Lấy các chỉ số (index) từ DataFrame. Chuyển thành int.
        p1_idx, p2_idx = df.index.get_loc(row["p1"]), df.index.get_loc(row["p2"])
        i1_idx, i2_idx = df.index.get_loc(row["i1"]), df.index.get_loc(row["i2"])

        # Vẽ đường phân kỳ trên panel giá (nối các đỉnh giá)
        price_y = [df["high"].iloc[p1_idx], df["high"].iloc[p2_idx]]
        price_x = [p1_idx, p2_idx]
        ax_price.plot(price_x, price_y, color="fuchsia", **divergence_linestyle, label = "Phân kỳ giảm" if first_sell else "")
        first_sell = False
        indicator_y = [df["macd"].iloc[i1_idx], df["macd"].iloc[i2_idx]]
        indicator_x = [i1_idx, i2_idx]
        ax_macd.plot(indicator_x, indicator_y, color="fuchsia", **divergence_linestyle)

    # Lấy các ngày có tín hiệu (dưới dạng Timestamps)
    buy_signal_timestamps = df.index[df[f"{signal}_buy_signals"].notna()]
    sell_signal_timestamps = df.index[df[f"{signal}_sell_signals"].notna()]

    # Vẽ đường kẻ tín hiệu mua (màu xanh)
    for date_ts in buy_signal_timestamps:
        try:
            # Lấy vị trí SỐ NGUYÊN của ngày trong DataFrame
            x_coordinate = df.index.get_loc(date_ts)
            ax_price.axvline(
                x_coordinate,
                label="Tín hiệu mua" if date_ts == buy_signal_timestamps[0] else "",
                **buy_signal_linestyle,
            )
            ax_macd.axvline(x_coordinate, **buy_signal_linestyle)
            # print(f"Vẽ tín hiệu mua tại ngày {date_ts} (tọa độ x: {x_coordinate})") # Để debug
        except KeyError:
            print(
                f"Cảnh báo: Ngày {date_ts} của tín hiệu mua không tìm thấy trong index của DataFrame."
            )
        except Exception as e:
            print(f"Lỗi khi vẽ tín hiệu mua tại {date_ts}: {e}")

    # Vẽ đường kẻ tín hiệu bán (màu đỏ)
    for date_ts in sell_signal_timestamps:
        try:
            # Lấy vị trí SỐ NGUYÊN của ngày trong DataFrame
            x_coordinate = df.index.get_loc(date_ts)
            ax_price.axvline(
                x_coordinate,
                label="Tín hiệu bán" if date_ts == sell_signal_timestamps[0] else "",
                **sell_signal_linestyle,
            )
            ax_macd.axvline(x_coordinate, **sell_signal_linestyle)
            # print(f"Vẽ tín hiệu bán tại ngày {date_ts} (tọa độ x: {x_coordinate})") # Để debug
        except KeyError:
            print(
                f"Cảnh báo: Ngày {date_ts} của tín hiệu bán không tìm thấy trong index của DataFrame."
            )
        except Exception as e:
            print(f"Lỗi khi vẽ tín hiệu bán tại {date_ts}: {e}")

    # handles, labels = ax_price.get_legend_handles_labels()
    # if handles:
    #     ax_price.legend(handles=handles, labels=labels)
    ax_price.legend()

    # Hiển thị biểu đồ
    mpf.show()

def plot_crossover(df: pd.DataFrame, indicator: str, plot_title=""):

    plot_title = Path(plot_title).stem
    plot_title = f"{plot_title} - {df.index.min().strftime('%Y-%m-%d')} - {df.index.max().strftime('%Y-%m-%d')}"

    apd = [
        mpf.make_addplot(
            df["buy_mark"],
            panel=0,
            color="blue",
            type="scatter",
            markersize=200,
            alpha=0.8,
            label="Điểm mua"
        ),
        mpf.make_addplot(
            df["sell_mark"],
            panel=0,
            color="red",
            type="scatter",
            markersize=200,
            alpha=0.8,
            label="Điểm bán"
        ),
    ]

    if indicator == "macd":
        apd.extend(
            [
                mpf.make_addplot(df["macd"], panel=1, color="blue", label="MACD"),
                mpf.make_addplot(
                    df["macdsignal"],
                    panel=1,
                    color="orange",
                    label="Signal",
                    secondary_y=False,
                ),
                mpf.make_addplot(
                    df["macdhist"],
                    type="bar",
                    panel=1,
                    color="gray",
                    alpha=0.5,
                    secondary_y=False,
                ),
                mpf.make_addplot(df["rsi"], panel=2, color="black", label="RSI")
            ]
        )
    elif indicator == "rsi":
        apd.extend([mpf.make_addplot(df["rsi"], panel=1, color="blue", label="RSI")])
    elif indicator == "cci":
        apd.extend([mpf.make_addplot(df["cci"], panel=1, color="blue", label="CCI")])
    elif indicator == "stoch":
        apd.extend(
            [
                mpf.make_addplot(df["slowk"], panel=1, color="blue", label="%K"),
                mpf.make_addplot(
                    df["slowd"], panel=1, color="orange", label="%D", secondary_y=False
                ),
            ]
        )

    # Vẽ biểu đồ và lấy Figure, Axes
    fig, axes = mpf.plot(
        df,
        style="starsandstripes",
        type="candle",
        addplot=apd,
        volume=False,
        panel_ratios=(2, 1),
        figsize=(16, 6),
        figscale=2,
        returnfig=True,
        title=plot_title
    )

    print(axes)
    # Xác định các Axes cần vẽ
    ax_price = axes[0]
    ax_macd = axes[2]
    if indicator == "macd":
        ax_rsi = axes[4]

    # Lấy các ngày có tín hiệu (dưới dạng Timestamps)
    buy_signal_timestamps = df.index[df[f"{indicator}_crossover_buy_signals"].notna()]
    sell_signal_timestamps = df.index[df[f"{indicator}_crossover_sell_signals"].notna()]

    # Vẽ đường kẻ tín hiệu mua (màu xanh)
    for date_ts in buy_signal_timestamps:
        try:
            # Lấy vị trí SỐ NGUYÊN của ngày trong DataFrame
            x_coordinate = df.index.get_loc(date_ts)
            ax_price.axvline(
                x_coordinate,
                label="Tín hiệu mua" if date_ts == buy_signal_timestamps[0] else "",
                **buy_signal_linestyle,
            )
            ax_macd.axvline(x_coordinate, **buy_signal_linestyle)
            if indicator == "macd":
                ax_rsi.axvline(x_coordinate, **buy_signal_linestyle)
            # print(f"Vẽ tín hiệu mua tại ngày {date_ts} (tọa độ x: {x_coordinate})") # Để debug
        except KeyError:
            print(
                f"Cảnh báo: Ngày {date_ts} của tín hiệu mua không tìm thấy trong index của DataFrame."
            )
        except Exception as e:
            print(f"Lỗi khi vẽ tín hiệu mua tại {date_ts}: {e}")

    # Vẽ đường kẻ tín hiệu bán (màu đỏ)
    for date_ts in sell_signal_timestamps:
        try:
            # Lấy vị trí SỐ NGUYÊN của ngày trong DataFrame
            x_coordinate = df.index.get_loc(date_ts)
            ax_price.axvline(
                x_coordinate,
                label="Tín hiệu bán" if date_ts == sell_signal_timestamps[0] else "",
                **sell_signal_linestyle,
            )
            ax_macd.axvline(x_coordinate, **sell_signal_linestyle)
            if indicator == "macd":
                ax_rsi.axvline(x_coordinate, **sell_signal_linestyle)
            # print(f"Vẽ tín hiệu bán tại ngày {date_ts} (tọa độ x: {x_coordinate})") # Để debug
        except KeyError:
            print(
                f"Cảnh báo: Ngày {date_ts} của tín hiệu bán không tìm thấy trong index của DataFrame."
            )
        except Exception as e:
            print(f"Lỗi khi vẽ tín hiệu bán tại {date_ts}: {e}")

    hlinestyle = {
        "alpha": 0.8
    }

    if indicator == "macd":
        ax_macd.axhline(y = 0, **hlinestyle)
        ax_rsi.axhline(y = 45, **hlinestyle)
        ax_rsi.axhline(y = 55, **hlinestyle)
    if indicator == "rsi":
        ax_macd.axhline(y = 30, **hlinestyle)
        ax_macd.axhline(y = 70, **hlinestyle)
    if indicator == "cci":
        ax_macd.axhline(y = -100, **hlinestyle)
        ax_macd.axhline(y = 100, **hlinestyle)
    if indicator == "stoch":
        ax_macd.axhline(y = 80, **hlinestyle)
        ax_macd.axhline(y = 20, **hlinestyle)
    # handles, labels = ax_price.get_legend_handles_labels()
    # if handles:
    #     ax_price.legend(handles=handles, labels=labels)
    ax_price.legend()

    # Hiển thị biểu đồ
    mpf.show()


if __name__ == "__main__":
    filepath = "stock_data_VN30/ACB.csv"
    df = strat.load_and_filter_from_csv(filepath)
    signal = "macd_divergence"
    results = strat.bns1(df, filepath, [signal])
    result_reader.read_PnL_results(results)
    plot_divergence_onesignal(df.loc["2008-01-01":"2010-05-01"], signal)

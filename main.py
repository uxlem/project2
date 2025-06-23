import pandas as pd
import numpy as np
import strat_v4_vectorized as strats
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import result_reader
import time

DATA_DIR = "stock_data_VN30"
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
signals = ["macd_divergence", "rsi_divergence", "cci_divergence", "stoch_divergence",
             "macd_crossover", "rsi_crossover", "cci_crossover", "stoch_crossover"]

signals_xover_only = ["macd_crossover", "rsi_crossover", "cci_crossover", "stoch_crossover"]
vnindex = strats.load_and_filter_from_csv("VNINDEX.csv")
start_date = vnindex.index.min()
end_date = vnindex.index.max()

def runStrategyFromCSV(file_path: str, signals: list):
    df = strats.load_and_filter_from_csv(file_path)
    df = df.loc[start_date:end_date].copy()
    test_results = strats.b1s1(df, file_path, signals, 1, (0.5, 0.2))
    # print(f"Done running strategy on {file_path}")
    return test_results

def runAsync():
    columns = [
    "Tín hiệu",
    "Số lần giao dịch",
    "Lãi",
    "profit rate",
    "Hoà vốn",
    "breakeven rate",
    "Lỗ",
    "loss rate",
    "PnL trung bình",
    "Avg cumulative return",
    "Avg VN Index cumulative return",
    "avgalpha",
    "wins against VN Index",
    "winrate against VN Index",
    "avgdaysdiff",
    "returnperyear"
    ]
    table = pd.DataFrame(columns=columns)

    for signal in signals: 
        total_result = []
        _return = []
        wins_against_VN_Index = 0
        sum_VN_Index_returns = 0
        completed_futures = 0
        avg_alpha = 0
        avg_days_diff = 0

        print(f"\n\nThử bất đồng bộ tín hiệu {signal}")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(runStrategyFromCSV, file, [signal]) for file in csv_files]
            for future in as_completed(futures):
                result = future.result()
                total_result.extend(result["PnL"])
                _return.append(result["return"])

                first_entry = pd.to_datetime(result["first_buy_date"])
                last_exit = pd.to_datetime(result["last_sell_date"])

                # Find the closest available date before or equal to first_entry
                first_entry_idx = vnindex.index.get_indexer([first_entry], method='ffill')[0]
                first_entry_actual = vnindex.index[first_entry_idx]

                # Find the closest available date after or equal to last_exit
                last_exit_idx = vnindex.index.get_indexer([last_exit], method='bfill')[0]
                last_exit_actual = vnindex.index[last_exit_idx]

                VN_Index_return = vnindex.at[last_exit_actual, "close"] / vnindex.at[first_entry_actual, "close"]
                sum_VN_Index_returns += VN_Index_return
                avg_alpha += (result["return"]-VN_Index_return)
                avg_days_diff += (last_exit - first_entry).days
                if (result["return"] > VN_Index_return):
                    wins_against_VN_Index += 1
                completed_futures += 1
                # print(f"{completed_futures}")

        avg_alpha = avg_alpha/completed_futures
        avg_days_diff = avg_days_diff/completed_futures
        profit_count, breakeven_count, loss_count, total_count, avg_PnL = result_reader.read_PnL_results(total_result).values()
        # total_result.to_csv("total_results_v4.csv", header=["PnL"])
        _return = pd.Series(_return)
        print(f"{"Avg cumulative return:":<20}{(_return.values.mean()-1)*100:.2f}%")
        print(f"Avg VN Index cumulative return: {(sum_VN_Index_returns/completed_futures - 1)*100:.2f}%")
        print(f"Thắng VN Index {wins_against_VN_Index}/{completed_futures} lần ({(wins_against_VN_Index/completed_futures*100):.2f}%)")
        table.loc[len(table)] = [
            signal,
            total_count,
            profit_count,
            f"{profit_count/total_count*100:.2f}%",
            breakeven_count,
            f"{breakeven_count/total_count*100:.2f}%",
            loss_count,
            f"{loss_count*100/total_count:.2f}%",
            f"{avg_PnL:.2f}%",
            f"{(_return.values.mean()-1)*100:.2f}%",
            f"{(sum_VN_Index_returns/completed_futures - 1)*100:.2f}%",
            f"{avg_alpha*100:.2f}%",
            wins_against_VN_Index,
            f"{(wins_against_VN_Index/completed_futures*100):.2f}%",
            f"{avg_days_diff:.2f}",
            f"{(_return.values.mean()**(365/avg_days_diff) - 1)*100:.2f}%"
        ]
    
    return table

def main():
    stime = time.time()
    table = runAsync()
    etime = time.time()
    table.to_excel("Result.xlsx")
    print(f"Time elapsed: {etime - stime}")

if __name__ == "__main__":
    main()

import pandas as pd
import strat_v4_vectorized as strats
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import result_reader
import time

DATA_DIR = "stock_data"
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
signals = ["macd_divergence", "rsi_divergence", "cci_divergence", "stoch_divergence",
             "macd_crossover", "rsi_crossover", "cci_crossover", "stoch_crossover"]

signals_xover_only = ["macd_crossover", "rsi_crossover", "cci_crossover", "stoch_crossover"]

def runStrategyFromCSV(file_path: str, signals: list):
    df = pd.read_csv(file_path, index_col='time')
    test_results = strats.strategy(df, file_path, signals)
    # print(f"Done running strategy on {file_path}")
    return test_results

def runAsync():
    for signal in signals: 
        total_result = []
        completed_futures = 0
        print(f"Thử bất đồng bộ tín hiệu {signal}")
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(runStrategyFromCSV, file, [signal]) for file in csv_files]
            for future in as_completed(futures):
                total_result.extend(future.result())
                completed_futures += 1
                # print(f"{completed_futures}")

        result_reader.read_PnL_results(total_result)
        # total_result.to_csv("total_results_v4.csv", header=["PnL"])

def main():
    stime = time.time()
    runAsync()
    etime = time.time()
    print(f"Time elapsed: {etime - stime}")

if __name__ == "__main__":
    main()

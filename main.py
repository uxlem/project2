import pandas as pd
import strat_v4a as strats
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

DATA_DIR = "precalc_data"
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def runStrategyFromCSV(file_path):
    df = pd.read_csv(file_path, index_col='time')
    # test_results = strategies.strategy_two(df)
    test_results = strats.strategy(df)
    print(f"Done running strategy on {file_path}")
    return test_results


def main():
    total_result = pd.Series()
    # total_profit = pd.Series()
    # total_breakeven = pd.Series()
    # total_loss = pd.Series()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(runStrategyFromCSV, file) for file in csv_files]
        for future in as_completed(futures):
            total_result = pd.concat([total_result, future.result()], ignore_index=True)

    total_result.to_csv("total_results_stoch_v4a.csv")

if __name__ == "__main__":
    main()

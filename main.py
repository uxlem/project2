import talib as ta 
import pandas as pd
import numpy as np
import strategies
import os
from concurrent.futures import ProcessPoolExecutor  # or ProcessPoolExecutor

DATA_DIR = "stock_data"
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def runStrategyFromCSV(file_path):
    df = pd.read_csv(file_path, index_col='time')
    # df['time'] = pd.DatetimeIndex(df['time'])
    # df.set_index('time', inplace = True)
    strategies.strategyOne(df)
    print(f"Done running strategy on {file_path}")


def main():
    results = []

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(runStrategyFromCSV, file) for file in csv_files]
        for future in futures:
            results.append(future.result())

    # Do something with results
    # for r in results:
    #     print(r)

if __name__ == "__main__":
    main()

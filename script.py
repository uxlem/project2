import pandas as pd
import talib as ta
import os
from concurrent.futures import ThreadPoolExecutor

DATA_DIR = "stock_data"
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

def preCalc(f):
    df = pd.read_csv(f)
    # df['time'] = pd.DatetimeIndex(df['time'])
    # df.set_index('time', inplace=True)
    df['cci'] = ta.CCI(df['close'], df['low'], df['high'])
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(df['close'])
    df['rsi'] = ta.RSI(df['close'])

    file_path = os.path.split(f)[1]
    file_path = os.path.join('precalc', file_path)
    df.to_csv(file_path, index=False)
    
def main():
    # results = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(preCalc, f) for f in csv_files}
        # for future in futures:
        #     results.append(future.result())

if __name__ == "__main__":
    main()
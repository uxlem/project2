import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from vnstock import Vnstock
from vnstock import Listing
import os

# Parameters
start_date = '2018-01-01'
end_date = '2025-01-01'
max_workers = 5  # number of threads per batch (small because server is sensitive)
batch_size = 5   # fetch 5 stocks at a time
delay_seconds = 10  # wait 10 seconds between batches
output_folder = 'stock_data'
retry_limit = 3

listing = Listing()
symbols = listing.all_symbols()['symbol']

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

def fetch_symbol(symbol, start, end, retries=retry_limit):
    for attempt in range(retries):
        try:
            stock = Vnstock().stock(symbol)
            df = stock.quote.history(start=start, end=end, interval='1D')
            if not df.empty:
                return symbol, df
        except Exception as e:
            print(f"[{symbol}] Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    print(f"[{symbol}] Failed after {retries} attempts.")
    return symbol, None

def fetch_batch(batch_symbols, start, end):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_symbol, symbol, start, end): symbol for symbol in batch_symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                sym, df = future.result()
                if df is not None:
                    results[sym] = df
            except Exception as e:
                print(f"[{symbol}] Unexpected error: {e}")
    return results

def save_data(results, folder):
    for symbol, df in results.items():
        file_path = os.path.join(folder, f"{symbol}.csv")
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    total_symbols = len(symbols)
    all_data = {}
    for i in tqdm(range(0, total_symbols, batch_size), desc="Fetching Stocks in Batches"):
        batch = symbols[i:i+batch_size]
        batch_data = fetch_batch(batch, start_date, end_date)
        save_data(batch_data, output_folder)
        all_data.update(batch_data)
        if i + batch_size < total_symbols:
            print(f"Waiting {delay_seconds} seconds before next batch...")
            time.sleep(delay_seconds)
    
    print(f"Finished fetching {len(all_data)} stocks. Saved to '{output_folder}/'")
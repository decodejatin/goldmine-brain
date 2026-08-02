import os
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "goldmine_features_1s.parquet")

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def process_file(csv_file: str):
    print(f"[*] Processing {csv_file}...")
    # BookTicker CSV usually: update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time
    col_names = ["update_id", "bid_price", "bid_qty", "ask_price", "ask_qty", "transaction_time"]
    
    # Depending on the month, there might be a header or not. 
    # Try reading first row to check
    df = pd.read_csv(csv_file, names=col_names, header=0 if "update_id" in open(csv_file).readline() else None)
    
    # Convert transaction_time to datetime
    if 'transaction_time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['transaction_time'], unit='ms')
    else:
        # Fallback if columns differ
        df['timestamp'] = pd.to_datetime(df.iloc[:, 5], unit='ms')
        df.rename(columns={df.columns[1]: "bid_price", df.columns[3]: "ask_price"}, inplace=True)
    
    df.set_index('timestamp', inplace=True)
    
    # Calculate Mid Price
    df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2.0
    
    # 1-second aggregation
    print("[*] Resampling to 1s intervals...")
    ohlc = df['mid_price'].resample('1s').ohlc()
    ohlc.ffill(inplace=True) # Forward fill missing seconds
    
    # Add spread
    df_spread = df[['bid_price', 'ask_price']].resample('1s').last().ffill()
    ohlc['spread_bps'] = ((df_spread['ask_price'] - df_spread['bid_price']) / df_spread['bid_price']) * 10000.0
    
    # Compute Indicators
    print("[*] Computing Technical Indicators (RSI, ATR, Z-Score)...")
    ohlc['rsi_14'] = compute_rsi(ohlc['close'], 14)
    ohlc['atr_14'] = compute_atr(ohlc['high'], ohlc['low'], ohlc['close'], 14)
    
    roll_mean = ohlc['close'].rolling(window=50).mean()
    roll_std = ohlc['close'].rolling(window=50).std()
    ohlc['z_score_50'] = (ohlc['close'] - roll_mean) / roll_std
    
    # Drop NAs from indicator window warmup
    ohlc.dropna(inplace=True)
    
    # Save to Parquet
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    ohlc.to_parquet(OUTPUT_FILE, engine='pyarrow', compression='snappy')
    print(f"[+] Successfully saved features to {OUTPUT_FILE}")
    print(f"[+] Total Rows: {len(ohlc)}")

if __name__ == "__main__":
    # Test on a downloaded file
    for file in os.listdir(RAW_DATA_DIR):
        if file.endswith(".csv"):
            process_file(os.path.join(RAW_DATA_DIR, file))
            break

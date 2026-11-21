"""
Data Engineering at Scale (DA2310) - Project Data Acquisition
Option C: High-Frequency Tick Data (Open Public AWS S3 Archive)

Downloads 1-minute high-frequency tick data from data.binance.vision (AWS S3).
Zero API keys needed. Each month contains ~43,200 records per asset.
"""

import os
import sys
import io
import zipfile
import time
from datetime import datetime
import pandas as pd
import requests
import urllib3
from tqdm import tqdm

# Disable SSL verification warnings for public AWS archive
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Target high-liquidity crypto assets
TARGET_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]

# Target months available in the public archive (2024)
TARGET_MONTHS = [
    ("2024", "01"),
    ("2024", "02"),
    ("2024", "03"),
    ("2024", "04"),
    ("2024", "05"),
    ("2024", "06")
]

# Output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "high_freq_ticks")
RAW_DIR = os.path.join(DATA_DIR, "raw_csv")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore"
]


def download_monthly_kline(symbol, year, month):
    filename = f"{symbol}-1m-{year}-{month}.zip"
    url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{filename}"
    
    try:
        resp = requests.get(url, stream=True, timeout=30, verify=False)
        if resp.status_code != 200:
            return None
            
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            csv_names = z.namelist()
            if not csv_names:
                return None
            with z.open(csv_names[0]) as f:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
                
                # Filter out header string row if present
                is_num = pd.to_numeric(df["open_time"], errors="coerce").notnull()
                df = df[is_num].copy()
                
                df["open_time"] = pd.to_numeric(df["open_time"])
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                df["ticker"] = symbol
                df["year"] = int(year)
                df["month"] = int(month)
                df["date"] = df["timestamp"].dt.date
                
                for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["trades_count"] = pd.to_numeric(df["trades_count"], errors="coerce").fillna(0).astype(int)
                
                typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
                df["typical_price"] = typical_price
                df["volume_price"] = typical_price * df["volume"]
                
                df = df.drop(columns=["open_time", "close_time", "ignore"])
                return df
                
    except Exception as e:
        print(f"[x] Error downloading {filename}: {e}")
        return None


def run_batch_download():
    print("=" * 70)
    print("  DA2310: DATA ACQUISITION - OPTION C (HIGH-FREQUENCY 1-MIN TICKS)")
    print("=" * 70)
    print(f"[+] Target Symbols: {TARGET_SYMBOLS}")
    print(f"[+] Months: {len(TARGET_MONTHS)} months per symbol")
    print(f"[+] Output RAW: {RAW_DIR}")
    print(f"[+] Output PARQUET: {PARQUET_DIR}")
    print("-" * 70)
    
    all_dfs = []
    
    total_downloads = len(TARGET_SYMBOLS) * len(TARGET_MONTHS)
    with tqdm(total=total_downloads, desc="Downloading 1m Ticks") as pbar:
        for symbol in TARGET_SYMBOLS:
            symbol_dfs = []
            for year, month in TARGET_MONTHS:
                df = download_monthly_kline(symbol, year, month)
                if df is not None and not df.empty:
                    symbol_dfs.append(df)
                    all_dfs.append(df)
                pbar.update(1)
                time.sleep(0.1)
                
            if symbol_dfs:
                sym_combined = pd.concat(symbol_dfs, ignore_index=True)
                csv_path = os.path.join(RAW_DIR, f"{symbol}_1m_history.csv")
                sym_combined.to_csv(csv_path, index=False)
                print(f"\n[+] Saved {symbol}: {len(sym_combined):,} 1-minute ticks ({os.path.getsize(csv_path)/(1024*1024):.2f} MB)")
                
    if all_dfs:
        print("\n[+] Consolidating Master High-Frequency Dataset...")
        master_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[+] Total High-Frequency Records: {len(master_df):,}")
        
        master_csv = os.path.join(DATA_DIR, "high_freq_ticks_master.csv")
        master_df.to_csv(master_csv, index=False)
        print(f"[+] Saved Master CSV: {master_csv} ({os.path.getsize(master_csv)/(1024*1024):.2f} MB)")
        
        try:
            parquet_path = os.path.join(PARQUET_DIR, "ticks_1m.parquet")
            master_df.to_parquet(
                parquet_path, 
                partition_cols=["year", "ticker"], 
                engine="pyarrow", 
                coerce_timestamps="us", 
                allow_truncated_timestamps=True
            )
            print(f"[+] Exported Partitioned Parquet to: {parquet_path}")
        except Exception as e:
            print(f"[!] Parquet export note: {e}")
            
    print("\n[v] Data Acquisition for Option C Completed Successfully!")


if __name__ == "__main__":
    run_batch_download()

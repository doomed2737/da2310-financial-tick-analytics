"""
Data Engineering at Scale (DA2310) - Project Data Acquisition
Option B: US Equities (S&P 500 & NASDAQ Mega-Caps) Batch Downloader

This script downloads historical and intraday financial market data for major US
benchmark ETFs and mega-cap tech/financial equities, organizing them into raw CSV
and partitioned Parquet.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import requests
from tqdm import tqdm

# Top S&P 500 / NASDAQ Benchmarks and Mega-Cap Equities
US_TICKERS = [
    "SPY",      # S&P 500 ETF Benchmark
    "QQQ",      # NASDAQ 100 ETF Benchmark
    "DIA",      # Dow Jones Industrial Average ETF
    "IWM",      # Russell 2000 Small Cap ETF
    "AAPL",     # Apple
    "MSFT",     # Microsoft
    "NVDA",     # NVIDIA (High volatility / semiconductor)
    "AMZN",     # Amazon
    "GOOGL",    # Alphabet
    "META",     # Meta Platforms
    "TSLA",     # Tesla (Very high retail volatility)
    "JPM",      # JPMorgan Chase (Banking)
    "V",        # Visa (Payments)
    "UNH",      # UnitedHealth
    "XOM",      # Exxon Mobil (Energy)
    "WMT",      # Walmart (Consumer)
    "PG",       # Procter & Gamble
    "LLY",      # Eli Lilly (Healthcare)
    "AMD",      # Advanced Micro Devices
    "NFLX"      # Netflix
]

# Output directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "us_equities")
RAW_DIR = os.path.join(DATA_DIR, "raw_csv")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)


def download_via_direct_api(ticker, interval="5m", range_param="60d"):
    """
    Downloads market data directly from Yahoo Finance v8 Chart API via requests.
    Zero external finance library requirements.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    params = {
        "interval": interval,
        "range": range_param,
        "includePrePost": "false",
        "events": "div,splits"
    }
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            print(f"[!] Warning: {ticker} HTTP status {resp.status_code}")
            return None
            
        data = resp.json()
        result = data.get("chart", {}).get("result")
        if not result:
            print(f"[!] No result found for {ticker}")
            return None
            
        result_data = result[0]
        timestamps = result_data.get("timestamp", [])
        if not timestamps:
            print(f"[!] No timestamps for {ticker}")
            return None
            
        quote = result_data.get("indicators", {}).get("quote", [{}])[0]
        
        df = pd.DataFrame({
            "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps],
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", [])
        })
        
        df["ticker"] = ticker
        df["year"] = df["timestamp"].dt.year
        df["date"] = df["timestamp"].dt.date
        
        # Drop rows with null close/volume
        df = df.dropna(subset=["open", "close"])
        
        # Compute VWAP proxy metrics
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        df["typical_price"] = typical_price
        df["volume_price"] = typical_price * df["volume"]
        
        return df
        
    except Exception as e:
        print(f"[x] Error downloading {ticker}: {e}")
        return None


def run_batch_download():
    print("=" * 70)
    print("  DA2310: DATA ACQUISITION - OPTION B (US EQUITIES / S&P 500 & NASDAQ)")
    print("=" * 70)
    print(f"[+] Target Tickers: {len(US_TICKERS)} symbols")
    print(f"[+] Raw CSV Directory: {RAW_DIR}")
    print(f"[+] Parquet Directory: {PARQUET_DIR}")
    print("-" * 70)
    
    all_dfs = []
    
    # 1. Download 5-minute Intraday Data (Last 60 Days)
    print("\n[1/2] Fetching 5-minute Intraday Ticks (Last 60 Days)...")
    for ticker in tqdm(US_TICKERS, desc="Intraday 5m"):
        df = download_via_direct_api(ticker, interval="5m", range_param="60d")
        if df is not None and not df.empty:
            csv_path = os.path.join(RAW_DIR, f"{ticker}_5m_60d.csv")
            df.to_csv(csv_path, index=False)
            all_dfs.append(df)
        time.sleep(0.3)  # Rate limiting
        
    # 2. Download Multi-Year Daily Data (Last 5 Years)
    print("\n[2/2] Fetching Multi-Year Historical Daily Bars (5 Years)...")
    for ticker in tqdm(US_TICKERS, desc="Daily 5Y"):
        df = download_via_direct_api(ticker, interval="1d", range_param="5y")
        if df is not None and not df.empty:
            csv_path = os.path.join(RAW_DIR, f"{ticker}_daily_5y.csv")
            df.to_csv(csv_path, index=False)
        time.sleep(0.3)
        
    # 3. Consolidate & Export to Partitioned Parquet
    if all_dfs:
        print("\n[+] Consolidating US intraday dataset...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[+] Total Intraday Records Downloaded: {len(combined_df):,}")
        
        master_csv = os.path.join(DATA_DIR, "us_intraday_master.csv")
        combined_df.to_csv(master_csv, index=False)
        print(f"[+] Saved Master CSV: {master_csv} ({os.path.getsize(master_csv) / (1024*1024):.2f} MB)")
        
        try:
            parquet_path = os.path.join(PARQUET_DIR, "intraday_5m.parquet")
            combined_df.to_parquet(
                parquet_path, 
                partition_cols=["year", "ticker"], 
                engine="pyarrow", 
                coerce_timestamps="us", 
                allow_truncated_timestamps=True
            )
            print(f"[+] Exported Partitioned Parquet to: {PARQUET_DIR}")
        except Exception as e:
            print(f"[!] Note: Parquet export skipped ({e}). Run 'pip install pyarrow' to enable.")
            
    print("\n[v] Data Acquisition for Option B Completed Successfully!")


if __name__ == "__main__":
    run_batch_download()

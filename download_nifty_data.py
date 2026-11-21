"""
Data Engineering at Scale (DA2310) - Project Data Acquisition
Option A: Indian Equities (NSE / NIFTY 50) Batch Downloader

This script downloads historical and intraday financial market data for top NIFTY 50
constituents and index benchmarks, then organizes them into raw CSV and partitioned Parquet.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import requests
from tqdm import tqdm

# Top NIFTY 50 Constituents on National Stock Exchange (NSE)
# Suffix .NS is standard for National Stock Exchange in financial APIs
NIFTY_TICKERS = [
    "^NSEI",          # NIFTY 50 Benchmark Index
    "^NSEBANK",       # NIFTY Bank Benchmark Index
    "RELIANCE.NS",    # Energy / Conglomerate
    "TCS.NS",         # IT Services
    "HDFCBANK.NS",    # Banking & Financials
    "INFY.NS",        # IT Services
    "ICICIBANK.NS",   # Banking
    "HINDUNILVR.NS",  # FMCG
    "ITC.NS",         # FMCG
    "SBIN.NS",        # PSU Bank
    "BHARTIARTL.NS",  # Telecom
    "LICI.NS",        # Insurance
    "LT.NS",          # Infrastructure / Capital Goods
    "BAJFINANCE.NS",  # NBFC
    "TATAMOTORS.NS",  # Automotive
    "SUNPHARMA.NS",   # Pharmaceuticals
    "MARUTI.NS",      # Automotive
    "AXISBANK.NS",    # Banking
    "TITAN.NS",       # Consumer Durables
    "ONGC.NS",        # Oil & Gas
    "NTPC.NS",        # Power
    "POWERGRID.NS",   # Power Transmission
    "TATASTEEL.NS",   # Metals & Mining
    "ADANIENT.NS",    # Metals & Trading
    "KOTAKBANK.NS"    # Banking
]

# Output directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "nifty")
RAW_DIR = os.path.join(DATA_DIR, "raw_csv")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)


def download_via_direct_api(ticker, interval="5m", range_param="60d"):
    """
    Downloads market data directly from Yahoo Finance v8 Chart API via requests.
    Requires ZERO external finance libraries (only standard requests + pandas).
    
    Interval options:
      - '1m': 1-minute intraday (max range: 7d)
      - '5m': 5-minute intraday (max range: 60d)
      - '15m': 15-minute intraday (max range: 60d)
      - '60m' or '1h': 1-hour intraday (max range: 730d / 2 years)
      - '1d': Daily bars (max range: 10y or max)
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
        
        # Clean ticker symbol for filename and metadata
        clean_ticker = ticker.replace("^", "").replace(".NS", "")
        df["ticker"] = clean_ticker
        df["year"] = df["timestamp"].dt.year
        df["date"] = df["timestamp"].dt.date
        
        # Drop rows with null close/volume
        df = df.dropna(subset=["open", "close"])
        
        # Compute VWAP proxy: (High + Low + Close) / 3 * Volume
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        df["typical_price"] = typical_price
        df["volume_price"] = typical_price * df["volume"]
        
        return df
        
    except Exception as e:
        print(f"[x] Error downloading {ticker}: {e}")
        return None


def run_batch_download():
    print("=" * 70)
    print("  DA2310: DATA ACQUISITION - OPTION A (INDIAN EQUITIES / NIFTY 50)")
    print("=" * 70)
    print(f"[+] Target Tickers: {len(NIFTY_TICKERS)} symbols")
    print(f"[+] Raw CSV Directory: {RAW_DIR}")
    print(f"[+] Parquet Directory: {PARQUET_DIR}")
    print("-" * 70)
    
    all_dfs = []
    
    # 1. Download 5-minute Intraday Data (Last 60 Days)
    print("\n[1/2] Fetching 5-minute Intraday Ticks (Last 60 Days)...")
    for ticker in tqdm(NIFTY_TICKERS, desc="Intraday 5m"):
        df = download_via_direct_api(ticker, interval="5m", range_param="60d")
        if df is not None and not df.empty:
            clean_ticker = ticker.replace("^", "").replace(".NS", "")
            csv_path = os.path.join(RAW_DIR, f"{clean_ticker}_5m_60d.csv")
            df.to_csv(csv_path, index=False)
            all_dfs.append(df)
        time.sleep(0.3)  # Gentle rate-limiting to avoid HTTP 429
        
    # 2. Download Multi-Year Daily Data (Last 5 Years for Macro Trends)
    print("\n[2/2] Fetching Multi-Year Historical Daily Bars (5 Years)...")
    for ticker in tqdm(NIFTY_TICKERS, desc="Daily 5Y"):
        df = download_via_direct_api(ticker, interval="1d", range_param="5y")
        if df is not None and not df.empty:
            clean_ticker = ticker.replace("^", "").replace(".NS", "")
            csv_path = os.path.join(RAW_DIR, f"{clean_ticker}_daily_5y.csv")
            df.to_csv(csv_path, index=False)
        time.sleep(0.3)
        
    # 3. Consolidate & Export to Master CSV & Partitioned Parquet
    if all_dfs:
        print("\n[+] Consolidating intraday dataset...")
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"[+] Total Intraday Records Downloaded: {len(combined_df):,}")
        
        # Save monolithic raw CSV
        master_csv = os.path.join(DATA_DIR, "nifty_intraday_master.csv")
        combined_df.to_csv(master_csv, index=False)
        print(f"[+] Saved Master CSV: {master_csv} ({os.path.getsize(master_csv) / (1024*1024):.2f} MB)")
        
        # Try exporting to Parquet if pyarrow/fastparquet is installed
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
            print(f"[!] Note: Parquet export skipped ({e}). Run 'pip install pyarrow' to enable Parquet partitioning.")
            
    print("\n[v] Data Acquisition for Option A Completed Successfully!")


if __name__ == "__main__":
    run_batch_download()

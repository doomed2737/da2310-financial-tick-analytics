"""
Data Engineering at Scale (DA2310) - Project Data Acquisition
Enhanced US Equities Downloader with Market Cap Categorization
(Large Cap, Mid Cap, Small Cap)

Downloads both 5-minute intraday (60 days) and 1-hour bars (730 days) across
60+ US stocks, scaling the dataset to ~500,000 records.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import requests
from tqdm import tqdm

US_EQUITY_CATALOG = [
    # --- LARGE CAP (S&P 500 Mega / Large Caps) ---
    {"ticker": "AAPL", "name": "Apple Inc.", "cap": "Large Cap", "sector": "Technology"},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "cap": "Large Cap", "sector": "Technology"},
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "cap": "Large Cap", "sector": "Semiconductors"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "cap": "Large Cap", "sector": "Consumer Discretionary"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "cap": "Large Cap", "sector": "Communication Services"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "cap": "Large Cap", "sector": "Communication Services"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "cap": "Large Cap", "sector": "Automobile"},
    {"ticker": "JPM", "name": "JPMorgan Chase", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "V", "name": "Visa Inc.", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "UNH", "name": "UnitedHealth Group", "cap": "Large Cap", "sector": "Healthcare"},
    {"ticker": "XOM", "name": "Exxon Mobil", "cap": "Large Cap", "sector": "Energy"},
    {"ticker": "WMT", "name": "Walmart Inc.", "cap": "Large Cap", "sector": "Consumer Staples"},
    {"ticker": "PG", "name": "Procter & Gamble", "cap": "Large Cap", "sector": "Consumer Staples"},
    {"ticker": "LLY", "name": "Eli Lilly", "cap": "Large Cap", "sector": "Healthcare"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "cap": "Large Cap", "sector": "Semiconductors"},
    {"ticker": "NFLX", "name": "Netflix Inc.", "cap": "Large Cap", "sector": "Communication Services"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "cap": "Large Cap", "sector": "Healthcare"},
    {"ticker": "HD", "name": "Home Depot", "cap": "Large Cap", "sector": "Consumer Discretionary"},
    {"ticker": "BAC", "name": "Bank of America", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "COST", "name": "Costco Wholesale", "cap": "Large Cap", "sector": "Consumer Staples"},

    # --- MID CAP (S&P MidCap 400 High-Volume Equities) ---
    {"ticker": "DECK", "name": "Deckers Outdoor", "cap": "Mid Cap", "sector": "Consumer Discretionary"},
    {"ticker": "ELF", "name": "e.l.f. Beauty", "cap": "Mid Cap", "sector": "Consumer Staples"},
    {"ticker": "SAIA", "name": "Saia Inc.", "cap": "Mid Cap", "sector": "Industrials"},
    {"ticker": "WSM", "name": "Williams-Sonoma", "cap": "Mid Cap", "sector": "Consumer Discretionary"},
    {"ticker": "MANH", "name": "Manhattan Associates", "cap": "Mid Cap", "sector": "Technology"},
    {"ticker": "BLDR", "name": "Builders FirstSource", "cap": "Mid Cap", "sector": "Industrials"},
    {"ticker": "FIX", "name": "Comfort Systems USA", "cap": "Mid Cap", "sector": "Industrials"},
    {"ticker": "TREX", "name": "Trex Company", "cap": "Mid Cap", "sector": "Industrials"},
    {"ticker": "ONTO", "name": "Onto Innovation", "cap": "Mid Cap", "sector": "Semiconductors"},
    {"ticker": "LNW", "name": "Light & Wonder", "cap": "Mid Cap", "sector": "Gaming / Tech"},
    {"ticker": "CROX", "name": "Crocs Inc.", "cap": "Mid Cap", "sector": "Consumer Discretionary"},
    {"ticker": "GDDY", "name": "GoDaddy Inc.", "cap": "Mid Cap", "sector": "Technology"},
    {"ticker": "CHTR", "name": "Charter Communications", "cap": "Mid Cap", "sector": "Communication Services"},
    {"ticker": "MTCH", "name": "Match Group", "cap": "Mid Cap", "sector": "Communication Services"},
    {"ticker": "WIX", "name": "Wix.com", "cap": "Mid Cap", "sector": "Technology"},
    {"ticker": "RGEN", "name": "Repligen Corp.", "cap": "Mid Cap", "sector": "Healthcare"},
    {"ticker": "FIVE", "name": "Five Below", "cap": "Mid Cap", "sector": "Consumer Discretionary"},
    {"ticker": "DAR", "name": "Darling Ingredients", "cap": "Mid Cap", "sector": "Consumer Staples"},
    {"ticker": "COLM", "name": "Columbia Sportswear", "cap": "Mid Cap", "sector": "Consumer Discretionary"},
    {"ticker": "POST", "name": "Post Holdings", "cap": "Mid Cap", "sector": "Consumer Staples"},

    # --- SMALL CAP (Russell 2000 Dynamic High-Beta Equities) ---
    {"ticker": "CALX", "name": "Calix Inc.", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "POWI", "name": "Power Integrations", "cap": "Small Cap", "sector": "Semiconductors"},
    {"ticker": "AMBA", "name": "Ambarella Inc.", "cap": "Small Cap", "sector": "Semiconductors"},
    {"ticker": "SMTC", "name": "Semtech Corp.", "cap": "Small Cap", "sector": "Semiconductors"},
    {"ticker": "EXTR", "name": "Extreme Networks", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "ACAD", "name": "ACADIA Pharmaceuticals", "cap": "Small Cap", "sector": "Healthcare"},
    {"ticker": "HALO", "name": "Halozyme Therapeutics", "cap": "Small Cap", "sector": "Healthcare"},
    {"ticker": "KURA", "name": "Kura Oncology", "cap": "Small Cap", "sector": "Healthcare"},
    {"ticker": "KRYS", "name": "Krystal Biotech", "cap": "Small Cap", "sector": "Healthcare"},
    {"ticker": "BOOT", "name": "Boot Barn Holdings", "cap": "Small Cap", "sector": "Consumer Discretionary"},
    {"ticker": "SHOO", "name": "Steven Madden", "cap": "Small Cap", "sector": "Consumer Discretionary"},
    {"ticker": "PLAY", "name": "Dave & Buster's", "cap": "Small Cap", "sector": "Entertainment"},
    {"ticker": "SSTK", "name": "Shutterstock", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "MOD", "name": "Modine Manufacturing", "cap": "Small Cap", "sector": "Industrials"},
    {"ticker": "HURN", "name": "Huron Consulting", "cap": "Small Cap", "sector": "Consulting"},
    {"ticker": "PRDO", "name": "Perdoceo Education", "cap": "Small Cap", "sector": "Education"},
    {"ticker": "PLUS", "name": "ePlus Inc.", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "APAM", "name": "Artisan Partners", "cap": "Small Cap", "sector": "Financials"},
    {"ticker": "VRRM", "name": "Verra Mobility", "cap": "Small Cap", "sector": "Industrials"},
    {"ticker": "NVRI", "name": "Enviri Corp.", "cap": "Small Cap", "sector": "Industrials"}
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "us_multicap")
RAW_DIR = os.path.join(DATA_DIR, "raw_csv")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)


def fetch_chart(ticker, meta, interval, range_param):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    params = {"interval": interval, "range": range_param, "includePrePost": "false"}
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result")
        if not result:
            return None
        res_data = result[0]
        timestamps = res_data.get("timestamp", [])
        if not timestamps:
            return None
        quote = res_data.get("indicators", {}).get("quote", [{}])[0]
        
        df = pd.DataFrame({
            "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps],
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", [])
        })
        
        df["ticker"] = ticker
        df["company_name"] = meta["name"]
        df["market_cap_category"] = meta["cap"]
        df["sector"] = meta["sector"]
        df["interval"] = interval
        df["year"] = df["timestamp"].dt.year
        df["date"] = df["timestamp"].dt.date
        
        df = df.dropna(subset=["open", "close"])
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        df["typical_price"] = typical_price
        df["volume_price"] = typical_price * df["volume"]
        return df
    except Exception as e:
        return None


def run():
    print("=" * 75)
    print("  DA2310: US EQUITIES MULTI-CAP DATA ACQUISITION")
    print("  Large Cap | Mid Cap | Small Cap (5m Intraday + 1h 2-Year History)")
    print("=" * 75)
    print(f"[+] Total Tickers: {len(US_EQUITY_CATALOG)}")
    
    # Save Security Master Dimension Table
    sec_master = pd.DataFrame(US_EQUITY_CATALOG)
    sec_master_path = os.path.join(DATA_DIR, "security_master_us.csv")
    sec_master.to_csv(sec_master_path, index=False)
    print(f"[+] Saved Security Master Dimension Table: {sec_master_path}")
    
    all_dfs = []
    
    # 1. 5-minute Intraday (Last 60 Days)
    print("\n[1/2] Downloading 5-minute Intraday Data (60 Days)...")
    for item in tqdm(US_EQUITY_CATALOG, desc="5m Intraday"):
        df = fetch_chart(item["ticker"], item, interval="5m", range_param="60d")
        if df is not None and not df.empty:
            all_dfs.append(df)
        time.sleep(0.15)
        
    # 2. 1-hour Bars (Last 730 Days / 2 Years)
    print("\n[2/2] Downloading 1-hour Historical Bars (2 Years)...")
    for item in tqdm(US_EQUITY_CATALOG, desc="1h Historical"):
        df = fetch_chart(item["ticker"], item, interval="60m", range_param="730d")
        if df is not None and not df.empty:
            all_dfs.append(df)
        time.sleep(0.15)
        
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n[+] Total Records Downloaded: {len(combined):,}")
        
        master_csv = os.path.join(DATA_DIR, "us_multicap_master.csv")
        combined.to_csv(master_csv, index=False)
        print(f"[+] Saved Master CSV: {master_csv} ({os.path.getsize(master_csv)/(1024*1024):.2f} MB)")
        
        try:
            parquet_path = os.path.join(PARQUET_DIR, "multicap.parquet")
            combined.to_parquet(
                parquet_path, 
                partition_cols=["market_cap_category", "ticker"], 
                engine="pyarrow", 
                coerce_timestamps="us", 
                allow_truncated_timestamps=True
            )
            print(f"[+] Exported Partitioned Parquet to: {parquet_path}")
        except Exception as e:
            print(f"[!] Parquet note: {e}")
            
    print("\n[v] Multi-Cap US Data Acquisition Completed!")


if __name__ == "__main__":
    run()

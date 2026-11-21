"""
Data Engineering at Scale (DA2310) - Project Data Acquisition
Enhanced Indian Equities Downloader with Market Cap Categorization
(Large Cap, Mid Cap, Small Cap)

Downloads both 5-minute intraday (60 days) and 1-hour bars (730 days) across
60+ top Indian stocks, scaling the dataset to ~500,000 records.
"""

import os
import sys
import time
from datetime import datetime
import pandas as pd
import requests
from tqdm import tqdm

# Curated Universe: 60+ Indian Stocks across Market Cap Tiers
EQUITY_CATALOG = [
    # --- LARGE CAP (NIFTY 50 Top Weights) ---
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "cap": "Large Cap", "sector": "Energy"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "cap": "Large Cap", "sector": "Technology"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "INFY.NS", "name": "Infosys", "cap": "Large Cap", "sector": "Technology"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "cap": "Large Cap", "sector": "Consumer Goods"},
    {"ticker": "ITC.NS", "name": "ITC Limited", "cap": "Large Cap", "sector": "Consumer Goods"},
    {"ticker": "SBIN.NS", "name": "State Bank of India", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "cap": "Large Cap", "sector": "Telecom"},
    {"ticker": "LT.NS", "name": "Larsen & Toubro", "cap": "Large Cap", "sector": "Construction"},
    {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors", "cap": "Large Cap", "sector": "Automobile"},
    {"ticker": "MARUTI.NS", "name": "Maruti Suzuki", "cap": "Large Cap", "sector": "Automobile"},
    {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma", "cap": "Large Cap", "sector": "Healthcare"},
    {"ticker": "AXISBANK.NS", "name": "Axis Bank", "cap": "Large Cap", "sector": "Financials"},
    {"ticker": "TITAN.NS", "name": "Titan Company", "cap": "Large Cap", "sector": "Consumer Goods"},
    {"ticker": "ONGC.NS", "name": "ONGC", "cap": "Large Cap", "sector": "Energy"},
    {"ticker": "NTPC.NS", "name": "NTPC Limited", "cap": "Large Cap", "sector": "Utilities"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel", "cap": "Large Cap", "sector": "Materials"},
    {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "cap": "Large Cap", "sector": "Financials"},

    # --- MID CAP (NIFTY Midcap 50 Growth Stocks) ---
    {"ticker": "POLYCAB.NS", "name": "Polycab India", "cap": "Mid Cap", "sector": "Capital Goods"},
    {"ticker": "TRENT.NS", "name": "Trent Limited", "cap": "Mid Cap", "sector": "Consumer Goods"},
    {"ticker": "TATAPOWER.NS", "name": "Tata Power", "cap": "Mid Cap", "sector": "Utilities"},
    {"ticker": "FEDERALBNK.NS", "name": "Federal Bank", "cap": "Mid Cap", "sector": "Financials"},
    {"ticker": "VOLTAS.NS", "name": "Voltas Limited", "cap": "Mid Cap", "sector": "Consumer Goods"},
    {"ticker": "JUBLFOOD.NS", "name": "Jubilant FoodWorks", "cap": "Mid Cap", "sector": "Consumer Goods"},
    {"ticker": "COFORGE.NS", "name": "Coforge", "cap": "Mid Cap", "sector": "Technology"},
    {"ticker": "PERSISTENT.NS", "name": "Persistent Systems", "cap": "Mid Cap", "sector": "Technology"},
    {"ticker": "MPHASIS.NS", "name": "Mphasis", "cap": "Mid Cap", "sector": "Technology"},
    {"ticker": "ASTRAL.NS", "name": "Astral Limited", "cap": "Mid Cap", "sector": "Building Materials"},
    {"ticker": "AUROPHARMA.NS", "name": "Aurobindo Pharma", "cap": "Mid Cap", "sector": "Healthcare"},
    {"ticker": "BHARATFORG.NS", "name": "Bharat Forge", "cap": "Mid Cap", "sector": "Automobile"},
    {"ticker": "ASHOKLEY.NS", "name": "Ashok Leyland", "cap": "Mid Cap", "sector": "Automobile"},
    {"ticker": "CUMMINSIND.NS", "name": "Cummins India", "cap": "Mid Cap", "sector": "Capital Goods"},
    {"ticker": "IDFCFIRSTB.NS", "name": "IDFC First Bank", "cap": "Mid Cap", "sector": "Financials"},
    {"ticker": "OBEROIRLTY.NS", "name": "Oberoi Realty", "cap": "Mid Cap", "sector": "Real Estate"},
    {"ticker": "GODREJPROP.NS", "name": "Godrej Properties", "cap": "Mid Cap", "sector": "Real Estate"},
    {"ticker": "PAGEIND.NS", "name": "Page Industries", "cap": "Mid Cap", "sector": "Textiles"},
    {"ticker": "DEEPAKNTR.NS", "name": "Deepak Nitrite", "cap": "Mid Cap", "sector": "Chemicals"},
    {"ticker": "APOLLOTYRE.NS", "name": "Apollo Tyres", "cap": "Mid Cap", "sector": "Automobile"},

    # --- SMALL CAP (NIFTY Smallcap High-Beta Equities) ---
    {"ticker": "CDSL.NS", "name": "Central Depository Services", "cap": "Small Cap", "sector": "Financials"},
    {"ticker": "BSE.NS", "name": "BSE Limited", "cap": "Small Cap", "sector": "Financials"},
    {"ticker": "ANGELONE.NS", "name": "Angel One", "cap": "Small Cap", "sector": "Financials"},
    {"ticker": "HAPPSTMNDS.NS", "name": "Happiest Minds", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "ROUTE.NS", "name": "Route Mobile", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "TANLA.NS", "name": "Tanla Platforms", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "SONACOMS.NS", "name": "Sona BLW", "cap": "Small Cap", "sector": "Automobile"},
    {"ticker": "RADICO.NS", "name": "Radico Khaitan", "cap": "Small Cap", "sector": "Consumer Goods"},
    {"ticker": "KAYNES.NS", "name": "Kaynes Technology", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "SUZLON.NS", "name": "Suzlon Energy", "cap": "Small Cap", "sector": "Utilities"},
    {"ticker": "ZENSARTECH.NS", "name": "Zensar Technologies", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "KPITTECH.NS", "name": "KPIT Technologies", "cap": "Small Cap", "sector": "Technology"},
    {"ticker": "CENTURYPLY.NS", "name": "Century Plyboards", "cap": "Small Cap", "sector": "Building Materials"},
    {"ticker": "PVRINOX.NS", "name": "PVR INOX", "cap": "Small Cap", "sector": "Media & Ent."},
    {"ticker": "TEJASNET.NS", "name": "Tejas Networks", "cap": "Small Cap", "sector": "Telecom"},
    {"ticker": "LEMONTREE.NS", "name": "Lemon Tree Hotels", "cap": "Small Cap", "sector": "Hospitality"},
    {"ticker": "CEATLTD.NS", "name": "CEAT Limited", "cap": "Small Cap", "sector": "Automobile"},
    {"ticker": "CASTROLIND.NS", "name": "Castrol India", "cap": "Small Cap", "sector": "Oil & Gas"},
    {"ticker": "AMBER.NS", "name": "Amber Enterprises", "cap": "Small Cap", "sector": "Consumer Durables"},
    {"ticker": "JBCHEPHARM.NS", "name": "JB Chemicals", "cap": "Small Cap", "sector": "Healthcare"}
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "nifty_multicap")
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
        
        clean_ticker = ticker.replace(".NS", "")
        df["ticker"] = clean_ticker
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
    print("  DA2310: INDIAN EQUITIES MULTI-CAP DATA ACQUISITION")
    print("  Large Cap | Mid Cap | Small Cap (5m Intraday + 1h 2-Year History)")
    print("=" * 75)
    print(f"[+] Total Tickers: {len(EQUITY_CATALOG)}")
    
    # Save Security Master Dimension Table
    sec_master = pd.DataFrame(EQUITY_CATALOG)
    sec_master["clean_ticker"] = sec_master["ticker"].str.replace(".NS", "")
    sec_master_path = os.path.join(DATA_DIR, "security_master_india.csv")
    sec_master.to_csv(sec_master_path, index=False)
    print(f"[+] Saved Security Master Dimension Table: {sec_master_path}")
    
    all_dfs = []
    
    # 1. 5-minute Intraday (Last 60 Days)
    print("\n[1/2] Downloading 5-minute Intraday Data (60 Days)...")
    for item in tqdm(EQUITY_CATALOG, desc="5m Intraday"):
        df = fetch_chart(item["ticker"], item, interval="5m", range_param="60d")
        if df is not None and not df.empty:
            all_dfs.append(df)
        time.sleep(0.15)
        
    # 2. 1-hour Bars (Last 730 Days / 2 Years) to boost volume
    print("\n[2/2] Downloading 1-hour Historical Bars (2 Years)...")
    for item in tqdm(EQUITY_CATALOG, desc="1h Historical"):
        df = fetch_chart(item["ticker"], item, interval="60m", range_param="730d")
        if df is not None and not df.empty:
            all_dfs.append(df)
        time.sleep(0.15)
        
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\n[+] Total Records Downloaded: {len(combined):,}")
        
        master_csv = os.path.join(DATA_DIR, "nifty_multicap_master.csv")
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
            
    print("\n[v] Multi-Cap India Data Acquisition Completed!")


if __name__ == "__main__":
    run()

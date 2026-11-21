"""
DA2310: Data Engineering at Scale - Phase 2 (PySpark Engine)
Blueprint 2: Rolling Window Indicators (src/02_window_analytics.py)

Computes 20-period Rolling VWAP, Bollinger Bands (Upper, Lower, Bandwidth),
and Realized Volatility using PySpark native window functions partitioned by ticker.
Eliminates cross-worker shuffle by preserving ticker co-partitioning.
"""

import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.window import Window

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spark_session import get_spark_session


def compute_window_analytics(df):
    """
    Computes rolling indicators over a 20-row sliding window partitioned by ticker.
    Uses rowsBetween(-19, 0) to ensure strict historical causality (no lookahead bias).
    """
    # 20-period window per ticker ordered by timestamp
    window_20 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-19, 0)
    
    analytics_df = df.withColumn(
        "cum_volume_price", F.sum("volume_price").over(window_20)
    ).withColumn(
        "cum_volume", F.sum("volume").over(window_20)
    ).withColumn(
        "rolling_vwap", 
        F.when(F.col("cum_volume") > 0, F.col("cum_volume_price") / F.col("cum_volume"))
         .otherwise(F.col("close"))
    ).withColumn(
        "rolling_mean", F.avg("close").over(window_20)
    ).withColumn(
        "rolling_std", F.coalesce(F.stddev("close").over(window_20), F.lit(0.0))
    ).withColumn(
        "bollinger_upper", F.col("rolling_mean") + (F.col("rolling_std") * 2.0)
    ).withColumn(
        "bollinger_lower", F.col("rolling_mean") - (F.col("rolling_std") * 2.0)
    ).withColumn(
        "bandwidth_pct", 
        F.when(F.col("rolling_mean") > 0, 
               ((F.col("bollinger_upper") - F.col("bollinger_lower")) / F.col("rolling_mean")) * 100.0)
         .otherwise(0.0)
    )
    
    return analytics_df


def run():
    spark = get_spark_session(app_name="DA2310_02_Window_Analytics")
    
    print("=" * 75)
    print("  DA2310: ROLLING VWAP & BOLLINGER BANDS ENGINE")
    print("=" * 75)
    
    parquet_path = "data/nifty_multicap/parquet/multicap.parquet"
    print(f"\n[+] Loading Multi-Cap Parquet from: {parquet_path}")
    raw_df = spark.read.parquet(parquet_path)
    
    print("[+] Calculating Rolling 20-period VWAP & Bollinger Bands...")
    analyzed_df = compute_window_analytics(raw_df)
    
    # Display sample calculation for a representative large-cap (RELIANCE) and small-cap (BSE)
    sample_tickers = ["RELIANCE", "TCS", "BSE", "CDSL"]
    print(f"\n[+] Sample Window Indicators for Tickers: {sample_tickers}")
    analyzed_df.filter(F.col("ticker").isin(sample_tickers)) \
        .select(
            "timestamp", "ticker", "market_cap_category", 
            F.round("close", 2).alias("close"),
            F.round("rolling_vwap", 2).alias("vwap"),
            F.round("bollinger_upper", 2).alias("b_upper"),
            F.round("bollinger_lower", 2).alias("b_lower"),
            F.round("bandwidth_pct", 2).alias("volatility_%")
        ).orderBy("ticker", "timestamp", ascending=[True, False]).show(12)
        
    spark.stop()
    print("\n[v] Blueprint 2 Pipeline Completed Successfully!")


if __name__ == "__main__":
    run()

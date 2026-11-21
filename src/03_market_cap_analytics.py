"""
DA2310: Data Engineering at Scale - Phase 2 (PySpark Engine)
Blueprint 3: Market Cap Volatility Regime Comparison (src/03_market_cap_analytics.py)

Empirically evaluates the core research hypothesis:
"Do Small-Cap equities exhibit statistically higher intraday realized volatility
and wider Bollinger Band spreads than Large-Cap mega equities?"

Applies normalized volatility metrics (Coefficient of Variation and Bandwidth %)
across both Indian (NSE) and US (NASDAQ/S&P) multi-cap universes.
"""

import os
import sys
from pyspark.sql import functions as F
from pyspark.sql.window import Window

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spark_session import get_spark_session


def analyze_market_cap_regimes(parquet_path, market_label):
    spark = get_spark_session(app_name=f"DA2310_03_Market_Cap_{market_label}")
    
    print("\n" + "=" * 75)
    print(f"  DA2310: MARKET CAP VOLATILITY COMPARISON — {market_label.upper()}")
    print("=" * 75)
    
    df = spark.read.parquet(parquet_path)
    
    # 20-period moving window per ticker
    window_20 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-19, 0)
    
    enriched_df = df.withColumn(
        "rolling_mean", F.avg("close").over(window_20)
    ).withColumn(
        "rolling_std", F.coalesce(F.stddev("close").over(window_20), F.lit(0.0))
    ).withColumn(
        "bandwidth_pct",
        F.when(F.col("rolling_mean") > 0,
               (F.col("rolling_std") * 4.0 / F.col("rolling_mean")) * 100.0)
         .otherwise(0.0)
    )
    
    # Aggregate by Market Cap Category
    cap_summary = enriched_df.groupBy("market_cap_category").agg(
        F.count("*").alias("total_ticks"),
        F.countDistinct("ticker").alias("ticker_count"),
        F.round(F.avg("volume"), 0).alias("mean_volume"),
        F.round(F.avg("rolling_std"), 2).alias("raw_std_pts"),
        F.round(F.avg("bandwidth_pct"), 3).alias("normalized_volatility_%"),
        F.round(F.max("bandwidth_pct"), 2).alias("peak_volatility_spike_%")
    ).orderBy(F.col("normalized_volatility_%").desc())
    
    cap_summary.show(truncate=False)
    
    # Top 5 most volatile tickers in this market
    print(f"[+] Top 5 Highest Volatility Equities in {market_label}:")
    top_volatile = enriched_df.groupBy("ticker", "market_cap_category", "sector").agg(
        F.round(F.avg("bandwidth_pct"), 3).alias("avg_band_volatility_%"),
        F.round(F.max("bandwidth_pct"), 2).alias("max_spike_%")
    ).orderBy(F.col("avg_band_volatility_%").desc()).limit(5)
    
    top_volatile.show(truncate=False)


def run():
    # Analyze Indian Equities
    analyze_market_cap_regimes("data/nifty_multicap/parquet/multicap.parquet", "India Multi-Cap (NSE)")
    
    # Analyze US Equities
    analyze_market_cap_regimes("data/us_multicap/parquet/multicap.parquet", "US Multi-Cap (NASDAQ/S&P)")
    
    print("\n[v] Blueprint 3 Market Cap Analysis Completed Successfully!")


if __name__ == "__main__":
    run()

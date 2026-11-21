"""
DA2310: Data Engineering at Scale - Phase 3 (Distributed MLlib Engine)
Blueprint 8: Indian Equities Gradient-Boosted Trees (GBT) Alpha Regressor
(src/08_indian_equities_gbt_alpha.py)

Trains a distributed Gradient-Boosted Trees Regressor (GBTRegressor) specifically
on the 60+ Indian Multi-Cap equities dataset (Large Cap, Mid Cap, Small Cap).
Forecasts continuous forward percentage returns (Expected Alpha), evaluates R2/RMSE/MAE,
extracts non-linear feature importances, and produces rank-ordered Invest vs. Divest targets.
"""

import os
import sys
import time
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, IntegerType, DateType, TimestampNTZType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Ensure local imports work cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from spark_session import get_spark_session


def get_tick_schema():
    """Explicit DDL schema for Star Schema fact table."""
    return StructType([
        StructField("timestamp", TimestampNTZType(), False),
        StructField("open", DoubleType(), True),
        StructField("high", DoubleType(), True),
        StructField("low", DoubleType(), True),
        StructField("close", DoubleType(), True),
        StructField("volume", DoubleType(), True),
        StructField("company_name", StringType(), True),
        StructField("sector", StringType(), True),
        StructField("interval", StringType(), True),
        StructField("year", IntegerType(), True),
        StructField("date", DateType(), True),
        StructField("typical_price", DoubleType(), True),
        StructField("volume_price", DoubleType(), True),
        StructField("market_cap_category", StringType(), True),
        StructField("ticker", StringType(), False)
    ])


def build_alpha_features(df):
    """
    Computes continuous multi-factor technical indicators and forward returns.
    """
    w_ticker = Window.partitionBy("ticker").orderBy("timestamp")
    w_20 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-19, 0)
    w_5 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-4, 0)
    
    print("[+] Computing continuous alpha features across Indian equity partitions...")
    
    # 1. Rolling VWAP & Bollinger Bands
    feat_df = df.withColumn(
        "cum_vp_20", F.sum("volume_price").over(w_20)
    ).withColumn(
        "cum_vol_20", F.sum("volume").over(w_20)
    ).withColumn(
        "rolling_vwap",
        F.when(F.col("cum_vol_20") > 0, F.col("cum_vp_20") / F.col("cum_vol_20")).otherwise(F.col("close"))
    ).withColumn(
        "rolling_mean_20", F.avg("close").over(w_20)
    ).withColumn(
        "rolling_std_20", F.coalesce(F.stddev("close").over(w_20), F.lit(0.0))
    ).withColumn(
        "rolling_mean_5", F.avg("close").over(w_5)
    ).withColumn(
        "bollinger_upper", F.col("rolling_mean_20") + (F.col("rolling_std_20") * 2.0)
    ).withColumn(
        "bollinger_lower", F.col("rolling_mean_20") - (F.col("rolling_std_20") * 2.0)
    )
    
    # 2. Continuous Factor Metrics (Normalized in %)
    feat_df = feat_df.withColumn(
        "bandwidth_pct",
        F.when(F.col("rolling_mean_20") > 0,
               ((F.col("bollinger_upper") - F.col("bollinger_lower")) / F.col("rolling_mean_20")) * 100.0
        ).otherwise(0.0)
    ).withColumn(
        "bollinger_pct_b",
        F.when((F.col("bollinger_upper") - F.col("bollinger_lower")) > 0,
               (F.col("close") - F.col("bollinger_lower")) / (F.col("bollinger_upper") - F.col("bollinger_lower"))
        ).otherwise(0.5)
    ).withColumn(
        "vwap_deviation_pct",
        F.when(F.col("rolling_vwap") > 0,
               ((F.col("close") - F.col("rolling_vwap")) / F.col("rolling_vwap")) * 100.0
        ).otherwise(0.0)
    ).withColumn(
        "momentum_5_20_pct",
        F.when(F.col("rolling_mean_20") > 0,
               ((F.col("rolling_mean_5") / F.col("rolling_mean_20")) - 1.0) * 100.0
        ).otherwise(0.0)
    ).withColumn(
        "vol_mean_20", F.avg("volume").over(w_20)
    ).withColumn(
        "volume_surge",
        F.when(F.col("vol_mean_20") > 0, F.col("volume") / F.col("vol_mean_20")).otherwise(1.0)
    ).withColumn(
        "high_low_spread_pct",
        F.when(F.col("close") > 0, ((F.col("high") - F.col("low")) / F.col("close")) * 100.0).otherwise(0.0)
    )
    
    # 3. Rolling 1-period Return Volatility
    feat_df = feat_df.withColumn(
        "prev_close", F.lag("close", 1).over(w_ticker)
    ).withColumn(
        "bar_return_pct",
        F.when((F.col("prev_close").isNotNull()) & (F.col("prev_close") > 0),
               ((F.col("close") - F.col("prev_close")) / F.col("prev_close")) * 100.0
        ).otherwise(0.0)
    ).withColumn(
        "rolling_volatility_pct",
        F.coalesce(F.stddev("bar_return_pct").over(w_20), F.lit(0.0))
    )
    
    # 4. Continuous Forward Target (Horizon H = 10 bars ahead) in %
    feat_df = feat_df.withColumn(
        "future_close", F.lead("close", 10).over(w_ticker)
    ).withColumn(
        "forward_return_pct",
        F.when(F.col("close") > 0,
               ((F.col("future_close") - F.col("close")) / F.col("close")) * 100.0
        ).otherwise(0.0)
    ).withColumn("label", F.col("forward_return_pct"))
    
    return feat_df


def run_indian_gbt_pipeline():
    start_time = time.time()
    spark = get_spark_session(app_name="DA2310_08_Indian_GBT_Alpha", master="local[4]")
    
    print("=" * 80)
    print("  DA2310: INDIAN EQUITIES GRADIENT-BOOSTED TREES (GBT) ALPHA ENGINE")
    print("  Continuous Forward Return Prediction across Large, Mid & Small Cap (NSE)")
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    india_parquet = os.path.join(project_root, "data", "nifty_multicap", "parquet", "multicap.parquet")
    
    # 1. Load Data
    print(f"\n[+] Ingesting Indian Multi-Cap Parquet from: {india_parquet}")
    india_df = spark.read.schema(get_tick_schema()).parquet(india_parquet)
    raw_count = india_df.count()
    print(f"[v] Loaded Indian Equities Records: {raw_count:,}")
    
    # Clean rows with positive close and volume
    cleaned_df = india_df.filter(
        F.col("close").isNotNull() & (F.col("close") > 0) &
        F.col("volume").isNotNull() & F.col("timestamp").isNotNull()
    )
    
    # 2. Build Alpha Factor Pipeline
    raw_dataset = build_alpha_features(cleaned_df)
    
    # Column pruning for high memory efficiency
    needed_cols = [
        "ticker", "company_name", "market_cap_category", "sector", "timestamp",
        "close", "rolling_vwap", "bandwidth_pct", "bollinger_pct_b",
        "vwap_deviation_pct", "momentum_5_20_pct", "volume_surge",
        "high_low_spread_pct", "rolling_volatility_pct",
        "future_close", "label"
    ]
    dataset = raw_dataset.select(needed_cols)
    
    # Training data (historical bars where forward 10-bar price is known)
    labeled_data = dataset.filter(F.col("future_close").isNotNull())
    latest_inference_bars = dataset.filter(F.col("future_close").isNull())
    
    total_labeled = labeled_data.count()
    print(f"[v] Labeled Training & Testing Instances: {total_labeled:,}")
    
    # Summary of Forward Return Distribution in Indian Equities
    print("\n[+] Indian Equities Forward Return Statistics (Target %):")
    labeled_data.select(
        F.round(F.mean("label"), 4).alias("mean_return_%"),
        F.round(F.stddev("label"), 4).alias("std_return_%"),
        F.round(F.min("label"), 2).alias("min_return_%"),
        F.round(F.max("label"), 2).alias("max_return_%")
    ).show()
    
    # 3. Construct PySpark MLlib Pipeline
    feature_cols = [
        "bandwidth_pct",
        "bollinger_pct_b",
        "vwap_deviation_pct",
        "momentum_5_20_pct",
        "volume_surge",
        "high_low_spread_pct",
        "rolling_volatility_pct",
        "cap_idx",
        "sector_idx"
    ]
    
    cap_indexer = StringIndexer(inputCol="market_cap_category", outputCol="cap_idx", handleInvalid="keep")
    sector_indexer = StringIndexer(inputCol="sector", outputCol="sector_idx", handleInvalid="keep")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    
    # Train / Test Split (80% Train, 20% Test)
    train_data, test_data = labeled_data.randomSplit([0.8, 0.2], seed=42)
    print(f"[+] Train Partition Size: {train_data.count():,} rows | Test Partition Size: {test_data.count():,} rows")
    
    # 4. Train Distributed Gradient-Boosted Trees Regressor
    print("\n[+] Training Distributed GBTRegressor (maxIter=25, maxDepth=5, stepSize=0.1)...")
    gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=25, maxDepth=5, stepSize=0.1, seed=42)
    
    gbt_pipeline = Pipeline(stages=[cap_indexer, sector_indexer, assembler, scaler, gbt])
    gbt_model = gbt_pipeline.fit(train_data)
    
    # 5. Evaluate on Out-Of-Sample Test Set
    test_preds = gbt_model.transform(test_data).cache()
    
    eval_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    eval_mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    eval_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    
    rmse = eval_rmse.evaluate(test_preds)
    mae = eval_mae.evaluate(test_preds)
    r2 = eval_r2.evaluate(test_preds)
    
    print("\n" + "=" * 55)
    print("      INDIAN EQUITIES GBT REGRESSION METRICS")
    print("=" * 55)
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}%")
    print(f"  Mean Absolute Error (MAE):       {mae:.4f}%")
    print(f"  Coefficient of Det. (R2):       {r2:.4f}")
    print("=" * 55)
    
    # 6. Non-Linear Feature Importance Ranking
    gbt_stage = gbt_model.stages[-1]
    importances = gbt_stage.featureImportances.toArray()
    print("\n[+] GBT Factor Importance Ranking for Indian Equities:")
    feat_importance_list = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for rank, (feat, score) in enumerate(feat_importance_list, 1):
        bar = "#" * int(score * 40)
        print(f"  {rank}. {feat:<24}: {score * 100:6.2f}% | {bar}")
        
    # 7. Generate Rank-Ordered Invest and Divest Recommendations for Current Equities
    print("\n[+] Generating Actionable Invest & Divest Recommendations for Current Indian Stocks...")
    
    # Get latest active bar per ticker
    w_latest = Window.partitionBy("ticker").orderBy(F.col("timestamp").desc())
    latest_ticks = latest_inference_bars.withColumn("rn", F.row_number().over(w_latest)).filter(F.col("rn") == 1)
    
    inference_preds = gbt_model.transform(latest_ticks)
    
    inference_preds = inference_preds.withColumn(
        "expected_alpha_pct", F.round(F.col("prediction"), 3)
    ).withColumn(
        "recommendation",
        F.when(F.col("expected_alpha_pct") >= 0.25, F.lit("STRONG INVEST (LONG)"))
         .when(F.col("expected_alpha_pct") >= 0.10, F.lit("MODERATE ACCUMULATE"))
         .when(F.col("expected_alpha_pct") <= -0.25, F.lit("STRONG DIVEST (SELL/EXIT)"))
         .when(F.col("expected_alpha_pct") <= -0.10, F.lit("MODERATE REDUCE"))
         .otherwise(F.lit("HOLD / NEUTRAL"))
    )
    
    summary_cols = [
        "ticker", "company_name", "market_cap_category", "sector",
        "close", "rolling_vwap", "bandwidth_pct", "rolling_volatility_pct",
        "expected_alpha_pct", "recommendation"
    ]
    recs_df = inference_preds.select(summary_cols).orderBy(F.col("expected_alpha_pct").desc()).toPandas()
    
    # Save recommendations to CSV
    recs_csv_path = os.path.join(project_root, "data", "indian_equities_gbt_alpha_recommendations.csv")
    recs_df.to_csv(recs_csv_path, index=False)
    print(f"\n[+] Exported Indian GBT Recommendations to: {recs_csv_path}")
    
    # Print Top 10 Invest Recommendations
    top_invest = recs_df.head(10)
    print("\n" + "=" * 95)
    print("          TOP 10 INDIAN STOCKS TO INVEST (HIGHEST EXPECTED ALPHA %)")
    print("=" * 95)
    print(top_invest[["ticker", "company_name", "market_cap_category", "close", "expected_alpha_pct", "recommendation"]].to_string(index=False))
    
    # Print Top 10 Divest Recommendations
    top_divest = recs_df.tail(10).iloc[::-1]
    print("\n" + "=" * 95)
    print("          TOP 10 INDIAN STOCKS TO DIVEST (LOWEST/NEGATIVE EXPECTED ALPHA %)")
    print("=" * 95)
    print(top_divest[["ticker", "company_name", "market_cap_category", "close", "expected_alpha_pct", "recommendation"]].to_string(index=False))
    print("=" * 95)
    
    # Breakdown by Market Cap Tier
    print("\n[+] Average Expected Alpha by Market Cap Tier:")
    cap_summary = recs_df.groupby("market_cap_category")["expected_alpha_pct"].agg(["mean", "std", "count"]).reset_index()
    cap_summary.columns = ["Tier", "Avg Alpha (%)", "Alpha Spread (Std)", "Tickers"]
    print(cap_summary.to_string(index=False))
    
    elapsed = time.time() - start_time
    print(f"\n[v] Blueprint 8 Indian GBT Alpha Pipeline Completed in {elapsed:.2f} seconds!")
    spark.stop()


if __name__ == "__main__":
    run_indian_gbt_pipeline()

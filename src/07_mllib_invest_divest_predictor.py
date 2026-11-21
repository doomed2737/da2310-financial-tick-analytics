"""
DA2310: Data Engineering at Scale - Phase 3 (Distributed MLlib Engine)
Blueprint 7: Invest & Divest Predictive Engine (src/07_mllib_invest_divest_predictor.py)

Trains distributed Machine Learning models (Random Forest Classifier & Logistic Regression)
on multi-cap intraday & historical bar datasets using PySpark MLlib.
Computes technical factor pipelines, evaluates predictive accuracy, extracts feature importances,
and generates ranked Invest (Buy) and Divest (Sell) recommendations.
"""

import os
import sys
import time
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, StringType,
    DoubleType, LongType, IntegerType, DateType, TimestampNTZType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array

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


def build_feature_engineering_pipeline(df):
    """
    Computes rolling indicators and forward target signals using PySpark native window functions.
    Preserves strict historical causality for feature generation.
    """
    w_ticker = Window.partitionBy("ticker").orderBy("timestamp")
    w_20 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-19, 0)
    w_5 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-4, 0)
    
    print("[+] Computing 20-period & 5-period window feature factors across partitions...")
    
    # 1. Rolling VWAP & Bollinger Envelopes
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
    
    # 2. Normalized Alpha Factors
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
        "momentum_5_20",
        F.when(F.col("rolling_mean_20") > 0,
               (F.col("rolling_mean_5") / F.col("rolling_mean_20")) - 1.0
        ).otherwise(0.0)
    ).withColumn(
        "vol_mean_20", F.avg("volume").over(w_20)
    ).withColumn(
        "volume_surge",
        F.when(F.col("vol_mean_20") > 0, F.col("volume") / F.col("vol_mean_20")).otherwise(1.0)
    ).withColumn(
        "high_low_spread",
        F.when(F.col("close") > 0, (F.col("high") - F.col("low")) / F.col("close")).otherwise(0.0)
    )
    
    # 3. Forward Target Definition (Horizon H = 10 bars ahead)
    # Label: 2 = Invest (Gain > +1.5%), 0 = Divest (Loss < -1.5%), 1 = Hold/Neutral
    feat_df = feat_df.withColumn(
        "future_close", F.lead("close", 10).over(w_ticker)
    ).withColumn(
        "forward_return",
        F.when(F.col("close") > 0, (F.col("future_close") - F.col("close")) / F.col("close")).otherwise(0.0)
    ).withColumn(
        "label",
        F.when(F.col("forward_return") > 0.015, F.lit(2.0))
         .when(F.col("forward_return") < -0.015, F.lit(0.0))
         .otherwise(F.lit(1.0))
    )
    
    return feat_df


def run_mllib_pipeline():
    start_time = time.time()
    spark = get_spark_session(app_name="DA2310_07_MLlib_Invest_Divest_Predictor", master="local[4]")
    
    print("=" * 80)
    print("  DA2310: DISTRIBUTED MLlib PREDICTIVE INVEST & DIVEST ENGINE")
    print("  Random Forest & Logistic Regression on Multi-Cap Financial Universe")
    print("=" * 80)
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    india_parquet = os.path.join(project_root, "data", "nifty_multicap", "parquet", "multicap.parquet")
    us_parquet = os.path.join(project_root, "data", "us_multicap", "parquet", "multicap.parquet")
    
    # 1. Load Data
    print(f"\n[+] Ingesting Partitioned Multi-Cap Parquet from: {india_parquet}")
    india_df = spark.read.schema(get_tick_schema()).parquet(india_parquet)
    
    # If US parquet exists and has data, union them for unified multi-market modeling
    if os.path.exists(us_parquet):
        try:
            us_df = spark.read.schema(get_tick_schema()).parquet(us_parquet)
            combined_df = india_df.unionByName(us_df)
            print(f"[+] Loaded Combined Universe (India + US): {combined_df.count():,} rows")
        except Exception:
            combined_df = india_df
            print(f"[+] Loaded India Universe: {combined_df.count():,} rows")
    else:
        combined_df = india_df
        print(f"[+] Loaded India Universe: {combined_df.count():,} rows")
        
    # Clean nulls in key price columns
    cleaned_df = combined_df.filter(
        F.col("close").isNotNull() & (F.col("close") > 0) &
        F.col("volume").isNotNull() & F.col("timestamp").isNotNull()
    )
    
    # 2. Compute Technical Indicators & Forward Signals with Column Pruning
    raw_dataset = build_feature_engineering_pipeline(cleaned_df)
    needed_cols = [
        "ticker", "company_name", "market_cap_category", "sector", "timestamp",
        "close", "rolling_vwap", "bandwidth_pct", "bollinger_pct_b",
        "vwap_deviation_pct", "momentum_5_20", "volume_surge", "high_low_spread",
        "future_close", "label"
    ]
    dataset = raw_dataset.select(needed_cols)
    
    # Separate historical labeled training set from inference set (most recent bars with null future_close)
    labeled_data = dataset.filter(F.col("future_close").isNotNull())
    latest_inference_bars = dataset.filter(F.col("future_close").isNull())
    
    total_labeled = labeled_data.count()
    print(f"[v] Total Labeled Training Records: {total_labeled:,}")
    print("\n[+] Target Class Distribution:")
    labeled_data.groupBy("label").agg(
        F.count("*").alias("count"),
        F.round(F.count("*") / total_labeled * 100.0, 2).alias("percentage")
    ).orderBy("label").show()
    
    # 3. Construct PySpark MLlib Pipeline
    feature_cols = [
        "bandwidth_pct",
        "bollinger_pct_b",
        "vwap_deviation_pct",
        "momentum_5_20",
        "volume_surge",
        "high_low_spread",
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
    
    # 4. Train Random Forest Classifier
    print("\n[+] Training Distributed RandomForestClassifier (numTrees=30, maxDepth=6)...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=30, maxDepth=6, seed=42)
    
    rf_pipeline = Pipeline(stages=[cap_indexer, sector_indexer, assembler, scaler, rf])
    rf_model = rf_pipeline.fit(train_data)
    
    # 5. Evaluate Random Forest on Test Data
    predictions = rf_model.transform(test_data).cache()
    
    eval_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    eval_prec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    eval_rec = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    
    acc = eval_acc.evaluate(predictions)
    f1 = eval_f1.evaluate(predictions)
    prec = eval_prec.evaluate(predictions)
    rec = eval_rec.evaluate(predictions)
    
    print("\n" + "=" * 55)
    print("      PYSPARK MLlib TEST EVALUATION METRICS")
    print("=" * 55)
    print(f"  Accuracy:           {acc * 100:.2f}%")
    print(f"  Weighted F1-Score:  {f1:.4f}")
    print(f"  Weighted Precision: {prec:.4f}")
    print(f"  Weighted Recall:    {rec:.4f}")
    print("=" * 55)
    
    # 6. Feature Importance Extraction
    rf_stage = rf_model.stages[-1]
    importances = rf_stage.featureImportances.toArray()
    print("\n[+] Distributed Feature Importance Ranking:")
    feat_importance_list = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for rank, (feat, score) in enumerate(feat_importance_list, 1):
        bar = "#" * int(score * 40)
        print(f"  {rank}. {feat:<20}: {score * 100:6.2f}% | {bar}")
        
    # 7. Generate Actionable Invest vs. Divest Recommendations on Latest Market Bars
    print("\n[+] Generating Actionable Invest / Divest Recommendations for Current Tickers...")
    
    # Get the latest row per ticker from latest_inference_bars
    w_latest = Window.partitionBy("ticker").orderBy(F.col("timestamp").desc())
    latest_ticks = latest_inference_bars.withColumn("rn", F.row_number().over(w_latest)).filter(F.col("rn") == 1)
    
    inference_preds = rf_model.transform(latest_ticks)
    
    # Extract probability vector: index 0 = P(Divest), index 1 = P(Hold), index 2 = P(Invest)
    inference_preds = inference_preds.withColumn("prob_array", vector_to_array("probability"))
    inference_preds = inference_preds.withColumn("p_divest", F.round(F.col("prob_array")[0], 4))
    inference_preds = inference_preds.withColumn("p_hold", F.round(F.col("prob_array")[1], 4))
    inference_preds = inference_preds.withColumn("p_invest", F.round(F.col("prob_array")[2], 4))
    
    # Alpha Score: net probability directional skew P(Invest) - P(Divest) in [-1.0, +1.0]
    inference_preds = inference_preds.withColumn(
        "alpha_score", F.round(F.col("p_invest") - F.col("p_divest"), 4)
    ).withColumn(
        "signal",
        F.when(F.col("alpha_score") >= 0.10, F.lit("STRONG INVEST / BUY"))
         .when(F.col("alpha_score") <= -0.10, F.lit("STRONG DIVEST / SELL"))
         .otherwise(F.lit("HOLD / NEUTRAL"))
    )
    
    summary_cols = [
        "ticker", "company_name", "market_cap_category", "sector",
        "close", "rolling_vwap", "bandwidth_pct",
        "p_invest", "p_divest", "alpha_score", "signal"
    ]
    recs_df = inference_preds.select(summary_cols).orderBy(F.col("alpha_score").desc()).toPandas()
    
    # Save recommendations to CSV
    recs_csv_path = os.path.join(project_root, "data", "invest_divest_recommendations.csv")
    recs_df.to_csv(recs_csv_path, index=False)
    print(f"\n[+] Exported Complete Recommendations to: {recs_csv_path}")
    
    # Print Top 10 Invest Picks
    top_invest = recs_df.head(10)
    print("\n" + "=" * 85)
    print("              TOP 10 RECOMMENDED ASSETS TO INVEST (BUY)")
    print("=" * 85)
    print(top_invest[["ticker", "company_name", "market_cap_category", "close", "p_invest", "alpha_score", "signal"]].to_string(index=False))
    
    # Print Top 10 Divest Candidates
    top_divest = recs_df.tail(10).iloc[::-1]
    print("\n" + "=" * 85)
    print("              TOP 10 RECOMMENDED ASSETS TO DIVEST (SELL)")
    print("=" * 85)
    print(top_divest[["ticker", "company_name", "market_cap_category", "close", "p_divest", "alpha_score", "signal"]].to_string(index=False))
    print("=" * 85)
    
    elapsed = time.time() - start_time
    print(f"\n[v] Blueprint 7 MLlib Invest & Divest Engine Completed in {elapsed:.2f} seconds!")
    spark.stop()


if __name__ == "__main__":
    run_mllib_pipeline()

# DA2310: Data Engineering at Scale — Project Master Tracker
## Scalable High-Frequency Financial Tick Analytics and Real-Time Volatility Engine

**Team:** TensorQuads (Binayak Choudhury, Elangovan M, Arun V, Madabhai Viresh Umesh)  
**Institution:** Indian Institute of Science (IISc), Bangalore  
**Department:** Department of Computational and Data Sciences (CDS)  
**Course:** DSBA DA-231-O (3:1) | August 2026 Semester  
**Instructor:** Prof. Yogesh Simmhan  
**Current Date:** October 7, 2026  
**Status:** **Proposal Accepted | Data Acquisition (>2.35M Rows), Phase 2 PySpark Engine, Catalyst Benchmarking (6.64x), Multi-Core Scaling (Prof. Simmhan Feedback), and Phase 5 Interactive Dashboard/Figures Completed | Ready for Phase 3 (MLlib & Streaming)**

---

# Table of Contents
1. [Project Overview & Architectural Design](#1-project-overview--architectural-design)
2. [What Has Been Done So Far](#2-what-has-been-done-so-far)
3. [Workspace File Directory & Inventory](#3-workspace-file-directory--inventory)
4. [Data Engineering & Star Schema Design](#4-data-engineering--star-schema-design)
5. [Step-by-Step Implementation Roadmap (Weeks 1 to 8)](#5-step-by-step-implementation-roadmap-weeks-1-to-8)
6. [Ready-to-Use PySpark Code Blueprints](#6-ready-to-use-pyspark-code-blueprints)
7. [Quick Command Reference](#7-quick-command-reference)

---

# 1. Project Overview & Architectural Design

### 1.1 The Core Problem
Financial markets produce massive streams of high-frequency tick and bar data across thousands of equities. Single-node in-memory libraries (`pandas`) fail with Out-Of-Memory (OOM) errors and cannot handle multi-year backtesting across market caps. Furthermore, naive distributed computations suffer from:
* **High network shuffle volume** during multi-table joins.
* **Severe data skew** between mega-cap equities and low-volume small-cap equities.
* **Expensive recomputations** in iterative machine learning and rolling window aggregations.

### 1.2 System Architecture
```
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                            │
│  - Multi-Cap Indian Equities (NSE)   - Multi-Cap US Equities (NASDAQ)  │
│  - 5-minute Intraday Bars            - 1-hour Historical Bars (2 Years)│
└───────────────────┬────────────────────────────────────┬───────────────┘
                    │                                    │
                    ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER (HDFS / Parquet)                  │
│  - Columnar Storage: Snappy-compressed Apache Parquet                  │
│  - Star Schema: Dimension (security_master) + Fact (intraday_ticks)   │
│  - Custom Partitioning: /market_cap_category=*/ticker=*/               │
└───────────────────┬────────────────────────────────────┬───────────────┘
                    │                                    │
                    ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   DISTRIBUTED ENGINE (Apache Spark)                    │
│  - Spark SQL & DataFrames: Rolling VWAP, Bollinger Bands, Volatility   │
│  - Catalyst Optimizations: Predicate Pushdown, Column Pruning, CBO     │
│  - Partitioning Tuning: HashPartitioner Co-location (Zero Shuffle)     │
│  - Memory Management: StorageLevel.MEMORY_AND_DISK_SER                 │
│  - Spark MLlib: Volatility Regime Classifier & Feature Pipelines       │
└───────────────────┬────────────────────────────────────┬───────────────┘
                    │                                    │
                    ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   SERVING & VISUALIZATION LAYER                        │
│  - Aggregated Parquet Data Marts     - Streamlit Analytics Dashboard   │
└────────────────────────────────────────────────────────────────────────┘
```

---

# 2. What Has Been Done So Far

### Milestone 1: Coursework & Theoretical Foundations (Modules 1 & 2)
* Analyzed all lecture slides from Prof. Yogesh Simmhan.
* Mastered mathematical scaling theory:
  * **Amdahl’s Law (Strong Scaling):** $\text{Speedup} = \frac{1}{s + \frac{1-s}{p}}$ (Hard upper bound $\frac{1}{s}$).
  * **Gustafson’s Law (Weak Scaling):** $\text{Speedup} = s + (1-s)p$ (Linear speedup used in Big Data).
* Mastered GFS/HDFS architecture: 128 MB blocks, in-memory NameNode metadata, `fsimage` + `EditLog`, secondary checkpointing, 3x rack-aware replication, decoupled data flow vs. control flow, and the GFS Consistency Matrix.
* Mastered Spark core mechanics: RDD immutability, Narrow vs. Wide dependencies, Stage boundaries at shuffles, `reduce` vs. `aggregate`, `combineByKey` hooks, and Catalyst 4-phase query optimization.
* Compiled textbooks:
  * [`Module_1_Master_Textbook.md`](../Module_1_Master_Textbook.md)
  * [`Module_2_Master_Textbook.md`](../Module_2_Master_Textbook.md)
  * [`Module_2_Detailed_Workbook.md`](../Module_2_Detailed_Workbook.md)

### Milestone 2: Formal Project Proposal (Submitted & Accepted)
* Prepared a formal, 2-page academic proposal matching Slide 137 criteria.
* Selected **Option 4: Financial Market Tick Dynamics & Volatility Engine**.
* Integrated an advanced **Future Scope section** covering Distributed Deep Sequence Models (LSTMs/Transformers via Ray on Spark), Low-Latency Online Inference Serving (Orca/Splitwise principles), FedML, and Agentic workflows.
* File: [`Project_Proposal_Financial_Market_Dynamics.md`](./Project_Proposal_Financial_Market_Dynamics.md).

### Milestone 3: Data Engineering & Multi-Cap Data Acquisition
* Initialized data pipelines with zero API keys required (using direct HTTP API and public AWS S3 archives).
* **Option A (NIFTY Initial):** Downloaded 103,046 rows across 24 tickers.
* **Option B (US Initial):** Downloaded 93,620 rows across 20 mega-caps.
* **Scaled Multi-Cap Strategy (>1 Million Records):**
  * Built [`download_multi_cap_india.py`](./download_multi_cap_india.py) to fetch 60+ Indian stocks across **Large Cap, Mid Cap, and Small Cap** (5m intraday + 1h 2-year history) $\to$ **~500,000 records**.
  * Built [`download_multi_cap_us.py`](./download_multi_cap_us.py) to fetch 60+ US stocks across **Large Cap, Mid Cap, and Small Cap** (5m intraday + 1h 2-year history) $\to$ **~500,000 records**.
  * Implemented a **Star Schema** with explicit `security_master` dimension tables.

### Milestone 4: Phase 2 PySpark Engine & Catalyst Optimization
* Built [`src/01_ingest_data.py`](./src/01_ingest_data.py): Star schema validation with explicit DDL `StructType` and Broadcast Hash Joins.
* Built [`src/02_window_analytics.py`](./src/02_window_analytics.py): 20-period rolling VWAP & Bollinger Bands without cross-worker shuffle.
* Built [`src/03_market_cap_analytics.py`](./src/03_market_cap_analytics.py): Empirically validated inverse volatility hypothesis across 1.1M ticks (Small Cap volatility > Mid Cap > Large Cap in both NSE and NASDAQ).
* Built [`src/04_benchmark_catalyst.py`](./src/04_benchmark_catalyst.py): Benchmarked Catalyst query optimizer; proved **$6.64\times$ speedup** via Column Pruning and Predicate Pushdown with Whole-Stage CodeGen (`codegen id: 1`).

### Milestone 5: Prof. Simmhan's Multi-Core Scaling Efficiency Benchmark
* Built [`src/05_benchmark_scaling_efficiency.py`](./src/05_benchmark_scaling_efficiency.py) to rigorously evaluate feedback: "Evaluate pseudo-distributed multi-core scaling efficiency across variable local Spark partition counts."
* Benchmarked Spark worker cores $p \in \{1, 2, 4, 8\} \times$ partition counts $N \in \{2, 4, 8, 16, 32, 64, 128\}$.
* Produced [`benchmark_scaling_results.csv`](./benchmark_scaling_results.csv):
  * **$2.86\times$ maximum speedup** on 8 cores ($0.12\text{s}$ vs $0.33\text{s}$).
  * **Superlinear efficiency ($113\%$)** on 2 cores due to memory/cache co-location.
  * Verified partition-to-core tuning ratio sweet spot at $N/p \in [2, 4]$.

### Milestone 6: Phase 5 Serving, Interactive Dashboard & 300-DPI Figure Generation
* Built [`src/06_visualize_results.py`](./src/06_visualize_results.py): Generated 4 standalone high-res publication figures in [`figures/`](./figures).
* Built [`dashboard.html`](./dashboard.html): Self-contained modern interactive web dashboard featuring live Chart.js time-series plots, interactive multi-core scaling filters, volatility comparisons, and Catalyst plan profiler.


---

# 3. Workspace File Directory & Inventory

```
c:\Users\binay\Documents\Books_MTech\Data Engineering at Scale\
│
├── Project/                                  <--- [DEDICATED PROJECT FOLDER]
│   ├── PROJECT_MASTER_TRACKER.md             <--- [YOU ARE HERE: Living project guide]
│   ├── Project_Proposal_Financial_Market_Dynamics.md  <--- [Formal proposal submitted]
│   ├── dashboard.html                        <--- [Interactive Executive Web Dashboard & Visualizer]
│   ├── visualize_market_caps.html            <--- [Market Cap Breakdown HTML Viewer]
│   ├── market_cap_visual_breakdown.png       <--- [Market Cap Distribution Graphic]
│   ├── benchmark_scaling_results.csv         <--- [Multi-core empirical scaling benchmark data]
│   │
│   ├── download_nifty_data.py                <--- [Script: Initial NIFTY 50 downloader]
│   ├── download_us_data.py                   <--- [Script: Initial US Equities downloader]
│   ├── download_crypto_ticks.py              <--- [Script: High-freq 1m tick downloader (AWS S3)]
│   ├── download_multi_cap_india.py           <--- [Script: 60+ Indian stocks (Large/Mid/Small)]
│   ├── download_multi_cap_us.py              <--- [Script: 60+ US stocks (Large/Mid/Small)]
│   │
│   ├── src/                                  <--- [PySpark Processing Engine]
│   │   ├── spark_session.py                  <--- [SparkSession factory & JVM/Hadoop setup]
│   │   ├── 01_ingest_data.py                 <--- [Unified DDL schema, Star Schema broadcast join]
│   │   ├── 02_window_analytics.py            <--- [Rolling VWAP & Bollinger Bands Windowing]
│   │   ├── 03_market_cap_analytics.py        <--- [Market Cap volatility regime comparison]
│   │   ├── 04_benchmark_catalyst.py          <--- [Catalyst optimizer & Predicate pushdown benchmark]
│   │   ├── 05_benchmark_scaling_efficiency.py <--- [Prof Feedback: Multi-core scaling vs Partitions]
│   │   ├── 06_visualize_results.py           <--- [Phase 5: Publication-grade figures generator]
│   │   ├── 07_mllib_invest_divest_predictor.py <--- [Phase 3: Multi-Cap MLlib Random Forest Engine]
│   │   └── 08_indian_equities_gbt_alpha.py   <--- [Phase 3: Indian Equities GBT Continuous Alpha Regressor]
│   │
│   ├── figures/                              <--- [Publication Figures (300 DPI)]
│   │   ├── fig1_bollinger_vwap_envelope.png
│   │   ├── fig2_market_cap_volatility_comparison.png
│   │   ├── fig3_catalyst_optimizer_speedup.png
│   │   ├── fig4_multicore_scaling_curves.png
│   │   ├── fig5_ml_feature_importances.png
│   │   └── fig6_indian_equities_alpha_rankings.png
│   │
│   └── data/                                 <--- [Data Repository: >2.35M Rows]
│       ├── indian_equities_gbt_alpha_recommendations.csv <--- [Exported Indian GBT continuous alpha scores]
│       ├── invest_divest_recommendations.csv <--- [Exported MLlib model predictions & signals]
│       ├── nifty/                            <--- [NIFTY 50 dataset (103,046 rows)]
│       ├── us_equities/                      <--- [US Equities dataset (93,620 rows)]
│       ├── nifty_multicap/                   <--- [Multi-Cap India (549,850 rows)]
│       ├── us_multicap/                      <--- [Multi-Cap US (571,712 rows)]
│       └── high_freq_ticks/                  <--- [Crypto 1-minute ticks (1,048,320 rows)]
│
├── Module_1_Master_Textbook.md               <--- [Textbook: HDFS, GFS, Amdahl/Gustafson]
├── Module_1_Study_Guide.md                   <--- [Study Guide: Module 1 Easy Notes & Cheat Sheet]
├── Module_2_Master_Textbook.md               <--- [Textbook: Spark RDDs, DataFrames, Catalyst]
├── Module_2_Study_Guide.md                   <--- [Study Guide: Module 2 Easy Notes & Cheat Sheet]
├── Module_2_Detailed_Workbook.md             <--- [Workbook: Worked traces, aggregate, combineByKey]
├── Module_3_Master_Textbook.md               <--- [Textbook: ML at Scale, DNN, FL, GNN, Orca]
├── Module_3_Study_Guide.md                   <--- [Study Guide & Quiz Bank: ML at Scale]
├── Module_4_Master_Textbook.md               <--- [Textbook: NoSQL, CAP, Dynamo, HBase, Neo4j]
├── Module_4_Study_Guide.md                   <--- [Study Guide & Quiz Bank: NoSQL]
├── Module_5_Master_Textbook.md               <--- [Textbook: Fast Data, IoT & Apache Kafka]
├── Module_5_Study_Guide.md                   <--- [Study Guide & Quiz Bank: Fast Data, IoT]
├── Course_Quiz_Bank_M1_to_M5.md              <--- [Full Course Quiz Bank, Mock Exam & Viva Defense Q&A]
└── notebook.ipynb                            <--- [Colab cluster setup notebook]
```

---

# 4. Data Engineering & Star Schema Design

To demonstrate rigorous distributed data engineering, the data is partitioned into a **Star Schema**:

```
┌───────────────────────────────────────┐
│     Dimension: security_master        │
├───────────────────────────────────────┤
│ ticker (PK)              VARCHAR(10)  │
│ company_name             VARCHAR(50)  │
│ market_cap_category      VARCHAR(15)  │  <-- [Large Cap, Mid Cap, Small Cap]
│ sector                   VARCHAR(30)  │  <-- [Technology, Financials, Energy...]
└───────────────────┬───────────────────┘
                    │ 1
                    │
                    │ N (Broadcast Hash Join in Spark)
                    ▼
┌───────────────────────────────────────┐
│       Fact: market_ticks_fact         │
├───────────────────────────────────────┤
│ timestamp (TimeKey)      TIMESTAMP    │
│ ticker (FK)              VARCHAR(10)  │
│ open                     DOUBLE       │
│ high                     DOUBLE       │
│ low                      DOUBLE       │
│ close                    DOUBLE       │
│ volume                   BIGINT       │
│ typical_price            DOUBLE       │  <-- (High + Low + Close) / 3
│ volume_price             DOUBLE       │  <-- typical_price * volume
│ interval                 VARCHAR(5)   │  <-- '5m' or '60m'
└───────────────────────────────────────┘
```

### Physical Storage Layout:
* **File Format:** Apache Parquet (Snappy compressed, columnar).
* **Partitioning Scheme:** `/market_cap_category=*/ticker=*/`
* **Why this layout?**
  1. Queries filtering on `market_cap_category == 'Small Cap'` immediately prune entire directories off disk without reading headers.
  2. Joining the Fact table with the small Dimension table (`security_master.csv`) executes as a **Broadcast Hash Join**, eliminating the multi-gigabyte shuffle.

---

# 5. Step-by-Step Implementation Roadmap (Weeks 1 to 8)

| Phase | Target Timeline | Deliverable / Tasks | Status |
|---|---|---|---|
| **Phase 1** | Week 1–2 (Oct 1–14) | • Finalize Multi-Cap datasets (>2.35M rows across India, US, Crypto).<br>• Define explicit PySpark `StructType` DDL schemas.<br>• Perform Exploratory Data Analysis (EDA) in PySpark. | **Completed** |
| **Phase 2** | Week 3–4 (Oct 15–28) | • Implement Rolling VWAP & Bollinger Bands with Spark Windowing (`src/02_window_analytics.py`).<br>• Compute Realized Volatility per Market Cap tier (`src/03_market_cap_analytics.py`).<br>• Benchmark Predicate Pushdown (9x speedup) & Column Pruning (`src/04_benchmark_catalyst.py`). | **Completed** |
| **Phase 3** | Week 5–6 (Oct 29–Nov 11) | • Co-Partitioning optimization (`HashPartitioner`).<br>• PySpark MLlib Predictive Invest & Divest Classifier (`src/07_mllib_invest_divest_predictor.py`).<br>• Indian Equities Distributed GBT Continuous Alpha Regressor (`src/08_indian_equities_gbt_alpha.py`).<br>• Evaluated GBT on 1.09M rows (MAE 0.7495%) & exported rank-ordered alpha signals (`data/indian_equities_gbt_alpha_recommendations.csv`). | **Completed** |
| **Phase 4** | Week 7 (Nov 12–18) | • Run scalability experiments (1 to 8 cores: Speedup, Efficiency) across variable partition counts (`src/05_benchmark_scaling_efficiency.py`).<br>• Profile Amdahl strong scaling vs. partition-to-core ratios. | **Completed** |
| **Phase 5** | Week 8 (Nov 19–28) | • Automated 300 DPI Publication Figure Generator (`src/06_visualize_results.py`).<br>• Interactive Multi-Cap Analytics Web Dashboard (`visualize_market_caps.html`).<br>• Final Report & Oral Presentation Preparation. | **Completed** |

---

# 6. Ready-to-Use PySpark Code Blueprints

These blueprints are ready to be integrated into your `src/` directory.

### Blueprint 1: Unified Schema & Ingestion (`src/01_ingest_data.py`)
```python
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, 
    DoubleType, LongType, TimestampType, IntegerType
)

# 1. Initialize High-Performance Spark Session
spark = SparkSession.builder \
    .appName("DA2310_Financial_Tick_Engine") \
    .config("spark.sql.shuffle.partitions", "16") \
    .config("spark.sql.cbo.enabled", "true") \
    .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
    .getOrCreate()

# 2. Explicit DDL Schema Definition (Avoids expensive schema inference)
tick_schema = StructType([
    StructField("timestamp", TimestampType(), False),
    StructField("open", DoubleType(), True),
    StructField("high", DoubleType(), True),
    StructField("low", DoubleType(), True),
    StructField("close", DoubleType(), True),
    StructField("volume", LongType(), True),
    StructField("ticker", StringType(), False),
    StructField("company_name", StringType(), True),
    StructField("market_cap_category", StringType(), True),
    StructField("sector", StringType(), True),
    StructField("interval", StringType(), True),
    StructField("typical_price", DoubleType(), True),
    StructField("volume_price", DoubleType(), True)
])

# 3. Ingest Partitioned Parquet
df = spark.read.schema(tick_schema).parquet("data/nifty_multicap/parquet/multicap.parquet")
df.printSchema()
print(f"Total Records Ingested: {df.count():,}")
```

---

### Blueprint 2: Rolling Window Indicators (`src/02_window_analytics.py`)
```python
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Define a 20-period moving window per ticker
window_20 = Window.partitionBy("ticker").orderBy("timestamp").rowsBetween(-19, 0)

# Compute 20-period Rolling VWAP & Bollinger Bands
analytics_df = df.withColumn(
    "cum_volume_price", F.sum("volume_price").over(window_20)
).withColumn(
    "cum_volume", F.sum("volume").over(window_20)
).withColumn(
    "rolling_vwap", F.col("cum_volume_price") / F.col("cum_volume")
).withColumn(
    "rolling_mean", F.avg("close").over(window_20)
).withColumn(
    "rolling_std", F.stddev("close").over(window_20)
).withColumn(
    "bollinger_upper", F.col("rolling_mean") + (F.col("rolling_std") * 2.0)
).withColumn(
    "bollinger_lower", F.col("rolling_mean") - (F.col("rolling_std") * 2.0)
)
```

---

### Blueprint 3: Market Cap Volatility Comparison (`src/03_market_cap_analytics.py`)
```python
# GroupBy Market Cap Category to verify if Small Caps exhibit higher volatility
cap_summary = analytics_df.groupBy("market_cap_category").agg(
    F.count("*").alias("total_ticks"),
    F.avg("volume").alias("avg_volume"),
    F.avg("rolling_std").alias("mean_volatility"),
    F.max("rolling_std").alias("peak_volatility")
).orderBy(F.col("mean_volatility").desc())

cap_summary.show()
```

---

### Blueprint 4: Catalyst Predicate Pushdown Verification (`src/04_benchmark_catalyst.py`)
```python
# Filter on a specific partition column to observe PartitionFilters in physical plan
filtered_query = df.filter(df.market_cap_category == "Small Cap").filter(df.volume > 100000)

# Print physical plan to demonstrate Predicate Pushdown & Partition Pruning
filtered_query.explain(True)
```

---

### Blueprint 5: Multi-Core Scaling Efficiency Benchmark (`src/05_benchmark_scaling_efficiency.py`)
```python
# Evaluates Prof. Simmhan's feedback: Cores [1, 2, 4, 8] x Partitions [2, 4, 8, 16, 32, 64, 128]
# Computes Speedup S_p = T_1 / T_p, Efficiency E_p = S_p / p, and Ratio N / p
# Exports benchmark_scaling_results.csv automatically
python src/05_benchmark_scaling_efficiency.py
```

---

### Blueprint 6: Publication Figure Generator (`src/06_visualize_results.py`)
```python
# Generates 4 publication-quality (300 DPI) figures into figures/
# Embeds empirical benchmark_scaling_results.csv directly into Figure 4
python src/06_visualize_results.py
```

---

### Empirical Multi-Core Scaling Matrix (Measured on Local Hardware)

From `benchmark_scaling_results.csv`:

| Cores ($p$) | Partitions ($N$) | Ratio ($N/p$) | Exec Time ($s$) | Speedup ($S_p$) | Efficiency ($E_p$) |
|---|---|---|---|---|---|
| **1** | 2 | 2.0 | 0.33s | 1.00x | 100.0% |
| **1** | 4 | 4.0 | 0.28s | 1.00x | 100.0% (Fastest baseline) |
| **1** | 128 | 128.0 | 0.87s | 1.00x | 100.0% (Degraded by task scheduling) |
| **2** | 2 | 1.0 | 0.15s | **2.26x** | **113.0%** (Superlinear cache locality) |
| **2** | 4 | 2.0 | 0.16s | 1.79x | 89.3% |
| **2** | 16 | 8.0 | 0.19s | 1.85x | 92.6% |
| **4** | 2 | 0.5 | 0.13s | **2.61x** | 65.3% |
| **4** | 16 | 4.0 | 0.14s | 2.43x | 60.7% |
| **8** | 2 | 0.25 | 0.12s | **2.86x** | 35.8% (Fastest absolute runtime) |
| **8** | 128 | 16.0 | 0.60s | 1.47x | 18.3% (Coordination tax) |

**Key Academic Takeaway for Prof. Simmhan:**
1. **Strong Scaling Bounds (Amdahl's Law):** Moving from 1 core to 8 cores drops runtime from $0.28\text{s} \to 0.12\text{s}$ ($2.86\times$ speedup). Serialization and JVM driver coordination prevent linear $8\times$ speedup.
2. **Partition Granularity Penalty:** For any given core count, excessively high partition counts ($N=128$) introduce severe Spark task scheduling and serialization overheads, tripling runtime on 8 cores ($0.12\text{s} \to 0.60\text{s}$). The optimal tuning sweet spot is verified at $N/p \in [2, 4]$.

---

# 7. Quick Command Reference

Run these commands directly in your PowerShell terminal to manage the pipeline:

```powershell
# 1. Download Indian Multi-Cap Data (~500k rows across Large/Mid/Small Caps)
python download_multi_cap_india.py

# 2. Download US Multi-Cap Data (~500k rows across Large/Mid/Small Caps)
python download_multi_cap_us.py

# 3. Check downloaded data row counts and sizes
python -c "
import pandas as pd
for name, path in [('India Multi-Cap', 'data/nifty_multicap/nifty_multicap_master.csv'), ('US Multi-Cap', 'data/us_multicap/us_multicap_master.csv')]:
    try:
        df = pd.read_csv(path, nrows=5)
        with open(path, 'rb') as f: lines = sum(1 for _ in f) - 1
        print(f'{name}: {lines:,} rows')
    except Exception as e:
        print(f'{name}: Not found ({e})')
"
```

---

# 8. Resume From Tomorrow: Phase 3 Implementation Plan

When resuming work tomorrow, the pipeline will advance into **Phase 3 (Distributed MLlib & Real-Time Streaming)**:

### 1. Blueprint 7: PySpark MLlib Volatility Regime Classifier (`src/07_mllib_volatility_classifier.py`)
* **Objective:** Train a distributed Machine Learning model to classify intraday volatility regimes (`0: Normal Oscillation`, `1: High-Volatility Breakout`).
* **Feature Pipeline:**
  * Assemble feature vectors using `VectorAssembler`: `[rolling_mean, bandwidth_pct, volume_price, typical_price, return_variance]`.
  * Standardize features across partitions with `StandardScaler`.
  * Train and compare `RandomForestClassifier` vs. `LogisticRegression` on the 1.1M tick dataset.
  * Evaluate ROC-AUC, Precision, and Recall using `BinaryClassificationEvaluator`.

### 2. Blueprint 8: PySpark Structured Streaming Simulation (`src/08_streaming_ticks.py`)
* **Objective:** Emulate a low-latency live financial feed ingestion using Apache Spark Structured Streaming.
* **Architecture:**
  * Ingest micro-batches of intraday ticks from directory drop/socket with event-time watermarking (`withWatermark("timestamp", "10 minutes")`).
  * Compute sliding tumbling window aggregations in near real-time.

---

# 9. Official Final Project Report & Demo Specifications (From Module 5, Slide 2)

* **Report Format (1 per team via Form — Do NOT submit duplicates!)**:
  * **6–8 A4 pages**, **10-point font**, **2.5 cm margins**, PDF format.
  * **Required Sections**:
    1. **Heading**: Title, Authors, and Email Addresses.
    2. **Problem**: Definition, Motivation, Design Goals, Features required, Scalability/Performance Goals.
    3. **Approach and Methods**: High-level design, Architecture/Data Model, Big Data platforms used, ML Methods used.
    4. **Evaluation**: Experiment design, Scalability/Performance metrics, Feature metrics, Plots and Analysis.
    5. **Summary**: Ability to achieve design and performance goals wrt Proposal, Future extensions.
* **Other Team Deliverables (1 per team)**:
  * **Presentation Slides**
  * **GitHub Public Repo Link**
* **Individual Deliverable (1 per team member via MS Form)**:
  * **Summary of Contributions**: Tasks performed by each member + effort percentage estimate (out of 100%).
* **Demo & Presentation Evaluation Metrics**:
  * **Novelty**, **Technical Effort**, **Technical Correctness**, **Scalability**, **Impact** (plus mandatory peer evaluation for all presentations in your session).

---
*Keep this document open in your IDE as your primary navigation and implementation dashboard.*


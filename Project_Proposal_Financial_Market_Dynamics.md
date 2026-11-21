# Scalable High-Frequency Financial Tick Analytics and Real-Time Volatility Engine

**Course:** DSBA DA-231-O (3:1): Data Engineering at Scale (August 2026)  
**Department:** Department of Computational and Data Sciences (CDS), Indian Institute of Science, Bangalore  
**Instructor:** Prof. Yogesh Simmhan  

**Team Members (Team of 4):**
1. **[Your Name]** (Lead / Data Systems Architect) — <email>@iisc.ac.in
2. **[Team Member 2]** (Data Engineering / ETL & Partitioning) — <email>@iisc.ac.in
3. **[Team Member 3]** (Distributed ML & Analytics) — <email>@iisc.ac.in
4. **[Team Member 4]** (Streaming, Visualization & Benchmarking) — <email>@iisc.ac.in

---

## 1. Problem Statement

### 1.1 Problem Definition
Modern equity and derivatives exchanges generate millions of high-frequency market quotes and transaction ticks daily. Processing, aggregating, and modeling this unstructured and semi-structured time-series data presents severe computational challenges:
1. **I/O and Storage Bottlenecks:** Naive storage formats (uncompressed CSV/JSON) lead to massive disk footprints and exorbitant sequential scan times across multi-year tick horizons.
2. **Shuffle and Skew Inefficiencies:** Grouping and windowed aggregation by equity tickers (e.g., highly traded equities vs. illiquid stocks) causes severe data skew and network shuffle storms during distributed joins.
3. **Dual Processing Latency:** Financial institutions require both **historical deep-backtesting** (batch over terabytes) and **real-time risk surveillance** (sub-second streaming for Value-at-Risk and volatility spikes).

This project designs and implements an end-to-end, scalable big data engineering architecture to ingest, store, partition, process, and analyze multi-year high-frequency financial tick datasets for market dynamics and volatility modeling.

### 1.2 Motivation
Traditional single-node data science environments (e.g., pandas, SQLite) fail with out-of-memory (OOM) errors when processing tick-level intraday historical data ($>10\text{ GB}$ to terabytes). Distributed processing frameworks—specifically Apache Spark and distributed columnar storage—are essential to parallelize windowed analytics (e.g., Rolling Volume-Weighted Average Price [VWAP], Exponential Moving Averages, and Realized Volatility) and train predictive models across thousands of equities simultaneously.

### 1.3 Design Goals & Core Features
1. **Distributed Columnar Storage & Ingestion:** Ingest historical minute-level tick data into HDFS/cloud storage formatted in Snappy-compressed **Apache Parquet**, utilizing custom hierarchical partitioning (year/ticker).
2. **Optimized Batch Analytics via Spark SQL / DataFrames:** Compute complex, multi-scale technical indicators (VWAP, Bollinger Bands, Rolling Beta, Realized Volatility) using Spark Windowing without cross-worker shuffle repartitioning.
3. **Real-Time Stream Simulation:** Simulate live market tick feeds via **Apache Kafka** and **Spark Structured Streaming**, implementing stateful sliding-window aggregations with event-time watermarking to handle late-arriving ticks.
4. **Distributed Volatility Regime Modeling:** Develop a **Spark MLlib** pipeline to classify high-volatility breakout regimes across equity sectors.
5. **System Tuning & Catalyst Optimization:** Profile and quantify the impact of **Predicate Pushdown**, **Column Pruning**, and **Co-Partitioning (partitionBy)** on network shuffle reduction.

---

## 2. Technical Approach & Architecture

### 2.1 Big Data Platforms & System Architecture
The proposed architecture follows a modern Big Data Lambda/Lakehouse pattern:

`
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                            │
│  Historical Archive (Parquet/CSV)  │  Real-Time Ticks (Kafka Producer) │
└───────────────────┬────────────────┴───────────────────┬───────────────┘
                    │                                    │
                    ▼                                    ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│       BATCH STORAGE (HDFS)           │  │   STREAMING INGESTION        │
│  - 128 MB Block Size                 │  │   - Apache Kafka Broker      │
│  - Snappy Columnar Parquet           │  │   - Partitioned by Ticker    │
│  - Partitioned by: /year/ticker/     │  │   - Event-time Watermarking  │
└───────────────────┬──────────────────┘  └──────────────┬───────────────┘
                    │                                    │
                    ▼                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   DISTRIBUTED COMPUTE LAYER (SPARK)                    │
│  - Spark SQL & Catalyst Optimizer (Predicate Pushdown, Column Pruning) │
│  - PySpark Window Functions (Rolling VWAP, Bollinger Bands)            │
│  - Spark Structured Streaming (Stateful Window Aggregations)           │
│  - Spark MLlib Pipeline (Volatility Regime Classification)             │
│  - Memory Management: Tuning MEMORY_AND_DISK_SER & HashPartitioner     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      SERVING & VISUALIZATION LAYER                     │
│  - Parquet Aggregated Marts  │  Streamlit Interactive Dashboards       │
└────────────────────────────────────────────────────────────────────────┘
`

### 2.2 Data Sources & Data Model
* **Data Source:** High-Frequency Intraday Equity Tick Dataset (S&P 500 / NIFTY 500 intraday 1-minute bar data from Kaggle / Yahoo Finance / Alpha Vantage archives; approximately \text{--}20\text{ GB}$, $\sim 40\text{ million records}$).
* **Schema Definition (DDL / StructType):**
  * 	imestamp: TimestampType (ISO-8601 millisecond event time)
  * 	icker: StringType (e.g., AAPL, RELIANCE, INFY)
  * open: DoubleType, high: DoubleType, low: DoubleType, close: DoubleType
  * olume: LongType
  * 	rades_count: IntegerType
  * wap: DoubleType

### 2.3 Optimization Strategies (Direct Alignment with Course Modules)
1. **Co-Partitioning to Eliminate Shuffle:** Heavy joins between static security master tables (sector, market cap) and high-volume tick data will be optimized using HashPartitioner(numPartitions) and .persist(), ensuring co-located joins without network shuffle.
2. **Catalyst Optimizer Benchmarking:** We will inspect physical execution plans (df.explain(True)) to verify whole-stage code generation, filter pushdowns to Parquet row groups, and projection pruning.
3. **Avoiding groupByKey:** All distributed aggregations will strictly use 
educeByKey, combineByKey, or DataFrame native expressions (sum, vg) to leverage map-side combiners.

---

## 3. Evaluation Approach & Success Metrics

### 3.1 Experimental Plan
* **Baseline Comparison:** Single-node execution using Python pandas / sqlite3 vs. Distributed PySpark cluster (scaling across 1, 2, 4, and 8 executor cores).
* **Workload Scenarios:**
  1. Multi-scale rolling window computations (30-day rolling volatility across 500 tickers).
  2. Join benchmark: Unpartitioned full-shuffle join vs. Pre-partitioned Broadcast Hash Join.
  3. Streaming benchmark: Sustained tick ingestion rate under variable watermark delays (10s to 5m).

### 3.2 Performance Metrics (Scalability & Systems)
* **Speedup ($) & Efficiency ($):** Strong scaling evaluation for a fixed 10 GB dataset as executor cores increase from 1 to 8.
* **Shuffle Read/Write Volume (MB):** Measuring network data movement via Spark UI / Metrics Listener.
* **Memory & GC Pressure:** Tracking JVM Garbage Collection pause times under MEMORY_ONLY vs. MEMORY_ONLY_SER.

### 3.3 Feature & Analytical Metrics (Domain Correctness)
* **Mathematical Accuracy:** Cross-validation of Spark-computed VWAP and Bollinger Bands against ground truth quantitative baselines.
* **Model Evaluation:** Precision, Recall, and ROC-AUC of the Spark MLlib volatility regime classifier.
* **Streaming Latency:** End-to-end latency from Kafka production to window materialization.

---

## 4. Timeline & 8-Week Milestones

| Timeline | Milestone / Deliverable | Primary Responsibility |
|---|---|---|
| **Week 1–2 (Oct 1 – Oct 14)** | Ingestion of raw tick data; conversion to Parquet; HDFS/local directory partitioning; baseline exploratory data analysis (EDA). | [Team Member 2] & [Your Name] |
| **Week 3–4 (Oct 15 – Oct 28)** | Spark DataFrame & SQL batch processing engine; Rolling indicator implementations (VWAP, Realized Volatility); Partitioning & Catalyst optimization profiling. | [Your Name] & [Team Member 2] |
| **Week 5–6 (Oct 29 – Nov 11)** | Kafka tick generator simulation; Spark Structured Streaming pipeline with stateful windowing; Spark MLlib feature extraction and classifier training. | [Team Member 3] & [Team Member 4] |
| **Week 7 (Nov 12 – Nov 18)** | End-to-end benchmarking (Speedup, Shuffle volume, GC profiling); Final Report and Public Code Repository submission (**Due Nov 15**). | **All Members** (Led by [Your Name]) |
| **Week 8 (Nov 19 – Nov 28)** | Streamlit interactive visualization dashboard; Slide deck creation; Rehearsal for Final Project Demo & Presentation (**Nov 28/29**). | [Team Member 4] & All Members |

---

## 5. Division of Responsibilities

* **[Your Name] (Lead / Systems Architect):** Core PySpark pipeline architecture, Catalyst query plan optimization, memory persistence tuning, and benchmarking coordinator.
* **[Team Member 2] (Data Engineering & Storage):** HDFS storage layout, Parquet schema design, partitioning strategies (HashPartitioner), and data cleaning.
* **[Team Member 3] (Distributed ML & Analytics):** Feature engineering pipeline (VectorAssembler), volatility regime classification model with Spark MLlib, and analytical validation.
* **[Team Member 4] (Streaming & Visualization):** Kafka producer simulation, Spark Structured Streaming watermarking, and Streamlit monitoring dashboard.

---
## 6. Future Scope & Extensibility: ML-Based Predictive Modeling & Distributed Inference

While the primary 8-week deliverable focuses on core data engineering, distributed windowing, and baseline Spark MLlib classification, the architecture is designed to serve as the scalable backbone for advanced machine learning extensions (aligning with Module 3 of DA2310):

1. **Distributed Deep Sequence Modeling (LSTMs & Temporal Transformers):**  
   Extend the historical feature mart to train deep time-series models (e.g., Temporal Fusion Transformers, GRUs) using distributed deep learning runtimes such as **Ray on Spark** or **Distributed TensorFlow**. This enables multi-step-ahead price forecasting and non-linear order-flow imbalance modeling without moving data outside the cluster.

2. **Ultra-Low-Latency Online Inference & Model Serving:**  
   Incorporate serialized model weights (via ONNX runtime or TorchScript) directly inside **Spark Structured Streaming** micro-batches for sub-millisecond tick classification. For heavy deep models, evaluate decoupled distributed inference architectures (adopting batching and KV-cache serving principles similar to *Orca* and *Splitwise* for high-throughput streaming evaluation).

3. **Federated & Privacy-Preserving Market Intelligence (FedML):**  
   Enable cross-institutional risk and liquidity modeling across disparate brokerages and clearing houses using **Federated Learning (FedML)**, training global anomaly detectors without sharing proprietary client order books.

4. **Agentic Quantitative Workflows:**  
   Integrate autonomous LLM agents to continuously monitor real-time streaming anomalies, synthesize market summary telemetry, and dynamically adjust execution routing parameters.

---
**Code & Data Reproducibility:** All source code, Docker compose environment files, PySpark notebooks, and benchmark scripts will be published in a public GitHub repository upon completion.

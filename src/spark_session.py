"""
DA2310: Data Engineering at Scale
High-Performance SparkSession Initializer for Local & Cluster Environments
"""

import os
import sys
from pyspark.sql import SparkSession

# Standardized runtime paths
DEFAULT_JDK = r"C:\Users\binay\AppData\Local\Programs\jdk-17"
DEFAULT_HADOOP = r"C:\Users\binay\AppData\Local\Programs\hadoop"

def init_environment():
    """Configures environment variables required for PySpark 4 on Windows."""
    if os.path.exists(DEFAULT_JDK):
        os.environ.setdefault("JAVA_HOME", DEFAULT_JDK)
    if os.path.exists(DEFAULT_HADOOP):
        os.environ.setdefault("HADOOP_HOME", DEFAULT_HADOOP)
        hadoop_bin = os.path.join(DEFAULT_HADOOP, "bin")
        if hadoop_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = hadoop_bin + ";" + os.environ.get("PATH", "")
            
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def get_spark_session(app_name="DA2310_Financial_Tick_Engine", master="local[*]"):
    """
    Creates or retrieves an optimized SparkSession with Catalyst CBO,
    proper Windows driver IP binding, and tuned shuffle partitions.
    """
    init_environment()
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.sql.shuffle.partitions", "16") \
        .config("spark.sql.cbo.enabled", "true") \
        .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    return spark

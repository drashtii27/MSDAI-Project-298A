"""
Clean and transform job data using PySpark.
Outputs to silver layer as partitioned parquet.
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, to_timestamp, to_date, lower, trim, regexp_replace,
    when, length, sha2, concat_ws, lit, row_number
)
import pyspark.sql.functions as F
import glob


def main():
    """Main Spark cleaning pipeline."""
    
    # Initialize Spark
    spark = (SparkSession.builder
             .appName("clean_jobs")
             .config("spark.sql.session.timeZone", "UTC")
             .getOrCreate())
    
    # Find bronze files
    paths = glob.glob("data/bronze/jobs/date=*/*.jsonl")
    if not paths:
        raise SystemExit("❌ No bronze job files found. Run 01_fetch_jobs_adzuna.py first.")
    
    print(f"Found {len(paths)} bronze files")
    
    # Read raw data
    df = spark.read.json(paths)
    
    # Select and rename columns
    df = df.select(
        col("id").cast("string").alias("job_id"),
        trim(col("title")).alias("title"),
        col("company").alias("company"),
        col("location").alias("location"),
        col("latitude").cast("double").alias("lat"),
        col("longitude").cast("double").alias("lon"),
        col("description").alias("description_raw"),
        col("category").alias("category"),
        col("salary_min").cast("double").alias("salary_min"),
        col("salary_max").cast("double").alias("salary_max"),
        col("created").alias("posted_raw"),
        col("redirect_url").alias("url")
    )
    
    # Clean text fields
    clean = (df
        .withColumn("title", lower(trim(col("title"))))
        .withColumn("company", trim(col("company")))
        .withColumn("location", trim(col("location")))
        .withColumn("description",
            regexp_replace(lower(col("description_raw")), r"\s+", " "))
        .drop("description_raw")
        .withColumn("posted_ts", to_timestamp("posted_raw"))
        .withColumn("dt", to_date("posted_ts"))
    )
    
    # Calculate mid salary
    clean = clean.withColumn("salary_mid",
        when(col("salary_min").isNotNull() & col("salary_max").isNotNull(),
             (col("salary_min") + col("salary_max")) / 2
        ).otherwise(col("salary_max"))
    )
    
    # Remove duplicates based on title+company+location
    clean = (clean
        .withColumn("dup_key", sha2(concat_ws("||",
            col("title"), col("company"), col("location")), 256))
        .orderBy(col("posted_ts").desc())
        .dropDuplicates(["dup_key"])
        .drop("dup_key")
    )
    
    # Outlier removal for salary
    quantiles = clean.approxQuantile("salary_mid", [0.25, 0.75], 0.05)
    if quantiles and len(quantiles) == 2 and quantiles[0] is not None:
        q1, q3 = quantiles
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5*iqr, q3 + 1.5*iqr
        clean = clean.withColumn("salary_mid",
            when(col("salary_mid").between(lower_bound, upper_bound), col("salary_mid"))
            .otherwise(lit(None).cast("double"))
        )
    
    # Keep only most recent per job_id
    window = Window.partitionBy("job_id").orderBy(F.col("posted_ts").desc_nulls_last())
    clean = (clean
        .withColumn("rn", row_number().over(window))
        .filter("rn = 1")
        .drop("rn")
    )
    
    # Write to silver layer
    output_path = "data/silver/jobs"
    (clean.write
     .mode("overwrite")
     .partitionBy("dt")
     .parquet(output_path))
    
    print(f"✓ Wrote {clean.count()} cleaned jobs to {output_path}")
    
    spark.stop()


if __name__ == "__main__":
    main()
"""
Prefect workflow for job data pipeline (fetch -> clean -> skills -> embeddings -> Neon)
"""

from __future__ import annotations
import sys
import os
from pathlib import Path
import subprocess
import logging

import pandas as pd
from prefect import flow, task

from src.config import SILVER_JOBS
from src.models.embeddings import embed_texts
from src.models.skill_extractor import extract_skills_batch
from src.warehouse.load_to_neon import (
    setup_database,
    insert_jobs,
    insert_job_skills,
    insert_embeddings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("job_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@task
def t_fetch_jobs():
    logger.info("📥 Fetching jobs from Adzuna...")
    script = PROJECT_ROOT / "src" / "01_fetch_jobs_adzuna.py"
    subprocess.check_call([sys.executable, str(script)])
    logger.info("✓ Job fetch complete")


@task
def t_clean_jobs():
    logger.info("🧹 Cleaning jobs with Spark...")
    script = PROJECT_ROOT / "src" / "03_clean_jobs_spark.py"
    subprocess.check_call(["spark-submit", str(script)])
    logger.info("✓ Job cleaning complete")


@task
def t_skill_and_embeddings():
    logger.info("📁 Loading cleaned jobs from silver layer...")
    df_jobs = pd.read_parquet(SILVER_JOBS)
    logger.info(f"Loaded {len(df_jobs)} jobs")

    # Prepare texts
    job_texts = (df_jobs["title"].fillna("") + ". " + df_jobs["description"].fillna("")).tolist()

    logger.info("🧠 Extracting technical skills with hirly-ner-multi + filters...")
    skills_lists = extract_skills_batch(job_texts, batch_size=16)

    pairs = []
    for job_id, skills in zip(df_jobs["job_id"], skills_lists):
        for s in skills:
            pairs.append({"job_id": job_id, "skill": s})
    df_skills = pd.DataFrame(pairs).drop_duplicates()
    logger.info(f"✓ Extracted {len(df_skills)} job-skill pairs")

    logger.info("🔢 Computing embeddings for jobs (bge-base-en-v1.5)...")
    embed_texts_input = (df_jobs["title"].fillna("") + " | " + df_jobs["description"].fillna("")).tolist()
    emb = embed_texts(embed_texts_input)
    df_emb = pd.DataFrame({"job_id": df_jobs["job_id"].tolist(), "embedding": list(emb)})

    return df_jobs, df_skills, df_emb


@task
def t_load_to_neon(df_jobs: pd.DataFrame, df_skills: pd.DataFrame, df_emb: pd.DataFrame):
    logger.info("🗄 Setting up Neon database...")
    setup_database()

    logger.info("⬆ Inserting jobs...")
    insert_jobs(df_jobs)

    logger.info("⬆ Inserting job skills...")
    insert_job_skills(df_skills)

    logger.info("⬆ Inserting job embeddings...")
    insert_embeddings("job_embeddings", "job_id", df_emb)

    logger.info("✓ Data successfully loaded to Neon")


@flow
def job_pipeline():
    print("=" * 70)
    print("JOB DATA PIPELINE - START")
    print("=" * 70)

    t_fetch_jobs()
    t_clean_jobs()
    df_jobs, df_skills, df_emb = t_skill_and_embeddings()
    t_load_to_neon(df_jobs, df_skills, df_emb)

    print("=" * 70)
    print("JOB DATA PIPELINE - COMPLETE ✓")
    print("=" * 70)


if __name__ == "__main__":
    job_pipeline()

from __future__ import annotations

"""
Load processed data to Neon Postgres with pgvector.
Handles jobs, skills, embeddings, interview questions & skills.
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import PG_URL, TBL_INTERVIEW_SKILLS
from src.models.embeddings import load_embedding_model

# Single global engine
engine = create_engine(PG_URL, future=True)


# ---------------------------------------------------------------------
# Helper: always get the *current* embedding dimension
# ---------------------------------------------------------------------
def get_embed_dim() -> int:
    """Ensure model is loaded and return its embedding dimension."""
    model = load_embedding_model()
    return model.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------
# Schema setup
# ---------------------------------------------------------------------
def ensure_extensions():
    with engine.begin() as con:
        con.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    print("✓ pgvector extension ensured")


def ensure_tables():
    embed_dim = get_embed_dim()

    with engine.begin() as con:
        # Jobs
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT,
                company TEXT,
                location TEXT,
                lat DOUBLE PRECISION,
                lon DOUBLE PRECISION,
                description TEXT,
                category TEXT,
                salary_min DOUBLE PRECISION,
                salary_max DOUBLE PRECISION,
                salary_mid DOUBLE PRECISION,
                posted_raw TEXT,
                posted_ts TIMESTAMP,
                dt DATE,
                url TEXT
            );
        """))

        # Job skills
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS job_skills_ner (
                job_id TEXT,
                skill TEXT,
                PRIMARY KEY (job_id, skill)
            );
        """))

        # Job embeddings
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS job_embeddings (
                job_id TEXT PRIMARY KEY,
                embedding VECTOR({embed_dim})
            );
        """))

        # Interview questions
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS interview_questions (
                q_id SERIAL PRIMARY KEY,
                question TEXT NOT NULL,
                topic TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT now()
            );
        """))

        # Interview embeddings
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS interview_embeddings (
                q_id INTEGER PRIMARY KEY,
                embedding VECTOR({embed_dim})
            );
        """))

        # Interview skills
        con.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TBL_INTERVIEW_SKILLS} (
                q_id INTEGER,
                skill TEXT,
                PRIMARY KEY (q_id, skill)
            );
        """))

    print("✓ All tables ensured")


# ---------------------------------------------------------------------
# Inserts for jobs & skills
# ---------------------------------------------------------------------
def insert_jobs(df_jobs: pd.DataFrame):
    df = df_jobs.copy()

    if "posted_ts" in df.columns:
        df["posted_ts"] = pd.to_datetime(df["posted_ts"], errors="coerce")
        try:
            df["posted_ts"] = df["posted_ts"].dt.tz_localize(None)
        except TypeError:
            pass

    if "dt" not in df.columns:
        if "posted_ts" in df.columns:
            df["dt"] = df["posted_ts"].dt.date
        else:
            df["dt"] = pd.Timestamp.utcnow().date()
    else:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.date

    with engine.begin() as con:
        df.to_sql("_jobs_stage", con, if_exists="replace", index=False)
        con.execute(text("""
            INSERT INTO jobs AS t
            (job_id, title, company, location, lat, lon, description, category,
             salary_min, salary_max, salary_mid, posted_raw, posted_ts, dt, url)
            SELECT job_id, title, company, location, lat, lon, description, category,
                   salary_min, salary_max, salary_mid, posted_raw, posted_ts, dt, url
            FROM _jobs_stage
            ON CONFLICT (job_id) DO UPDATE SET
                title = EXCLUDED.title,
                company = EXCLUDED.company,
                location = EXCLUDED.location,
                description = EXCLUDED.description,
                salary_mid = EXCLUDED.salary_mid,
                posted_ts = EXCLUDED.posted_ts,
                dt = EXCLUDED.dt;
            DROP TABLE _jobs_stage;
        """))
    print(f"✓ Inserted/updated {len(df)} jobs")


def insert_job_skills(df_skills: pd.DataFrame):
    if df_skills.empty:
        print("⚠ No job skills to insert")
        return
    df = df_skills[["job_id", "skill"]].drop_duplicates()
    with engine.begin() as con:
        df.to_sql("_skills_stage", con, if_exists="replace", index=False)
        con.execute(text("""
            INSERT INTO job_skills_ner (job_id, skill)
            SELECT job_id, skill FROM _skills_stage
            ON CONFLICT (job_id, skill) DO NOTHING;
            DROP TABLE _skills_stage;
        """))
    print(f"✓ Inserted {len(df)} job-skill pairs")


def insert_interview_skills(df_skills: pd.DataFrame):
    """Insert question-skill pairs into interview_skills table."""
    if df_skills.empty:
        print("⚠ No interview skills to insert")
        return
    df = df_skills[["q_id", "skill"]].drop_duplicates()
    with engine.begin() as con:
        df.to_sql("_int_skills_stage", con, if_exists="replace", index=False)
        con.execute(text(f"""
            INSERT INTO {TBL_INTERVIEW_SKILLS} (q_id, skill)
            SELECT q_id, skill FROM _int_skills_stage
            ON CONFLICT (q_id, skill) DO NOTHING;
            DROP TABLE _int_skills_stage;
        """))
    print(f"✓ Inserted {len(df)} interview-skill pairs")


# ---------------------------------------------------------------------
# Generic embeddings upsert
# ---------------------------------------------------------------------
def insert_embeddings(table_name: str, id_column: str, df_emb: pd.DataFrame) -> None:
    """
    df_emb must have columns:
      - id_column (e.g. 'job_id' or 'q_id')
      - 'embedding' (1D array / list of floats)

    Works for both jobs and interview questions.
    """
    embed_dim = get_embed_dim()

    df = df_emb.copy()

    # -------------------------------------------------------------
    # Make id type match the real table
    #   - job_embeddings.job_id  -> TEXT
    #   - interview_embeddings.q_id -> INTEGER
    # -------------------------------------------------------------
    if id_column == "q_id":
        # interview_embeddings: q_id is INTEGER
        df[id_column] = df[id_column].astype("int64")
        id_sql_type = "INTEGER"
        select_id_expr = id_column          # already INTEGER in temp table
    else:
        # job_embeddings: job_id is TEXT (can be long numeric strings)
        df[id_column] = df[id_column].astype(str)
        id_sql_type = "TEXT"
        select_id_expr = id_column          # NO cast to INT here

    # Ensure embedding is stored as list[float] and will serialize to a string
    df["embedding"] = df["embedding"].apply(
        lambda v: list(map(float, np.asarray(v, dtype=np.float32)))
    )

    with engine.begin() as con:
        # fresh temp table
        con.execute(text("DROP TABLE IF EXISTS _emb_stage;"))
        con.execute(text(f"""
            CREATE TEMP TABLE _emb_stage (
                {id_column} {id_sql_type},
                embedding   TEXT
            );
        """))

        # load into temp table (will append into existing schema)
        df.to_sql("_emb_stage", con, if_exists="append", index=False)

        # upsert into real table
        upsert_sql = """
        INSERT INTO job_embeddings (job_id, embedding)
        SELECT
            job_id,
            REPLACE(REPLACE(embedding, '{', '['), '}', ']')::vector(1024)
        FROM _emb_stage
        ON CONFLICT (job_id) DO UPDATE
            SET embedding = EXCLUDED.embedding;
        DROP TABLE _emb_stage;
    """

        con.execute(text(upsert_sql))

    print(f"✓ Inserted {len(df)} embeddings into {table_name}")

# ---------------------------------------------------------------------
# Top-level setup
# ---------------------------------------------------------------------
def setup_database():
    print("=" * 70)
    print("SETTING UP NEON DATABASE")
    print("=" * 70)
    ensure_extensions()
    ensure_tables()
    print("✓ Database setup complete!")


if __name__ == "__main__":
    setup_database()

"""
Ingest interview questions from CSV files, compute embeddings, and load into Neon.

- Reads all CSVs in `interview_data/` (skips ones with 'behavioral' in the name)
- Keeps only technical questions (topic contains 'tech')
- Truncates existing interview tables
- Inserts questions + pgvector embeddings

You can run this directly:

    (venv) python src/interview_ingest_clean_load.py

or via Prefect:

    (venv) python -m src.flows.interview_pipeline_prefect
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path BEFORE importing src.*
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../career_mentor_modern
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PG_URL,
    TBL_INTERVIEW_QUESTIONS,
    TBL_INTERVIEW_EMBEDDINGS,
)
from src.models.embeddings import embed_texts  # uses EMBED_MODEL_NAME + EMBED_DIM from .env
from src.warehouse.load_to_neon import insert_embeddings

# Folder that holds the CSVs
INTERVIEW_DIR = PROJECT_ROOT / "interview_data"


# ---------------------------------------------------------------------------
# CSV LOADING HELPERS
# ---------------------------------------------------------------------------

def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    """Try reading CSV with different encodings."""
    last_error: Optional[Exception] = None
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    warnings.warn(f"Failed to parse {path.name}: {last_error}")
    return None


def load_interview_sources() -> pd.DataFrame:
    """Load **technical** interview questions from all CSV files."""
    print(f"Loading interview questions from: {INTERVIEW_DIR}")

    if not INTERVIEW_DIR.exists():
        raise RuntimeError(f"Interview directory does not exist: {INTERVIEW_DIR}")

    all_dfs: list[pd.DataFrame] = []

    for csv_path in INTERVIEW_DIR.glob("*.csv"):
        # Completely skip behavioral files
        if "behavioral" in csv_path.name.lower():
            print(f"Skipping behavioral file: {csv_path.name}")
            continue

        print(f"Reading: {csv_path.name}")
        df = _read_csv_safe(csv_path)
        if df is None:
            continue

        # Normalize column names
        df.columns = [c.lower().strip() for c in df.columns]

        # Find question column
        q_col = None
        for candidate in ("question", "questions", "interview_question"):
            if candidate in df.columns:
                q_col = candidate
                break

        if not q_col:
            print(f"  ⚠ No 'question' column found in {csv_path.name}, skipping")
            continue

        df = df.rename(columns={q_col: "question"})

        # Topic
        if "topic" in df.columns:
            df["topic"] = df["topic"].fillna("technical")
        else:
            df["topic"] = "technical"

        df["source"] = csv_path.name

        all_dfs.append(df[["question", "topic", "source"]])

    if not all_dfs:
        raise RuntimeError(f"No valid interview CSV files found in {INTERVIEW_DIR}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["question"] = combined["question"].astype(str).str.strip()

    # Drop empty / nan questions
    combined = combined[
        combined["question"].ne("") & combined["question"].ne("nan")
    ]

    # Keep only technical topics
    combined = combined[combined["topic"].str.contains("tech", case=False, na=False)]

    # Drop duplicates
    combined = combined.drop_duplicates(subset=["question"]).reset_index(drop=True)

    print(f"✓ Loaded {len(combined)} unique technical interview questions")
    return combined


# ---------------------------------------------------------------------------
# LOAD TO NEON (POSTGRES + PGVECTOR)
# ---------------------------------------------------------------------------

def load_to_postgres(df_questions: pd.DataFrame) -> None:
    """Truncate and reload interview_questions + interview_embeddings."""
    print("\nConnecting to Neon...")
    engine = create_engine(PG_URL)

    tbl_q = TBL_INTERVIEW_QUESTIONS
    tbl_e = TBL_INTERVIEW_EMBEDDINGS

    print(f"Using tables: {tbl_q}, {tbl_e}")

    # 1) Clear + insert questions in ONE short transaction
    with engine.begin() as con:
        print("Clearing existing interview data...")
        con.execute(text(f"TRUNCATE TABLE {tbl_e} CASCADE"))
        con.execute(text(f"TRUNCATE TABLE {tbl_q} RESTART IDENTITY CASCADE"))

        print("Inserting interview questions...")
        df_questions[["question", "topic", "source"]].to_sql(
            tbl_q.split(".")[-1],  # table name only
            con,
            if_exists="append",
            index=False,
        )

    # 2) Fetch q_id + question in a fresh, short transaction
    with engine.begin() as con:
        rows = con.execute(
            text(f"SELECT q_id, question FROM {tbl_q}")
        ).fetchall()

    q_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    # 3) Compute embeddings OUTSIDE any DB transaction
    print(f"Computing embeddings for {len(texts)} questions...")
    embeddings = embed_texts(texts)  # np.ndarray [N, EMBED_DIM]

    # Build DataFrame compatible with insert_embeddings()
    # Make sure these are plain Python lists, not raw numpy arrays
    df_emb = pd.DataFrame({
        "q_id": q_ids,
        "embedding": [emb.tolist() for emb in embeddings],
    })

    # 4) Insert embeddings using its own transaction inside insert_embeddings()
    print("Inserting embeddings (pgvector)...")
    insert_embeddings(tbl_e, "q_id", df_emb)

    print("✓ Successfully loaded interview data to Neon")



# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 70)
    print("INTERVIEW DATA PIPELINE - START")
    print("=" * 70 + "\n")

    df_questions = load_interview_sources()
    load_to_postgres(df_questions)

    print("\n" + "=" * 70)
    print("INTERVIEW DATA PIPELINE - COMPLETE ✓")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

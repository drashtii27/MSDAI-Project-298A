"""
Prefect flow for the interview data pipeline.

Run with:

    (venv) python -m src.flows.interview_pipeline_prefect

or:

    (venv) python src/flows/interview_pipeline_prefect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from prefect import flow, task

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interview_ingest_clean_load import main as interview_main


@task(name="t_ingest_clean_load")
def t_ingest_clean_load():
    """Run the interview ingestion + embeddings pipeline."""
    interview_main()


@flow(name="interview-pipeline")
def interview_pipeline():
    t_ingest_clean_load()


if __name__ == "__main__":
    interview_pipeline()

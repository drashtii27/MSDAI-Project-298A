"""
AI-Driven Career Mentor Streamlit Application — FIXED / STABLE VERSION
"""

import os
import sys
from typing import List
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Add src/ to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

# ============================================================================
# SAFE IMPORTS
# ============================================================================

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    st.error("❌ Install psycopg2-binary: pip install psycopg2-binary")

try:
    from src.models.embeddings import embed_query
    EMBEDDINGS_AVAILABLE = True
except Exception as e:
    EMBEDDINGS_AVAILABLE = False
    st.error(f"❌ Could not import embeddings: {e}")

try:
    from src.models.roadmap_generator import generate_career_roadmap
    ROADMAP_AVAILABLE = True
except Exception as e:
    ROADMAP_AVAILABLE = False
    st.error(f"❌ Could not import roadmap module: {e}")

# ============================================================================
# CONFIG
# ============================================================================

PG_URL = os.getenv("PG_URL")

if not PG_URL:
    st.error("❌ PG_URL missing in .env")
    st.stop()

# ============================================================================
# DATABASE UTILITIES — FIXED (NO CACHED CONNECTION)
# ============================================================================

def get_connection():
    """
    Open a NEW connection for every query.
    Do NOT cache connections. Neon auto-closes idle connections.
    """
    try:
        return psycopg2.connect(PG_URL, sslmode="require")
    except Exception as e:
        st.error(f"❌ Database connection error: {e}")
        return None


def run_query(sql: str, params=None, return_dict=False):
    """Safe DB query wrapper."""
    conn = get_connection()
    if conn is None:
        return None

    try:
        cursor_factory = RealDictCursor if return_dict else None
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            cur.execute(sql, params or ())
            rows = cur.fetchall()
            return rows
    except Exception as e:
        st.error(f"❌ Database query failed: {e}")
        return None
    finally:
        conn.close()


def test_db():
    """Check DB health."""
    result = run_query("SELECT 1")
    return result is not None


# ============================================================================
# QUERY FUNCTIONS — FIXED
# ============================================================================

def get_similar_jobs(target_role: str, user_skills: str, top_k: int = 30) -> pd.DataFrame:
    if not EMBEDDINGS_AVAILABLE:
        return pd.DataFrame()

    query_text = f"Target role: {target_role}. Current skills: {user_skills}"
    query_emb = embed_query(query_text).tolist()

    rows = run_query(
        """
        SELECT
            j.job_id,
            j.title,
            j.company,
            j.location,
            j.salary_mid,
            j.url,
            e.embedding <-> %s::vector AS distance
        FROM job_embeddings e
        JOIN jobs j ON j.job_id = e.job_id
        WHERE j.title IS NOT NULL
        ORDER BY e.embedding <-> %s::vector
        LIMIT %s;
        """,
        (query_emb, query_emb, top_k)
    )

    if not rows:
        return pd.DataFrame()

    cols = ["job_id", "title", "company", "location", "salary_mid", "url", "distance"]
    return pd.DataFrame(rows, columns=cols)


def get_top_skills_for_jobs(job_ids: List[str], top_n: int = 30) -> pd.DataFrame:
    if not job_ids:
        return pd.DataFrame()

    rows = run_query(
        """
        SELECT skill, COUNT(*) AS count
        FROM job_skills_ner
        WHERE job_id = ANY(%s)
        GROUP BY skill
        ORDER BY count DESC
        LIMIT %s;
        """,
        (job_ids, top_n)
    )

    return pd.DataFrame(rows, columns=["skill", "count"]) if rows else pd.DataFrame()


def get_interview_questions(target_role: str, user_skills: str, top_k: int = 15) -> pd.DataFrame:
    if not EMBEDDINGS_AVAILABLE:
        return pd.DataFrame()

    query_text = f"Technical interview questions for {target_role}. Candidate skills: {user_skills}"
    query_emb = embed_query(query_text).tolist()

    rows = run_query(
        """
        SELECT
            q.q_id,
            q.question,
            q.topic,
            q.source,
            e.embedding <-> %s::vector AS distance
        FROM interview_embeddings e
        JOIN interview_questions q ON q.q_id = e.q_id
        ORDER BY e.embedding <-> %s::vector
        LIMIT %s;
        """,
        (query_emb, query_emb, top_k)
    )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["q_id", "question", "topic", "source", "distance"])
    df["similarity"] = 1 / (1 + df["distance"])
    return df


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="AI Career Mentor",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 AI-Driven Career Mentor & Market Analyzer")

with st.expander("🔧 System Status"):
    col1, col2, col3, col4 = st.columns(4)
    col1.write("✓ psycopg2" if PSYCOPG2_AVAILABLE else "❌ psycopg2")
    col2.write("✓ Embeddings" if EMBEDDINGS_AVAILABLE else "❌ Embeddings")
    col3.write("✓ Roadmap LLM" if ROADMAP_AVAILABLE else "❌ Roadmap")
    col4.write("✓ Database" if test_db() else "❌ Database")

st.sidebar.header("🎯 Your Career Goals")
target_role = st.sidebar.text_input("Target Role", value="Machine Learning Engineer")
user_skills_str = st.sidebar.text_area("Your Current Skills", value="Python, SQL, Pandas")
top_k_jobs = st.sidebar.slider("Number of similar jobs", 10, 100, 30, 5)
top_k_questions = st.sidebar.slider("Number of interview questions", 5, 50, 15, 5)
analyze_button = st.sidebar.button("🔍 Analyze Career Path", type="primary")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if analyze_button:

    # -------------------------------
    # 1. Similar Jobs
    # -------------------------------
    st.subheader("📊 Representative Jobs")

    similar_jobs = get_similar_jobs(target_role, user_skills_str, top_k_jobs)

    if similar_jobs.empty:
        st.error("❌ No jobs found. Run: python src/flows/job_pipeline_prefect.py")
        st.stop()

    st.success(f"Found {len(similar_jobs)} similar jobs")
    st.dataframe(similar_jobs.head(15), hide_index=True, use_container_width=True)

    # -------------------------------
    # 2. Skills
    # -------------------------------
    st.subheader("💼 Top Skills")

    job_ids = similar_jobs["job_id"].tolist()
    top_skills = get_top_skills_for_jobs(job_ids)

    if top_skills.empty:
        st.warning("⚠️ No skills found")
    else:
        st.bar_chart(top_skills.set_index("skill")["count"].head(20))

    # -------------------------------
    # 3. Interview Questions
    # -------------------------------
    st.subheader("❓ Interview Questions")

    questions = get_interview_questions(target_role, user_skills_str, top_k_questions)

    if questions.empty:
        st.error("⚠️ No questions found. Run interview pipeline.")
    else:
        for idx, row in questions.iterrows():
            st.markdown(f"**{idx+1}. {row['question']}**")
            st.caption(f"Topic: {row['topic']} | Source: {row['source']} | Similarity: {row['similarity']:.3f}")
            st.markdown("---")

    # -------------------------------
    # 4. Roadmap Generator
    # -------------------------------
    st.subheader("🗺️ Career Roadmap")

    if ROADMAP_AVAILABLE:
        roadmap = generate_career_roadmap(
            target_role,
            [s.strip() for s in user_skills_str.split(",")],
            top_skills["skill"].head(10).tolist(),
            similar_jobs["title"].unique().tolist(),
            questions["question"].head(5).tolist()
        )
        st.markdown(roadmap)
    else:
        st.error("Roadmap module unavailable.")

else:
    st.info("👈 Enter details and click **Analyze Career Path**")


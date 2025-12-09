import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = BASE_DIR
DATA_DIR = BASE_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

for p in [BRONZE_DIR, SILVER_DIR, GOLD_DIR]:
    p.mkdir(parents=True, exist_ok=True)

INTERVIEW_DIR = BASE_DIR / "interview_data"
INTERVIEW_MAX = int(os.getenv("INTERVIEW_MAX", "500000"))

SILVER_JOBS = SILVER_DIR / "jobs"
SILVER_JOB_SKILLS = SILVER_DIR / "job_skills_ner"
SILVER_JOB_EMBEDDINGS = SILVER_DIR / "job_embeddings"

# ============================================================================
# DATABASE (NEON POSTGRES)
# ============================================================================
PG_URL = os.getenv("PG_URL")
if not PG_URL:
    raise RuntimeError("Missing PG_URL in .env file")

TBL_JOBS = "jobs"
TBL_JOB_SKILLS = "job_skills_ner"
TBL_JOB_EMBEDDINGS = "job_embeddings"
TBL_INTERVIEW_QUESTIONS = "interview_questions"
TBL_INTERVIEW_EMBEDDINGS = "interview_embeddings"
TBL_INTERVIEW_SKILLS = "interview_skills_ner"

# ============================================================================
# API KEYS
# ============================================================================
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "us")
ADZUNA_WHERE = os.getenv("ADZUNA_WHERE", "United States")
ADZUNA_MAX_PAGES = int(os.getenv("ADZUNA_MAX_PAGES", "40"))
ADZUNA_RESULTS_PER_PAGE = int(os.getenv("ADZUNA_RESULTS_PER_PAGE", "50"))

HF_TOKEN = os.getenv("HF_TOKEN")

PREFECT_API_KEY = os.getenv("PREFECT_API_KEY")
PREFECT_API_URL = os.getenv("PREFECT_API_URL")

# ============================================================================
# AI MODELS (NEW)
# ============================================================================

# 1) Skill extraction – HR CV/JD NER
SKILL_NER_MODEL = os.getenv(
    "SKILL_NER_MODEL",
    "feliponi/hirly-ner-multi"
)

# 2) Embeddings – unified for jobs & questions
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "BAAI/bge-base-en-v1.5"
)
SENTENCE_EMB_MODEL = EMBED_MODEL
EMBED_DIM = 768  # bge-base-en-v1.5

# 3) Roadmap LLM – modern instruct model
CAREER_LLM_MODEL = os.getenv(
    "CAREER_LLM_MODEL",
    "mistralai/Mistral-Nemo-Instruct-2407"
)

# Optional tiny model used by llm_gap_analyzer as fallback
CAREER_MENTOR_LLM_MODEL = os.getenv(
    "CAREER_MENTOR_LLM_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct"
)

SKILL_EXTRACTION_BATCH_SIZE = 16
EMBEDDING_BATCH_SIZE = 32
LLM_MAX_NEW_TOKENS = 1600
LLM_TEMPERATURE = 0.6

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "BRONZE_DIR", "SILVER_DIR", "GOLD_DIR",
    "SILVER_JOBS", "SILVER_JOB_SKILLS", "SILVER_JOB_EMBEDDINGS",
    "INTERVIEW_DIR", "INTERVIEW_MAX",
    "PG_URL",
    "TBL_JOBS", "TBL_JOB_SKILLS", "TBL_JOB_EMBEDDINGS",
    "TBL_INTERVIEW_QUESTIONS", "TBL_INTERVIEW_EMBEDDINGS", "TBL_INTERVIEW_SKILLS",
    "ADZUNA_APP_ID", "ADZUNA_APP_KEY", "ADZUNA_COUNTRY", "ADZUNA_WHERE",
    "ADZUNA_MAX_PAGES", "ADZUNA_RESULTS_PER_PAGE",
    "HF_TOKEN",
    "SKILL_NER_MODEL", "EMBED_MODEL", "SENTENCE_EMB_MODEL", "EMBED_DIM",
    "CAREER_LLM_MODEL", "CAREER_MENTOR_LLM_MODEL",
    "SKILL_EXTRACTION_BATCH_SIZE", "EMBEDDING_BATCH_SIZE",
    "LLM_MAX_NEW_TOKENS", "LLM_TEMPERATURE",
    "PREFECT_API_KEY", "PREFECT_API_URL",
]

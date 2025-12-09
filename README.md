# AI-driven Autonomous Career Mentor & Market Analyzer (Modern LLM Edition)

This repo is a **from-scratch, runnable project directory** that implements:

- Cloud data pipelines (jobs + interview questions) orchestrated by **Prefect**.
- A **Neon Postgres** warehouse with `pgvector` for embeddings.
- Modern **Hugging Face models only**:
  - DeBERTa-based **skill NER** for job descriptions.
  - **BGE** sentence embeddings for jobs & interview questions.
  - An instruction **LLM** (Mistral) for career trajectory & networking advice.
- A **Streamlit UI** that, given a role like `ML engineer`:
  - Analyzes job market trends & required skills.
  - Surfaces interview questions to prepare.
  - Generates a personalised skill-gap plan and networking actions.

No classic regression / TF‑IDF classifiers / regex-based models are used in the main stack.

---

## 1. Quick start

### 1.1. Clone & install

```bash
cd career_mentor_modern
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1.2. Create `.env`

Copy the template and fill in:

```bash
cp .env.example .env
```

Edit `.env` and set:

- `ADZUNA_APP_ID`, `ADZUNA_APP_KEY`
- `PG_URL` (Neon connection string)
- `HF_TOKEN` (for private HF models like Mistral)
- `PREFECT_API_KEY` / `PREFECT_API_URL` (optional but useful)
- `INTERVIEW_DIR` (default: `interview_data` inside this repo)

### 1.3. Get a free Neon Postgres database

1. Go to https://neon.tech and sign up (GitHub / Google login is fine).
2. Create a new **project** and **database**.
3. In the **Connection details** panel, copy the **Postgres connection string**.
   - It will look like:  
     `postgresql://user:pass@ep-xxx.eu-central-1.aws.neon.tech/neondb`
4. Turn it into a SQLAlchemy URL in `.env`:

```env
PG_URL=postgresql+psycopg2://user:pass@ep-xxx.eu-central-1.aws.neon.tech/neondb?sslmode=require
```

5. In the Neon SQL console, run once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Our loader also tries to create it, but this ensures it's enabled.

### 1.4. Prepare interview dataset

Place your interview files (PDF, CSV, TXT, HTML) into the folder:

```text
interview_data/
 ├── deeplearning_questions.csv
 ├── ml_questions.txt
 ├── dl_questions.txt
 ├── Software Questions.csv
 └── Behavioral_HR Interview Questions.pdf
```

You can add more files as long as they contain questions.

---

## 2. Pipelines

### 2.1 Job pipeline (Adzuna → Spark → NER → embeddings → Neon)

Run:

```bash
python -m src.flows.job_pipeline_prefect
```

This will:

1. Call `src/01_fetch_jobs_adzuna.py` to pull many pages from Adzuna
   into `data/bronze/jobs/date=YYYY-MM-DD/adzuna.jsonl`.
2. Run `src/03_clean_jobs_spark.py` to create cleaned parquet at `data/silver/jobs/`.
3. Apply **skill NER** (`gmay29/ner_model_final`) to extract skills.
4. Create **BGE embeddings** for job title+description.
5. Load jobs, `job_skills_ner`, and embeddings into Neon (`jobs`, `job_skills_ner`, `job_embeddings`).

### 2.2 Interview pipeline (files → cleaned parquet → embeddings → Neon)

Run:

```bash
python -m src.flows.interview_pipeline_prefect
```

This will:

1. Parse all files in `interview_data/` using `src/interview_ingest_clean_load.py`.
2. Write silver/gold parquet under `data/interview/`.
3. Compute BGE embeddings for each question.
4. Load questions + embeddings into Neon (`interview_questions`, `interview_embeddings`).

You can wire both flows into **Prefect Cloud** for scheduling:

- Create a Prefect Cloud workspace.
- Run `prefect cloud login -k <PREFECT_API_KEY>`.
- Then schedule these flows through the Prefect UI.

---

## 3. Streamlit UI (Career Mentor)

Run:

```bash
streamlit run ui/career_mentor_app.py
```

Then open the URL (typically `http://localhost:8501`).

**What it does:**

1. User enters a target role (e.g., `ML engineer`) and their current skills.
2. App uses **BGE embeddings + pgvector** to retrieve the most similar jobs & interview questions.
3. Aggregated skills from `job_skills_ner` show which skills are most demanded.
4. An instruction **LLM** (Mistral) generates:
   - Skill gaps grouped by theme.
   - A 3–6 month learning roadmap.
   - Networking actions.
   - Similar alternative roles.

---

## 4. Superset (optional, for dashboards)

If you want a BI dashboard on top of the same Neon DB:

1. Install Superset (see official docs) and configure `SQLALCHEMY_DATABASE_URI` to your `PG_URL`.
2. Create datasets on tables:
   - `jobs`
   - `job_skills_ner`
   - `interview_questions`
3. Optionally create a view:

```sql
CREATE OR REPLACE VIEW v_role_skill_demand AS
SELECT
  lower(j.title) AS role_title,
  s.skill,
  COUNT(*) AS demand_count
FROM jobs j
JOIN job_skills_ner s USING (job_id)
GROUP BY 1,2;
```

Use this for a dashboard showing **which skills are needed for a given role**.

---

## 5. Folder structure

```text
career_mentor_modern/
├── .env.example
├── requirements.txt
├── README.md
├── data/
│   ├── bronze/jobs/...
│   ├── silver/jobs/...
│   └── interview/{bronze,silver,gold}/...
├── interview_data/            # put raw interview files here
├── src/
│   ├── __init__.py
│   ├── 01_fetch_jobs_adzuna.py
│   ├── 03_clean_jobs_spark.py
│   ├── interview_ingest_clean_load.py
│   ├── models/
│   │   ├── skill_ner_hf.py
│   │   ├── embeddings_bge.py
│   │   └── llm_gap_analyzer.py
│   ├── warehouse/
│   │   └── load_to_neon.py
│   └── flows/
│       ├── job_pipeline_prefect.py
│       └── interview_pipeline_prefect.py
└── ui/
    └── career_mentor_app.py
```

This zip you downloaded from ChatGPT already contains all those files;
you only need to add `.env` and your interview files, then run the pipelines + UI.

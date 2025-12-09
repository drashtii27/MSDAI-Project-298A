"""
Simple evaluation script for:
- Skill extractor (rule-based + technical filters)
- Embeddings (BAAI/bge-large-en-v1.5)
- Roadmap LLM (mistralai/Mistral-7B-Instruct-v0.3)

Usage (from project root, venv activated):
    python -m src.eval.eval_roadmap_llm

Results will be saved to:
    artifacts/model_evaluation_results.json
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Make sure we can import src.*
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.skill_extractor import extract_skills_from_text
from src.models.embeddings import embed_texts, embed_query
from src.models.roadmap_generator import generate_career_roadmap

# Read model names from env (so they match your config)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
CAREER_LLM_MODEL_NAME = os.getenv(
    "CAREER_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"
)

# ---------------------------------------------------------------------------
# Tiny test data for evaluation
# ---------------------------------------------------------------------------

SAMPLE_JOB_DESCRIPTIONS = [
    "Senior Software Engineer working with Python, React, AWS, Docker, Kubernetes, PostgreSQL, Redis.",
    "Machine Learning Engineer using PyTorch, TensorFlow, MLOps, Airflow, Python, SQL.",
    "Data Engineer with Spark, Kafka, Airflow, Python, SQL, Snowflake, dbt.",
]

GROUND_TRUTH_SKILLS = [
    ["python", "react", "aws", "docker", "kubernetes", "postgresql", "redis"],
    ["pytorch", "tensorflow", "python", "sql", "airflow"],
    ["spark", "kafka", "airflow", "python", "sql", "snowflake", "dbt"],
]


# ---------------------------------------------------------------------------
# 1. Evaluate skill extractor
# ---------------------------------------------------------------------------

def evaluate_skill_extraction() -> Dict[str, Any]:
    print("=" * 70)
    print("EVALUATING SKILL EXTRACTION (rule-based + technical filters)")
    print("=" * 70)

    tp = fp = fn = 0
    times: List[float] = []

    for text, gt in zip(SAMPLE_JOB_DESCRIPTIONS, GROUND_TRUTH_SKILLS):
        start = time.time()
        pred = extract_skills_from_text(text)
        times.append(time.time() - start)

        gt_set = set(gt)
        pred_set = set(pred)

        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)

        print(f"Text: {text}")
        print(f"Ground truth: {sorted(gt_set)}")
        print(f"Predicted   : {sorted(pred_set)}\n")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "model": "rule_based_skill_extractor + technical filters",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_inference_time_ms": float(np.mean(times) * 1000) if times else 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Evaluate embeddings
# ---------------------------------------------------------------------------

def evaluate_embeddings() -> Dict[str, Any]:
    print("=" * 70)
    print(f"EVALUATING EMBEDDINGS ({EMBED_MODEL_NAME})")
    print("=" * 70)

    docs = [
        "Python backend engineer with Django and REST APIs",
        "Data scientist working on machine learning and deep learning",
        "Frontend engineer with React and TypeScript",
    ]

    start = time.time()
    emb_docs = embed_texts(docs)
    elapsed = time.time() - start

    query = "Python developer"
    emb_q = embed_query(query)
    sims = cosine_similarity([emb_q], emb_docs)[0]

    return {
        "model": EMBED_MODEL_NAME,
        "embedding_dimension": int(emb_docs.shape[1]),
        "avg_inference_time_ms": float(elapsed / len(docs) * 1000),
        "similarity_scores": sims.tolist(),
        "docs": docs,
        "query": query,
    }


# ---------------------------------------------------------------------------
# 3. Evaluate roadmap LLM
# ---------------------------------------------------------------------------

def evaluate_llm() -> Dict[str, Any]:
    print("=" * 70)
    print(f"EVALUATING ROADMAP LLM ({CAREER_LLM_MODEL_NAME})")
    print("=" * 70)

    start = time.time()
    roadmap = generate_career_roadmap(
        target_role="Machine Learning Engineer",
        user_skills=["python", "pandas", "sql"],
        target_skills=["pytorch", "tensorflow", "docker", "kubernetes", "aws"],
        similar_roles=["Data Scientist", "ML Ops Engineer"],
        sample_questions=["Explain backpropagation", "What is gradient descent?"],
    )
    elapsed = time.time() - start

    word_count = len(roadmap.split())
    has_sections = all(
        h in roadmap.lower()
        for h in [
            "top skill gaps",
            "12-week learning roadmap",
            "networking",
            "alternative roles",
        ]
    )

    print("\n--- Roadmap (first 600 chars) ---")
    print(roadmap[:600] + ("..." if len(roadmap) > 600 else ""))
    print("---------------------------------\n")

    return {
        "model": CAREER_LLM_MODEL_NAME,
        "inference_time_seconds": elapsed,
        "output_word_count": word_count,
        "contains_expected_sections": has_sections,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_evaluation() -> Dict[str, Any]:
    results = {
        "skill_extraction": evaluate_skill_extraction(),
        "embeddings": evaluate_embeddings(),
        "llm_roadmap": evaluate_llm(),
    }
    return results


if __name__ == "__main__":
    import json

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    out_path = artifacts_dir / "model_evaluation_results.json"

    res = run_full_evaluation()
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)

    print(f"✓ Saved evaluation to {out_path}")

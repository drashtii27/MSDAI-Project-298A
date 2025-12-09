"""
Model evaluation framework for all components.
Evaluates: Skill extraction, embeddings, and LLM roadmap generation.
"""

from __future__ import annotations
import time
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.models.skill_extractor import extract_skills_from_text
from src.models.embeddings import embed_texts, embed_query
from src.models.roadmap_generator import generate_career_roadmap


# ============================================================================
# TEST DATA
# ============================================================================

SAMPLE_JOB_DESCRIPTIONS = [
    """Senior Software Engineer - We seek an experienced engineer with Python, 
    React, and AWS expertise. Must have strong knowledge of Docker, Kubernetes, 
    and CI/CD pipelines. Experience with PostgreSQL and Redis required.""",
    
    """Machine Learning Engineer - Build ML models using PyTorch and TensorFlow. 
    Need experience with MLOps, Airflow, and cloud platforms (AWS/GCP). 
    Strong Python and SQL skills essential.""",
    
    """Data Engineer - Design data pipelines with Spark, Kafka, and Airflow. 
    Proficiency in Python, SQL, and Snowflake. Experience with dbt and 
    data modeling required.""",
]

GROUND_TRUTH_SKILLS = [
    ["python", "react", "aws", "docker", "kubernetes", "postgresql", "redis"],
    ["pytorch", "tensorflow", "python", "sql", "airflow", "mlops"],
    ["spark", "kafka", "airflow", "python", "sql", "snowflake", "dbt"],
]


# ============================================================================
# SKILL EXTRACTION EVALUATION
# ============================================================================

def evaluate_skill_extraction() -> Dict[str, Any]:
    """
    Evaluate JobBERT skill extraction model.
    Metrics: Precision, Recall, F1, Inference Time
    """
    print("\n" + "="*70)
    print("EVALUATING SKILL EXTRACTION MODEL (JobBERT)")
    print("="*70)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    inference_times = []
    
    for i, (job_desc, true_skills) in enumerate(zip(SAMPLE_JOB_DESCRIPTIONS, GROUND_TRUTH_SKILLS)):
        print(f"\nSample {i+1}:")
        print(f"Text: {job_desc[:100]}...")
        
        start = time.time()
        predicted_skills = extract_skills_from_text(job_desc)
        elapsed = time.time() - start
        inference_times.append(elapsed)
        
        # Calculate metrics
        true_set = set(s.lower() for s in true_skills)
        pred_set = set(s.lower() for s in predicted_skills)
        
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        print(f"True skills: {true_skills}")
        print(f"Predicted: {predicted_skills}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Inference time: {elapsed:.3f}s")
    
    # Overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_time = np.mean(inference_times)
    
    results = {
        "model": "jjzha/jobbert_skill_extraction",
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "avg_inference_time_ms": avg_time * 1000,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
    
    print("\n" + "-"*70)
    print("OVERALL RESULTS:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Avg Inference Time: {avg_time*1000:.2f}ms")
    print("-"*70)
    
    return results


# ============================================================================
# EMBEDDING EVALUATION
# ============================================================================

def evaluate_embeddings() -> Dict[str, Any]:
    """
    Evaluate BGE embedding model.
    Metrics: Embedding dimension, inference time, semantic similarity
    """
    print("\n" + "="*70)
    print("EVALUATING EMBEDDING MODEL (BGE-Large)")
    print("="*70)
    
    test_texts = [
        "Python software engineer with React experience",
        "Backend developer using Python and Node.js",
        "Data scientist specializing in machine learning",
    ]
    
    # Embedding inference
    start = time.time()
    embeddings = embed_texts(test_texts)
    elapsed = time.time() - start
    
    print(f"\nEmbedding dimension: {embeddings.shape[1]}")
    print(f"Inference time for {len(test_texts)} texts: {elapsed:.3f}s")
    print(f"Avg time per text: {elapsed/len(test_texts)*1000:.2f}ms")
    
    # Semantic similarity test
    query = "Looking for a Python developer"
    query_emb = embed_query(query)
    
    similarities = cosine_similarity([query_emb], embeddings)[0]
    
    print(f"\nQuery: '{query}'")
    print("Similarities:")
    for text, sim in zip(test_texts, similarities):
        print(f"  {sim:.4f} - {text}")
    
    results = {
        "model": "BAAI/bge-large-en-v1.5",
        "embedding_dimension": int(embeddings.shape[1]),
        "avg_inference_time_ms": (elapsed / len(test_texts)) * 1000,
        "similarity_scores": similarities.tolist(),
    }
    
    print("-"*70)
    
    return results


# ============================================================================
# LLM ROADMAP EVALUATION
# ============================================================================

def evaluate_llm_roadmap() -> Dict[str, Any]:
    """
    Evaluate Mistral roadmap generation.
    Metrics: Output length, structure quality, inference time
    """
    print("\n" + "="*70)
    print("EVALUATING LLM ROADMAP GENERATION (Mistral-7B)")
    print("="*70)
    
    test_case = {
        "target_role": "Machine Learning Engineer",
        "user_skills": ["python", "pandas", "sql"],
        "target_skills": ["pytorch", "tensorflow", "mlops", "kubernetes", "airflow"],
        "similar_roles": ["Data Scientist", "ML Researcher", "AI Engineer"],
        "sample_questions": [
            "Explain backpropagation in neural networks",
            "How do you prevent overfitting in ML models?",
            "What is the difference between L1 and L2 regularization?",
        ],
    }
    
    print(f"\nTest case:")
    print(f"Target role: {test_case['target_role']}")
    print(f"User skills: {test_case['user_skills']}")
    print(f"Target skills: {test_case['target_skills']}")
    
    start = time.time()
    roadmap = generate_career_roadmap(**test_case)
    elapsed = time.time() - start
    
    # Analyze output
    word_count = len(roadmap.split())
    line_count = len(roadmap.split('\n'))
    has_sections = any(header in roadmap.lower() for header in [
        "skill gap", "roadmap", "week", "project", "networking"
    ])
    
    print(f"\nGeneration time: {elapsed:.2f}s")
    print(f"Output length: {word_count} words, {line_count} lines")
    print(f"Contains expected sections: {has_sections}")
    print(f"\nFirst 500 chars of output:\n{roadmap[:500]}...")
    
    results = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "inference_time_seconds": elapsed,
        "output_word_count": word_count,
        "output_line_count": line_count,
        "contains_expected_sections": has_sections,
    }
    
    print("-"*70)
    
    return results


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

def run_full_evaluation() -> Dict[str, Any]:
    """Run complete evaluation of all models."""
    print("\n" + "="*70)
    print("CAREER MENTOR SYSTEM - FULL MODEL EVALUATION")
    print("="*70)
    
    results = {
        "skill_extraction": evaluate_skill_extraction(),
        "embeddings": evaluate_embeddings(),
        "llm_roadmap": evaluate_llm_roadmap(),
    }
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import json
    
    results = run_full_evaluation()
    
    # Save results
    with open("model_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Results saved to model_evaluation_results.json")
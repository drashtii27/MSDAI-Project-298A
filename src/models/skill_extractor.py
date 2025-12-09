"""
Rule-based technical skill extractor.

We intentionally avoid any HF NER pipeline here to keep the
pipeline simple and robust. This module exposes the same
API as before:

- extract_skills_from_text(text: str) -> List[str]
- extract_skills_batch(texts: List[str], batch_size: int = ...) -> List[List[str]]

Only *technical* skills are returned.
"""

from __future__ import annotations
from typing import List, Dict, Set
import re

from src.config import SKILL_EXTRACTION_BATCH_SIZE


# ============================================================================
# TECHNICAL SKILL VOCABULARY
# ============================================================================

CS_FOUNDATIONS = [
    "computer science fundamentals",
    "data structures",
    "arrays",
    "linked lists",
    "stacks",
    "queues",
    "hash tables",
    "trees",
    "binary trees",
    "binary search trees",
    "tries",
    "heaps",
    "priority queues",
    "graphs",
    "adjacency matrix",
    "adjacency list",
    "dynamic programming",
    "recursion",
    "divide and conquer",
    "greedy algorithms",
    "backtracking",
    "time complexity",
    "space complexity",
    "big o notation",
    "algorithm design",
    "computational complexity",
    "memory management",
]

ALGORITHMS = [
    "sorting algorithms",
    "merge sort",
    "quick sort",
    "heap sort",
    "bubble sort",
    "insertion sort",
    "selection sort",
    "binary search",
    "graph algorithms",
    "breadth first search",
    "bfs",
    "depth first search",
    "dfs",
    "shortest path",
    "dijkstra algorithm",
    "bellman ford",
    "minimum spanning tree",
    "kruskal",
    "prim",
    "string algorithms",
    "kmp",
    "rabin karp",
    "suffix arrays",
    "search algorithms",
    "path finding algorithms",
]

PROGRAMMING_PARADIGMS = [
    "object oriented programming",
    "oop",
    "inheritance",
    "polymorphism",
    "abstraction",
    "encapsulation",
    "functional programming",
    "event driven programming",
    "asynchronous programming",
    "concurrency",
    "parallelism",
    "multithreading",
    "synchronization",
    "message passing",
    "design patterns",
    "singleton",
    "factory pattern",
    "observer pattern",
    "adapter pattern",
]

SYSTEM_DESIGN = [
    "system design",
    "distributed systems",
    "scalability",
    "fault tolerance",
    "load balancing",
    "caching",
    "content delivery networks",
    "cdn",
    "database sharding",
    "partitioning",
    "replication",
    "microservices",
    "event driven architecture",
    "message queues",
    "pubsub systems",
    "api design",
    "rest apis",
    "grpc",
    "rate limiting",
    "high availability",
    "eventual consistency",
]

DATABASE_SKILLS = [
    "sql",
    "postgres",
    "mysql",
    "sql server",
    "oracle",
    "nosql",
    "mongodb",
    "redis",
    "dynamodb",
    "cassandra",
    "elasticsearch",
    "bigtable",
    "data modeling",
    "query optimization",
    "indexing",
]

BACKEND_SKILLS = [
    "backend engineering",
    "python",
    "java",
    "ruby",
    "nodejs",
    "php",
    "c#",
    "api development",
    "authentication",
    "authorization",
    "jwt",
    "oauth",
    "session management",
    "web frameworks",
    "django",
    "flask",
    "fastapi",
    "spring boot",
    "rails",
]

FRONTEND_SKILLS = [
    "frontend engineering",
    "javascript",
    "typescript",
    "react",
    "vue",
    "angular",
    "svelte",
    "css",
    "html",
    "responsive design",
    "web accessibility",
    "ui design",
    "ux design",
    "web performance optimization",
    "webpack",
    "vite",
]

DEVOPS_CLOUD = [
    "devops",
    "cloud computing",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "terraform",
    "ci cd",
    "github actions",
    "jenkins",
    "monitoring",
    "prometheus",
    "grafana",
    "logging",
    "elk stack",
    "load balancing",
    "reverse proxy",
    "nginx",
    "serverless computing",
]

SECURITY_NETWORKING = [
    "computer networks",
    "tcp/ip",
    "http",
    "https",
    "dns",
    "ssl",
    "tls",
    "network security",
    "encryption",
    "authentication protocols",
    "penetration testing",
    "vulnerability scanning",
    "secure coding",
]

ML_DS = [
    "machine learning",
    "deep learning",
    "data science",
    "statistics",
    "probability",
    "linear regression",
    "logistic regression",
    "random forest",
    "xgboost",
    "svm",
    "neural networks",
    "nlp",
    "computer vision",
    "matplotlib",
    "pandas",
    "numpy",
    "scikit learn",
    "tensorflow",
    "pytorch",
]

SOFTWARE_PRACTICES = [
    "unit testing",
    "integration testing",
    "test driven development",
    "tdd",
    "version control",
    "git",
    "clean code",
    "refactoring",
    "debugging",
    "deployment pipelines",
    "architecture documentation",
]

FULL_CS_TAXONOMY = (
    CS_FOUNDATIONS
    + ALGORITHMS
    + PROGRAMMING_PARADIGMS
    + SYSTEM_DESIGN
    + DATABASE_SKILLS
    + BACKEND_SKILLS
    + FRONTEND_SKILLS
    + DEVOPS_CLOUD
    + SECURITY_NETWORKING
    + ML_DS
    + SOFTWARE_PRACTICES
)

# canonical_skill -> list of phrases to match
TECHNICAL_SKILL_PATTERNS: Dict[str, List[str]] = {
    # Programming languages
    "python": ["python"],
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript"],
    "c++": ["c++"],
    "c#": ["c#"],
    "go": ["golang", "go"],
    "rust": ["rust"],
    "scala": ["scala"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "ruby": ["ruby"],
    "php": ["php"],
    "r": [" r "],
    "matlab": ["matlab"],

    # Databases / SQL
    "sql": ["sql"],
    "mysql": ["mysql"],
    "postgresql": ["postgresql", "postgres"],
    "sqlite": ["sqlite"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "cassandra": ["cassandra"],
    "elasticsearch": ["elasticsearch", "elastic search"],

    # Web / backend frameworks
    "html": ["html"],
    "css": ["css", "scss", "sass"],
    "react": ["react", "react.js", "reactjs"],
    "next.js": ["next.js", "nextjs"],
    "vue": ["vue", "vue.js", "vuejs"],
    "angular": ["angular", "angular.js", "angularjs"],
    "node.js": ["node.js", "nodejs", "node js"],
    "express": ["express", "express.js", "expressjs"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "spring": ["spring", "spring boot"],

    # Data / ML
    "pandas": ["pandas"],
    "numpy": ["numpy", "np"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow", "tf"],
    "keras": ["keras"],
    "xgboost": ["xgboost"],
    "lightgbm": ["lightgbm", "lgbm"],
    "spark": ["spark"],
    "pyspark": ["pyspark"],
    "airflow": ["airflow"],
    "kafka": ["kafka"],
    "mlflow": ["mlflow"],
    "dbt": ["dbt"],
    "hadoop": ["hadoop"],
    "snowflake": ["snowflake"],
    "bigquery": ["bigquery"],
    "redshift": ["redshift"],

    # Cloud / devops
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "helm": ["helm"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],
    "jenkins": ["jenkins"],
    "github actions": ["github actions"],
    "gitlab ci": ["gitlab ci"],
    "linux": ["linux"],

    # Tools / analytics
    "git": ["git"],
    "jira": ["jira"],
    "confluence": ["confluence"],
    "tableau": ["tableau"],
    "power bi": ["power bi"],
    "looker": ["looker"],
    "superset": ["superset"],

    # MLOps / infra
    "mlops": ["mlops"],
    "feature store": ["feature store"],
    "kubeflow": ["kubeflow"],
    "ray": ["ray"],
}

# ---------------------------------------------------------------------------
# Add FULL_CS_TAXONOMY phrases as additional skills to match
# without changing existing technical skills.
# ---------------------------------------------------------------------------

for phrase in FULL_CS_TAXONOMY:
    canonical = phrase.lower()
    # Don't override any existing hand-tuned patterns
    if canonical not in TECHNICAL_SKILL_PATTERNS:
        TECHNICAL_SKILL_PATTERNS[canonical] = [canonical]


# Non-technical words to ignore if they accidentally match
SKILL_STOPWORDS: Set[str] = {
    "r",
    "go",
    "spring",  # quite ambiguous in plain text
}

# Precompile regex patterns for speed: canonical -> list[compiled_regex]
COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {}

for canonical, phrases in TECHNICAL_SKILL_PATTERNS.items():
    compiled_list: List[re.Pattern] = []
    for phrase in phrases:
        phrase = phrase.strip()
        if phrase == "r":
            regex = re.compile(r"\br\b", flags=re.IGNORECASE)
        else:
            regex = re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
        compiled_list.append(regex)
    COMPILED_PATTERNS[canonical] = compiled_list


# ============================================================================
# CORE EXTRACTION LOGIC
# ============================================================================

def _extract_from_single_text(text: str) -> List[str]:
    """Extract technical skills from a single piece of text."""
    if not text:
        return []

    found: Set[str] = set()
    for canonical, regex_list in COMPILED_PATTERNS.items():
        for regex in regex_list:
            if regex.search(text):
                if canonical not in SKILL_STOPWORDS:
                    found.add(canonical)
                break  # stop after first matching phrase for this canonical skill

    return sorted(found)


# ============================================================================
# PUBLIC API
# ============================================================================

def extract_skills_from_text(text: str) -> List[str]:
    """
    Extract skills from a single job description / interview question.
    Returns a sorted list of canonical technical skills.
    """
    return _extract_from_single_text(text)


def extract_skills_batch(
    texts: List[str],
    batch_size: int = SKILL_EXTRACTION_BATCH_SIZE,
) -> List[List[str]]:
    """
    Batch version for compatibility with the rest of the pipeline.
    We keep batch_size arg just to match the old signature.
    """
    if not texts:
        return []

    results: List[List[str]] = []
    for text in texts:
        results.append(_extract_from_single_text(text))
    return results

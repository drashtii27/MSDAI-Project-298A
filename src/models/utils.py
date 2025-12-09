"""Utility functions for text processing and skill cleaning."""

import re
from typing import Set, List

def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.strip()).lower()

def clean_job_description(desc: str) -> str:
    if not desc:
        return ""
    desc = re.sub(r'<[^>]+>', ' ', desc)            # HTML
    desc = re.sub(r'http\S+|www\.\S+', ' ', desc)   # URLs
    desc = re.sub(r'\S+@\S+', ' ', desc)            # emails
    desc = re.sub(r'\s+', ' ', desc)
    return desc.strip().lower()

def deduplicate_skills(skills: List[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for skill in skills:
        s = skill.lower().strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return sorted(result)

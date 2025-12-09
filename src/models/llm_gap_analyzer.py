# src/models/llm_gap_analyzer.py

import os
from textwrap import dedent
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Small default model; you can override with an env var if you want
LLM_MODEL = os.getenv("CAREER_MENTOR_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

_tokenizer = None
_model = None


def _get_llm():
    """
    Lazy-load tokenizer & model once and reuse them.
    """
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    # Ensure we have a pad token
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Load model
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    return _tokenizer, _model


def build_prompt(
    role: str,
    user_skills: List[str],
    target_skills: List[str],
    similar_roles: List[str],
    sample_questions: List[str],
) -> str:
    user_skills_str = ", ".join(sorted(set(user_skills))) if user_skills else "not specified"
    target_skills_str = ", ".join(target_skills[:25]) if target_skills else "N/A"
    similar_roles_str = ", ".join(similar_roles[:5]) if similar_roles else "N/A"
    sample_questions_str = "\n".join(f"- {q}" for q in sample_questions[:8]) if sample_questions else "N/A"

    prompt = f"""
    You are an AI career mentor.

    Target job role: {role}

    User current skills: {user_skills_str}

    Skills required for this role (aggregated from live job postings):
    {target_skills_str}

    Similar roles from the market:
    {similar_roles_str}

    Representative interview questions:
    {sample_questions_str}

    TASK:
    1. List the top skill gaps the user should close (group them as 'Core ML', 'MLOps',
       'Data Engineering', 'Soft skills').
    2. Propose a 3–6 month learning roadmap with concrete actions.
    3. Suggest 3–5 networking actions (meetups, open-source, writing, etc.) tailored to this role.
    4. Mention 3 closely related alternative roles the user could target and why.

    Output in markdown with clear headings and bullet points.
    """
    return dedent(prompt).strip()


def analyze_career(
    role: str,
    user_skills: List[str],
    target_skills: List[str],
    similar_roles: List[str],
    sample_questions: List[str],
) -> str:
    """
    Main entrypoint used by Streamlit. Returns markdown text.
    """
    tokenizer, model = _get_llm()
    prompt = build_prompt(role, user_skills, target_skills, similar_roles, sample_questions)

    # Tokenize with truncation so the prompt doesn't blow up
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,          # limit output size so it can't hang forever
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Some models echo the prompt; strip it if present
    if full_text.startswith(prompt):
        full_text = full_text[len(prompt):]

    return full_text.strip()

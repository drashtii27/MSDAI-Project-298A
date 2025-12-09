"""
Simple deterministic career roadmap generator.

This version avoids heavy LLMs and tokenizer issues.
It builds a roadmap from your job / skills / questions data.
"""

from __future__ import annotations
from typing import List


def _format_list(items: List[str], bullet: str = "-") -> str:
    if not items:
        return "_None identified yet._"
    return "\n".join(f"{bullet} {item}" for item in items)


def _compute_skill_gaps(user_skills: List[str], target_skills: List[str]) -> List[str]:
    user = {s.strip().lower() for s in user_skills if s.strip()}
    gaps = []
    for skill in target_skills:
        s_norm = skill.strip().lower()
        if s_norm and s_norm not in user:
            gaps.append(skill)
    return gaps


def generate_career_roadmap(
    target_role: str,
    user_skills: List[str],
    target_skills: List[str],
    similar_roles: List[str],
    sample_questions: List[str],
) -> str:
    """
    Build a text roadmap using simple template logic.
    This matches the interface expected by the Streamlit app.
    """

    if not target_role:
        target_role = "your target role"

    user_skills_clean = [s.strip() for s in user_skills if s.strip()]
    target_skills_clean = [s.strip() for s in target_skills if s.strip()]
    gaps = _compute_skill_gaps(user_skills_clean, target_skills_clean)
    top_gaps = gaps[:8]

    roadmap_lines: List[str] = []

    # ------------------------------------------------------------------ #
    # TITLE & SUMMARY
    # ------------------------------------------------------------------ #
    roadmap_lines.append(f"# Career Roadmap: {target_role}\n")

    roadmap_lines.append("## 1. Snapshot of Your Current Profile\n")
    roadmap_lines.append("**Current skills:**")
    roadmap_lines.append(_format_list(user_skills_clean))
    roadmap_lines.append("")

    if similar_roles:
        roadmap_lines.append("**Representative job titles in the market:**")
        roadmap_lines.append(_format_list(similar_roles[:6]))
        roadmap_lines.append("")
    else:
        roadmap_lines.append(
            "No similar job titles were found in the current dataset, "
            "but we can still build a solid roadmap based on skills."
        )
        roadmap_lines.append("")

    # ------------------------------------------------------------------ #
    # SKILL GAP ANALYSIS
    # ------------------------------------------------------------------ #
    roadmap_lines.append("## 2. Skill Gap Analysis\n")

    if target_skills_clean:
        roadmap_lines.append("**Skills frequently requested in job postings:**")
        roadmap_lines.append(_format_list(target_skills_clean[:20]))
        roadmap_lines.append("")
    else:
        roadmap_lines.append(
            "We couldn't extract specific skills from the job data, "
            "so this roadmap focuses on strengthening and deepening your current stack."
        )
        roadmap_lines.append("")

    if gaps:
        roadmap_lines.append("**High-priority skill gaps to focus on next:**")
        roadmap_lines.append(_format_list(top_gaps))
        roadmap_lines.append("")
    else:
        roadmap_lines.append(
            "Based on the job data we have, your current skills overlap strongly "
            "with market requirements. The roadmap below focuses on **depth, projects, "
            "and interview excellence** rather than new tools."
        )
        roadmap_lines.append("")

    # ------------------------------------------------------------------ #
    # 12-WEEK LEARNING PLAN
    # ------------------------------------------------------------------ #
    roadmap_lines.append("## 3. 12-Week Learning & Project Plan\n")

    roadmap_lines.append("### Phase 1 (Weeks 1–3): Foundations & Gaps\n")
    roadmap_lines.append(
        "- Pick **2–3 core skills** from the gap list above and plan to cover them in depth.\n"
        "- For each new skill, follow this pattern:\n"
        "  1. Take a focused course / official tutorial.\n"
        "  2. Implement a **mini-project** (e.g., a small ML model, API, or dashboard).\n"
        "  3. Write short notes or a blog post summarizing what you learned.\n"
        "- Refresh your **Python + SQL** fundamentals with 30–45 minutes of practice per day.\n"
        "- Start a log (Notion / Google Doc) to track topics, links, and questions."
    )
    roadmap_lines.append("")

    roadmap_lines.append("### Phase 2 (Weeks 4–6): Portfolio Project\n")
    roadmap_lines.append(
        "- Design **one flagship project** aligned with your target role, for example:\n"
        "  - An end-to-end ML pipeline (data ingestion → feature engineering → model → evaluation).\n"
        "  - A small recommendation / ranking system using real-world data.\n"
        "  - A job-market analytics dashboard using public datasets.\n"
        "- Put all code in a clean, well-documented GitHub repo.\n"
        "- Add unit tests, a clear README, and screenshots of outputs.\n"
        "- Aim to touch the main tools you see in job postings (e.g., scikit-learn, pandas, SQL, Git)."
    )
    roadmap_lines.append("")

    roadmap_lines.append("### Phase 3 (Weeks 7–9): System Design & Interview Prep\n")
    roadmap_lines.append(
        "- For each week, pick **3–5 interview questions** and write:\n"
        "  - A clear explanation of the concept.\n"
        "  - At least one coded example (in Python or your main language).\n"
        "- Practice explaining your flagship project out loud:\n"
        "  - Problem, data, approach, metrics, trade-offs, and limitations.\n"
        "- Review common topics for your role (e.g., ML fundamentals, evaluation metrics,\n"
        "  feature engineering, basic statistics, and SQL joins/window functions)."
    )
    roadmap_lines.append("")

    roadmap_lines.append("### Phase 4 (Weeks 10–12): Applications & Networking\n")
    roadmap_lines.append(
        "- Polish your **CV and LinkedIn** with:\n"
        "  - Bullet points that highlight measurable impact (metrics, improvements).\n"
        "  - Links to your GitHub projects and portfolio.\n"
        "- Set a weekly target, e.g., **10–15 tailored applications per week**.\n"
        "- Reach out to people with titles similar to your target role:\n"
        "  - Send short, specific messages asking for a 15-minute coffee chat.\n"
        "  - Prepare 3–4 good questions (team structure, tech stack, what they look for)."
    )
    roadmap_lines.append("")

    # ------------------------------------------------------------------ #
    # INTERVIEW QUESTION FOCUS
    # ------------------------------------------------------------------ #
    roadmap_lines.append("## 4. Interview Question Focus\n")

    if sample_questions:
        roadmap_lines.append(
            "Here are some representative questions from the dataset that you can "
            "use to drive your prep. For each question, aim to:\n"
            "- Write a **clear, concise answer**.\n"
            "- Implement a **code example**, where applicable.\n"
            "- Note down **follow-up questions** you might get."
        )
        roadmap_lines.append("")
        for i, q in enumerate(sample_questions[:10], start=1):
            roadmap_lines.append(f"{i}. {q}")
        roadmap_lines.append("")
    else:
        roadmap_lines.append(
            "We could not load representative questions from the database. You can still practice by:\n"
            "- Searching for '[target_role] interview questions' on reputable sites.\n"
            "- Categorizing questions into: algorithms/coding, ML theory, SQL, and system design.\n"
            "- Maintaining a personal Q&A document with your best answers."
        )
        roadmap_lines.append("")

    # ------------------------------------------------------------------ #
    # DAILY / WEEKLY HABITS
    # ------------------------------------------------------------------ #
    roadmap_lines.append("## 5. Weekly Routine Template\n")
    roadmap_lines.append(
        "- **3× per week – Deep Learning Blocks (60–90 min):**\n"
        "  Study a specific topic or gap skill and extend your project.\n"
        "- **2× per week – Coding / SQL Practice (45–60 min):**\n"
        "  LeetCode, HackerRank, or similar; focus on problems relevant to data / ML.\n"
        "- **1× per week – Portfolio & Community (60 min):**\n"
        "  Improve your GitHub repo, write a short article, or share progress on LinkedIn.\n"
        "- **1× per week – Networking (30–45 min):**\n"
        "  Reach out to professionals, send follow-ups, or schedule chats."
    )
    roadmap_lines.append("")

    # ------------------------------------------------------------------ #
    # FINAL ENCOURAGEMENT
    # ------------------------------------------------------------------ #
    roadmap_lines.append("## 6. How to Use This Roadmap\n")
    roadmap_lines.append(
        "Use this roadmap as a **living document**:\n"
        "- At the end of each week, quickly review what you did and update the plan.\n"
        "- When a topic feels weak, schedule another focused session for it.\n"
        "- Keep all notes, links, and code in a place you can easily show during interviews."
    )
    roadmap_lines.append("")
    roadmap_lines.append(
        "If you stay consistent with this plan, you will steadily build the skills, "
        "projects, and confidence needed to move into a strong "
        f"**{target_role}** role."
    )

    return "\n".join(roadmap_lines)

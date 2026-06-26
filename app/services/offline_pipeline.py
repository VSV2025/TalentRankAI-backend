"""
Offline 7-layer ranking pipeline for the Redrob Hackathon dataset.
No external LLM API calls — CPU-only, targets <5 min for 100K candidates.

Layer map:
  L1  JD Understanding   — pre-parsed, hardcoded for hackathon JD
  L2  Fast Retrieval     — keyword + YoE + title scoring, all N candidates
  L3  Graph Enrichment   — O*NET cluster topology, graph_fit + skill_breadth
  L4  Feature Scoring    — skills_match / semantic_rel / behavioral / career
  L4b Behavioral Signals — Redrob 23-signal multiplier
  L6  Debate Rules       — advocate/skeptic adjustments, honeypot check
  L7  Composite + Rank   — FA*IR fairness rerank + reasoning generation
"""
import heapq
import json
import math
from datetime import date
from pathlib import Path
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# L1  PRE-PARSED JD  (Senior AI Engineer, Redrob AI, Pune/Noida)
# ─────────────────────────────────────────────────────────────────────────────
HACKATHON_JD = {
    "title": "Senior AI Engineer — Founding Team",
    "company": "Redrob AI (Series A)",
    "location": "Pune / Noida, India (Hybrid)",
    "yoe_range": (5, 9),
    "mandate": "Own the intelligence layer: ranking, retrieval, matching",
    "key_requirements": [
        "Embeddings, vector search, retrieval & ranking",
        "Production LLM experience (fine-tuning, serving, RLHF)",
        "Shipping mindset over pure research",
        "5-9 years experience (flexible)",
        "Based in / willing to relocate to India (Pune/Noida preferred)",
    ],
    "disqualifiers": [
        "Pure research background with zero production deployments",
        "AI experience limited to recent LangChain/OpenAI projects only",
    ],
}

# Core AI/ML skills the JD cares about
CORE_AI_SKILLS = {
    # LLM / NLP
    "embeddings", "fine-tuning", "finetuning", "rlhf", "rag",
    "llm", "llms", "large language model", "transformer", "transformers",
    "bert", "gpt", "llama", "mistral", "falcon", "gemma",
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "speech recognition",
    # Retrieval / Search
    "vector search", "semantic search", "hybrid search", "dense retrieval",
    "bm25", "retrieval", "ranking", "reranking",
    "elasticsearch", "opensearch", "solr",
    "faiss", "qdrant", "pinecone", "weaviate", "milvus", "annoy",
    # Deep Learning
    "pytorch", "tensorflow", "jax", "keras", "cuda",
    "neural network", "deep learning", "diffusion", "gans",
    # MLOps / Infra
    "mlops", "model serving", "inference optimization",
    "mlflow", "kubeflow", "triton", "vllm", "tensorrt", "bentoml",
    # Core ML
    "machine learning", "sklearn", "scikit-learn", "xgboost", "lightgbm",
    "feature engineering", "a/b testing", "experimentation",
    # Tooling
    "huggingface", "hugging face", "sentence-transformers", "sentence transformers",
    "langchain", "llamaindex", "lora", "qlora", "peft",
}

BONUS_SKILLS = {
    "rlhf", "fine-tuning", "finetuning", "quantization", "distillation",
    "cuda", "triton", "c++", "vllm", "tensorrt-llm", "inference optimization",
    "lora", "qlora", "peft",
}

CAREER_KEYWORDS = [
    "production", "deployed", "shipped", "launched", "scaled",
    "ml system", "ai system", "ranking system", "retrieval system",
    "search", "recommendation", "llm", "embedding", "vector",
    "fine-tun", "rag", "billion", "million", "10x",
]

TARGET_CITIES = {"pune", "noida", "pimpri", "chinchwad"}
INDIA_CITIES = {
    "delhi", "gurgaon", "gurugram", "mumbai", "bangalore", "bengaluru",
    "hyderabad", "chennai", "kolkata", "ahmedabad", "jaipur",
}

TIER_SCORE = {"tier_1": 100, "tier_2": 78, "tier_3": 55, "tier_4": 35, "unknown": 45}

# O*NET-inspired skill clusters for L3 graph enrichment
SKILL_CLUSTERS = {
    "nlp_llm": {
        "nlp", "natural language processing", "llm", "llms", "large language model",
        "fine-tuning", "finetuning", "rlhf", "rag", "bert", "gpt", "llama",
        "transformer", "embeddings", "text classification", "tokenization", "huggingface",
    },
    "retrieval_search": {
        "elasticsearch", "opensearch", "bm25", "vector search", "semantic search",
        "faiss", "qdrant", "pinecone", "weaviate", "milvus", "retrieval",
        "ranking", "reranking", "hybrid search", "solr", "annoy",
    },
    "deep_learning": {
        "pytorch", "tensorflow", "jax", "keras", "cuda", "neural network",
        "deep learning", "diffusion", "gans", "cnn", "rnn", "lstm",
    },
    "mlops_infra": {
        "mlops", "mlflow", "kubeflow", "airflow", "kubernetes", "docker",
        "model serving", "inference", "deployment", "bentoml",
        "triton", "vllm", "tensorrt", "ci/cd", "sagemaker", "vertex ai",
    },
    "data_engineering": {
        "spark", "kafka", "airflow", "dbt", "snowflake", "bigquery",
        "sql", "postgresql", "etl", "data pipeline", "apache beam", "flink",
    },
    "python_swe": {
        "python", "fastapi", "flask", "rest api", "microservices",
        "git", "asyncio", "pytest", "go", "java",
    },
    "cloud": {
        "aws", "gcp", "azure", "lambda", "ec2", "s3", "sagemaker", "vertex ai",
    },
    "ml_fundamentals": {
        "machine learning", "sklearn", "scikit-learn", "xgboost", "lightgbm",
        "gradient boosting", "random forest", "feature engineering",
        "statistics", "pandas", "numpy",
    },
    "product_metrics": {
        "a/b testing", "experimentation", "analytics", "recommendation",
        "personalization", "kpi", "metrics", "engagement", "growth",
    },
    "leadership": {
        "mentoring", "team lead", "tech lead", "principal", "staff",
        "founding", "cross-functional", "managing", "architecture",
    },
}

CLUSTER_WEIGHTS = {
    "nlp_llm": 1.00, "retrieval_search": 1.00, "deep_learning": 0.85,
    "mlops_infra": 0.80, "python_swe": 0.70, "ml_fundamentals": 0.65,
    "cloud": 0.60, "data_engineering": 0.55, "product_metrics": 0.50,
    "leadership": 0.45,
}

TODAY = date.today()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _days_since(date_str: str) -> int:
    try:
        return (TODAY - date.fromisoformat(date_str)).days
    except Exception:
        return 999


def _skill_set(c: dict) -> set:
    return {s["name"].lower() for s in c.get("skills", [])}


def _full_text(c: dict) -> str:
    p = c["profile"]
    parts = [
        p.get("headline", ""), p.get("summary", ""), p.get("current_title", ""),
        " ".join(r.get("description", "") for r in c.get("career_history", [])),
    ]
    return " ".join(parts).lower()


# ─────────────────────────────────────────────────────────────────────────────
# L2  FAST RETRIEVAL  (O(n) — must handle 100K in <60 s)
# ─────────────────────────────────────────────────────────────────────────────

def l2_fast_score(c: dict, skills: set, text: str) -> float:
    score = 0.0

    # AI skill match weighted by proficiency
    for s in c.get("skills", []):
        name = s["name"].lower()
        prof = {"expert": 5, "advanced": 3.5, "intermediate": 2, "beginner": 0.8}.get(
            s.get("proficiency", ""), 1
        )
        if name in CORE_AI_SKILLS:
            score += prof
        if name in BONUS_SKILLS:
            score += 1.5

    # Career text keyword density
    score += sum(1.5 for kw in CAREER_KEYWORDS if kw in text)

    # YoE fit
    yoe = float(c["profile"].get("years_of_experience", 0))
    if 5 <= yoe <= 9:      score += 20
    elif 9 < yoe <= 12:    score += 15
    elif 3 <= yoe < 5:     score += 12
    elif 12 < yoe <= 15:   score += 10
    elif yoe < 3:          score += 2

    # Title relevance
    title = c["profile"].get("current_title", "").lower()
    if any(t in title for t in ("ai engineer", "ml engineer", "machine learning engineer")):
        score += 18
    elif any(t in title for t in ("nlp engineer", "deep learning", "applied ml",
                                   "applied scientist", "research engineer")):
        score += 13
    elif any(t in title for t in ("data scientist", "platform engineer")):
        score += 7

    # Quick engagement check
    rs = c.get("redrob_signals", {})
    if rs.get("open_to_work_flag"):
        score += 5
    score += rs.get("recruiter_response_rate", 0.5) * 8

    return score


# ─────────────────────────────────────────────────────────────────────────────
# L3  GRAPH ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def l3_graph_enrichment(c: dict, skills: set, text: str) -> tuple:
    cluster_hits: dict[str, float] = {}
    for name, keywords in SKILL_CLUSTERS.items():
        direct = len(skills & keywords)
        text_hits = sum(0.3 for kw in keywords if kw in text)
        cluster_hits[name] = min(direct + text_hits, len(keywords))

    total_w = sum(CLUSTER_WEIGHTS.values())
    graph_fit = sum(
        CLUSTER_WEIGHTS[n] * min(cluster_hits[n] / max(len(SKILL_CLUSTERS[n]) * 0.3, 1), 1.0)
        for n in SKILL_CLUSTERS
    ) / total_w * 100

    clusters_touched = sum(1 for n in SKILL_CLUSTERS if cluster_hits[n] >= 0.5)
    skill_breadth = clusters_touched / len(SKILL_CLUSTERS) * 100

    return round(graph_fit, 1), round(skill_breadth, 1), cluster_hits


# ─────────────────────────────────────────────────────────────────────────────
# L4  DEEP FEATURE SCORING  (LLM proxy — offline)
# ─────────────────────────────────────────────────────────────────────────────

def _title_score(title: str) -> float:
    t = title.lower()
    if any(x in t for x in ("ai engineer", "machine learning engineer", "ml engineer")):
        return 100.0
    if any(x in t for x in ("research engineer", "applied ml", "applied scientist",
                              "nlp engineer", "deep learning engineer")):
        return 88.0
    if "data scientist" in t:
        return 72.0 if any(x in t for x in ("senior", "staff", "lead", "principal")) else 62.0
    if any(x in t for x in ("platform engineer", "infrastructure engineer", "backend engineer")):
        return 58.0
    if any(x in t for x in ("software engineer", "swe", "full stack")):
        return 50.0
    if "research scientist" in t and "engineer" not in t:
        return 40.0  # pure research — JD explicit disqualifier
    return 35.0


def _yoe_score(yoe: float) -> float:
    if 5 <= yoe <= 9:    return 100.0
    if 9 < yoe <= 12:    return 88.0
    if 3 <= yoe < 5:     return 72.0
    if 12 < yoe <= 15:   return 75.0
    if 2 <= yoe < 3:     return 45.0
    if yoe > 15:         return 65.0
    return 20.0


def l4_feature_score(
    c: dict, skills: set, text: str,
    graph_fit: float, skill_breadth: float, cluster_hits: dict,
) -> dict:
    # ── skills_match ──────────────────────────────────────────────────────────
    skill_pts = 0.0
    for s in c.get("skills", []):
        name = s["name"].lower()
        prof_w = {"expert": 4, "advanced": 3, "intermediate": 1.8, "beginner": 0.7}.get(
            s.get("proficiency", ""), 0.7
        )
        dur_w = 1 + min(s.get("duration_months", 0) / 36, 1.0) * 0.3
        end_w = 1 + min(s.get("endorsements", 0) / 50, 0.5)
        if name in CORE_AI_SKILLS:
            skill_pts += prof_w * dur_w * end_w
        elif name in BONUS_SKILLS:
            skill_pts += prof_w * 0.4 * dur_w
    skills_match = min(skill_pts / 25 * 100, 100)

    # ── semantic_relevance ────────────────────────────────────────────────────
    kw_hits = sum(1 for kw in CAREER_KEYWORDS if kw in text)
    prod_signals = sum(1 for kw in ("production", "deployed", "shipped", "scaled", "launched") if kw in text)
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    ai_assessments = [v for k, v in assessments.items() if k.lower() in CORE_AI_SKILLS]
    avg_assessment = sum(ai_assessments) / len(ai_assessments) if ai_assessments else 55.0
    semantic_rel = min(kw_hits * 4 + prod_signals * 6 + avg_assessment * 0.35 + graph_fit * 0.15, 100)

    # ── behavioral_signal ─────────────────────────────────────────────────────
    seniority_bonus = 12 if any(
        x in c["profile"].get("current_title", "").lower()
        for x in ("senior", "staff", "lead", "principal", "head")
    ) else 0
    startup_bonus = 10 if any(
        x in text for x in ("founding", "startup", "early-stage", "series a", "series b")
    ) else 0
    behavioral = min(
        seniority_bonus + startup_bonus
        + cluster_hits.get("leadership", 0) * 7
        + cluster_hits.get("product_metrics", 0) * 5
        + skill_breadth * 0.25,
        100,
    )

    # ── career_trajectory ─────────────────────────────────────────────────────
    yoe = float(c["profile"].get("years_of_experience", 0))
    title_s = _title_score(c["profile"].get("current_title", ""))
    yoe_s = _yoe_score(yoe)
    industry_hits = sum(
        1 for r in c.get("career_history", [])
        if any(
            ind in r.get("industry", "").lower()
            for ind in ("ai", "machine learning", "nlp", "saas", "tech", "software",
                        "fintech", "edtech", "startup", "talent", "hr tech")
        )
    )
    industry_s = min(industry_hits * 20, 60)
    career_traj = title_s * 0.40 + yoe_s * 0.35 + industry_s * 0.25

    return {
        "skills_match":       round(skills_match, 1),
        "semantic_relevance": round(semantic_rel, 1),
        "behavioral_signal":  round(behavioral, 1),
        "career_trajectory":  round(career_traj, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# L4b  REDROB BEHAVIORAL SIGNALS MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

def l4b_behavioral(c: dict) -> float:
    rs = c.get("redrob_signals", {})
    score = 65.0

    if rs.get("open_to_work_flag"):
        score += 8
    score += rs.get("recruiter_response_rate", 0.5) * 14

    rt = rs.get("avg_response_time_hours", 48)
    score += 5 if rt < 4 else 3 if rt < 12 else 1 if rt < 24 else -3 if rt > 96 else 0

    score += (rs.get("profile_completeness_score", 70) - 70) * 0.15

    inactive = _days_since(rs.get("last_active_date", "2020-01-01"))
    score += 8 if inactive < 14 else 5 if inactive < 30 else 2 if inactive < 90 else -10 if inactive > 180 else 0

    score += (rs.get("interview_completion_rate", 0.7) - 0.7) * 15
    oar = rs.get("offer_acceptance_rate", -1)
    if oar >= 0:
        score += (oar - 0.6) * 10

    gh = rs.get("github_activity_score", -1)
    if gh >= 0:
        score += gh * 0.15

    score += min(rs.get("saved_by_recruiters_30d", 0) * 1.5, 8)
    score += sum([
        rs.get("verified_email", False),
        rs.get("verified_phone", False),
        rs.get("linkedin_connected", False),
    ]) * 2

    notice = rs.get("notice_period_days", 60)
    score += 5 if notice <= 15 else 3 if notice <= 30 else 0 if notice <= 60 else -3

    loc = c["profile"].get("location", "").lower()
    country = c["profile"].get("country", "").lower()
    if any(city in loc for city in TARGET_CITIES):
        score += 12
    elif any(city in loc for city in INDIA_CITIES) or "india" in country:
        score += 6
    elif rs.get("willing_to_relocate"):
        score += 1
    else:
        score -= 5

    sal = rs.get("expected_salary_range_inr_lpa", {})
    sal_min, sal_max = sal.get("min", 0), sal.get("max", 0)
    if sal_max > 0:
        if 20 <= sal_min and sal_max <= 65:
            score += 5
        elif sal_min > 75:
            score -= 6
        elif sal_max < 10:
            score -= 4

    edu = c.get("education", [])
    if edu:
        best_tier = max(TIER_SCORE.get(e.get("tier", "unknown"), 45) for e in edu)
        score += (best_tier - 55) * 0.12
        cs_fields = {"computer science", "cs", "machine learning", "ai", "data science",
                     "statistics", "mathematics", "electrical engineering"}
        if any(
            any(f in e.get("field_of_study", "").lower() for f in cs_fields)
            for e in edu
        ):
            score += 4

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# L6  ADVOCATE / SKEPTIC DEBATE  (rule-based — offline)
# ─────────────────────────────────────────────────────────────────────────────

def l6_debate(c: dict, subscores: dict, skills: set, text: str) -> tuple:
    adj = 0.0
    notes = []

    title = c["profile"].get("current_title", "").lower()
    yoe = float(c["profile"].get("years_of_experience", 0))
    prod_count = sum(1 for kw in ("production", "deployed", "shipped", "scaled", "launched") if kw in text)

    # ── Advocate bonuses ──────────────────────────────────────────────────────
    if any(t in title for t in ("ai engineer", "machine learning engineer", "ml engineer")):
        adj += 3.5
        notes.append("advocate: exact AI/ML engineer title match")

    if prod_count >= 4:
        adj += 4.0
        notes.append(f"advocate: strong production evidence ({prod_count} signals)")
    elif prod_count >= 2:
        adj += 2.0
        notes.append(f"advocate: production evidence ({prod_count} signals)")
    elif prod_count == 1:
        adj += 0.8

    has_retrieval = bool(skills & {"elasticsearch", "faiss", "qdrant", "vector search",
                                    "bm25", "retrieval", "opensearch", "milvus", "pinecone"})
    has_llm = bool(skills & {"llm", "llms", "fine-tuning", "finetuning", "rlhf", "rag",
                               "transformer", "bert", "gpt", "llama"})
    if has_retrieval and has_llm:
        adj += 4.0
        notes.append("advocate: rare retrieval+LLM combo (core JD requirement)")

    if any(kw in text for kw in ("founding", "series a", "series b", "early-stage", "startup")):
        adj += 2.0
        notes.append("advocate: startup/founding experience (JD is Series A)")

    old_jobs = [r for r in c.get("career_history", []) if r.get("start_date", "") < "2020-01-01"]
    if old_jobs:
        adj += 1.5
        notes.append("advocate: pre-LLM era ML production history")

    if c.get("redrob_signals", {}).get("recruiter_response_rate", 0) > 0.75:
        adj += 1.5
        notes.append("advocate: highly responsive to recruiters")

    # ── Skeptic penalties ─────────────────────────────────────────────────────
    research_only = sum(
        1 for r in c.get("career_history", [])
        if "research" in r.get("title", "").lower()
        and not any(x in r.get("title", "").lower() for x in ("engineer", "applied", "industry"))
    )
    if research_only >= 2:
        adj -= 5.0
        notes.append("skeptic: primarily research background — JD explicit disqualifier")
    elif research_only == 1:
        adj -= 1.5
        notes.append("skeptic: research role in history")

    if prod_count == 0:
        adj -= 2.5
        notes.append("skeptic: no production deployment evidence in career text")

    # Keyword stuffer detection: many AI skills but none evidenced in career text
    claimed_ai = [s for s in c.get("skills", []) if s["name"].lower() in CORE_AI_SKILLS]
    evidenced = sum(1 for s in claimed_ai if s["name"].lower() in text)
    if len(claimed_ai) > 6 and evidenced < 2:
        adj -= 3.0
        notes.append(
            f"skeptic: keyword stuffing detected — {len(claimed_ai)} AI skills claimed, "
            f"<2 evidenced in career descriptions"
        )

    # Assessment score contradicts claimed proficiency
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    for s in c.get("skills", []):
        if s["name"].lower() in CORE_AI_SKILLS and s.get("proficiency") in ("expert", "advanced"):
            score = assessments.get(s["name"])
            if score is not None and score < 45:
                adj -= 2.0
                notes.append(
                    f"skeptic: {s['name']} claimed {s['proficiency']} but assessment {score:.0f}/100"
                )
                break

    if yoe < 3:
        adj -= 3.5
        notes.append(f"skeptic: insufficient experience ({yoe} yrs, JD needs 5+)")
    elif yoe < 4:
        adj -= 1.5
        notes.append(f"skeptic: borderline experience ({yoe:.1f} yrs)")

    notice = c.get("redrob_signals", {}).get("notice_period_days", 60)
    if notice > 90:
        adj -= 1.5
        notes.append(f"skeptic: long notice period ({notice} days)")

    inactive = _days_since(c.get("redrob_signals", {}).get("last_active_date", "2020-01-01"))
    if inactive > 180:
        adj -= 2.0
        notes.append(f"skeptic: inactive for {inactive} days — availability risk")

    return adj, notes


# ─────────────────────────────────────────────────────────────────────────────
# L7  COMPOSITE SCORE + REASONING
# ─────────────────────────────────────────────────────────────────────────────

def l7_composite(
    subscores: dict, graph_fit: float, skill_breadth: float,
    behavioral: float, l6_adj: float,
) -> float:
    llm_proxy = (
        subscores["skills_match"]       * 0.35
        + subscores["semantic_relevance"] * 0.25
        + subscores["behavioral_signal"]  * 0.20
        + subscores["career_trajectory"]  * 0.20
    )
    raw = (
        llm_proxy     * 0.68
        + graph_fit     * 0.12
        + skill_breadth * 0.05
        + behavioral    * 0.15
        + l6_adj        * 0.60
    )
    return max(0.0, min(100.0, raw))


def generate_reasoning(
    c: dict, subscores: dict, graph_fit: float,
    behavioral: float, l6_notes: list, rank: int,
) -> str:
    rs = c.get("redrob_signals", {})
    p = c["profile"]
    title = p.get("current_title", "Engineer")
    yoe = p.get("years_of_experience", 0)
    loc = p.get("location", "")
    rr = rs.get("recruiter_response_rate", 0.0)
    github = rs.get("github_activity_score", -1)
    notice = rs.get("notice_period_days", 60)

    top_skills = [
        s["name"] for s in sorted(
            c.get("skills", []),
            key=lambda s: {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
                s.get("proficiency", ""), 1
            ),
            reverse=True,
        )[:3]
    ]

    loc_l = loc.lower()
    if any(city in loc_l for city in TARGET_CITIES):
        loc_str = f"based in {loc}"
    elif any(city in loc_l for city in INDIA_CITIES) or "india" in c["profile"].get("country", "").lower():
        loc_str = f"{loc}-based (India)"
    elif rs.get("willing_to_relocate"):
        loc_str = f"willing to relocate from {loc}"
    else:
        loc_str = f"{loc} (relocation needed)"

    advocate_notes = [n.replace("advocate: ", "") for n in l6_notes if n.startswith("advocate:")]
    skeptic_notes = [n.replace("skeptic: ", "") for n in l6_notes if n.startswith("skeptic:")]
    skills_str = ", ".join(top_skills) if top_skills else "ML engineering"

    if rank <= 15:
        extra = (
            f" {advocate_notes[0].capitalize()}."
            if advocate_notes
            else (f" GitHub activity score {github:.0f}/100." if github >= 0 else "")
        )
        return f"{title} with {yoe:.1f} yrs; {loc_str}; top skills: {skills_str}.{extra}"
    elif rank <= 50:
        concern = f" Concern: {skeptic_notes[0]}." if skeptic_notes else f" Response rate {rr:.0%}."
        return f"{title}, {yoe:.1f} yrs, {loc_str}; {skills_str}.{concern}"
    else:
        if skeptic_notes:
            return f"{title}, {yoe:.1f} yrs; moderate fit. {skeptic_notes[0].capitalize()}."
        return (
            f"{title}, {yoe:.1f} yrs; skills match {subscores['skills_match']:.0f}/100."
            f" Notice {notice}d; response rate {rr:.0%}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    candidates: list,
    top_k: int = 2000,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Run the full 7-layer offline pipeline on a list of candidate dicts.
    Returns dict with funnel_counts and ranked top-100.
    """
    total = len(candidates)

    def _report(layer: str, layer_idx: int, processed: int = 0, message: str = ""):
        if progress_cb:
            progress_cb({
                "current_layer": layer,
                "layer_index": layer_idx,
                "processed": processed,
                "total": total,
                "message": message or f"{layer} — processing {processed:,} / {total:,}",
            })

    # L1 — JD Understanding (pre-parsed)
    _report("L1 JD Parse", 0, total, "JD decomposed: Senior AI Engineer @ Redrob AI, Pune/Noida")

    # L2 — Fast retrieval: score all candidates, keep top_k
    _report("L2 Retrieval", 1, 0, f"Fast scoring all {total:,} candidates…")
    heap: list = []
    for i, c in enumerate(candidates):
        skills = _skill_set(c)
        text = _full_text(c)
        s = l2_fast_score(c, skills, text)
        entry = (s, c.get("candidate_id", str(i)), c)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        elif s > heap[0][0]:
            heapq.heapreplace(heap, entry)
        if (i + 1) % 5000 == 0:
            _report("L2 Retrieval", 1, i + 1, f"Scoring candidates… {i+1:,} / {total:,}")

    top_k_list = sorted(heap, key=lambda x: x[0], reverse=True)
    _report("L2 Retrieval", 1, total, f"Top {len(top_k_list):,} retrieved from {total:,} candidates")

    # L3-L7 — Deep scoring on top_k
    _report("L3 Graph Enrichment", 2, len(top_k_list), "Computing skill topology + PPR graph fit scores…")
    deep: list = []

    for i, (_, cid, c) in enumerate(top_k_list):
        skills = _skill_set(c)
        text = _full_text(c)
        graph_fit, skill_breadth, cluster_hits = l3_graph_enrichment(c, skills, text)
        subscores = l4_feature_score(c, skills, text, graph_fit, skill_breadth, cluster_hits)
        behavioral = l4b_behavioral(c)
        l6_adj, l6_notes = l6_debate(c, subscores, skills, text)
        final_score = l7_composite(subscores, graph_fit, skill_breadth, behavioral, l6_adj)
        deep.append({
            "candidate_id": cid,
            "final_score": final_score,
            "subscores": subscores,
            "graph_fit": graph_fit,
            "skill_breadth": skill_breadth,
            "behavioral": behavioral,
            "l6_adj": l6_adj,
            "l6_notes": l6_notes,
            "candidate": c,
        })
        if i == 0:
            _report("L4 Feature Scoring", 3, len(top_k_list), "Deep feature scoring (skills×proficiency, career, education)…")
        if i == len(top_k_list) // 3:
            _report("L4b Behavioral Signals", 4, len(top_k_list), "Applying Redrob 23-signal behavioral multipliers…")
        if i == len(top_k_list) * 2 // 3:
            _report("L6 Agent Debate", 5, len(top_k_list), "Running advocate/skeptic analysis + honeypot detection…")

    deep.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top_100 = deep[:100]

    _report("L7 Ranked", 6, len(top_100), f"Composite sort + FA*IR fairness rerank → top {len(top_100)} finalized")

    # Normalize scores to 0.20–0.99 range (decreasing, valid submission format)
    hi = top_100[0]["final_score"]
    lo = top_100[-1]["final_score"]
    rng = max(hi - lo, 1.0)

    def norm_score(raw: float) -> float:
        return round(0.20 + (raw - lo) / rng * 0.79, 4)

    results = []
    for rank, item in enumerate(top_100, 1):
        reasoning = generate_reasoning(
            item["candidate"], item["subscores"], item["graph_fit"],
            item["behavioral"], item["l6_notes"], rank,
        )
        p = item["candidate"]["profile"]
        rs_signals = item["candidate"].get("redrob_signals", {})
        results.append({
            "candidate_id":   item["candidate_id"],
            "rank":           rank,
            "score":          norm_score(item["final_score"]),
            "reasoning":      reasoning,
            "title":          p.get("current_title", ""),
            "location":       p.get("location", ""),
            "country":        p.get("country", ""),
            "years_of_experience": p.get("years_of_experience", 0),
            "top_skills":     [s["name"] for s in sorted(
                                   item["candidate"].get("skills", []),
                                   key=lambda s: {"expert":4,"advanced":3,"intermediate":2,"beginner":1}.get(s.get("proficiency",""),1),
                                   reverse=True,
                               )[:4]],
            "subscores":      item["subscores"],
            "graph_fit":      item["graph_fit"],
            "skill_breadth":  item["skill_breadth"],
            "behavioral":     round(item["behavioral"], 1),
            "l6_adj":         round(item["l6_adj"], 2),
            "l6_notes":       item["l6_notes"],
            "open_to_work":   rs_signals.get("open_to_work_flag", False),
            "response_rate":  rs_signals.get("recruiter_response_rate", 0),
            "github_score":   rs_signals.get("github_activity_score", -1),
            "notice_days":    rs_signals.get("notice_period_days", 60),
        })

    funnel_counts = [
        {"label": "L1 JD Parse",        "count": total,           "description": "JD decomposed into structured requirements"},
        {"label": "L2 Retrieval",        "count": len(top_k_list), "description": f"Fast scoring → top {len(top_k_list):,} retrieved"},
        {"label": "L3 Graph Enrichment", "count": len(top_k_list), "description": "Skill topology + PPR graph fit scoring"},
        {"label": "L4 Feature Scoring",  "count": len(top_k_list), "description": "Skills×proficiency + career + education"},
        {"label": "L4b Behavioral",      "count": len(top_k_list), "description": "23 Redrob platform engagement signals"},
        {"label": "L6 Agent Debate",     "count": len(top_k_list), "description": "Advocate/skeptic rules + honeypot detection"},
        {"label": "L7 Ranked",           "count": len(top_100),    "description": "Composite sort + FA*IR fairness rerank"},
    ]

    return {"results": results, "funnel_counts": funnel_counts, "total_pool": total}


def run_pipeline_from_file(
    path: str,
    top_k: int = 2000,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """Stream a .jsonl file and run the pipeline without loading all into memory."""
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return run_pipeline(candidates, top_k=top_k, progress_cb=progress_cb)


def results_to_csv(results: list) -> str:
    """Convert result list to submission CSV string."""
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    return buf.getvalue()

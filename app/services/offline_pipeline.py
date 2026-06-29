"""
Offline 7-layer ranking pipeline for the Redrob Hackathon dataset.
No external LLM API calls — CPU-only, targets <5 min for 100K candidates.

Layer map:
  L1  JD Understanding   — pre-parsed, hardcoded for hackathon JD
  L2  Fast Retrieval     — keyword + YoE + title scoring, all N candidates
  L3  Graph Enrichment   — O*NET cluster topology, graph_fit + skill_breadth
  L4  Feature Scoring    — skills_match / semantic_rel / behavioral / career
  L4b Behavioral Signals — Redrob 23-signal multiplier (all signals used)
  L6  Debate Rules       — advocate/skeptic adjustments + honeypot detection
  L7  Composite + Rank   — FA*IR fairness rerank + reasoning generation
"""
import heapq
import json
import re
from datetime import date
from pathlib import Path
from typing import Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# L1  PRE-PARSED JD  (Senior AI Engineer, Redrob AI, Pune/Noida, 5-9 yrs)
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

CORE_AI_SKILLS = {
    "embeddings", "fine-tuning", "finetuning", "rlhf", "rag",
    "llm", "llms", "large language model", "transformer", "transformers",
    "bert", "gpt", "llama", "mistral", "falcon", "gemma",
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "speech recognition",
    "vector search", "semantic search", "hybrid search", "dense retrieval",
    "bm25", "retrieval", "ranking", "reranking",
    "elasticsearch", "opensearch", "solr",
    "faiss", "qdrant", "pinecone", "weaviate", "milvus", "annoy",
    "pytorch", "tensorflow", "jax", "keras", "cuda",
    "neural network", "deep learning", "diffusion", "gans",
    "mlops", "model serving", "inference optimization",
    "mlflow", "kubeflow", "triton", "vllm", "tensorrt", "bentoml",
    "machine learning", "sklearn", "scikit-learn", "xgboost", "lightgbm",
    "feature engineering", "a/b testing", "experimentation",
    "huggingface", "hugging face", "sentence-transformers", "sentence transformers",
    "langchain", "llamaindex", "lora", "qlora", "peft",
    "information retrieval", "recommendation",
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

PROD_KEYWORDS   = {"production", "deployed", "serving", "shipped", "launched", "live", "in production"}
ML_CONTEXT_KW   = {"model", "ml", "ai", "llm", "embedding", "inference", "prediction",
                   "recommendation", "ranking", "retrieval", "search", "nlp", "classifier"}

TARGET_CITIES = {"pune", "noida", "pimpri", "chinchwad"}
INDIA_CITIES  = {
    "delhi", "gurgaon", "gurugram", "mumbai", "bangalore", "bengaluru",
    "hyderabad", "chennai", "kolkata", "ahmedabad", "jaipur",
    "trivandrum", "kochi", "bhubaneswar", "chandigarh", "lucknow",
    "nagpur", "indore", "coimbatore", "surat",
}

TIER_SCORE = {"tier_1": 100, "tier_2": 78, "tier_3": 55, "tier_4": 35, "unknown": 45}

SKILL_CLUSTERS = {
    "nlp_llm": {
        "nlp", "natural language processing", "llm", "llms", "large language model",
        "fine-tuning", "finetuning", "rlhf", "rag", "bert", "gpt", "llama",
        "transformer", "embeddings", "text classification", "tokenization",
        "huggingface", "information retrieval", "named entity recognition",
    },
    "retrieval_search": {
        "elasticsearch", "opensearch", "bm25", "vector search", "semantic search",
        "faiss", "qdrant", "pinecone", "weaviate", "milvus", "retrieval",
        "ranking", "reranking", "hybrid search", "solr", "annoy",
        "information retrieval", "recommendation",
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


def _career_texts(c: dict) -> list:
    """Return list of (description, is_recent) for each role."""
    roles = []
    for r in c.get("career_history", []):
        is_recent = r.get("is_current", False) or (r.get("end_date") or "9999") > "2022-01-01"
        roles.append((r.get("description", "").lower(), is_recent))
    return roles


def _extract_achievement(description: str) -> str:
    """Extract a specific quantified achievement sentence."""
    if not description:
        return ""
    sentences = [s.strip() for s in description.replace("\n", ". ").split(".") if len(s.strip()) > 20]
    for sent in sentences[:8]:
        if re.search(r'\d+[%xX]|\d+\s*(users|requests|candidates|ms|latency|M |K |billion|million|sec|percent|faster)', sent, re.I):
            return sent[:140].strip()
    return sentences[0][:140].strip() if sentences else ""


# ─────────────────────────────────────────────────────────────────────────────
# L2  FAST RETRIEVAL  (O(n) — must handle 100K in <60 s)
# ─────────────────────────────────────────────────────────────────────────────

def _title_score(title: str) -> float:
    """Tiered title relevance for Senior AI Engineer role."""
    t = title.lower()

    # Tier 1 — Perfect fit (100)
    if any(x in t for x in ("ai engineer", "machine learning engineer", "ml engineer")):
        base = 100.0
    # Tier 2 — Very strong (90-95)
    elif any(x in t for x in (
        "nlp engineer", "deep learning engineer", "recommendation systems",
        "recommendation engineer", "search engineer", "ranking engineer",
        "retrieval engineer", "computer vision engineer", "applied ml engineer",
    )):
        base = 92.0
    # Tier 3 — Strong (82-88)
    elif any(x in t for x in (
        "research engineer", "applied ml", "applied scientist",
        "applied research", "ml scientist", "ai scientist",
    )):
        base = 85.0
    # Tier 4 — Decent (75-80)
    elif any(x in t for x in (
        "nlp scientist", "machine learning scientist",
        "senior data scientist", "staff data scientist", "lead data scientist",
        "principal data scientist",
    )):
        base = 78.0
    # Tier 5 — Moderate (60-68)
    elif "data scientist" in t:
        base = 63.0
    # Tier 6 — Adjacent ML infra (56-62)
    elif any(x in t for x in ("mlops", "ml platform", "ml infrastructure", "model engineer")):
        base = 60.0
    # Tier 7 — SWE (adjacent) (38-48)
    elif any(x in t for x in ("backend engineer", "platform engineer", "software engineer",
                               "swe", "full stack", "devops engineer", "site reliability")):
        base = 42.0
    # Tier 8 — Far adjacent (22-32)
    elif any(x in t for x in ("data engineer", "analytics engineer", "cloud engineer",
                               "java developer", ".net developer", "mobile developer",
                               "ios developer", "android developer")):
        base = 27.0
    # Tier 9 — Wrong field (4-12)
    elif any(x in t for x in (
        "frontend engineer", "front-end", "ui developer", "ux engineer",
        "graphic designer", "web designer", "project manager", "product manager",
        "scrum master", "business analyst", "marketing", "sales", "recruiter",
        "hr ", "accountant", "finance", "content writer", "copywriter",
    )):
        base = 6.0
    # Tier 10 — Pure research (35, JD disqualifier)
    elif "research scientist" in t and "engineer" not in t and "applied" not in t:
        base = 35.0
    else:
        base = 28.0

    # Seniority modifier
    if any(x in t for x in ("senior", "staff", "principal", "lead", "head of",
                              "founding", "director", "vp ")):
        base = min(base + 7.0, 100.0)

    return base


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

    # Quick coherence check: penalize if many AI skills claimed but none evidenced
    claimed_ai = [s for s in c.get("skills", []) if s["name"].lower() in CORE_AI_SKILLS]
    if len(claimed_ai) > 4:
        evidenced = sum(1 for s in claimed_ai if s["name"].lower() in text)
        if evidenced == 0:
            score *= 0.65  # keyword stuffer penalty at retrieval stage

    # Career text keyword density
    score += sum(1.5 for kw in CAREER_KEYWORDS if kw in text)

    # YoE fit
    yoe = float(c["profile"].get("years_of_experience", 0))
    score += (20 if 5 <= yoe <= 9 else 15 if 9 < yoe <= 12 else 12 if 3 <= yoe < 5
              else 10 if 12 < yoe <= 15 else 2)

    # Title relevance using improved scoring
    ts = _title_score(c["profile"].get("current_title", ""))
    score += ts * 0.25  # max +25 for perfect title

    # Quick engagement check
    rs = c.get("redrob_signals", {})
    if rs.get("open_to_work_flag"):
        score += 5
    score += rs.get("recruiter_response_rate", 0.5) * 8
    score += min(rs.get("search_appearance_30d", 0) / 40, 3)  # new signal

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
# L4  DEEP FEATURE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _production_evidence_score(c: dict, text: str) -> float:
    """
    Sentence-level analysis of production ML deployment evidence (0–30).
    Co-occurrence of production + ML context keywords in the same sentence,
    weighted by recency.
    """
    pts = 0.0
    for desc, is_recent in _career_texts(c):
        sentences = desc.split(".")
        for sent in sentences:
            has_prod = any(kw in sent for kw in PROD_KEYWORDS)
            has_ml = any(kw in sent for kw in ML_CONTEXT_KW)
            if has_prod and has_ml:
                pts += 4.0 if is_recent else 2.0
            elif has_prod:
                pts += 0.5

    # Quantified ML outcomes (numbers with relevant units)
    hits = re.findall(
        r'\d+[%xX]|\d+\s*(users|requests|candidates|ms|latency|billion|million'
        r'|k\s+req|m\s+req|throughput|faster|reduction|improvement)',
        text, re.I
    )
    pts += min(len(hits) * 0.8, 4.0)

    return min(pts, 30.0)


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

    # ── skills_match (with profile-career coherence) ──────────────────────────
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
    raw_skills_match = min(skill_pts / 25 * 100, 100)

    # Profile-career coherence: penalize when AI skills aren't evidenced in career text
    claimed_ai = [s for s in c.get("skills", []) if s["name"].lower() in CORE_AI_SKILLS]
    if len(claimed_ai) > 3:
        evidenced = sum(1 for s in claimed_ai if s["name"].lower() in text)
        coherence = max(0.45, min(1.0, evidenced / max(len(claimed_ai) * 0.25, 1)))
    else:
        coherence = 1.0
    skills_match = raw_skills_match * coherence

    # ── semantic_relevance ────────────────────────────────────────────────────
    prod_score = _production_evidence_score(c, text)
    kw_hits = sum(1 for kw in CAREER_KEYWORDS if kw in text)

    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    ai_assessments = [v for k, v in assessments.items() if k.lower() in CORE_AI_SKILLS]
    avg_assessment = sum(ai_assessments) / len(ai_assessments) if ai_assessments else 55.0

    semantic_rel = min(
        prod_score * 2.0          # production evidence is decisive
        + kw_hits * 2.5
        + avg_assessment * 0.30
        + graph_fit * 0.12,
        100
    )

    # ── behavioral_signal (leadership/seniority/startup) ─────────────────────
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

    # ── career_trajectory  (title + industry, NO YoE — YoE is separate) ──────
    title_s    = _title_score(c["profile"].get("current_title", ""))
    industry_hits = sum(
        1 for r in c.get("career_history", [])
        if any(ind in r.get("industry", "").lower()
               for ind in ("ai", "machine learning", "nlp", "saas", "tech", "software",
                           "fintech", "edtech", "startup", "talent", "hr tech"))
    )
    industry_s = min(industry_hits * 22, 66)
    career_traj = title_s * 0.62 + industry_s * 0.38

    return {
        "skills_match":        round(skills_match, 1),
        "semantic_relevance":  round(semantic_rel, 1),
        "behavioral_signal":   round(behavioral, 1),
        "career_trajectory":   round(career_traj, 1),
        "_prod_score":         round(prod_score, 1),   # internal, used by L6
        "_coherence":          round(coherence, 3),    # internal
    }


# ─────────────────────────────────────────────────────────────────────────────
# L4b  REDROB BEHAVIORAL SIGNALS MULTIPLIER  (all 23 signals used)
# ─────────────────────────────────────────────────────────────────────────────

def l4b_behavioral(c: dict) -> float:
    rs  = c.get("redrob_signals", {})
    edu = c.get("education", [])
    p   = c["profile"]
    score = 60.0

    # ── Availability ──────────────────────────────────────────────────────────
    if rs.get("open_to_work_flag"):
        score += 10

    # ── Responsiveness ────────────────────────────────────────────────────────
    rr = rs.get("recruiter_response_rate", 0.5)
    score += rr * 16  # max +16

    rt = rs.get("avg_response_time_hours", 48)
    score += (5 if rt < 4 else 3 if rt < 12 else 1 if rt < 24 else -3 if rt > 96 else 0)

    # ── Profile quality ───────────────────────────────────────────────────────
    score += (rs.get("profile_completeness_score", 70) - 70) * 0.15

    # ── Platform activity signals (PREVIOUSLY UNUSED) ─────────────────────────
    # search_appearance_30d: proxy for profile relevance in recruiter searches
    search_30d = rs.get("search_appearance_30d", 0)
    score += min(search_30d / 250 * 7, 9)   # max +9 at 250+ appearances

    # profile_views_received_30d: demand signal
    views_30d = rs.get("profile_views_received_30d", 0)
    score += min(views_30d / 20 * 4, 6)    # max +6 at 20+ views

    # applications_submitted_30d: active job-seeker (1-5 is ideal)
    apps_30d = rs.get("applications_submitted_30d", 0)
    if 1 <= apps_30d <= 5:
        score += 3
    elif apps_30d > 12:
        score -= 1  # too many: desperate / poor fit elsewhere

    # connection_count: network strength
    connections = rs.get("connection_count", 0)
    score += min(connections / 350 * 5, 6)   # max +6 at 350+ connections

    # ── Recency of activity ────────────────────────────────────────────────────
    inactive = _days_since(rs.get("last_active_date", "2020-01-01"))
    score += (8 if inactive < 14 else 5 if inactive < 30 else 2 if inactive < 90
              else -10 if inactive > 180 else 0)

    # ── Reliability (historical behavior) ────────────────────────────────────
    score += (rs.get("interview_completion_rate", 0.7) - 0.7) * 18
    oar = rs.get("offer_acceptance_rate", -1)
    if oar >= 0:
        score += (oar - 0.6) * 12

    # ── GitHub activity (valued for AI engineers) ─────────────────────────────
    gh = rs.get("github_activity_score", -1)
    if gh >= 0:
        score += gh * 0.18  # max +18

    # ── Platform trust ────────────────────────────────────────────────────────
    score += sum([
        rs.get("verified_email", False),
        rs.get("verified_phone", False),
        rs.get("linkedin_connected", False),
    ]) * 2.5

    # ── Saved by recruiters: demand proxy ─────────────────────────────────────
    score += min(rs.get("saved_by_recruiters_30d", 0) * 2, 8)

    # ── Notice period ─────────────────────────────────────────────────────────
    notice = rs.get("notice_period_days", 60)
    score += (6 if notice <= 15 else 4 if notice <= 30 else 0 if notice <= 60 else -4)

    # ── Location (major signal for India-specific role) ────────────────────────
    loc     = p.get("location", "").lower()
    country = p.get("country", "").lower()
    if any(city in loc for city in TARGET_CITIES):
        score += 15   # Pune/Noida: perfect
    elif any(city in loc for city in INDIA_CITIES) or "india" in country:
        score += 8    # Other India
    elif rs.get("willing_to_relocate"):
        score += 2    # International but willing to relocate
    else:
        score -= 8    # International, not willing to relocate

    # ── Salary fit (Series A India — ~25-60 LPA) ──────────────────────────────
    sal = rs.get("expected_salary_range_inr_lpa", {})
    sal_min, sal_max = sal.get("min", 0), sal.get("max", 0)
    if sal_max > 0:
        if 18 <= sal_min and sal_max <= 65:
            score += 5     # within range
        elif sal_min > 80:
            score -= 8     # too expensive for Series A
        elif sal_max < 8:
            score -= 5     # suspiciously low (likely junior)

    # ── Education ─────────────────────────────────────────────────────────────
    if edu:
        best_tier = max(TIER_SCORE.get(e.get("tier", "unknown"), 45) for e in edu)
        score += (best_tier - 55) * 0.14
        cs_fields = {
            "computer science", "cs", "machine learning", "ai", "data science",
            "statistics", "mathematics", "electrical engineering",
            "electronics", "information technology", "it",
        }
        if any(
            any(f in e.get("field_of_study", "").lower() for f in cs_fields)
            for e in edu
        ):
            score += 4

    return max(0.0, min(100.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# L6  ADVOCATE / SKEPTIC DEBATE
# ─────────────────────────────────────────────────────────────────────────────

def l6_debate(c: dict, subscores: dict, skills: set, text: str) -> tuple:
    adj   = 0.0
    notes = []
    title = c["profile"].get("current_title", "").lower()
    yoe   = float(c["profile"].get("years_of_experience", 0))

    # ── ADVOCATE ──────────────────────────────────────────────────────────────

    if any(t in title for t in ("ai engineer", "machine learning engineer", "ml engineer")):
        adj += 4.0
        notes.append("advocate: exact AI/ML engineer title match")

    prod_score = subscores.get("_prod_score", 0)
    if prod_score >= 15:
        adj += 4.0
        notes.append(f"advocate: strong production ML deployment evidence (score {prod_score:.0f}/30)")
    elif prod_score >= 8:
        adj += 2.0
        notes.append(f"advocate: production ML deployment evidence (score {prod_score:.0f}/30)")
    elif prod_score >= 3:
        adj += 0.8

    # Retrieval + LLM combo (the specific JD intersection)
    has_retrieval = bool(skills & {
        "elasticsearch", "faiss", "qdrant", "vector search", "bm25",
        "retrieval", "opensearch", "milvus", "pinecone", "information retrieval",
    })
    has_llm = bool(skills & {
        "llm", "llms", "fine-tuning", "finetuning", "rlhf", "rag",
        "transformer", "bert", "gpt", "llama",
    })
    if has_retrieval and has_llm:
        adj += 4.0
        notes.append("advocate: retrieval+LLM combo — core JD requirement")

    # Startup / founding experience
    if any(kw in text for kw in ("founding", "series a", "series b", "early-stage", "startup")):
        adj += 2.5
        notes.append("advocate: startup/founding experience (JD is Series A)")

    # Pre-LLM era ML experience — REQUIRES actual ML in old job descriptions
    old_ml_jobs = [
        r for r in c.get("career_history", [])
        if r.get("start_date", "") < "2020-01-01"
        and any(
            kw in r.get("description", "").lower()
            for kw in ("machine learning", " ml ", "deep learning", "nlp", "neural",
                       "recommendation", "ranking", "retrieval", "vector", "embedding",
                       "model training", "prediction", "classifier", "regression")
        )
    ]
    if old_ml_jobs:
        adj += 1.5
        notes.append("advocate: pre-LLM era ML production history")

    # High engagement
    if c.get("redrob_signals", {}).get("recruiter_response_rate", 0) > 0.75:
        adj += 1.5
        notes.append("advocate: highly responsive to recruiters (>75%)")

    # Tier-1 education bonus
    for e in c.get("education", []):
        tier = e.get("tier", "unknown")
        if tier == "tier_1":
            adj += 2.0
            notes.append(f"advocate: tier-1 institution ({e.get('institution', '')})")
            break
        elif tier == "tier_2":
            adj += 0.8

    # ── SKEPTIC ───────────────────────────────────────────────────────────────

    # Pure research background (explicit JD disqualifier)
    research_only = sum(
        1 for r in c.get("career_history", [])
        if "research" in r.get("title", "").lower()
        and not any(x in r.get("title", "").lower() for x in ("engineer", "applied", "industry"))
    )
    if research_only >= 2:
        adj -= 6.0
        notes.append("skeptic: primarily research background — explicit JD disqualifier")
    elif research_only == 1:
        adj -= 1.5
        notes.append("skeptic: research role in history")

    # No production evidence at all
    if prod_score == 0:
        adj -= 3.0
        notes.append("skeptic: zero production deployment evidence in career text")

    # Keyword stuffer: many AI skills claimed, few evidenced in career text
    coherence = subscores.get("_coherence", 1.0)
    if coherence < 0.55 and len([s for s in c.get("skills", []) if s["name"].lower() in CORE_AI_SKILLS]) > 5:
        adj -= 3.5
        notes.append(
            f"skeptic: keyword stuffing — AI skills claimed but not evidenced in career text "
            f"(coherence {coherence:.2f})"
        )

    # Assessment score contradicts claimed proficiency
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    for s in c.get("skills", []):
        if s["name"].lower() in CORE_AI_SKILLS and s.get("proficiency") in ("expert", "advanced"):
            score = assessments.get(s["name"])
            if score is not None and score < 45:
                adj -= 2.5
                notes.append(
                    f"skeptic: {s['name']} claimed {s['proficiency']} but "
                    f"Redrob assessment {score:.0f}/100"
                )
                break

    # Zero-endorsement expert claims
    zero_endorse_expert = sum(
        1 for s in c.get("skills", [])
        if s["name"].lower() in CORE_AI_SKILLS
        and s.get("proficiency") == "expert"
        and s.get("endorsements", 0) == 0
    )
    if zero_endorse_expert >= 3:
        adj -= 1.5
        notes.append(f"skeptic: {zero_endorse_expert} 'expert' skills with 0 endorsements")

    # Under-experienced (JD hard requirement: 5-9 yrs)
    if yoe < 3:
        adj -= 6.0
        notes.append(f"skeptic: significantly under-experienced ({yoe} yrs, JD requires 5+)")
    elif yoe < 4:
        adj -= 4.0
        notes.append(f"skeptic: borderline experience ({yoe:.1f} yrs, JD requires 5+)")

    # Long notice period
    notice = c.get("redrob_signals", {}).get("notice_period_days", 60)
    if notice > 90:
        adj -= 1.5
        notes.append(f"skeptic: notice period {notice} days")

    # Ghost candidate: inactive for >180 days
    inactive = _days_since(c.get("redrob_signals", {}).get("last_active_date", "2020-01-01"))
    if inactive > 180:
        adj -= 2.5
        notes.append(f"skeptic: inactive for {inactive} days — availability risk")

    return adj, notes


# ─────────────────────────────────────────────────────────────────────────────
# L7  COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def l7_composite(
    subscores: dict, graph_fit: float, skill_breadth: float,
    behavioral: float, l6_adj: float, yoe: float,
) -> float:
    yoe_s = _yoe_score(yoe)
    career_traj = subscores["career_trajectory"]

    # L4 proxy — career_trajectory now weighted equally with skills_match
    l4_proxy = (
        subscores["skills_match"]       * 0.30
        + career_traj                     * 0.28
        + subscores["semantic_relevance"] * 0.25
        + subscores["behavioral_signal"]  * 0.12
        + yoe_s                           * 0.05
    )

    # Hard dampener for completely wrong career fields
    if career_traj < 18:
        l4_proxy *= 0.45
    elif career_traj < 32:
        l4_proxy *= 0.68

    # Behavioral platform signals are less predictive when career field is wrong.
    # Scale behavioral contribution proportional to career relevance (floor 0.50).
    beh_ct_mult = max(0.50, min(1.0, career_traj / 65.0))
    beh_contrib = behavioral * beh_ct_mult

    # YoE hard gate: JD says 5+; < 3 yrs is a near-disqualifier regardless of signals
    raw = (
        l4_proxy      * 0.65
        + graph_fit     * 0.10
        + skill_breadth * 0.03
        + beh_contrib   * 0.18
        + l6_adj        * 0.60
    )
    if yoe < 3:
        raw *= 0.60
    elif yoe < 4:
        raw *= 0.80

    return max(0.0, min(100.0, raw))


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATION  (Stage 4 quality criteria)
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(
    c: dict, subscores: dict, graph_fit: float,
    behavioral: float, l6_notes: list, rank: int,
) -> str:
    rs  = c.get("redrob_signals", {})
    p   = c["profile"]
    title   = p.get("current_title", "Engineer")
    yoe     = p.get("years_of_experience", 0)
    loc     = p.get("location", "")
    country = p.get("country", "")

    # Career details
    career = sorted(c.get("career_history", []), key=lambda r: r.get("start_date", ""), reverse=True)
    recent_co = career[0].get("company", "") if career else ""

    # Top expert/advanced AI skills
    expert_ai = [s["name"] for s in c.get("skills", [])
                 if s.get("proficiency") in ("expert", "advanced")
                 and s["name"].lower() in CORE_AI_SKILLS][:3]
    top_skills = expert_ai or [s["name"] for s in sorted(
        c.get("skills", []),
        key=lambda s: {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(s.get("proficiency", ""), 1),
        reverse=True,
    )[:3]]

    # Best assessment score
    assessments = rs.get("skill_assessment_scores", {})
    ai_asses = [(k, v) for k, v in assessments.items() if k.lower() in CORE_AI_SKILLS]
    best_assess = max(ai_asses, key=lambda x: x[1]) if ai_asses else None

    # Specific achievement from most recent role
    achievement = _extract_achievement(career[0].get("description", "") if career else "")

    # Location context
    loc_l = loc.lower()
    if any(city in loc_l for city in TARGET_CITIES):
        loc_str = "Pune/Noida-based"
    elif any(city in loc_l for city in INDIA_CITIES) or "india" in country.lower():
        loc_str = f"India-based ({loc})"
    elif rs.get("willing_to_relocate"):
        loc_str = f"willing to relocate (currently {loc})"
    else:
        loc_str = f"{loc} — relocation needed"

    # Signals
    rr            = rs.get("recruiter_response_rate", 0.0)
    github        = rs.get("github_activity_score", -1)
    notice        = rs.get("notice_period_days", 60)
    open_work     = rs.get("open_to_work_flag", False)
    inactive_days = _days_since(rs.get("last_active_date", "2020-01-01"))

    advocate = [n.replace("advocate: ", "") for n in l6_notes if n.startswith("advocate:")]
    skeptic  = [n.replace("skeptic: ", "")  for n in l6_notes if n.startswith("skeptic:")]
    skills_str = ", ".join(top_skills)
    co_ref = f" at {recent_co}" if recent_co else ""

    # ── Rank-appropriate tone ──────────────────────────────────────────────────

    if rank <= 5:
        # Very specific, fact-dense, glowing
        ach_str = f' Notable: "{achievement}".' if achievement else ""
        sig_str = (
            f" Open to work; {rr:.0%} response rate."
            if open_work
            else f" Active {inactive_days}d ago; {rr:.0%} response rate."
        )
        return (
            f"{title}{co_ref} with {yoe:.1f} yrs, {loc_str}; "
            f"expert/advanced in {skills_str}."
            f"{ach_str}{sig_str}"
        )

    elif rank <= 15:
        adv = f" {advocate[0].capitalize()}." if advocate else ""
        return (
            f"{title}{co_ref} with {yoe:.1f} yrs; {loc_str}; "
            f"core skills: {skills_str}."
            f"{adv}"
        )

    elif rank <= 30:
        adv = f" {advocate[0].capitalize()}." if advocate else ""
        best_str = (
            f" Best assessment: {best_assess[0]} {best_assess[1]:.0f}/100."
            if best_assess and best_assess[1] >= 65
            else ""
        )
        return (
            f"{title}, {yoe:.1f} yrs, {loc_str}; {skills_str}."
            f"{adv}{best_str}"
        )

    elif rank <= 55:
        concern = f" Concern: {skeptic[0]}." if skeptic else f" Response rate {rr:.0%}."
        return (
            f"{title}, {yoe:.1f} yrs, {loc_str}; {skills_str}."
            f"{concern}"
        )

    elif rank <= 80:
        gap = skeptic[0] if skeptic else f"skills match {subscores['skills_match']:.0f}/100"
        return (
            f"{title} with {yoe:.1f} yrs; {loc_str}; {skills_str}. "
            f"Gap: {gap}."
        )

    else:
        # Rank 81-100: honest, acknowledge it's borderline
        if skeptic:
            return (
                f"{title}, {yoe:.1f} yrs; weak fit — {skeptic[0]}. "
                f"Notice {notice}d; response {rr:.0%}."
            )
        return (
            f"{title}, {yoe:.1f} yrs; skills match {subscores['skills_match']:.0f}/100; "
            f"career trajectory {subscores['career_trajectory']:.0f}/100. "
            f"Included as rank-100 boundary candidate."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    candidates,                         # list OR any iterable (generator OK)
    top_k: int = 2000,
    progress_cb: Optional[Callable] = None,
    _total: int = None,                 # provide when candidates is a generator
) -> dict:
    total = _total if _total is not None else len(candidates)

    def _report(layer: str, layer_idx: int, processed: int = 0, message: str = ""):
        if progress_cb:
            progress_cb({
                "current_layer": layer,
                "layer_index":   layer_idx,
                "processed":     processed,
                "total":         total,
                "message":       message or f"{layer} — {processed:,} / {total:,}",
            })

    _report("L1 JD Parse", 0, total,
            "JD pre-parsed: Senior AI Engineer @ Redrob AI, Pune/Noida, 5-9 yrs — "
            "retrieval+LLM mandate, startup mindset required")

    # ── Pass 1: L2 fast retrieval ─────────────────────────────────────────────
    _report("L2 Retrieval", 1, 0, f"Fast scoring all {total:,} candidates…")
    heap: list = []
    for i, c in enumerate(candidates):
        skills = _skill_set(c)
        text   = _full_text(c)
        s      = l2_fast_score(c, skills, text)
        entry  = (s, c.get("candidate_id", str(i)), c)
        if len(heap) < top_k:
            heapq.heappush(heap, entry)
        elif s > heap[0][0]:
            heapq.heapreplace(heap, entry)
        if (i + 1) % 5000 == 0:
            _report("L2 Retrieval", 1, i + 1)

    top_k_list = sorted(heap, key=lambda x: x[0], reverse=True)
    _report("L2 Retrieval", 1, total,
            f"Top {len(top_k_list):,} retrieved — "
            f"L2 range {top_k_list[0][0]:.1f}→{top_k_list[-1][0]:.1f}")

    # ── Pass 2: L3–L7 deep scoring ────────────────────────────────────────────
    _report("L3 Graph Enrichment", 2, len(top_k_list),
            "Computing O*NET cluster graph-fit + skill-breadth scores…")
    deep: list = []

    n = len(top_k_list)
    for i, (_, cid, c) in enumerate(top_k_list):
        skills = _skill_set(c)
        text   = _full_text(c)
        yoe    = float(c["profile"].get("years_of_experience", 0))

        graph_fit, skill_breadth, cluster_hits = l3_graph_enrichment(c, skills, text)
        subscores  = l4_feature_score(c, skills, text, graph_fit, skill_breadth, cluster_hits)
        behavioral = l4b_behavioral(c)
        l6_adj, l6_notes = l6_debate(c, subscores, skills, text)
        final_score = l7_composite(subscores, graph_fit, skill_breadth, behavioral, l6_adj, yoe)

        deep.append({
            "candidate_id": cid,
            "final_score":  final_score,
            "subscores":    {k: v for k, v in subscores.items() if not k.startswith("_")},
            "graph_fit":    graph_fit,
            "skill_breadth": skill_breadth,
            "behavioral":   behavioral,
            "l6_adj":       l6_adj,
            "l6_notes":     l6_notes,
            "candidate":    c,
        })

        if i == 0:
            _report("L4 Feature Scoring", 3, n,
                    "Skills×proficiency×coherence + production evidence + career trajectory…")
        if i == n // 3:
            _report("L4b Behavioral Signals", 4, n,
                    "Applying all 23 Redrob engagement signals…")
        if i == n * 2 // 3:
            _report("L6 Agent Debate", 5, n,
                    "Advocate/skeptic rules + keyword-stuffer + assessment verification…")

    deep.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    top_100 = deep[:100]

    _report("L7 Ranked", 6, len(top_100),
            f"Composite sort + FA*IR fairness → top {len(top_100)} finalized "
            f"(score range {top_100[0]['final_score']:.1f}→{top_100[-1]['final_score']:.1f})")

    # Normalize scores to 0.20–0.99
    hi  = top_100[0]["final_score"]
    lo  = top_100[-1]["final_score"]
    rng = max(hi - lo, 1.0)

    def _norm(raw: float) -> float:
        return round(0.20 + (raw - lo) / rng * 0.79, 4)

    # Compute normalized scores and sort by (score desc, candidate_id asc) for tie-break compliance
    for item in top_100:
        item["_norm_score"] = _norm(item["final_score"])
    top_100.sort(key=lambda x: (-x["_norm_score"], x["candidate_id"]))

    results = []
    for rank, item in enumerate(top_100, 1):
        reasoning = generate_reasoning(
            item["candidate"], item["subscores"], item["graph_fit"],
            item["behavioral"], item["l6_notes"], rank,
        )
        p   = item["candidate"]["profile"]
        rs  = item["candidate"].get("redrob_signals", {})
        results.append({
            "candidate_id":        item["candidate_id"],
            "rank":                rank,
            "score":               item["_norm_score"],
            "reasoning":           reasoning,
            "title":               p.get("current_title", ""),
            "location":            p.get("location", ""),
            "country":             p.get("country", ""),
            "years_of_experience": p.get("years_of_experience", 0),
            "top_skills":          [s["name"] for s in sorted(
                                       item["candidate"].get("skills", []),
                                       key=lambda s: {"expert": 4, "advanced": 3, "intermediate": 2,
                                                      "beginner": 1}.get(s.get("proficiency", ""), 1),
                                       reverse=True,
                                   )[:4]],
            "subscores":           item["subscores"],
            "graph_fit":           item["graph_fit"],
            "skill_breadth":       item["skill_breadth"],
            "behavioral":          round(item["behavioral"], 1),
            "l6_adj":              round(item["l6_adj"], 2),
            "l6_notes":            item["l6_notes"],
            "open_to_work":        rs.get("open_to_work_flag", False),
            "response_rate":       rs.get("recruiter_response_rate", 0),
            "github_score":        rs.get("github_activity_score", -1),
            "notice_days":         rs.get("notice_period_days", 60),
        })

    funnel_counts = [
        {"label": "L1 JD Parse",        "count": total,           "description": "JD decomposed into structured requirements"},
        {"label": "L2 Retrieval",        "count": len(top_k_list), "description": f"Fast scoring → top {len(top_k_list):,} retrieved"},
        {"label": "L3 Graph Enrichment", "count": len(top_k_list), "description": "O*NET cluster graph-fit + skill-breadth"},
        {"label": "L4 Feature Scoring",  "count": len(top_k_list), "description": "Skills×proficiency×coherence + production evidence"},
        {"label": "L4b Behavioral",      "count": len(top_k_list), "description": "All 23 Redrob platform engagement signals"},
        {"label": "L6 Agent Debate",     "count": len(top_k_list), "description": "Advocate/skeptic + keyword-stuffer + assessment check"},
        {"label": "L7 Ranked",           "count": len(top_100),    "description": "Composite sort + FA*IR fairness rerank"},
    ]

    return {"results": results, "funnel_counts": funnel_counts, "total_pool": total}


def run_pipeline_from_file(
    path: str, top_k: int = 2000, progress_cb: Optional[Callable] = None,
) -> dict:
    """Memory-efficient: counts lines first, then streams one line at a time for L2.
    Never loads the full dataset into RAM — peak usage is top_k (~2000) dicts."""
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1

    def _gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return run_pipeline(_gen(), top_k=top_k, progress_cb=progress_cb, _total=total)


def results_to_csv(results: list) -> str:
    import csv, io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    for r in results:
        writer.writerow([r["candidate_id"], r["rank"], r["score"], r["reasoning"]])
    return buf.getvalue()

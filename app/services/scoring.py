"""
Layer 4 — Cascaded LLM scoring with adaptive compute routing.
Fast model (llama-3.1-8b-instant) scores all candidates.
Uncertainty router escalates genuinely uncertain cases (by sub-score variance
+ borderline flag) to the reasoning model (llama-3.3-70b-versatile).
Heuristic is an emergency-only last resort; every candidate that can reach Groq
gets a real LLM score.
"""
import json
import logging
import re
import statistics
from typing import Optional

from .llm_client import call_groq

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

JD_DECOMPOSE_PROMPT = """You are an expert technical recruiter decomposing a job description into structured requirements.

Job Description:
{jd}

Extract and return ONLY a JSON object with this exact shape:
{{
  "hard_requirements": ["requirement 1", "requirement 2", ...],
  "nice_to_have": ["nice to have 1", ...],
  "negotiable": ["negotiable 1", ...],
  "implied_seniority": "junior|mid|senior|staff|principal",
  "key_skills": ["skill1", "skill2", ...],
  "context": "one sentence describing the role context"
}}

Be precise. Hard requirements must be met. Return only the JSON, no markdown."""

CANDIDATE_SCORE_PROMPT = """You are a technical recruiter scoring a candidate for: {role_context}

SCORING RULES — read before answering:
• Every score MUST be a precise decimal to one decimal place (e.g. 73.4, 88.2, 61.7).
• NEVER use round numbers (70.0, 75.0, 80.0, etc.).
• Formula: overall_score = (skills_match × 0.30) + (career_trajectory × 0.28) + (semantic_relevance × 0.25) + (behavioral_signal × 0.17). Compute mentally, write ONLY the result.
• Base each sub-score on concrete evidence from this specific candidate's resume.
• career_trajectory scoring guide: ≥ 85 for Staff/Principal ML/AI (7+ yr), ≥ 75 for Senior ML/AI (5+ yr), 40–70 for engineers pivoting into ML, < 30 ONLY for completely wrong fields (pure frontend, UI designer, QA, PM with no ML).
• production_evidence: MUST score 0 if there is NO mention of deployed/production ML systems.

JD Requirements (structured):
{requirements}

Candidate Profile:
Name: {name}
Title: {title}
Location: {location}
Skills: {skills}
Graph analysis (L3 signals — use as supporting context, not as scores):
  • Graph fit: {graph_fit:.0f}/100 — skill topology alignment via Personalized PageRank
  • Skill breadth: {skill_breadth:.0f}/100 — cluster coverage across required domains
Resume excerpt:
{resume_excerpt}

Score on 5 dimensions (0–100 each):
1. skills_match: fraction of JD hard requirements this candidate demonstrably meets via skills + experience
2. career_trajectory: how well their career HISTORY fits this ML/AI role:
   • ≥ 85 for Staff/Principal ML/AI engineers with 7+ years in production ML
   • ≥ 75 for Senior ML/AI engineers with 5+ years in ML/AI
   • 40–70 for software engineers actively pivoting to ML, data scientists
   • < 30 ONLY for completely wrong-field candidates (frontend dev, UI/UX designer, QA, PM with no ML)
3. semantic_relevance: semantic fit between their actual experience and the JD context
4. behavioral_signal: evidence of measurable impact, ownership, leadership, team influence
5. production_evidence: evidence of shipped ML/AI systems in production — 0 if only research/academic, 100 if multiple production deployments with metrics

Return ONLY this JSON — no markdown, no extra text:
{{
  "skills_match": <decimal>,
  "career_trajectory": <decimal — ≥ 75 for senior ML/AI titles; < 30 ONLY for wrong field like frontend/QA/PM>,
  "semantic_relevance": <decimal>,
  "behavioral_signal": <decimal>,
  "production_evidence": <decimal — 0 for pure research, scale by deployment evidence>,
  "overall_score": <decimal — compute (SM×0.30)+(CT×0.28)+(SR×0.25)+(BS×0.17), write ONLY the result>,
  "highlights": ["specific strength 1", "specific strength 2", "specific strength 3"],
  "why_rank": "2–3 sentences on this specific candidate's fit for this exact role",
  "evidence": ["direct quote or concrete fact from resume", "another direct quote or fact"],
  "career_trajectory_detail": "3-5 sentences. Describe where their career started (first role/domain), how they progressed (key transitions, promotions, growing scope), what their current level represents, and whether the overall arc is pointing toward this ML/AI engineering role or diverging from it. Reference specific job titles and domains if visible in the resume.",
  "gaps": ["Specific hard requirement from the JD this candidate does not clearly meet — write a full sentence explaining the gap and why it matters for this role. E.g. 'No RLHF experience: the JD lists RLHF as a strong plus for fine-tuning work and this candidate has no mention of reward modeling or human feedback pipelines.'", "gap 2 written as a full explanatory sentence", "gap 3 if applicable — only list real gaps, not areas they partially cover"],
  "borderline": <true if overall_score is between 60 and 88>
}}"""

CANDIDATE_SCORE_DEEP_PROMPT = """You are a senior technical director doing a deep evaluation of a borderline candidate for: {role_context}
Initial fast-pass score was {fast_score:.1f}. Return a DISTINCT, evidence-based score — never match the initial score exactly.

IMPORTANT — output format: Write the JSON block FIRST (lines 1–15 of your response), then your step-by-step reasoning.

START YOUR RESPONSE WITH THIS JSON (fill in real decimal numbers, no formulas):
{{
  "skills_match": <0-100 decimal>,
  "career_trajectory": <≥ 85 for Staff/Principal ML/AI 7+ yr; ≥ 75 for Senior ML/AI 5+ yr; 40-70 for pivot; < 30 ONLY for wrong field (frontend/QA/PM)>,
  "semantic_relevance": <0-100 decimal>,
  "behavioral_signal": <0-100 decimal>,
  "production_evidence": <0 for pure research, scale by production deployment evidence>,
  "overall_score": <compute (SM×0.30)+(CT×0.28)+(SR×0.25)+(BS×0.17), write ONLY the result decimal>,
  "highlights": ["specific strength 1", "specific strength 2"],
  "why_rank": "2 sentences on fit vs. this specific JD",
  "evidence": ["direct quote or fact from resume", "direct quote or fact"],
  "career_trajectory_detail": "3-5 sentences. Describe where their career started (first role/domain), how they progressed (key transitions, promotions, growing scope), what their current level represents, and whether the overall arc is pointing toward this ML/AI engineering role or diverging from it. Reference specific job titles and domains if visible in the resume.",
  "gaps": ["Specific hard requirement from the JD this candidate does not clearly meet — write a full sentence explaining the gap and why it matters for this role.", "gap 2 as a full explanatory sentence", "gap 3 if applicable"],
  "borderline": <true/false>,
  "uncertainty_note": "one sentence on the key open question for this candidate"
}}

THEN explain your reasoning:
- For each hard requirement: MET / PARTIAL / MISSING with resume evidence
- How you derived each sub-score
- Why overall is not a round number

JD Requirements:
{requirements}

Candidate:
Name: {name} | Title: {title}
Skills: {skills}
Resume: {resume_excerpt}"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────────────────────────────────────

_SUB_KEYS = ("skills_match", "career_trajectory", "semantic_relevance", "behavioral_signal", "production_evidence")


def _normalize_score_scale(parsed: dict) -> dict:
    """
    Detect and correct when the model returned sub-scores in 0-1 range instead of 0-100.
    Checks sub-scores independently from overall_score (model sometimes mixes scales).
    """
    # Coerce all score values to float first
    for k in _SUB_KEYS + ("overall_score",):
        if k in parsed:
            try:
                parsed[k] = float(parsed[k])
            except (TypeError, ValueError):
                parsed[k] = 50.0

    sub_vals = [parsed[k] for k in _SUB_KEYS if k in parsed]
    if sub_vals and max(sub_vals) <= 1.5:
        # Sub-scores are clearly fractional — rescale to 0-100
        logger.warning("[scoring] 0-1 scale sub-scores detected — rescaling ×100")
        for k in _SUB_KEYS:
            if k in parsed:
                parsed[k] = round(parsed[k] * 100, 1)

    # Always recompute overall_score from sub-scores for formula consistency
    sm = parsed.get("skills_match", 50.0)
    ct = parsed.get("career_trajectory", 50.0)
    sr = parsed.get("semantic_relevance", 50.0)
    bs = parsed.get("behavioral_signal", 50.0)
    computed = round(sm * 0.30 + ct * 0.28 + sr * 0.25 + bs * 0.17, 1)
    model_overall = parsed.get("overall_score", computed)
    if abs(model_overall - computed) > 12:
        logger.warning(
            f"[scoring] overall_score mismatch: model={model_overall} formula={computed} — using formula"
        )
    parsed["overall_score"] = computed
    return parsed


def _extract_json_object(text: str, start: int) -> Optional[str]:
    """
    Bracket-match starting from text[start] and return just the JSON object,
    stopping at the matching '}'. Correctly handles nested objects and strings.
    Returns None if the brackets don't balance.
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if in_str:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def _parse_json_response(text: Optional[str]) -> Optional[dict]:
    """
    Extract the scoring JSON object from an LLM response.

    Handles all observed model response patterns:
    - Pure JSON (most common for fast model)
    - JSON wrapped in ``` fences with trailing analysis after closing ```
    - Analysis text BEFORE the JSON block
    - JSON first, then step-by-step analysis after (new deep prompt format)
    - Formula expression as overall_score value (cleaned before parse attempt)
    """
    if not text:
        return None

    # Pre-process: strip markdown fences and formula-in-JSON artefacts.
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Some fast models write: "overall_score": (73.4 × 0.35) + ... — not valid JSON
    cleaned = re.sub(r'"overall_score"\s*:\s*\([^)]*\)[^,\n]*,?\s*\n?', "", cleaned)
    # Fix split array syntax: "key": ["a"], ["b"] → "key": ["a", "b"]
    # Happens when model writes multi-item arrays as separate array literals
    cleaned = re.sub(r'\]\s*,\s*\n?\s*\[', ', ', cleaned)

    def _try(s: str) -> Optional[dict]:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return _normalize_score_scale(obj)
        except Exception:
            pass
        return None

    # 1. Direct parse — works when response is pure JSON (no extra text).
    result = _try(cleaned)
    if result:
        return result

    # 2. Bracket-match scan: for each '{' in the text (tried rightmost-first so
    #    the JSON block is found before any '{' characters inside analysis sections),
    #    extract exactly the balanced object ending at its matching '}', then parse
    #    just that substring.  This is the key fix for JSON-first responses where
    #    analysis text follows the closing '}', making `cleaned[start:]` unparseable.
    positions = [i for i, ch in enumerate(cleaned) if ch == "{"]
    for start in reversed(positions):
        candidate = _extract_json_object(cleaned, start)
        if candidate:
            result = _try(candidate)
            if result and "overall_score" in result:
                return result

    # 3. Regex rescue: response was truncated (e.g. highlights array cut off mid-string)
    # but all numeric scores are present. Extract them individually.
    _NUM_KEYS = ("skills_match", "career_trajectory", "semantic_relevance",
                 "behavioral_signal", "production_evidence", "overall_score")
    rescued: dict = {}
    for k in _NUM_KEYS:
        m = re.search(rf'"{k}"\s*:\s*([\d.]+)', cleaned)
        if m:
            rescued[k] = float(m.group(1))
    if "overall_score" in rescued and len(rescued) >= 5:
        logger.warning(f"[scoring] JSON truncated — rescued {len(rescued)} numeric fields via regex")
        rescued.setdefault("highlights", [])
        rescued.setdefault("why_rank", "")
        rescued.setdefault("evidence", [])
        # Extract why_rank if present (partial string is OK)
        wr = re.search(r'"why_rank"\s*:\s*"([^"]{20,})', cleaned)
        if wr:
            rescued["why_rank"] = wr.group(1)
        rescued["borderline"] = bool(60 <= rescued.get("overall_score", 0) <= 88)
        return _normalize_score_scale(rescued)

    logger.warning(f"[scoring] JSON parse failed: {text[:300]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty routing
# ─────────────────────────────────────────────────────────────────────────────

def _should_escalate(scored: dict, api_key: str) -> bool:
    """
    Return True if this candidate warrants deep reasoning evaluation.
    Criteria: score in borderline zone (60–88) AND either
      - the fast model flagged borderline=True, OR
      - sub-scores are inconsistent (std dev > 15), signalling genuine uncertainty.
    """
    if not api_key or scored.get("compute_path") != "fast-llm":
        return False
    overall = scored.get("overall_score", 0)
    if not (60 <= overall <= 88):
        return False
    sub = [
        scored.get("skills_match", 50),
        scored.get("semantic_relevance", 50),
        scored.get("behavioral_signal", 50),
        scored.get("career_trajectory", 50),
    ]
    sub_std = statistics.stdev(sub)
    return bool(scored.get("borderline", False)) or sub_std > 15


# ─────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def decompose_jd(
    jd_text: str,
    api_key: str,
    fast_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Layer 1: decompose JD into structured requirements via Groq."""
    prompt = JD_DECOMPOSE_PROMPT.format(jd=jd_text[:3000])
    response = call_groq(prompt, fast_model, api_key, base_url=base_url, max_tokens=512)
    parsed = _parse_json_response(response)
    if parsed:
        logger.info(f"[L1/Groq] JD decomposed: {len(parsed.get('hard_requirements', []))} hard reqs")
        return parsed
    logger.warning("[L1] Groq JD decompose failed — heuristic fallback")
    return _fallback_jd_decompose(jd_text)


def _fallback_jd_decompose(jd_text: str) -> dict:
    """Keyword-based JD decomposition — emergency fallback only."""
    lower = jd_text.lower()
    # Broad keyword list covering ML, data, software, and infrastructure roles
    _ALL_KW = [
        "python", "pytorch", "tensorflow", "jax", "keras", "scikit-learn",
        "llm", "large language model", "fine-tuning", "rlhf", "rag",
        "distributed training", "deepspeed", "cuda", "gpu",
        "mlops", "mlflow", "kubeflow", "airflow",
        "kubernetes", "docker", "aws", "gcp", "azure",
        "sql", "postgresql", "mongodb", "spark", "kafka",
        "react", "typescript", "javascript", "node.js", "fastapi",
        "machine learning", "deep learning", "neural network",
        "data science", "data engineering", "data pipeline",
        "rust", "go", "java", "c++",
    ]
    skills = [kw for kw in _ALL_KW if kw in lower]
    seniority = "principal" if any(w in lower for w in ["principal", "staff", "distinguished"]) else \
                "senior" if any(w in lower for w in ["senior", "lead", "head of"]) else \
                "junior" if any(w in lower for w in ["junior", "entry", "intern", "graduate"]) else "mid"
    context = "Technical role"
    if any(w in lower for w in ["machine learning", "ml engineer", "ai engineer"]):
        context = "ML/AI engineering role"
    elif any(w in lower for w in ["data science", "data scientist"]):
        context = "Data science role"
    elif any(w in lower for w in ["software engineer", "backend", "fullstack"]):
        context = "Software engineering role"
    return {
        "hard_requirements": skills[:6],
        "nice_to_have": skills[6:10],
        "negotiable": [],
        "implied_seniority": seniority,
        "key_skills": skills[:12],
        "context": f"{seniority.capitalize()} {context}.",
    }


def score_candidate_fast(
    candidate: dict,
    requirements: dict,
    api_key: str,
    fast_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Score a single candidate using the fast Groq model."""
    role_ctx = requirements.get("context", "Senior ML/AI Engineering role")
    prompt = CANDIDATE_SCORE_PROMPT.format(
        role_context=role_ctx,
        requirements=json.dumps(requirements, indent=2),
        name=candidate.get("name", ""),
        title=candidate.get("title", "Unknown title"),
        location=candidate.get("location", "Unknown"),
        skills=", ".join(candidate.get("skills", [])[:30]),
        graph_fit=float(candidate.get("graph_fit_score", 50)),
        skill_breadth=float(candidate.get("skill_breadth_score", 50)),
        resume_excerpt=(candidate.get("resume_text") or "")[:1500],
    )
    response = call_groq(prompt, fast_model, api_key, base_url=base_url, max_tokens=1200)
    parsed = _parse_json_response(response)
    if parsed:
        parsed["compute_path"] = "fast-llm"
        logger.info(f"[L4/fast] {candidate.get('name')} → overall={parsed.get('overall_score')} "
                    f"ct={parsed.get('career_trajectory')} prod={parsed.get('production_evidence')}")
        return parsed
    logger.warning(f"[L4] Fast LLM failed for {candidate.get('name')} — heuristic fallback")
    return _fallback_score(candidate, requirements)


def score_candidate_deep(
    candidate: dict,
    requirements: dict,
    fast_score: float,
    api_key: str,
    reasoning_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Score a borderline candidate using the reasoning Groq model."""
    role_ctx = requirements.get("context", "Senior ML/AI Engineering role")
    prompt = CANDIDATE_SCORE_DEEP_PROMPT.format(
        role_context=role_ctx,
        fast_score=fast_score,
        requirements=json.dumps(requirements, indent=2),
        name=candidate.get("name", ""),
        title=candidate.get("title", "Unknown title"),
        skills=", ".join(candidate.get("skills", [])[:30]),
        resume_excerpt=(candidate.get("resume_text") or "")[:2000],
    )
    response = call_groq(prompt, reasoning_model, api_key, base_url=base_url, max_tokens=1500)
    parsed = _parse_json_response(response)
    if parsed:
        parsed["compute_path"] = "reasoning-llm"
        logger.info(f"[L4/reasoning] {candidate.get('name')} → overall={parsed.get('overall_score')} (was {fast_score})")
        return parsed
    logger.warning(f"[L4] Reasoning LLM failed for {candidate.get('name')} — heuristic fallback")
    return _fallback_score(candidate, requirements)


def _fallback_score(candidate: dict, requirements: dict) -> dict:
    """Heuristic scoring — EMERGENCY FALLBACK ONLY when Groq is unreachable after all retries."""
    skills = set(s.lower() for s in (candidate.get("skills") or []))
    key_skills = set(s.lower() for s in (requirements.get("key_skills") or []))

    if key_skills:
        skills_match = round(len(skills & key_skills) / len(key_skills) * 100, 1)
    else:
        skills_match = 60.0

    exp_yrs = candidate.get("experience_years") if candidate.get("experience_years") is not None else 0.0
    exp = min(exp_yrs / 8.0, 1.0) * 30
    semantic_relevance = round(min(skills_match * 0.7 + exp + 20, 100), 1)
    behavioral = round(min(50 + len(skills) * 1.5, 90), 1)
    trajectory = round(min(40 + exp_yrs * 4, 95), 1)
    production_evidence = round(min(skills_match * 0.5 + exp * 0.5, 80), 1)
    overall = round(skills_match * 0.30 + trajectory * 0.28 + semantic_relevance * 0.25 + behavioral * 0.17, 1)

    logger.warning(
        f"[L4/HEURISTIC] {candidate.get('name')} scored via emergency fallback "
        f"(Groq unreachable after all retries) — overall={overall}"
    )
    return {
        "skills_match": skills_match,
        "career_trajectory": trajectory,
        "semantic_relevance": semantic_relevance,
        "behavioral_signal": behavioral,
        "production_evidence": production_evidence,
        "overall_score": overall,
        "highlights": list(skills)[:3] or ["profile analyzed"],
        "why_rank": f"[HEURISTIC FALLBACK] {candidate.get('name')} scored via keyword overlap ({len(skills)} skills matched).",
        "evidence": [f"Skills overlap: {', '.join(list(skills & key_skills)[:3]) or 'general fit'}"],
        "borderline": 60 <= overall <= 88,
        "compute_path": "heuristic",
    }


def score_all_candidates(
    candidates: list[dict],
    requirements: dict,
    api_key: str,
    fast_model: str,
    reasoning_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> list[dict]:
    """
    Score all candidates with adaptive compute routing.
    Fast-model calls run concurrently (the sliding-window rate limiter handles pacing
    across threads). Borderline escalations to the reasoning model also run concurrently.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not candidates:
        return []

    def _fast(cand: dict) -> tuple[dict, dict]:
        scored = (
            _fallback_score(cand, requirements)
            if not api_key
            else score_candidate_fast(cand, requirements, api_key, fast_model, base_url)
        )
        return cand, scored

    def _deep(pair: tuple[dict, dict]) -> dict:
        cand, fast_scored = pair
        fast_score = fast_scored.get("overall_score", 0)
        sub_std = statistics.stdev([
            fast_scored.get("skills_match", 50),
            fast_scored.get("semantic_relevance", 50),
            fast_scored.get("behavioral_signal", 50),
            fast_scored.get("career_trajectory", 50),
        ])
        logger.info(
            f"[L4] Escalating {cand.get('name')} (score={fast_score}, sub_std={sub_std:.1f}) "
            f"→ reasoning model"
        )
        deep_scored = score_candidate_deep(cand, requirements, fast_score, api_key, reasoning_model, base_url)
        return {**cand, **deep_scored}

    # Phase 1: concurrent fast scoring — 8 workers is safe given 25 RPM limit
    with ThreadPoolExecutor(max_workers=min(len(candidates), 8)) as ex:
        fast_pairs: list[tuple[dict, dict]] = list(ex.map(_fast, candidates))

    # Phase 2: route borderline candidates to deep reasoning (also concurrent)
    need_deep = [(c, s) for c, s in fast_pairs if _should_escalate(s, api_key)]
    results = [{**c, **s} for c, s in fast_pairs if not _should_escalate(s, api_key)]

    if need_deep:
        with ThreadPoolExecutor(max_workers=min(len(need_deep), 4)) as ex:
            results.extend(ex.map(_deep, need_deep))

    fast = sum(1 for c in results if c.get("compute_path") == "fast-llm")
    deep = sum(1 for c in results if c.get("compute_path") == "reasoning-llm")
    fallback = sum(1 for c in results if c.get("compute_path") == "heuristic")
    logger.info(f"[L4] Scored {len(results)}: fast-llm={fast} reasoning-llm={deep} heuristic={fallback}")
    if fallback:
        names = [c.get("name") for c in results if c.get("compute_path") == "heuristic"]
        logger.warning(f"[L4] HEURISTIC FALLBACK used for: {names} — Groq was unreachable for these candidates")
    return results

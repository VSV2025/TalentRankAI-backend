"""
Layer 4 — Cascaded LLM scoring with adaptive compute routing.
Fast model (llama-3.1-8b-instant) scores all candidates; uncertainty estimates route
borderline cases (score 65-85) to the reasoning model (llama-3.3-70b-versatile).
Uses Groq's OpenAI-compatible API. Results are cached so re-runs don't re-spend quota.
"""
import json
import logging
import re
import time
import hashlib
from typing import Optional

logger = logging.getLogger(__name__)

# ── In-memory prompt cache (prompt_hash → response text) ──────────────────────
_llm_cache: dict[str, str] = {}

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

CANDIDATE_SCORE_PROMPT = """You are a technical recruiter scoring a candidate against a job description.

JD Requirements (structured):
{requirements}

Candidate Profile:
Name: {name}
Title: {title}
Location: {location}
Skills: {skills}
Resume text (excerpt):
{resume_excerpt}

Score this candidate on 4 dimensions (0-100 each):
1. skills_match: How well their skills match hard requirements
2. semantic_relevance: Semantic fit with the JD context and role
3. behavioral_signal: Evidence of impact, leadership, ownership
4. career_trajectory: Career progression and seniority alignment

Also provide:
- overall_score: weighted average (skills_match*0.35 + semantic_relevance*0.30 + behavioral_signal*0.20 + career_trajectory*0.15)
- highlights: list of 3 key differentiators (strings)
- why_rank: 2-3 sentences explaining the fit score
- evidence: list of 2 quoted evidence snippets from the resume with why they're relevant
- borderline: true if overall_score is between 65 and 85

Return ONLY this JSON, no markdown:
{{
  "skills_match": 0-100,
  "semantic_relevance": 0-100,
  "behavioral_signal": 0-100,
  "career_trajectory": 0-100,
  "overall_score": 0-100,
  "highlights": ["...", "...", "..."],
  "why_rank": "...",
  "evidence": ["...", "..."],
  "borderline": true/false
}}"""

CANDIDATE_SCORE_DEEP_PROMPT = """You are a senior technical director with deep expertise in ML engineering evaluating a borderline candidate.
Think carefully and critically — this candidate is borderline and requires your best assessment.

JD Requirements:
{requirements}

Candidate:
Name: {name}
Title: {title}
Skills: {skills}
Full resume excerpt:
{resume_excerpt}

Perform a thorough, critical evaluation. Consider:
- Do their skills directly match the hard requirements? What is missing?
- What does their career trajectory signal about future performance?
- Are there red flags or strong differentiators?
- What is their true ceiling for this specific role?

Return ONLY this JSON:
{{
  "skills_match": 0-100,
  "semantic_relevance": 0-100,
  "behavioral_signal": 0-100,
  "career_trajectory": 0-100,
  "overall_score": 0-100,
  "highlights": ["...", "...", "..."],
  "why_rank": "...",
  "evidence": ["...", "..."],
  "borderline": true/false,
  "uncertainty_note": "one sentence on the key uncertainty"
}}"""


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()


def _call_llm(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str = "https://api.groq.com/openai/v1",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Groq via OpenAI-compatible client with retry-on-429 and result caching."""
    ck = _cache_key(prompt, model)
    if ck in _llm_cache:
        logger.debug(f"Cache hit for model={model}")
        return _llm_cache[ck]

    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        logger.error("openai package not installed — run: pip install openai")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content if resp.choices else None
            if text:
                _llm_cache[ck] = text
            return text

        except RateLimitError:
            wait = 2 ** attempt  # 2s, 4s, 8s
            if attempt == max_retries:
                logger.error(f"Rate limit hit on {model} — all {max_retries} retries exhausted")
                return None
            logger.warning(f"Groq rate limit (429) on {model} — retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)

        except Exception as e:
            logger.error(f"Groq call failed ({model}): {e}")
            return None

    return None


def _parse_json_response(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    logger.warning(f"Failed to parse JSON from LLM: {text[:200]}")
    return None


def decompose_jd(jd_text: str, api_key: str, fast_model: str, base_url: str = "https://api.groq.com/openai/v1") -> dict:
    """Layer 1: Decompose JD into structured requirements via Groq."""
    prompt = JD_DECOMPOSE_PROMPT.format(jd=jd_text[:3000])
    response = _call_llm(prompt, fast_model, api_key, base_url=base_url, max_tokens=512)
    parsed = _parse_json_response(response)
    if parsed:
        logger.info(f"[Groq/L1] JD decomposed: {len(parsed.get('hard_requirements', []))} hard requirements")
        return parsed
    logger.warning("[L1] Groq JD decompose failed — using heuristic fallback")
    return _fallback_jd_decompose(jd_text)


def _fallback_jd_decompose(jd_text: str) -> dict:
    """Keyword-based JD decomposition — emergency fallback only."""
    lower = jd_text.lower()
    skills = []
    for kw in ["python", "pytorch", "llm", "distributed training", "mlops", "cuda", "kubernetes"]:
        if kw in lower:
            skills.append(kw)

    seniority = "senior"
    if "principal" in lower or "staff" in lower:
        seniority = "principal"
    elif "junior" in lower or "entry" in lower:
        seniority = "junior"

    return {
        "hard_requirements": skills[:5],
        "nice_to_have": [],
        "negotiable": [],
        "implied_seniority": seniority,
        "key_skills": skills,
        "context": "ML engineering role requiring Python and LLM experience.",
    }


def score_candidate_fast(
    candidate: dict,
    requirements: dict,
    api_key: str,
    fast_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Score a single candidate using the fast Groq model."""
    prompt = CANDIDATE_SCORE_PROMPT.format(
        requirements=json.dumps(requirements, indent=2),
        name=candidate.get("name", ""),
        title=candidate.get("title", "Unknown title"),
        location=candidate.get("location", "Unknown"),
        skills=", ".join(candidate.get("skills", [])[:30]),
        resume_excerpt=(candidate.get("resume_text") or "")[:1500],
    )
    response = _call_llm(prompt, fast_model, api_key, base_url=base_url, max_tokens=600)
    parsed = _parse_json_response(response)
    if parsed:
        parsed["compute_path"] = "fast-llm"
        logger.info(f"[Groq/fast] {candidate.get('name')} → overall={parsed.get('overall_score')}")
        return parsed
    logger.warning(f"[L4] Fast LLM failed for {candidate.get('name')} — heuristic fallback")
    return _fallback_score(candidate, requirements)


def score_candidate_deep(
    candidate: dict,
    requirements: dict,
    api_key: str,
    reasoning_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Score a borderline candidate using the reasoning Groq model."""
    prompt = CANDIDATE_SCORE_DEEP_PROMPT.format(
        requirements=json.dumps(requirements, indent=2),
        name=candidate.get("name", ""),
        title=candidate.get("title", "Unknown title"),
        skills=", ".join(candidate.get("skills", [])[:30]),
        resume_excerpt=(candidate.get("resume_text") or "")[:2500],
    )
    response = _call_llm(prompt, reasoning_model, api_key, base_url=base_url, max_tokens=800)
    parsed = _parse_json_response(response)
    if parsed:
        parsed["compute_path"] = "reasoning-llm"
        logger.info(f"[Groq/reasoning] {candidate.get('name')} → overall={parsed.get('overall_score')}")
        return parsed
    logger.warning(f"[L4] Reasoning LLM failed for {candidate.get('name')} — heuristic fallback")
    return _fallback_score(candidate, requirements)


def _fallback_score(candidate: dict, requirements: dict) -> dict:
    """Heuristic scoring — emergency fallback when Groq is unreachable."""
    skills = set(s.lower() for s in (candidate.get("skills") or []))
    key_skills = set(s.lower() for s in (requirements.get("key_skills") or []))

    if key_skills:
        skills_match = round(len(skills & key_skills) / len(key_skills) * 100, 1)
    else:
        skills_match = 60.0

    exp = min((candidate.get("experience_years") or 3) / 8.0, 1.0) * 30
    semantic_relevance = round(min(skills_match * 0.7 + exp + 20, 100), 1)
    behavioral = round(min(50 + len(skills) * 1.5, 90), 1)
    trajectory = round(min(40 + (candidate.get("experience_years") or 3) * 4, 95), 1)

    overall = round(
        skills_match * 0.35 + semantic_relevance * 0.30 + behavioral * 0.20 + trajectory * 0.15,
        1,
    )

    logger.warning(f"[heuristic] {candidate.get('name')} scored via fallback: overall={overall}")
    return {
        "skills_match": skills_match,
        "semantic_relevance": semantic_relevance,
        "behavioral_signal": behavioral,
        "career_trajectory": trajectory,
        "overall_score": overall,
        "highlights": list(skills)[:3] or ["profile analyzed"],
        "why_rank": f"Candidate has {len(skills)} relevant skills for this role.",
        "evidence": [f"Skills overlap: {', '.join(list(skills & key_skills)[:3]) or 'general fit'}"],
        "borderline": 65 <= overall <= 85,
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
    Default path: fast-llm for bulk, reasoning-llm for borderlines (65-85).
    Falls back to heuristic only if Groq is unreachable.
    """
    results = []
    for i, cand in enumerate(candidates):
        # Small delay between candidates to respect Groq free-tier rate limits
        if i > 0 and api_key:
            time.sleep(0.3)

        if not api_key:
            scored = _fallback_score(cand, requirements)
        else:
            scored = score_candidate_fast(cand, requirements, api_key, fast_model, base_url)

        # Route borderline cases to deeper reasoning
        overall = scored.get("overall_score", 0)
        if api_key and 65 <= overall <= 85 and scored.get("compute_path") == "fast-llm":
            logger.info(f"Routing {cand.get('name')} (score={overall}) to reasoning model")
            scored = score_candidate_deep(cand, requirements, api_key, reasoning_model, base_url)

        cand_result = dict(cand)
        cand_result.update(scored)
        results.append(cand_result)

    fast = sum(1 for c in results if c.get("compute_path") == "fast-llm")
    deep = sum(1 for c in results if c.get("compute_path") == "reasoning-llm")
    fallback = sum(1 for c in results if c.get("compute_path") == "heuristic")
    logger.info(f"Scored {len(results)} candidates — fast-llm={fast} reasoning-llm={deep} heuristic={fallback}")
    return results

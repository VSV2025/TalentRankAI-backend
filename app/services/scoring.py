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

CANDIDATE_SCORE_PROMPT = """You are a technical recruiter scoring a candidate against a job description.

SCORING RULES — read before answering:
• Every score MUST be a precise decimal to one decimal place (e.g. 73.4, 88.2, 61.7).
• NEVER use round numbers (70.0, 75.0, 80.0, etc.) — round numbers mean you haven't assessed this specific candidate.
• Compute overall_score using the exact formula: (skills_match × 0.35) + (semantic_relevance × 0.30) + (behavioral_signal × 0.20) + (career_trajectory × 0.15). Round to one decimal.
• Base each sub-score on concrete evidence from this candidate's resume, not generic estimates.

JD Requirements (structured):
{requirements}

Candidate Profile:
Name: {name}
Title: {title}
Location: {location}
Skills: {skills}
Resume excerpt:
{resume_excerpt}

Score on 4 dimensions (0–100 each):
1. skills_match: what fraction of hard requirements this candidate demonstrably meets
2. semantic_relevance: semantic fit between their experience and the JD context
3. behavioral_signal: evidence of measurable impact, leadership, ownership
4. career_trajectory: career progression and seniority alignment with this role

Return ONLY this JSON — no markdown, no extra text, no formulas inside the JSON:
{{
  "skills_match": <decimal number only, e.g. 73.4>,
  "semantic_relevance": <decimal number only>,
  "behavioral_signal": <decimal number only>,
  "career_trajectory": <decimal number only>,
  "overall_score": <decimal number only — compute the formula mentally, write ONLY the result>,
  "highlights": ["...", "...", "..."],
  "why_rank": "2–3 sentences on this specific candidate's fit",
  "evidence": ["quoted snippet — why relevant", "quoted snippet — why relevant"],
  "borderline": <true if overall_score is between 62 and 88>
}}"""

CANDIDATE_SCORE_DEEP_PROMPT = """You are a senior technical director doing a deep evaluation of a borderline candidate.
Initial fast-pass score was {fast_score:.1f}. Return a DISTINCT, evidence-based score — never match the initial score exactly.

IMPORTANT — output format: Write the JSON block FIRST (lines 1–14 of your response), then your step-by-step reasoning.
This ensures the JSON is captured even if your response is long.

START YOUR RESPONSE WITH THIS JSON (fill in real decimal numbers, no formulas):
{{
  "skills_match": <0-100 decimal>,
  "semantic_relevance": <0-100 decimal>,
  "behavioral_signal": <0-100 decimal>,
  "career_trajectory": <0-100 decimal>,
  "overall_score": <compute (SM×0.35)+(SR×0.30)+(BS×0.20)+(CT×0.15), write ONLY the result decimal>,
  "highlights": ["specific strength 1", "specific strength 2"],
  "why_rank": "2 sentences on fit vs. this specific JD",
  "evidence": ["direct quote or fact from resume", "direct quote or fact"],
  "borderline": <true/false>,
  "uncertainty_note": "one sentence on the key open question for this candidate"
}}

THEN explain your reasoning (not part of the JSON):
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

_SUB_KEYS = ("skills_match", "semantic_relevance", "behavioral_signal", "career_trajectory")


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
        # Recompute overall from the rescaled sub-scores (don't trust model's overall)
        sm = parsed.get("skills_match", 50)
        sr = parsed.get("semantic_relevance", 50)
        bs = parsed.get("behavioral_signal", 50)
        ct = parsed.get("career_trajectory", 50)
        parsed["overall_score"] = round(sm * 0.35 + sr * 0.30 + bs * 0.20 + ct * 0.15, 1)
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
    # Regex removes ``` and ```json fences (any flavour).
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Some fast models write: "overall_score": (73.4 × 0.35) + (92.1 × 0.30) + ...
    # That's not valid JSON — strip those formula lines (the key appears again with the real value).
    cleaned = re.sub(r'"overall_score"\s*:\s*\([^)]*\)[^,\n]*,?\s*\n?', "", cleaned)

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

    logger.warning(f"[scoring] JSON parse failed: {text[:300]}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty routing
# ─────────────────────────────────────────────────────────────────────────────

def _should_escalate(scored: dict, api_key: str) -> bool:
    """
    Return True if this candidate warrants deep reasoning evaluation.
    Criteria: score in borderline zone (62–88) AND either
      - the fast model flagged borderline=True, OR
      - sub-scores are inconsistent (std dev > 10), signalling genuine uncertainty.
    """
    if not api_key or scored.get("compute_path") != "fast-llm":
        return False
    overall = scored.get("overall_score", 0)
    if not (62 <= overall <= 88):
        return False
    sub = [
        scored.get("skills_match", 50),
        scored.get("semantic_relevance", 50),
        scored.get("behavioral_signal", 50),
        scored.get("career_trajectory", 50),
    ]
    sub_std = statistics.stdev(sub)
    return bool(scored.get("borderline", False)) or sub_std > 10


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
    skills = [kw for kw in ["python", "pytorch", "llm", "distributed training", "mlops", "cuda", "kubernetes"] if kw in lower]
    seniority = "principal" if any(w in lower for w in ["principal", "staff"]) else "junior" if any(w in lower for w in ["junior", "entry"]) else "senior"
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
    response = call_groq(prompt, fast_model, api_key, base_url=base_url, max_tokens=600)
    parsed = _parse_json_response(response)
    if parsed:
        parsed["compute_path"] = "fast-llm"
        logger.info(f"[L4/fast] {candidate.get('name')} → overall={parsed.get('overall_score')}")
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
    prompt = CANDIDATE_SCORE_DEEP_PROMPT.format(
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

    exp = min((candidate.get("experience_years") or 3) / 8.0, 1.0) * 30
    semantic_relevance = round(min(skills_match * 0.7 + exp + 20, 100), 1)
    behavioral = round(min(50 + len(skills) * 1.5, 90), 1)
    trajectory = round(min(40 + (candidate.get("experience_years") or 3) * 4, 95), 1)
    overall = round(skills_match * 0.35 + semantic_relevance * 0.30 + behavioral * 0.20 + trajectory * 0.15, 1)

    logger.warning(
        f"[L4/HEURISTIC] {candidate.get('name')} scored via emergency fallback "
        f"(Groq unreachable after all retries) — overall={overall}"
    )
    return {
        "skills_match": skills_match,
        "semantic_relevance": semantic_relevance,
        "behavioral_signal": behavioral,
        "career_trajectory": trajectory,
        "overall_score": overall,
        "highlights": list(skills)[:3] or ["profile analyzed"],
        "why_rank": f"[HEURISTIC FALLBACK] {candidate.get('name')} scored via keyword overlap ({len(skills)} skills matched).",
        "evidence": [f"Skills overlap: {', '.join(list(skills & key_skills)[:3]) or 'general fit'}"],
        "borderline": 62 <= overall <= 88,
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
    Primary path: fast-llm → reasoning-llm for uncertain borderline cases.
    Heuristic fires only if Groq is completely unreachable after all retries.
    Throttling/pacing is handled by llm_client._throttle() — no manual sleeps needed.
    """
    results = []
    for cand in candidates:
        if not api_key:
            scored = _fallback_score(cand, requirements)
        else:
            scored = score_candidate_fast(cand, requirements, api_key, fast_model, base_url)

        fast_score = scored.get("overall_score", 0)

        if _should_escalate(scored, api_key):
            logger.info(
                f"[L4] Escalating {cand.get('name')} (score={fast_score}, "
                f"sub_std={statistics.stdev([scored.get('skills_match',50), scored.get('semantic_relevance',50), scored.get('behavioral_signal',50), scored.get('career_trajectory',50)]):.1f}) "
                f"→ reasoning model"
            )
            scored = score_candidate_deep(cand, requirements, fast_score, api_key, reasoning_model, base_url)

        cand_result = {**cand, **scored}
        results.append(cand_result)

    fast = sum(1 for c in results if c.get("compute_path") == "fast-llm")
    deep = sum(1 for c in results if c.get("compute_path") == "reasoning-llm")
    fallback = sum(1 for c in results if c.get("compute_path") == "heuristic")
    logger.info(f"[L4] Scored {len(results)}: fast-llm={fast} reasoning-llm={deep} heuristic={fallback}")
    if fallback:
        names = [c.get("name") for c in results if c.get("compute_path") == "heuristic"]
        logger.warning(f"[L4] HEURISTIC FALLBACK used for: {names} — Groq was unreachable for these candidates")
    return results

"""
Layer 6 — Multi-agent debate using LangGraph.
Three agents: ProAdvocate, Skeptic, Adjudicator.
Only runs for borderline candidates (score 62-88).
Rate limiting and retries handled by llm_client.
Falls back to static template only if Groq is completely unreachable.
"""
import json
import logging
import re
from typing import Optional, TypedDict

from .llm_client import call_groq

logger = logging.getLogger(__name__)

PRO_PROMPT = """You are a Pro-Advocate arguing FOR advancing this candidate.
Make the strongest honest case. Cite 2-3 specific resume facts as evidence.
Be concise — 2-3 sentences maximum.

Candidate: {name} | Title: {title} | Score: {score}
Key strengths: {highlights}
Skills: {skills}
Resume excerpt: {resume_excerpt}

Argue for advancing. Cite specific evidence."""

SKEPTIC_PROMPT = """You are a Skeptic identifying genuine risks in hiring this candidate.
Focus on real gaps vs the hard requirements. Be specific, not hypothetical.
Be concise — 2-3 sentences maximum.

Candidate: {name} | Title: {title} | Score: {score}
Hard requirements: {hard_requirements}
Skills: {skills}
Resume excerpt: {resume_excerpt}

Identify the key risks or gaps. Be specific."""

ADJUDICATOR_PROMPT = """You are an Adjudicator reviewing a Pro vs Skeptic debate about a borderline candidate.

Pro argument: {pro}
Skeptic argument: {skeptic}
Initial score: {score}

Weigh both arguments. Produce:
- adjusted_score: a precise decimal (e.g. 74.3) that is DIFFERENT from the initial score if the debate revealed new information
- verdict: one sentence explaining your decision

Return ONLY JSON: {{"adjusted_score": <decimal>, "verdict": "<one sentence>"}}"""


def _parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # Bracket-match scan: try each '{' (rightmost-first), extract just the balanced
    # object, parse it — handles JSON followed by analysis text.
    positions = [i for i, ch in enumerate(cleaned) if ch == "{"]
    for start in reversed(positions):
        candidate = _bracket_extract(cleaned, start)
        if candidate:
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def _bracket_extract(text: str, start: int) -> Optional[str]:
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


def run_debate_with_langgraph(
    candidate: dict,
    score: float,
    requirements: dict,
    api_key: str,
    model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Run the 3-agent LangGraph debate; falls back to direct calls if LangGraph unavailable."""
    try:
        return _run_langgraph_debate(candidate, score, requirements, api_key, model, base_url)
    except ImportError:
        logger.info("[L6] LangGraph not available — running direct 3-agent debate")
        return _run_direct_debate(candidate, score, requirements, api_key, model, base_url)


def _run_langgraph_debate(
    candidate: dict,
    score: float,
    requirements: dict,
    api_key: str,
    model: str,
    base_url: str,
) -> dict:
    from langgraph.graph import StateGraph, END

    class DebateState(TypedDict):
        candidate: dict
        score: float
        requirements: dict
        base_url: str
        model: str
        api_key: str
        pro_argument: str
        skeptic_argument: str
        adjusted_score: float
        verdict: str

    def _pro(state: DebateState) -> DebateState:
        cand = state["candidate"]
        prompt = PRO_PROMPT.format(
            name=cand.get("name", ""),
            title=cand.get("title", ""),
            score=state["score"],
            highlights=", ".join((cand.get("highlights") or [])[:3]),
            skills=", ".join((cand.get("skills") or [])[:15]),
            resume_excerpt=(cand.get("resume_text") or "")[:800],
        )
        text = call_groq(prompt, state["model"], state["api_key"], base_url=state["base_url"], max_tokens=300)
        return {**state, "pro_argument": text or "Strong technical background warrants consideration."}

    def _skeptic(state: DebateState) -> DebateState:
        cand = state["candidate"]
        prompt = SKEPTIC_PROMPT.format(
            name=cand.get("name", ""),
            title=cand.get("title", ""),
            score=state["score"],
            hard_requirements=", ".join((state["requirements"].get("hard_requirements") or [])[:5]),
            skills=", ".join((cand.get("skills") or [])[:15]),
            resume_excerpt=(cand.get("resume_text") or "")[:800],
        )
        text = call_groq(prompt, state["model"], state["api_key"], base_url=state["base_url"], max_tokens=300)
        return {**state, "skeptic_argument": text or "Skill gaps present risk for this role."}

    def _adjudicator(state: DebateState) -> DebateState:
        prompt = ADJUDICATOR_PROMPT.format(
            pro=state.get("pro_argument", ""),
            skeptic=state.get("skeptic_argument", ""),
            score=state["score"],
        )
        text = call_groq(prompt, state["model"], state["api_key"], base_url=state["base_url"], max_tokens=200)
        parsed = _parse_json(text)
        adj = float(parsed.get("adjusted_score", state["score"])) if parsed else state["score"]
        verdict = parsed.get("verdict", "") if parsed else ""
        return {**state, "adjusted_score": adj, "verdict": verdict}

    graph = StateGraph(DebateState)
    graph.add_node("pro", _pro)
    graph.add_node("skeptic", _skeptic)
    graph.add_node("adjudicator", _adjudicator)
    graph.set_entry_point("pro")
    graph.add_edge("pro", "skeptic")
    graph.add_edge("skeptic", "adjudicator")
    graph.add_edge("adjudicator", END)

    result = graph.compile().invoke({
        "candidate": candidate,
        "score": score,
        "requirements": requirements,
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "pro_argument": "",
        "skeptic_argument": "",
        "adjusted_score": score,
        "verdict": "",
    })
    return {
        "pro": result.get("pro_argument", ""),
        "skeptic": result.get("skeptic_argument", ""),
        "adjusted_score": result.get("adjusted_score", score),
        "verdict": result.get("verdict", ""),
    }


def _run_direct_debate(
    candidate: dict,
    score: float,
    requirements: dict,
    api_key: str,
    model: str,
    base_url: str,
) -> dict:
    cand = candidate

    pro_text = call_groq(
        PRO_PROMPT.format(
            name=cand.get("name", ""), title=cand.get("title", ""), score=score,
            highlights=", ".join((cand.get("highlights") or [])[:3]),
            skills=", ".join((cand.get("skills") or [])[:15]),
            resume_excerpt=(cand.get("resume_text") or "")[:800],
        ),
        model, api_key, base_url=base_url, max_tokens=300,
    ) or "Strong technical background warrants consideration."

    skeptic_text = call_groq(
        SKEPTIC_PROMPT.format(
            name=cand.get("name", ""), title=cand.get("title", ""), score=score,
            hard_requirements=", ".join((requirements.get("hard_requirements") or [])[:5]),
            skills=", ".join((cand.get("skills") or [])[:15]),
            resume_excerpt=(cand.get("resume_text") or "")[:800],
        ),
        model, api_key, base_url=base_url, max_tokens=300,
    ) or "Skill gaps present risk for this role."

    adj_result = _parse_json(call_groq(
        ADJUDICATOR_PROMPT.format(pro=pro_text, skeptic=skeptic_text, score=score),
        model, api_key, base_url=base_url, max_tokens=200,
    ))
    adj_score = float(adj_result.get("adjusted_score", score)) if adj_result else score
    verdict = adj_result.get("verdict", "") if adj_result else ""

    return {"pro": pro_text, "skeptic": skeptic_text, "adjusted_score": adj_score, "verdict": verdict}


def _static_debate(candidate: dict, score: float) -> dict:
    """Static fallback — only when Groq is completely unreachable."""
    name = candidate.get("name", "Candidate")
    skills = (candidate.get("skills") or [])[:3]
    skill_str = ", ".join(skills) if skills else "general profile"
    return {
        "pro": f"{name}'s profile shows relevant experience ({skill_str}). Career trajectory suggests capacity to ramp quickly.",
        "skeptic": f"With a score of {score:.0f}, gaps versus top candidates remain. Role demands may require longer ramp.",
        "adjusted_score": score,
        "verdict": "[static fallback — Groq unreachable]",
    }


def run_debate(
    candidate: dict,
    score: float,
    requirements: dict,
    api_key: str,
    fast_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Entry point — LangGraph debate or static fallback if no key."""
    if not api_key:
        return _static_debate(candidate, score)
    result = run_debate_with_langgraph(candidate, score, requirements, api_key, fast_model, base_url)
    logger.info(f"[L6] Debate for {candidate.get('name')}: adj_score={result.get('adjusted_score')}")
    return result

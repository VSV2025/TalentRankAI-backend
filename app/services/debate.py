"""
Layer 6 — Multi-agent debate using LangGraph.
Three agents: ProAdvocate, Skeptic, Adjudicator.
Only runs for borderline candidates (score 65-85).
Uses Groq's OpenAI-compatible API. Falls back to static template if Groq is unavailable.
"""
import json
import logging
import re
import time
from typing import Optional, TypedDict

logger = logging.getLogger(__name__)

PRO_PROMPT = """You are a Pro-Advocate agent arguing FOR hiring this candidate.
Your goal: make the strongest case for why this person should advance.
Cite specific resume evidence. Be compelling but honest.
Keep it to 2-3 sentences.

Candidate: {name}
Title: {title}
Skills: {skills}
Overall score: {score}
Key strengths: {highlights}
Resume excerpt: {resume_excerpt}

Argue for this candidate advancing. Cite specific evidence."""

SKEPTIC_PROMPT = """You are a Skeptic agent arguing AGAINST hiring this candidate.
Your goal: identify genuine risks, gaps, or concerns with this hire.
Be critical but fair — focus on real gaps, not hypotheticals.
Keep it to 2-3 sentences.

Candidate: {name}
Title: {title}
Skills: {skills}
Overall score: {score}
Role requirements (hard): {hard_requirements}
Resume excerpt: {resume_excerpt}

Identify the key risks or gaps. Be specific."""

ADJUDICATOR_PROMPT = """You are an Adjudicator reviewing a debate about a borderline candidate.

Pro argument: {pro}
Skeptic argument: {skeptic}

Candidate score: {score}

Provide a final adjusted score (0-100) and one sentence verdict.
Return only JSON: {{"adjusted_score": 75, "verdict": "one sentence"}}"""


def _call_llm_sync(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str = "https://api.groq.com/openai/v1",
    max_tokens: int = 400,
    max_retries: int = 3,
) -> Optional[str]:
    """Call Groq via OpenAI-compatible client with retry-on-429."""
    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        logger.error("openai package not installed")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content if resp.choices else None

        except RateLimitError:
            wait = 2 ** attempt
            if attempt == max_retries:
                logger.error(f"Debate: rate limit on {model} — retries exhausted")
                return None
            logger.warning(f"Debate: Groq rate limit (429) — retry {attempt}/{max_retries} in {wait}s")
            time.sleep(wait)

        except Exception as e:
            logger.error(f"Debate LLM call failed: {e}")
            return None

    return None


def _parse_json(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    text = re.sub(r"```(?:json)?\n?", "", text).strip().rstrip("`")
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]+\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def run_debate_with_langgraph(
    candidate: dict,
    score: float,
    requirements: dict,
    api_key: str,
    model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Run the 3-agent LangGraph debate. Falls back to direct calls if LangGraph unavailable."""
    try:
        return _run_langgraph_debate(candidate, score, requirements, api_key, model, base_url)
    except ImportError:
        logger.info("LangGraph not available — running direct 3-agent debate")
        return _run_direct_debate(candidate, score, requirements, api_key, model, base_url)


def _run_langgraph_debate(
    candidate: dict,
    score: float,
    requirements: dict,
    api_key: str,
    model: str,
    base_url: str,
) -> dict:
    """Stateful LangGraph debate graph."""
    from langgraph.graph import StateGraph, END

    class DebateState(TypedDict):
        candidate: dict
        score: float
        requirements: dict
        base_url: str
        pro_argument: str
        skeptic_argument: str
        adjusted_score: float
        verdict: str

    def pro_node(state: DebateState) -> DebateState:
        cand = state["candidate"]
        prompt = PRO_PROMPT.format(
            name=cand.get("name", ""),
            title=cand.get("title", ""),
            skills=", ".join((cand.get("skills") or [])[:15]),
            score=state["score"],
            highlights=", ".join((cand.get("highlights") or [])[:3]),
            resume_excerpt=(cand.get("resume_text") or "")[:800],
        )
        result = _call_llm_sync(prompt, model, api_key, base_url=state["base_url"])
        return {**state, "pro_argument": result or "Strong technical background warrants consideration."}

    def skeptic_node(state: DebateState) -> DebateState:
        cand = state["candidate"]
        prompt = SKEPTIC_PROMPT.format(
            name=cand.get("name", ""),
            title=cand.get("title", ""),
            skills=", ".join((cand.get("skills") or [])[:15]),
            score=state["score"],
            hard_requirements=", ".join((state["requirements"].get("hard_requirements") or [])[:5]),
            resume_excerpt=(cand.get("resume_text") or "")[:800],
        )
        result = _call_llm_sync(prompt, model, api_key, base_url=state["base_url"])
        return {**state, "skeptic_argument": result or "Skill gaps present risk for this role."}

    def adjudicator_node(state: DebateState) -> DebateState:
        prompt = ADJUDICATOR_PROMPT.format(
            pro=state.get("pro_argument", ""),
            skeptic=state.get("skeptic_argument", ""),
            score=state["score"],
        )
        result = _call_llm_sync(prompt, model, api_key, base_url=state["base_url"], max_tokens=200)
        parsed = _parse_json(result)
        adj_score = float(parsed.get("adjusted_score", state["score"])) if parsed else state["score"]
        verdict = parsed.get("verdict", "") if parsed else ""
        return {**state, "adjusted_score": adj_score, "verdict": verdict}

    graph = StateGraph(DebateState)
    graph.add_node("pro", pro_node)
    graph.add_node("skeptic", skeptic_node)
    graph.add_node("adjudicator", adjudicator_node)
    graph.set_entry_point("pro")
    graph.add_edge("pro", "skeptic")
    graph.add_edge("skeptic", "adjudicator")
    graph.add_edge("adjudicator", END)

    compiled = graph.compile()
    result = compiled.invoke({
        "candidate": candidate,
        "score": score,
        "requirements": requirements,
        "base_url": base_url,
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
    """Direct 3-call debate without LangGraph."""
    cand = candidate

    pro_prompt = PRO_PROMPT.format(
        name=cand.get("name", ""),
        title=cand.get("title", ""),
        skills=", ".join((cand.get("skills") or [])[:15]),
        score=score,
        highlights=", ".join((cand.get("highlights") or [])[:3]),
        resume_excerpt=(cand.get("resume_text") or "")[:800],
    )
    pro_text = _call_llm_sync(pro_prompt, model, api_key, base_url=base_url) or "Strong technical background."

    skeptic_prompt = SKEPTIC_PROMPT.format(
        name=cand.get("name", ""),
        title=cand.get("title", ""),
        skills=", ".join((cand.get("skills") or [])[:15]),
        score=score,
        hard_requirements=", ".join((requirements.get("hard_requirements") or [])[:5]),
        resume_excerpt=(cand.get("resume_text") or "")[:800],
    )
    skeptic_text = _call_llm_sync(skeptic_prompt, model, api_key, base_url=base_url) or "Skill gaps present risk."

    adj_prompt = ADJUDICATOR_PROMPT.format(pro=pro_text, skeptic=skeptic_text, score=score)
    adj_result = _parse_json(_call_llm_sync(adj_prompt, model, api_key, base_url=base_url, max_tokens=200))
    adj_score = float(adj_result.get("adjusted_score", score)) if adj_result else score
    verdict = adj_result.get("verdict", "") if adj_result else ""

    return {
        "pro": pro_text,
        "skeptic": skeptic_text,
        "adjusted_score": adj_score,
        "verdict": verdict,
    }


def _static_debate(candidate: dict, score: float) -> dict:
    """Fallback when Groq is unreachable."""
    name = candidate.get("name", "Candidate")
    skills = (candidate.get("skills") or [])[:3]
    skill_str = ", ".join(skills) if skills else "general profile"
    return {
        "pro": (
            f"{name}'s profile shows relevant experience and skills ({skill_str}). "
            "The career trajectory suggests capacity to grow into the role requirements quickly."
        ),
        "skeptic": (
            f"With a score of {score:.0f}, there are gaps versus the top candidates. "
            "The role's requirements are demanding and the skill delta may require a longer ramp time."
        ),
        "adjusted_score": score,
        "verdict": "",
    }


def run_debate(
    candidate: dict,
    score: float,
    requirements: dict,
    api_key: str,
    fast_model: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> dict:
    """Entry point — chooses LangGraph or fallback."""
    if not api_key:
        return _static_debate(candidate, score)
    return run_debate_with_langgraph(candidate, score, requirements, api_key, fast_model, base_url)

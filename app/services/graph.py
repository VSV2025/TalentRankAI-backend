"""
Layer 3 — Skills/career graph enrichment using NetworkX.
Builds a bipartite graph of candidates + skills, uses graph proximity
to infer unstated skills and compute trajectory/engagement signals.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

SKILL_ADJACENCY = {
    "python": ["pytorch", "tensorflow", "numpy", "pandas", "scikit-learn"],
    "pytorch": ["python", "cuda", "distributed training", "fine-tuning"],
    "tensorflow": ["python", "cuda", "mlops"],
    "llm": ["fine-tuning", "rlhf", "langchain", "transformers", "bert", "gpt"],
    "fine-tuning": ["llm", "pytorch", "rlhf", "transformers"],
    "rlhf": ["fine-tuning", "llm", "pytorch"],
    "distributed training": ["pytorch", "cuda", "kubernetes"],
    "mlops": ["mlflow", "kubeflow", "airflow", "kubernetes", "docker"],
    "mlflow": ["mlops", "python"],
    "kubeflow": ["mlops", "kubernetes", "python"],
    "kubernetes": ["docker", "mlops", "kubeflow"],
    "docker": ["kubernetes"],
    "vllm": ["inference optimization", "cuda", "python"],
    "inference optimization": ["vllm", "tensorrt", "cuda", "quantization"],
    "tensorrt": ["inference optimization", "cuda"],
    "quantization": ["inference optimization"],
    "cuda": ["pytorch", "tensorflow", "c++", "inference optimization"],
    "c++": ["cuda", "rust"],
    "nlp": ["bert", "transformers", "llm", "python"],
    "transformers": ["bert", "gpt", "llm", "python"],
    "rag": ["vector database", "llm", "langchain"],
    "langchain": ["llm", "rag", "python"],
    "research": ["published", "nlp", "pytorch"],
    "leadership": ["mentoring", "cross-functional"],
    "sql": ["postgresql", "pandas"],
}

IMPACT_WORDS = [
    "led", "launched", "built", "shipped", "designed", "architected",
    "scaled", "reduced", "improved", "increased", "optimized", "delivered",
    "implemented", "founded", "created", "owned", "managed", "mentored",
    "published", "authored", "contributed",
]


def build_skill_graph(candidates: list[dict]):
    """
    Build a NetworkX bipartite graph: candidate nodes ↔ skill nodes.
    Returns the graph object.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed — graph enrichment skipped")
        return None

    G = nx.Graph()

    # Skill adjacency edges (global knowledge graph)
    for skill, neighbors in SKILL_ADJACENCY.items():
        G.add_node(skill, type="skill")
        for n in neighbors:
            G.add_node(n, type="skill")
            G.add_edge(skill, n, weight=0.8)

    # Candidate → skill edges
    for cand in candidates:
        cid = f"candidate_{cand['id']}"
        G.add_node(cid, type="candidate", name=cand.get("name", ""))
        for skill in (cand.get("skills") or []):
            G.add_node(skill, type="skill")
            G.add_edge(cid, skill, weight=1.0)

    return G


def infer_skills(
    candidate_id: int,
    known_skills: list[str],
    graph,
    depth: int = 2,
) -> list[str]:
    """
    Walk the skill graph up to `depth` hops to infer likely unstated skills.
    Returns list of inferred skill names.
    """
    if graph is None:
        return []
    try:
        import networkx as nx
        cid_node = f"candidate_{candidate_id}"
        if cid_node not in graph:
            return []

        skill_nodes = {n for n, d in graph.nodes(data=True) if d.get("type") == "skill"}
        known_set = set(s.lower() for s in known_skills)

        inferred = []
        for skill_node in skill_nodes:
            if skill_node.lower() in known_set:
                continue
            try:
                path_len = nx.shortest_path_length(graph, cid_node, skill_node)
                if path_len <= depth:
                    inferred.append(skill_node)
            except nx.NetworkXNoPath:
                pass
        return inferred[:10]
    except Exception as e:
        logger.debug(f"Skill inference failed for {candidate_id}: {e}")
        return []


def career_trajectory_score(
    experience_years: float,
    skills: list[str],
    resume_text: str,
    title: Optional[str] = None,
) -> float:
    """
    Compute a 0-100 career trajectory score based on:
    - Seniority signals in title
    - Years of experience
    - Upward progression signals in resume text
    """
    score = 50.0

    # Years of experience factor (max boost at 8+ years)
    yoe_boost = min(experience_years / 8.0, 1.0) * 25
    score += yoe_boost

    # Seniority in title
    if title:
        tl = title.lower()
        if any(w in tl for w in ["principal", "staff", "distinguished", "fellow"]):
            score += 15
        elif any(w in tl for w in ["senior", "lead", "head"]):
            score += 10
        elif any(w in tl for w in ["junior", "associate", "intern"]):
            score -= 10

    # Progression signals in resume text
    if resume_text:
        lower = resume_text.lower()
        progression_signals = ["promoted", "promotion", "grew", "expanded scope", "increased responsibility"]
        for sig in progression_signals:
            if sig in lower:
                score += 3

    # Research/publication signal
    if any(s in ["research", "published", "neurips", "icml", "arxiv"] for s in skills):
        score += 5

    return round(min(max(score, 0), 100), 1)


def behavioral_signal_score(resume_text: str, skills: list[str]) -> float:
    """
    Compute a 0-100 behavioral signal from impact words, leadership, and quantified achievements.
    """
    if not resume_text:
        return 50.0

    lower = resume_text.lower()
    score = 45.0

    # Impact word density
    total_words = max(len(lower.split()), 1)
    impact_count = sum(lower.count(w) for w in IMPACT_WORDS)
    density = min(impact_count / total_words * 200, 1.0)
    score += density * 25

    # Quantified achievements
    import re
    quantified = len(re.findall(r"\d+[%xX]|\$\d+|\d+K|\d+M|\d+B|\d+ users|\d+ engineers|\d+ team", resume_text))
    score += min(quantified * 3, 20)

    # Leadership signals
    if any(s in lower for s in ["led", "managed", "mentored", "owned"]):
        score += 5

    # Open source / publication
    if any(s in lower for s in ["github", "open source", "contributed", "published"]):
        score += 5

    return round(min(max(score, 0), 100), 1)

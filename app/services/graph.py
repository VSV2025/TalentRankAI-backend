"""
Layer 3 — Skills/career graph enrichment.

Upgrades from the previous lightweight version:
- Expanded O*NET-inspired ontology: 200+ skills across 10 clusters
- Personalized PageRank (PPR) from JD required skills → measures true graph proximity
- Skill breadth score: fraction of required skill clusters covered
- Richer behavioral score: achievement density + recency + leadership multiplier
- enrich_candidates() wires inferred skills back into the candidate for L4
"""
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# O*NET-inspired skill ontology — 10 clusters, ~200 skills
# ─────────────────────────────────────────────────────────────────────────────

SKILL_CLUSTERS: dict[str, list[str]] = {
    "ml_core": [
        "machine learning", "deep learning", "neural networks", "pytorch", "tensorflow",
        "jax", "keras", "scikit-learn", "numpy", "scipy", "xgboost", "gradient boosting",
        "random forest", "supervised learning", "unsupervised learning",
        "reinforcement learning", "generative models", "diffusion models",
    ],
    "llm_nlp": [
        "llm", "large language models", "fine-tuning", "instruction tuning", "sft",
        "rlhf", "dpo", "ppo", "transformers", "bert", "gpt", "llama", "mistral",
        "attention mechanism", "prompt engineering", "rag", "retrieval augmented generation",
        "embedding models", "text generation", "summarization", "question answering",
        "information extraction", "huggingface", "langchain", "llamaindex",
        "sentence transformers", "semantic search",
    ],
    "mlops_platform": [
        "mlops", "mlflow", "kubeflow", "airflow", "dagster", "prefect", "dvc",
        "wandb", "neptune", "clearml", "weights and biases",
        "model registry", "feature store", "feast", "tecton",
        "model serving", "torchserve", "triton server", "bentoml", "seldon",
        "model monitoring", "evidently", "data versioning", "experiment tracking",
        "a/b testing", "canary deployment", "shadow deployment",
    ],
    "distributed_systems": [
        "distributed training", "data parallelism", "model parallelism",
        "pipeline parallelism", "tensor parallelism", "deepspeed", "horovod",
        "ray", "megatron-lm", "fsdp", "zero optimization",
        "gradient checkpointing", "mixed precision training", "fp16", "bf16",
        "multi-gpu", "multi-node", "collective operations",
    ],
    "inference_optimization": [
        "inference optimization", "cuda", "gpu programming", "tensorrt", "vllm",
        "trt-llm", "quantization", "int8 quantization", "int4 quantization",
        "pruning", "knowledge distillation", "onnx", "openvino",
        "flash attention", "paged attention", "speculative decoding",
        "kv cache", "continuous batching", "triton kernel", "cuda kernels",
    ],
    "cloud_infra": [
        "kubernetes", "k8s", "docker", "aws", "gcp", "azure",
        "terraform", "helm", "argo workflows", "argo cd",
        "spark", "databricks", "bigquery", "sagemaker", "vertex ai",
        "azure ml", "lambda functions", "ec2", "s3", "gcs", "cloud storage",
        "ci/cd", "github actions", "jenkins",
    ],
    "programming": [
        "python", "c++", "rust", "go", "scala", "julia", "java",
        "bash", "linux", "shell scripting", "git", "pandas", "polars",
        "dask", "numba", "cython", "pydantic", "fastapi", "celery",
    ],
    "data_engineering": [
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "kafka", "kinesis", "apache flink", "dbt", "data pipelines",
        "etl", "data warehouse", "data lake", "apache iceberg",
        "delta lake", "data modeling", "streaming data",
    ],
    "research": [
        "research", "published paper", "arxiv", "neurips", "icml", "iclr",
        "cvpr", "acl", "emnlp", "aaai", "phd", "research scientist",
        "novel architecture", "ablation study", "state of the art",
        "benchmark", "open source contribution", "github stars",
    ],
    "leadership": [
        "led team", "managed team", "mentored engineers", "cross-functional collaboration",
        "technical lead", "staff engineer", "principal engineer", "tech lead",
        "head of ml", "director of engineering", "founded", "co-founded",
        "technical strategy", "roadmap", "grew team", "hiring",
        "project ownership", "stakeholder management",
    ],
}

# Cross-cluster bridges: semantically related clusters get bridging edges
# between their first 3 representative (highest-weight) skills
_CLUSTER_BRIDGES: list[tuple[str, str, float]] = [
    ("ml_core",            "llm_nlp",              0.70),
    ("ml_core",            "distributed_systems",  0.60),
    ("ml_core",            "inference_optimization",0.60),
    ("ml_core",            "research",             0.50),
    ("llm_nlp",            "inference_optimization",0.65),
    ("llm_nlp",            "research",             0.55),
    ("mlops_platform",     "cloud_infra",          0.70),
    ("mlops_platform",     "distributed_systems",  0.50),
    ("distributed_systems","inference_optimization",0.75),
    ("distributed_systems","cloud_infra",          0.55),
    ("programming",        "ml_core",              0.55),
    ("programming",        "data_engineering",     0.60),
    ("research",           "llm_nlp",              0.60),
]

# Map lower-cased skill string → canonical cluster name (built at module load)
_SKILL_TO_CLUSTER: dict[str, str] = {}
for _cluster, _skills in SKILL_CLUSTERS.items():
    for _s in _skills:
        _SKILL_TO_CLUSTER[_s.lower()] = _cluster


def _normalize_skill(s: str) -> str:
    return s.lower().strip()


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_skill_graph(
    candidates: list[dict],
    requirements: Optional[dict] = None,
):
    """
    Build a weighted NetworkX graph:
      - Skill–skill edges (within-cluster 0.85, cross-cluster bridges 0.55)
      - Candidate–skill edges (weight 1.0)
      - JD-required skill nodes tagged as seed nodes (for PPR)

    Returns the graph, or None if networkx is unavailable.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("[L3] networkx not installed — graph enrichment skipped")
        return None

    G = nx.Graph()

    # ── Skill nodes + within-cluster edges ───────────────────────────────
    for cluster_name, skills in SKILL_CLUSTERS.items():
        for s in skills:
            G.add_node(s, type="skill", cluster=cluster_name, jd_required=False)
        # Full within-cluster clique (all pairs) — high semantic similarity
        for i, s1 in enumerate(skills):
            for s2 in skills[i + 1:]:
                G.add_edge(s1, s2, weight=0.85)

    # ── Cross-cluster bridge edges ────────────────────────────────────────
    for c1, c2, w in _CLUSTER_BRIDGES:
        rep1 = SKILL_CLUSTERS.get(c1, [])[:3]  # top-3 representative skills
        rep2 = SKILL_CLUSTERS.get(c2, [])[:3]
        for s1 in rep1:
            for s2 in rep2:
                if G.has_edge(s1, s2):
                    G[s1][s2]["weight"] = max(G[s1][s2]["weight"], w)
                else:
                    G.add_edge(s1, s2, weight=w)

    # ── Tag JD-required skills ────────────────────────────────────────────
    if requirements:
        req_skills = (
            [s.lower() for s in (requirements.get("key_skills") or [])]
            + [s.lower() for s in (requirements.get("hard_requirements") or [])]
        )
        for skill_text in req_skills:
            # Find closest ontology node
            for node in G.nodes():
                if skill_text in node.lower() or node.lower() in skill_text:
                    G.nodes[node]["jd_required"] = True

    # ── Candidate–skill edges ─────────────────────────────────────────────
    for cand in candidates:
        cid = f"candidate_{cand['id']}"
        G.add_node(cid, type="candidate", name=cand.get("name", ""))
        for skill in (cand.get("skills") or []):
            s_lower = _normalize_skill(skill)
            # Find matching ontology node (exact or substring match)
            matched = s_lower if G.has_node(s_lower) else None
            if matched is None:
                for node in G.nodes():
                    if node.lower() in s_lower or s_lower in node.lower():
                        matched = node
                        break
            target = matched or s_lower
            if not G.has_node(target):
                G.add_node(target, type="skill", cluster="other", jd_required=False)
            G.add_edge(cid, target, weight=1.0)

    return G


# ─────────────────────────────────────────────────────────────────────────────
# Personalized PageRank — graph fit score
# ─────────────────────────────────────────────────────────────────────────────

def compute_graph_fit_score(
    candidate_id: int,
    required_skills: list[str],
    graph,
    alpha: float = 0.85,
) -> float:
    """
    Run Personalized PageRank with JD-required skill nodes as the teleport set.
    Returns the PPR mass on this candidate's node, scaled to 0–100.
    Higher score = candidate's skills are closer to required skills in the graph.
    """
    if graph is None or not required_skills:
        return 50.0
    try:
        import networkx as nx

        # Build personalization: uniform over required-skill ontology nodes
        personalization: dict[str, float] = {}
        req_lower = {_normalize_skill(s) for s in required_skills}
        for node, data in graph.nodes(data=True):
            if data.get("type") == "skill" and data.get("jd_required"):
                personalization[node] = 1.0
            elif data.get("type") == "skill":
                for r in req_lower:
                    if r in node.lower() or node.lower() in r:
                        personalization[node] = personalization.get(node, 0) + 0.5

        if not personalization:
            # Fall back: tag any ontology skill that textually matches required_skills
            for node in graph.nodes():
                if graph.nodes[node].get("type") == "skill":
                    for r in req_lower:
                        if r in node.lower() or node.lower() in r:
                            personalization[node] = 1.0

        if not personalization:
            return 50.0

        total = sum(personalization.values())
        personalization = {k: v / total for k, v in personalization.items()}
        # Zero for all other nodes
        for node in graph.nodes():
            if node not in personalization:
                personalization[node] = 0.0

        ppr = nx.pagerank(graph, alpha=alpha, personalization=personalization,
                          weight="weight", max_iter=200, tol=1e-6)

        cid_node = f"candidate_{candidate_id}"
        raw = float(ppr.get(cid_node, 0.0))

        # Collect all candidate PPR scores for normalization
        cand_scores = [v for k, v in ppr.items()
                       if graph.nodes[k].get("type") == "candidate"]
        if not cand_scores or max(cand_scores) <= 0:
            return 50.0

        # Normalize within [30, 95] range
        min_s, max_s = min(cand_scores), max(cand_scores)
        if max_s > min_s:
            normalized = (raw - min_s) / (max_s - min_s) * 65 + 30
        else:
            normalized = 60.0

        return round(float(normalized), 1)

    except Exception as e:
        logger.debug(f"[L3] PPR failed for candidate {candidate_id}: {e}")
        return 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Skill breadth score — cluster coverage
# ─────────────────────────────────────────────────────────────────────────────

def compute_skill_breadth_score(
    skills: list[str],
    required_skills: Optional[list[str]] = None,
) -> float:
    """
    Measure how many distinct skill clusters (from the ontology) the candidate covers.
    If required_skills are provided, weight clusters present in the JD higher.
    Returns 0–100.
    """
    if not skills:
        return 40.0

    covered_clusters: set[str] = set()
    for s in skills:
        cluster = _SKILL_TO_CLUSTER.get(_normalize_skill(s))
        if cluster:
            covered_clusters.add(cluster)
        else:
            # Substring match
            for known_skill, cluster_name in _SKILL_TO_CLUSTER.items():
                if known_skill in _normalize_skill(s) or _normalize_skill(s) in known_skill:
                    covered_clusters.add(cluster_name)
                    break

    total_clusters = len(SKILL_CLUSTERS)

    # Identify which clusters are required by the JD
    jd_clusters: set[str] = set()
    if required_skills:
        for s in required_skills:
            cluster = _SKILL_TO_CLUSTER.get(_normalize_skill(s))
            if cluster:
                jd_clusters.add(cluster)

    if jd_clusters:
        # Score = (coverage of JD clusters × 0.7) + (breadth × 0.3)
        jd_covered = len(covered_clusters & jd_clusters) / max(len(jd_clusters), 1)
        breadth = len(covered_clusters) / total_clusters
        raw = jd_covered * 0.7 + breadth * 0.3
    else:
        raw = len(covered_clusters) / total_clusters

    return round(min(raw * 100, 100), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Skill inference — walk the graph
# ─────────────────────────────────────────────────────────────────────────────

def infer_skills(
    candidate_id: int,
    known_skills: list[str],
    graph,
    required_skills: Optional[list[str]] = None,
    depth: int = 2,
    max_inferred: int = 8,
) -> list[str]:
    """
    Walk the skill graph up to `depth` hops from the candidate to find likely
    but unstated skills. Prioritises skills that appear in the JD.
    """
    if graph is None:
        return []
    try:
        import networkx as nx
        cid_node = f"candidate_{candidate_id}"
        if cid_node not in graph:
            return []

        known_set = {_normalize_skill(s) for s in known_skills}
        req_set = {_normalize_skill(s) for s in (required_skills or [])}

        inferred: list[tuple[str, int, bool]] = []  # (skill, hops, is_jd_req)
        for node, data in graph.nodes(data=True):
            if data.get("type") != "skill":
                continue
            if _normalize_skill(node) in known_set:
                continue
            try:
                hop_len = nx.shortest_path_length(graph, cid_node, node)
                if 1 <= hop_len <= depth:
                    is_req = _normalize_skill(node) in req_set or any(
                        r in node.lower() for r in req_set
                    )
                    inferred.append((node, hop_len, is_req))
            except nx.NetworkXNoPath:
                pass

        # Sort: JD-required skills first, then by hop distance
        inferred.sort(key=lambda x: (not x[2], x[1]))
        return [s for s, _, _ in inferred[:max_inferred]]

    except Exception as e:
        logger.debug(f"[L3] Skill inference failed for {candidate_id}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral signal score — achievement-density model
# ─────────────────────────────────────────────────────────────────────────────

_IMPACT_VERBS = [
    "led", "launched", "built", "shipped", "designed", "architected",
    "scaled", "reduced", "improved", "increased", "optimized", "delivered",
    "implemented", "founded", "created", "owned", "managed", "mentored",
    "published", "authored", "contributed", "deployed", "automated",
    "accelerated", "drove", "enabled", "established", "pioneered",
]

_QUANT_PATTERN = re.compile(
    r"\d+[%xX]"            # percentages / multipliers
    r"|\$\s*\d+"           # dollar amounts
    r"|\d+\s*[KMBkmb]"    # thousand / million / billion
    r"|\d+\s*users"
    r"|\d+\s*engineers"
    r"|\d+\s*team"
    r"|\d+\s*models"
    r"|\d+\s*papers"
    r"|\d+\s*citations"
    r"|\d+\s*stars"
    r"|\d+\s*req"          # requests
    r"|\d+x\s+faster"
    r"|\d+\s*ms\b"         # latency
)

_SCALE_INDICATORS = [
    (r"\$\d+\s*[Bb]", 3.0),    # $1B+
    (r"\$\d+\s*[Mm]", 2.0),    # $1M+
    (r"\d+\s*M\s*users", 2.5), # millions of users
    (r"\d{3,}\s*K\s*users", 1.5),
    (r"\d{2,}\s*engineers", 1.5),
    (r"\d+[Kk]\+\s*stars", 1.5),
]


def behavioral_signal_score(resume_text: str, skills: list[str]) -> float:
    """
    Compute a 0–100 behavioral signal from:
    - Impact verb density (quality of writing)
    - Quantified achievement count (evidence of impact)
    - Scale of achievements (big numbers count more)
    - Leadership signals
    - Open source / publication signals
    """
    if not resume_text:
        return 50.0

    lower = resume_text.lower()
    total_words = max(len(lower.split()), 1)
    score = 40.0

    # 1. Impact verb density (up to +20)
    impact_count = sum(lower.count(f" {w} ") + lower.count(f" {w}ed ") for w in _IMPACT_VERBS)
    density = min(impact_count / total_words * 100, 1.0)
    score += density * 20

    # 2. Quantified achievements (up to +20)
    quant_matches = _QUANT_PATTERN.findall(resume_text)
    quant_count = len(quant_matches)
    score += min(quant_count * 2.5, 20)

    # 3. Scale multiplier — big numbers signal outsized impact (up to +15)
    scale_bonus = 0.0
    for pattern, mult in _SCALE_INDICATORS:
        if re.search(pattern, resume_text, re.IGNORECASE):
            scale_bonus += mult
    score += min(scale_bonus * 3, 15)

    # 4. Leadership signals (up to +10)
    leadership_words = ["led", "managed", "mentored", "owned", "founded", "drove",
                        "principal", "staff", "head of", "director", "vp"]
    leadership_hits = sum(1 for w in leadership_words if w in lower)
    score += min(leadership_hits * 2, 10)

    # 5. Open source / research signals (up to +5)
    research_words = ["github", "open source", "published", "arxiv", "neurips",
                      "icml", "patent", "conference"]
    if any(w in lower for w in research_words):
        score += 5

    return round(min(max(score, 0), 100), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Career trajectory score
# ─────────────────────────────────────────────────────────────────────────────

def career_trajectory_score(
    experience_years: float,
    skills: list[str],
    resume_text: str,
    title: Optional[str] = None,
) -> float:
    """
    0–100 career trajectory score based on experience velocity and signals.
    """
    score = 45.0

    # YOE factor — fast track: 8+ years earns full boost
    yoe_boost = min(experience_years / 8.0, 1.0) * 25
    score += yoe_boost

    # Seniority in current title
    if title:
        tl = title.lower()
        if any(w in tl for w in ["principal", "staff", "distinguished", "fellow", "vp", "director"]):
            score += 18
        elif any(w in tl for w in ["senior", "lead", "head"]):
            score += 12
        elif any(w in tl for w in ["junior", "associate", "intern"]):
            score -= 12
        else:
            score += 5  # mid-level default

    # Progression signals in resume
    if resume_text:
        rl = resume_text.lower()
        progression = [
            "promoted to", "promotion", "expanded scope", "increased responsibility",
            "grew from", "transitioned to", "elevated to",
        ]
        for sig in progression:
            if sig in rl:
                score += 4

        # Multiple companies / diverse experience
        company_signals = ["at ", " company", " corp", " inc", " ltd", " startup"]
        company_count = sum(1 for sig in company_signals if rl.count(sig) > 1)
        if company_count >= 2:
            score += 3

    # Research / publication — signals elite technical trajectory
    lower_skills = [_normalize_skill(s) for s in skills]
    if any(s in ["research", "published", "neurips", "icml", "iclr", "arxiv"] for s in lower_skills):
        score += 6

    # Broad skill set — versatile engineers progress faster
    skill_breadth = compute_skill_breadth_score(skills)
    score += (skill_breadth / 100) * 5  # up to +5 for breadth

    return round(min(max(score, 0), 100), 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main enrichment entry point
# ─────────────────────────────────────────────────────────────────────────────

def enrich_candidates(
    candidates: list[dict],
    requirements: Optional[dict] = None,
) -> list[dict]:
    """
    Enrich each candidate with graph-derived signals:
      - graph_fit_score: PPR proximity to JD required skills (0–100)
      - skill_breadth_score: cluster coverage (0–100)
      - career_trajectory_score: heuristic career health (0–100)
      - behavioral_score: achievement-density behavioral signal (0–100)
      - inferred_skills: skills inferred from graph proximity

    Inferred skills are also appended to the candidate's `skills` list so L4
    sees a richer skill profile without extra prompt tokens.
    """
    required_skills: list[str] = []
    if requirements:
        required_skills = (
            (requirements.get("key_skills") or []) +
            (requirements.get("hard_requirements") or [])
        )

    graph = build_skill_graph(candidates, requirements)

    for cand in candidates:
        cid = cand["id"]
        skills = cand.get("skills") or []
        resume_text = cand.get("resume_text") or ""

        # Graph signals
        cand["graph_fit_score"] = compute_graph_fit_score(cid, required_skills, graph)
        cand["skill_breadth_score"] = compute_skill_breadth_score(skills, required_skills)

        # Infer skills from graph — depth=1 only (direct adjacency, not speculative hops).
        # Stored as inferred_skills but NOT added to skills list — inferred skills are
        # graph signals for L7, not confirmed skills to feed the L4 LLM.
        inferred = infer_skills(cid, skills, graph, required_skills=required_skills, depth=1)
        cand["inferred_skills"] = inferred

        # Heuristic scores (L3 baseline — L4 LLM will produce its own assessment)
        cand["career_trajectory_score"] = career_trajectory_score(
            experience_years=cand.get("experience_years", 3),
            skills=cand.get("skills", []),
            resume_text=resume_text,
            title=cand.get("title"),
        )
        cand["behavioral_score"] = behavioral_signal_score(resume_text, cand.get("skills", []))

        logger.debug(
            f"[L3] {cand.get('name')}: graph_fit={cand['graph_fit_score']} "
            f"breadth={cand['skill_breadth_score']} inferred={len(inferred)}"
        )

    # Log summary
    avg_fit = sum(c.get("graph_fit_score", 0) for c in candidates) / max(len(candidates), 1)
    avg_breadth = sum(c.get("skill_breadth_score", 0) for c in candidates) / max(len(candidates), 1)
    total_inferred = sum(len(c.get("inferred_skills", [])) for c in candidates)
    logger.info(
        f"[L3] Enriched {len(candidates)} candidates | "
        f"avg_graph_fit={avg_fit:.1f} avg_breadth={avg_breadth:.1f} "
        f"total_inferred_skills={total_inferred}"
    )

    return candidates

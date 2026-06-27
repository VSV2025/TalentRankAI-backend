"""
Main pipeline orchestrator — runs all 7 layers in sequence,
logs per-layer timings, and returns a ranked shortlist.
"""
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

SMALL_POOL_THRESHOLD = 50  # skip Qdrant vector index for pools smaller than this

from ..config import get_settings
from . import embedding as emb_svc
from . import graph as graph_svc
from . import scoring as scoring_svc
from . import debate as debate_svc
from . import ranking as ranking_svc

logger = logging.getLogger(__name__)


def _timer():
    return time.perf_counter()


def run_pipeline(
    job_description: str,
    candidates: list[dict],
    progress_cb=None,
) -> dict:
    """progress_cb(layer_label: str, pct: int) — called after each layer completes."""
    """
    Execute all 7 pipeline layers on the provided candidates.

    Each candidate dict should have:
        id, name, email, title, location, skills (list),
        resume_text, experience_years

    Returns:
        {
          ranked: [ranked candidate dicts with scores],
          funnel_counts: [layer count progression],
          timings: {layer: seconds},
          requirements: {structured JD},
        }
    """
    settings = get_settings()
    api_key = settings.GROQ_API_KEY
    base_url = settings.GROQ_BASE_URL
    fast_model = settings.FAST_MODEL
    reasoning_model = settings.REASONING_MODEL
    timings = {}

    def _emit(layer: str, pct: int) -> None:
        if progress_cb:
            try:
                progress_cb(layer, pct)
            except Exception:
                pass

    total_pool = len(candidates)  # actual candidate pool size

    # ──────────────────────────────────────────────────────────────────
    # Layer 1: JD Understanding
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    if api_key:
        requirements = scoring_svc.decompose_jd(job_description, api_key, fast_model, base_url)
    else:
        requirements = scoring_svc._fallback_jd_decompose(job_description)
    timings["layer1_jd_understanding"] = round(_timer() - t0, 2)
    logger.info(f"Layer 1 done in {timings['layer1_jd_understanding']}s | "
                f"key_skills={requirements.get('key_skills', [])[:4]}")
    _emit("L1 JD Parse", 14)

    # ──────────────────────────────────────────────────────────────────
    # Layer 2: Fast Retrieval + Rerank
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    query = job_description + " " + " ".join(requirements.get("key_skills", []))

    qdrant_path = settings.QDRANT_PATH
    candidate_docs = [
        {"id": c["id"], "text": (c.get("resume_text") or "") + " " + " ".join(c.get("skills") or [])}
        for c in candidates
    ]

    if len(candidates) >= SMALL_POOL_THRESHOLD:
        # Large pool: build vector index (sentence-transformers → Qdrant ANN)
        emb_svc.build_index(candidate_docs, qdrant_path=qdrant_path)
        retrieved = emb_svc.retrieve_top(query, top_k=min(200, len(candidates)), qdrant_path=qdrant_path)
        retrieved_ids = {cid for cid, _ in retrieved}
        retrieved_candidates = [c for c in candidates if c["id"] in retrieved_ids]
        funnel_retrieved = min(200, len(retrieved_candidates))
    else:
        # Small pool: skip vector index — cross-encoder rerank directly is faster and accurate enough
        retrieved_candidates = candidates
        funnel_retrieved = len(candidates)
        logger.info(f"Layer 2: small pool ({len(candidates)}) — skipping Qdrant, using cross-encoder directly")

    # Cross-encoder rerank → top 30
    reranked = emb_svc.rerank_top(
        query,
        [{"id": c["id"], "text": (c.get("resume_text") or "") + " " + " ".join(c.get("skills") or [])}
         for c in retrieved_candidates],
        top_k=30,
    )
    reranked_ids = {cid for cid, _ in reranked}
    shortlisted_candidates = [c for c in retrieved_candidates if c["id"] in reranked_ids]
    timings["layer2_retrieval"] = round(_timer() - t0, 2)
    logger.info(f"Layer 2 done in {timings['layer2_retrieval']}s | "
                f"{funnel_retrieved}→{len(shortlisted_candidates)} candidates")
    _emit("L2 Retrieval", 28)

    # ──────────────────────────────────────────────────────────────────
    # Layer 3: Graph Enrichment (PPR + skill breadth + inferred skills)
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    shortlisted_candidates = graph_svc.enrich_candidates(
        shortlisted_candidates, requirements
    )
    timings["layer3_enrichment"] = round(_timer() - t0, 2)
    inferred_total = sum(len(c.get("inferred_skills", [])) for c in shortlisted_candidates)
    avg_fit = sum(c.get("graph_fit_score", 0) for c in shortlisted_candidates) / max(len(shortlisted_candidates), 1)
    logger.info(
        f"Layer 3 done in {timings['layer3_enrichment']}s | "
        f"graph_fit_avg={avg_fit:.1f} inferred_skills={inferred_total}"
    )
    _emit("L3 Graph Enrichment", 42)

    # ──────────────────────────────────────────────────────────────────
    # Layer 4: Cascade Scoring (fast model + routing to deep)
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    scored_candidates = scoring_svc.score_all_candidates(
        candidates=shortlisted_candidates,
        requirements=requirements,
        api_key=api_key,
        fast_model=fast_model,
        reasoning_model=reasoning_model,
        base_url=base_url,
    )
    timings["layer4_scoring"] = round(_timer() - t0, 2)
    fast_count = sum(1 for c in scored_candidates if c.get("compute_path") == "fast-llm")
    deep_count = sum(1 for c in scored_candidates if c.get("compute_path") == "reasoning-llm")
    logger.info(f"Layer 4 done in {timings['layer4_scoring']}s | "
                f"fast-llm={fast_count} reasoning-llm={deep_count} heuristic={len(scored_candidates)-fast_count-deep_count}")
    _emit("L4 Fast Scoring", 57)
    _emit("L4b Deep Reasoning", 71)

    # ──────────────────────────────────────────────────────────────────
    # Layer 5: Multimodal (skip — no portfolio images in seed data)
    # ──────────────────────────────────────────────────────────────────
    timings["layer5_multimodal"] = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Layer 6: Multi-Agent Debate (borderline candidates, concurrent)
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    borderline_cands = [
        (c, c.get("overall_score", 50))
        for c in scored_candidates
        if 65 <= c.get("overall_score", 50) <= 85
    ]
    for c in scored_candidates:
        if not (65 <= c.get("overall_score", 50) <= 85):
            c["debate"] = None

    def _run_debate(item: tuple) -> None:
        cand, overall = item
        try:
            result = debate_svc.run_debate(
                candidate=cand,
                score=overall,
                requirements=requirements,
                api_key=api_key,
                fast_model=fast_model,
                base_url=base_url,
            )
            cand["debate"] = {"pro": result["pro"], "skeptic": result["skeptic"]}
            adj_score = result.get("adjusted_score", overall)
            if abs(adj_score - overall) > 2:
                cand["overall_score"] = adj_score
                logger.info(f"Debate adjusted {cand.get('name')} score {overall:.1f}→{adj_score:.1f}")
        except Exception as e:
            logger.warning(f"[L6] Debate failed for {cand.get('name')}: {e}")
            cand["debate"] = None

    if borderline_cands:
        with ThreadPoolExecutor(max_workers=min(len(borderline_cands), 4)) as ex:
            list(ex.map(_run_debate, borderline_cands))

    borderline_count = len(borderline_cands)
    timings["layer6_debate"] = round(_timer() - t0, 2)
    logger.info(f"Layer 6 done in {timings['layer6_debate']}s | {borderline_count} debates")
    _emit("L6 Agent Debate", 85)

    # ──────────────────────────────────────────────────────────────────
    # Layer 7: LTR + Fairness
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    final_ranked = ranking_svc.compute_final_ranking(scored_candidates)
    timings["layer7_ltr_fairness"] = round(_timer() - t0, 2)
    logger.info(f"Layer 7 done in {timings['layer7_ltr_fairness']}s | final={len(final_ranked)}")
    _emit("L7 Rank & Fairness", 100)

    # ──────────────────────────────────────────────────────────────────
    # Build funnel counts — all 7 layers
    # ──────────────────────────────────────────────────────────────────
    deep_count = sum(1 for c in scored_candidates if c.get("compute_path") == "reasoning-llm")
    funnel_counts = [
        {"label": "L1 JD Parse",          "count": total_pool,                    "description": "JD decomposed into structured requirements"},
        {"label": "L2 Retrieval",          "count": funnel_retrieved,              "description": "Embedding + keyword search, top 200"},
        {"label": "L3 Graph Enrichment",   "count": len(shortlisted_candidates),   "description": "PPR graph fit + skill breadth scoring"},
        {"label": "L4 Fast LLM",           "count": len(shortlisted_candidates),   "description": "8B model semantic scoring for all candidates"},
        {"label": "L4b Reasoning LLM",     "count": deep_count,                    "description": "70B deep eval for borderline candidates"},
        {"label": "L6 Agent Debate",       "count": borderline_count,              "description": "Pro vs Skeptic debate with adjudicator"},
        {"label": "L7 Ranked",             "count": len(final_ranked),             "description": "Composite sort + FA*IR fairness rerank"},
    ]

    total_time = sum(timings.values())
    logger.info(f"Pipeline complete in {total_time:.2f}s total")

    return {
        "ranked": final_ranked,
        "funnel_counts": funnel_counts,
        "timings": timings,
        "requirements": requirements,
    }

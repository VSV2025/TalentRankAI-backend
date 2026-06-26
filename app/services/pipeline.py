"""
Main pipeline orchestrator — runs all 7 layers in sequence,
logs per-layer timings, and returns a ranked shortlist.
"""
import time
import logging
from typing import Optional

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
) -> dict:
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

    total_pool = 10000  # simulated full applicant pool

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

    # ──────────────────────────────────────────────────────────────────
    # Layer 2: Fast Retrieval + Rerank
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    query = job_description + " " + " ".join(requirements.get("key_skills", []))

    qdrant_path = settings.QDRANT_PATH

    # Build vector index (sentence-transformers → Qdrant ANN, falls back to TF-IDF)
    emb_svc.build_index(
        [{"id": c["id"], "text": (c.get("resume_text") or "") + " " + " ".join(c.get("skills") or [])}
         for c in candidates],
        qdrant_path=qdrant_path,
    )

    retrieved = emb_svc.retrieve_top(query, top_k=min(200, len(candidates)), qdrant_path=qdrant_path)
    retrieved_ids = {cid for cid, _ in retrieved}
    retrieved_candidates = [c for c in candidates if c["id"] in retrieved_ids]

    # Simulate 10K → 200 narrowing
    funnel_retrieved = min(200, len(retrieved_candidates))

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
    fast_count = sum(1 for c in scored_candidates if c.get("compute_path") == "fast")
    deep_count = sum(1 for c in scored_candidates if c.get("compute_path") == "deep")
    logger.info(f"Layer 4 done in {timings['layer4_scoring']}s | "
                f"fast={fast_count} deep={deep_count} fallback={len(scored_candidates)-fast_count-deep_count}")

    # ──────────────────────────────────────────────────────────────────
    # Layer 5: Multimodal (skip — no portfolio images in seed data)
    # ──────────────────────────────────────────────────────────────────
    timings["layer5_multimodal"] = 0.0

    # ──────────────────────────────────────────────────────────────────
    # Layer 6: Multi-Agent Debate (borderline candidates only)
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    borderline_count = 0
    for cand in scored_candidates:
        overall = cand.get("overall_score", 50)
        if 65 <= overall <= 85:
            borderline_count += 1
            debate_result = debate_svc.run_debate(
                candidate=cand,
                score=overall,
                requirements=requirements,
                api_key=api_key,
                fast_model=fast_model,
                base_url=base_url,
            )
            cand["debate"] = {"pro": debate_result["pro"], "skeptic": debate_result["skeptic"]}
            adj_score = debate_result.get("adjusted_score", overall)
            if abs(adj_score - overall) > 2:
                cand["overall_score"] = adj_score
                logger.info(f"Debate adjusted {cand.get('name')} score {overall:.1f}→{adj_score:.1f}")
        else:
            cand["debate"] = None

    timings["layer6_debate"] = round(_timer() - t0, 2)
    logger.info(f"Layer 6 done in {timings['layer6_debate']}s | {borderline_count} debates")

    # ──────────────────────────────────────────────────────────────────
    # Layer 7: LTR + Fairness
    # ──────────────────────────────────────────────────────────────────
    t0 = _timer()
    final_ranked = ranking_svc.compute_final_ranking(scored_candidates)
    timings["layer7_ltr_fairness"] = round(_timer() - t0, 2)
    logger.info(f"Layer 7 done in {timings['layer7_ltr_fairness']}s | final={len(final_ranked)}")

    # ──────────────────────────────────────────────────────────────────
    # Build funnel counts
    # ──────────────────────────────────────────────────────────────────
    funnel_counts = [
        {
            "label": "Fast Retrieval",
            "count": total_pool,
            "description": "Keyword + embedding pre-filter",
        },
        {
            "label": "Enrichment",
            "count": funnel_retrieved,
            "description": "Profile enrichment + deduplication",
        },
        {
            "label": "Deep Reasoning",
            "count": len(shortlisted_candidates),
            "description": "LLM semantic scoring + sub-scores",
        },
        {
            "label": "Ranked & Fairness-Checked",
            "count": len(final_ranked),
            "description": "Final shortlist with bias audit",
        },
    ]

    total_time = sum(timings.values())
    logger.info(f"Pipeline complete in {total_time:.2f}s total")

    return {
        "ranked": final_ranked,
        "funnel_counts": funnel_counts,
        "timings": timings,
        "requirements": requirements,
    }

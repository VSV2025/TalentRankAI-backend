"""
Layer 7 — LambdaMART (LightGBM) Learning-to-Rank + FA*IR-style fairness reranking.
Self-supervised: cascade scores are used as pseudo-labels for LTR training.
Falls back to weighted composite sort if LightGBM is unavailable.
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

PROTECTED_LOCATIONS = [
    # Geographic diversity signals — we track representation, not discriminate
    "remote", "uk", "de", "jp", "uae", "mx", "eu", "in", "sg", "au",
]


def _build_feature_matrix(candidates: list[dict]) -> np.ndarray:
    """
    Build feature matrix for LTR.
    L4 LLM sub-scores (4): skills_match, semantic_relevance, behavioral_signal, career_trajectory
    Raw profile features (2): experience_years, skills_count
    L3 graph signals (2): graph_fit_score (PPR), skill_breadth_score (cluster coverage)
    """
    rows = []
    for c in candidates:
        row = [
            # L4 LLM sub-scores
            float(c.get("skills_match", 0)) / 100.0,
            float(c.get("semantic_relevance", 0)) / 100.0,
            float(c.get("behavioral_signal", 0)) / 100.0,
            float(c.get("career_trajectory", 0)) / 100.0,
            # Raw profile
            min(float(c.get("experience_years", 0)) / 10.0, 1.0),
            min(float(len(c.get("skills") or [])) / 20.0, 1.0),
            # L3 graph signals
            float(c.get("graph_fit_score", 50)) / 100.0,
            float(c.get("skill_breadth_score", 50)) / 100.0,
        ]
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def _build_pseudo_labels(candidates: list[dict]) -> np.ndarray:
    """Use overall_score as pseudo-relevance labels (0-4 scale like NDCG)."""
    scores = np.array([float(c.get("overall_score", 50)) for c in candidates])
    # Map 0-100 → 0-4
    labels = np.clip((scores / 100.0) * 4, 0, 4).astype(int)
    return labels


def rank_with_lightgbm(candidates: list[dict]) -> list[dict]:
    """Train a LambdaMART ranker on the candidates and return them sorted.
    LambdaMART requires enough samples to learn meaningful feature weights —
    skip it for small sets where the composite sort is more reliable."""
    if len(candidates) < 20:
        logger.info(f"LambdaMART skipped ({len(candidates)} candidates < 20) — composite sort")
        return _fallback_rank(candidates)

    try:
        import lightgbm as lgb

        X = _build_feature_matrix(candidates)
        y = _build_pseudo_labels(candidates)

        group = np.array([len(candidates)])

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "label_gain": [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023],
            "learning_rate": 0.05,
            "num_leaves": 16,
            "min_data_in_leaf": 3,
            "verbose": -1,
        }
        dataset = lgb.Dataset(X, label=y, group=group)
        model = lgb.train(params, dataset, num_boost_round=100, valid_sets=[dataset], callbacks=[lgb.log_evaluation(-1)])

        scores = model.predict(X)
        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info(f"LambdaMART ranking complete ({len(candidates)} candidates)")
        return [c for c, _ in ranked]

    except ImportError:
        logger.warning("LightGBM not installed — using composite score sort")
        return _fallback_rank(candidates)
    except Exception as e:
        logger.warning(f"LambdaMART failed ({e}) — using composite score sort")
        return _fallback_rank(candidates)


def _fallback_rank(candidates: list[dict]) -> list[dict]:
    """Sort by blended composite: LLM sub-scores + L3 graph signals."""
    def composite(c: dict) -> float:
        llm = (
            float(c.get("skills_match", 0)) * 0.35
            + float(c.get("semantic_relevance", 0)) * 0.30
            + float(c.get("behavioral_signal", 0)) * 0.20
            + float(c.get("career_trajectory", 0)) * 0.15
        )
        graph_fit = float(c.get("graph_fit_score", 50))
        breadth = float(c.get("skill_breadth_score", 50))
        return llm * 0.85 + graph_fit * 0.10 + breadth * 0.05
    return sorted(candidates, key=composite, reverse=True)


def fairness_rerank(candidates: list[dict], p_min: float = 0.3) -> list[dict]:
    """
    FA*IR-inspired exposure fairness pass.
    Ensures geographic diversity is represented in the top-10 shortlist.
    Does not discriminate — only promotes under-represented candidates
    when their score is within a tolerable margin (10 points) of the next candidate.

    p_min: minimum proportion of diverse candidates in top-10.
    """
    if len(candidates) <= 2:
        return candidates

    # Tag candidates with diversity signal
    def is_diverse(c: dict) -> bool:
        loc = (c.get("location") or "").lower()
        return any(tag in loc for tag in PROTECTED_LOCATIONS)

    # Count current diversity in top-10
    top10 = candidates[:min(10, len(candidates))]
    n_diverse = sum(1 for c in top10 if is_diverse(c))
    min_diverse = max(1, int(len(top10) * p_min))

    if n_diverse >= min_diverse:
        logger.info(f"Fairness check: {n_diverse}/{len(top10)} diverse — no adjustment needed")
        return candidates

    # Find a qualifying diverse candidate to swap up
    result = list(candidates)
    for i in range(len(top10), len(result)):
        cand = result[i]
        if not is_diverse(cand):
            continue
        # Find the lowest-scoring non-diverse candidate in top-10 within margin
        for j in range(len(top10) - 1, -1, -1):
            incumbent = result[j]
            if not is_diverse(incumbent):
                score_gap = float(result[j].get("overall_score", 0)) - float(cand.get("overall_score", 0))
                if score_gap <= 10:
                    # Swap
                    result[j], result[i] = result[i], result[j]
                    logger.info(
                        f"Fairness rerank: promoted {cand.get('name')} (pos {i}→{j}), "
                        f"score gap={score_gap:.1f}"
                    )
                    n_diverse += 1
                    if n_diverse >= min_diverse:
                        return result
                    break
    return result


def compute_final_ranking(candidates: list[dict]) -> list[dict]:
    """
    Full Layer 7: LTR → fairness rerank → assign final ranks.
    Returns candidates with rank field set.
    """
    # LTR pass
    ranked = rank_with_lightgbm(candidates)
    # Fairness pass
    ranked = fairness_rerank(ranked)
    # Assign ranks
    for i, cand in enumerate(ranked):
        cand["rank"] = i + 1
    return ranked

"""
Layer 2 — ANN retrieval via sentence-transformers + Qdrant,
with cross-encoder/ms-marco reranker. Falls back to TF-IDF if
sentence-transformers or qdrant-client are unavailable.
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "candidates"


class VectorRetriever:
    """
    Real vector retrieval: sentence-transformers encodes text,
    Qdrant stores and queries embeddings, cross-encoder reranks.
    Falls back to TF-IDF if heavy deps unavailable.
    """

    def __init__(self):
        self._embed_model = None
        self._cross_encoder = None
        self._qdrant = None
        self._tfidf_fallback = None
        self._tfidf_matrix = None
        self._candidate_ids: list[int] = []
        self._use_vector = False

    # ── lazy loaders ──────────────────────────────────────────────────

    def _get_embed_model(self):
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {EMBED_MODEL}")
            self._embed_model = SentenceTransformer(EMBED_MODEL)
        return self._embed_model

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from sentence_transformers.cross_encoder import CrossEncoder
                logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
                self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            except Exception as e:
                logger.warning(f"Cross-encoder load failed ({e}) — reranking with cosine only")
        return self._cross_encoder

    def _get_qdrant(self, qdrant_path: str = "./qdrant_data"):
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            self._qdrant = QdrantClient(path=qdrant_path)
            # Ensure collection exists
            collections = [c.name for c in self._qdrant.get_collections().collections]
            if COLLECTION_NAME not in collections:
                self._qdrant.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                )
                logger.info(f"Created Qdrant collection '{COLLECTION_NAME}' (384-dim cosine)")
        return self._qdrant

    # ── indexing ──────────────────────────────────────────────────────

    def index(self, candidates: list[dict], qdrant_path: str = "./qdrant_data") -> None:
        if not candidates:
            return
        self._candidate_ids = [c["id"] for c in candidates]

        try:
            model = self._get_embed_model()
            qdrant = self._get_qdrant(qdrant_path)

            texts = [c.get("text", "") or "" for c in candidates]
            embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

            from qdrant_client.models import PointStruct
            points = [
                PointStruct(id=c["id"], vector=emb.tolist(), payload={"cid": c["id"]})
                for c, emb in zip(candidates, embeddings)
            ]
            # Upsert so re-indexing is idempotent
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            self._use_vector = True
            logger.info(f"Indexed {len(candidates)} candidates into Qdrant ({EMBED_MODEL})")

        except Exception as e:
            logger.warning(f"Vector indexing failed ({e}) — falling back to TF-IDF")
            self._build_tfidf(candidates)

    def _build_tfidf(self, candidates: list[dict]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(ngram_range=(1, 2), max_features=8000, sublinear_tf=True, min_df=1)
        texts = [c.get("text", "") or "" for c in candidates]
        self._tfidf_matrix = vect.fit_transform(texts)
        self._tfidf_fallback = vect
        self._candidate_ids = [c["id"] for c in candidates]
        logger.info(f"TF-IDF fallback index built ({len(candidates)} docs)")

    # ── retrieval ─────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 200,
                 qdrant_path: str = "./qdrant_data") -> list[tuple[int, float]]:
        if self._use_vector:
            return self._retrieve_vector(query, top_k, qdrant_path)
        return self._retrieve_tfidf(query, top_k)

    def _retrieve_vector(self, query: str, top_k: int,
                         qdrant_path: str) -> list[tuple[int, float]]:
        try:
            model = self._get_embed_model()
            qdrant = self._get_qdrant(qdrant_path)
            q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()
            results = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=q_emb,
                limit=min(top_k, 1000),
                with_payload=False,
            )
            return [(r.id, float(r.score)) for r in results]
        except Exception as e:
            logger.error(f"Qdrant retrieval failed ({e}) — TF-IDF fallback")
            return self._retrieve_tfidf(query, top_k)

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if self._tfidf_matrix is None or self._tfidf_fallback is None:
            return [(cid, 0.5) for cid in self._candidate_ids[:top_k]]
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._tfidf_fallback.transform([query])
        sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]
        pairs = sorted(zip(self._candidate_ids, sims.tolist()), key=lambda x: x[1], reverse=True)
        return pairs[:top_k]

    # ── reranking ─────────────────────────────────────────────────────

    def rerank(self, query: str, candidate_docs: list[dict],
               top_k: int = 30) -> list[tuple[int, float]]:
        if not candidate_docs:
            return []

        ce = self._get_cross_encoder()
        if ce is not None:
            return self._rerank_cross_encoder(query, candidate_docs, top_k, ce)
        return self._rerank_tfidf(query, candidate_docs, top_k)

    def _rerank_cross_encoder(self, query: str, candidate_docs: list[dict],
                               top_k: int, ce) -> list[tuple[int, float]]:
        try:
            pairs = [(query, c.get("text", "")[:512]) for c in candidate_docs]
            scores = ce.predict(pairs, show_progress_bar=False)
            ranked = sorted(
                zip([c["id"] for c in candidate_docs], scores.tolist()),
                key=lambda x: x[1], reverse=True,
            )
            logger.info("Cross-encoder rerank complete")
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"Cross-encoder rerank failed ({e})")
            return self._rerank_tfidf(query, candidate_docs, top_k)

    def _rerank_tfidf(self, query: str, candidate_docs: list[dict],
                      top_k: int) -> list[tuple[int, float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        texts = [c.get("text", "") or "" for c in candidate_docs]
        vect = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1)
        mat = vect.fit_transform(texts)
        q_vec = vect.transform([query])
        sims = cosine_similarity(q_vec, mat)[0]
        pairs = sorted(
            zip([c["id"] for c in candidate_docs], sims.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return pairs[:top_k]


# Module-level singleton — shared across one process lifetime
_retriever = VectorRetriever()


def build_index(candidates: list[dict], qdrant_path: str = "./qdrant_data") -> None:
    _retriever.index(candidates, qdrant_path=qdrant_path)


def retrieve_top(query: str, top_k: int = 200,
                 qdrant_path: str = "./qdrant_data") -> list[tuple[int, float]]:
    return _retriever.retrieve(query, top_k=top_k, qdrant_path=qdrant_path)


def rerank_top(query: str, candidate_docs: list[dict],
               top_k: int = 30) -> list[tuple[int, float]]:
    return _retriever.rerank(query, candidate_docs, top_k=top_k)

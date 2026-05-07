"""Dense, lexical, and hybrid retriever implementations for the MS MARCO RAG project."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from data_loader import Passage

try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:  # pragma: no cover
    CrossEncoder = None
    SentenceTransformer = None


class TextPreprocessor:
    """Reusable text preprocessing helpers for retrieval, query expansion, and UI facets."""

    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
    QUESTION_PREFIX_PATTERN = re.compile(
        r"^(what\s+is|what\s+are|what\s+was|what\s+were|who\s+is|who\s+was|"
        r"define|explain|describe|how\s+does|how\s+do|why\s+does|why\s+do)\s+",
        flags=re.IGNORECASE,
    )
    STOP_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """
        Normalizes repeated whitespace in a text string.

        @param text: Input text.
        @return: Text with repeated whitespace collapsed.
        """

        return re.sub(r"\s+", " ", text or "").strip()

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """
        Tokenizes text into lowercase alphanumeric terms.

        @param text: Input text.
        @return: List of lowercase tokens.
        """

        return [token.lower() for token in cls.TOKEN_PATTERN.findall(text or "")]

    @classmethod
    def content_tokens(cls, text: str) -> List[str]:
        """
        Tokenizes text and removes common question and function words.

        @param text: Input text.
        @return: Content-heavy tokens useful for simplified query variants.
        """

        return [token for token in cls.tokenize(text) if token not in cls.STOP_WORDS]

    @classmethod
    def build_query_variants(cls, query: str, max_variants: int = 4) -> List[str]:
        """
        Builds lightweight query variants for multi-query retrieval.

        This is intentionally simple and deterministic. It improves weaker manual
        queries by searching both the original wording and a shorter keyword form.

        @param query: Original user query.
        @param max_variants: Maximum number of unique variants to return.
        @return: Ordered list of query variants.
        """

        normalized = cls.normalize_whitespace(query)
        if not normalized:
            return []

        variants: List[str] = [normalized]
        without_prefix = cls.QUESTION_PREFIX_PATTERN.sub("", normalized).strip(" ?.!\t")
        if without_prefix and without_prefix.lower() != normalized.lower():
            variants.append(without_prefix)

        lowered = normalized.lower()
        if lowered.startswith("what causes "):
            topic = normalized[12:].strip(" ?.!\t")
            if topic:
                variants.extend([f"causes {topic}", f"symptoms {topic}"])
        elif " role of " in lowered and " in " in lowered:
            tokens = cls.content_tokens(normalized)
            if tokens:
                variants.append(" ".join(tokens))
        else:
            tokens = cls.content_tokens(normalized)
            if len(tokens) >= 2:
                variants.append(" ".join(tokens))

        unique_variants: List[str] = []
        seen = set()
        for variant in variants:
            cleaned = cls.normalize_whitespace(variant.strip(" ?.!\t"))
            key = cleaned.lower()
            if cleaned and key not in seen:
                unique_variants.append(cleaned)
                seen.add(key)
            if len(unique_variants) >= max_variants:
                break

        return unique_variants


@dataclass
class RetrievalResult:
    """
    Represents a single ranked retrieval result.

    @param pid: Passage id.
    @param text: Passage text.
    @param score: Retriever score or fused score.
    @param rank: One-based rank.
    @param source: Retriever source label, such as dense, bm25, or hybrid.
    @param score_details: Optional component scores used by hybrid retrieval.
    """

    pid: str
    text: str
    score: float
    rank: int
    source: str = "unknown"
    score_details: Dict[str, float] = field(default_factory=dict)


class SentenceBERTRetriever:
    """Dense retriever using Sentence-BERT embeddings and FAISS inner-product search."""

    def __init__(
        self,
        embedding_model_name: str,
        faiss_index_path: str,
        metadata_parquet_path: str,
        dense_embeddings_path: str,
    ) -> None:
        """
        Initializes the dense retriever.

        @param embedding_model_name: SentenceTransformer model name.
        @param faiss_index_path: Path to the saved FAISS index.
        @param metadata_parquet_path: Path to passage metadata saved beside the index.
        @param dense_embeddings_path: Path to the saved dense embedding matrix.
        @return: None.
        @raises ImportError: If sentence-transformers or FAISS is unavailable.
        """

        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required.")
        if faiss is None:
            raise ImportError("faiss is required.")

        self.model = SentenceTransformer(embedding_model_name)
        self.faiss_index_path = faiss_index_path
        self.metadata_parquet_path = metadata_parquet_path
        self.dense_embeddings_path = dense_embeddings_path
        self.passages: List[Passage] = []
        self.index = None

    def build_index(self, passages: Sequence[Passage], batch_size: int = 128) -> None:
        """
        Builds and saves the dense FAISS index.

        @param passages: Passage collection.
        @param batch_size: Encoding batch size.
        @return: None.
        """

        self.passages = list(passages)
        texts = [passage.text for passage in self.passages]
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        Path(self.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.metadata_parquet_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.dense_embeddings_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, self.faiss_index_path)
        pd.DataFrame(
            [{"pid": passage.pid, "text": passage.text} for passage in self.passages]
        ).to_parquet(self.metadata_parquet_path, index=False)
        np.save(self.dense_embeddings_path, embeddings)

    def load_index(self) -> None:
        """
        Loads a previously saved dense index and passage metadata.

        @return: None.
        @raises FileNotFoundError: If the index or metadata file is missing.
        """

        if not Path(self.faiss_index_path).exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")
        if not Path(self.metadata_parquet_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_parquet_path}")

        self.index = faiss.read_index(self.faiss_index_path)
        metadata = pd.read_parquet(self.metadata_parquet_path)
        self.passages = [
            Passage(pid=str(row.pid), text=str(row.text))
            for row in metadata.itertuples(index=False)
        ]

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Retrieves top-k dense results for a query.

        @param query: Query text.
        @param top_k: Number of passages to return.
        @return: Ranked dense retrieval results.
        @raises ValueError: If the dense index has not been loaded.
        """

        if self.index is None:
            raise ValueError("Dense index not loaded. Build or load the index first.")
        if top_k <= 0:
            return []

        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results: List[RetrievalResult] = []
        for rank, (score, index_value) in enumerate(zip(scores[0], indices[0]), start=1):
            if index_value < 0:
                continue
            passage = self.passages[int(index_value)]
            results.append(
                RetrievalResult(
                    pid=passage.pid,
                    text=passage.text,
                    score=float(score),
                    rank=rank,
                    source="dense",
                    score_details={"dense": float(score)},
                )
            )
        return results


class BM25Retriever:
    """Lexical BM25 baseline retriever."""

    def __init__(self, passages: Sequence[Passage]) -> None:
        """
        Initializes the BM25 retriever.

        @param passages: Passage collection.
        @return: None.
        @raises ImportError: If rank-bm25 is unavailable.
        """

        if BM25Okapi is None:
            raise ImportError("rank-bm25 is required.")

        self.passages = list(passages)
        self.tokenized_corpus = [TextPreprocessor.tokenize(passage.text) for passage in self.passages]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Retrieves top-k BM25 results for a query.

        @param query: Query text.
        @param top_k: Number of passages to return.
        @return: Ranked BM25 results.
        """

        if top_k <= 0:
            return []

        query_tokens = TextPreprocessor.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievalResult] = []
        for rank, index_value in enumerate(ranked_indices, start=1):
            score = float(scores[int(index_value)])
            passage = self.passages[int(index_value)]
            results.append(
                RetrievalResult(
                    pid=passage.pid,
                    text=passage.text,
                    score=score,
                    rank=rank,
                    source="bm25",
                    score_details={"bm25": score},
                )
            )
        return results


class HybridRetriever:
    """Hybrid dense plus BM25 retriever using reciprocal rank fusion."""

    def __init__(
        self,
        dense_retriever: SentenceBERTRetriever,
        bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 5,
        max_query_variants: int = 4,
    ) -> None:
        """
        Initializes the hybrid retriever.

        @param dense_retriever: Loaded dense retriever.
        @param bm25_retriever: Initialized BM25 retriever.
        @param rrf_k: Reciprocal rank fusion smoothing constant.
        @param candidate_multiplier: Candidate pool multiplier before final top-k truncation.
        @param max_query_variants: Maximum number of query variants used for fusion.
        @return: None.
        """

        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = max(1, int(rrf_k))
        self.candidate_multiplier = max(1, int(candidate_multiplier))
        self.max_query_variants = max(1, int(max_query_variants))

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Retrieves top-k fused results using dense, BM25, and query variants.

        @param query: Query text.
        @param top_k: Number of passages to return.
        @return: Ranked hybrid retrieval results.
        """

        if top_k <= 0:
            return []

        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        query_variants = TextPreprocessor.build_query_variants(query, self.max_query_variants)
        fused: Dict[str, Dict[str, object]] = {}

        for variant_index, variant in enumerate(query_variants):
            variant_weight = 1.0 / (1.0 + (0.25 * variant_index))
            self._add_results(
                fused=fused,
                results=self.dense_retriever.retrieve(variant, candidate_k),
                source="dense",
                variant_weight=variant_weight,
            )
            self._add_results(
                fused=fused,
                results=self.bm25_retriever.retrieve(variant, candidate_k),
                source="bm25",
                variant_weight=variant_weight,
            )

        ranked_items = sorted(
            fused.values(),
            key=lambda item: (-float(item["score"]), str(item["pid"])),
        )[:top_k]

        results: List[RetrievalResult] = []
        for rank, item in enumerate(ranked_items, start=1):
            results.append(
                RetrievalResult(
                    pid=str(item["pid"]),
                    text=str(item["text"]),
                    score=float(item["score"]),
                    rank=rank,
                    source="hybrid",
                    score_details=dict(item["score_details"]),
                )
            )
        return results

    def _add_results(
        self,
        fused: Dict[str, Dict[str, object]],
        results: Sequence[RetrievalResult],
        source: str,
        variant_weight: float,
    ) -> None:
        """
        Adds ranked results to the reciprocal-rank-fusion accumulator.

        @param fused: Mutable accumulator keyed by passage id.
        @param results: Ranked results from one retriever and query variant.
        @param source: Source retriever label.
        @param variant_weight: Weight for this query variant.
        @return: None.
        """

        for result in results:
            contribution = variant_weight / float(self.rrf_k + result.rank)
            if result.pid not in fused:
                fused[result.pid] = {
                    "pid": result.pid,
                    "text": result.text,
                    "score": 0.0,
                    "score_details": {},
                }

            fused[result.pid]["score"] = float(fused[result.pid]["score"]) + contribution
            details = fused[result.pid]["score_details"]
            details[source] = float(details.get(source, 0.0)) + contribution


class CrossEncoderRerankRetriever:
    """Hybrid candidate retriever followed by MS MARCO cross-encoder reranking."""

    def __init__(
        self,
        base_retriever: HybridRetriever,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_k: int = 50,
    ) -> None:
        """
        Initializes the cross-encoder reranking retriever.

        This retriever first asks the hybrid retriever for a larger candidate pool.
        It then scores each query-passage pair with a cross-encoder model trained
        for MS MARCO-style relevance ranking. This usually gives better ordering
        than using dense, BM25, or simple reciprocal-rank fusion alone, but it is
        slower and may need to download the reranker model the first time it runs.

        @param base_retriever: Hybrid retriever used to collect candidates.
        @param model_name: Sentence-Transformers cross-encoder model name.
        @param candidate_k: Number of candidates to rerank before returning top-k.
        @return: None.
        @raises ImportError: If sentence-transformers CrossEncoder is unavailable.
        """

        if CrossEncoder is None:
            raise ImportError("sentence-transformers CrossEncoder is required for reranking.")

        self.base_retriever = base_retriever
        self.model_name = model_name
        self.candidate_k = max(1, int(candidate_k))
        self.model = CrossEncoder(model_name)

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Retrieves top-k results using hybrid retrieval plus cross-encoder reranking.

        @param query: Query text.
        @param top_k: Number of final passages to return.
        @return: Ranked reranked retrieval results.
        """

        if top_k <= 0:
            return []

        candidate_count = max(top_k, self.candidate_k)
        candidates = self.base_retriever.retrieve(query, candidate_count)
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = self.model.predict(pairs)

        reranked_items = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]

        results: List[RetrievalResult] = []
        for rank, (candidate, rerank_score) in enumerate(reranked_items, start=1):
            score_details = dict(candidate.score_details)
            score_details["reranker"] = float(rerank_score)
            score_details["pre_rerank_score"] = float(candidate.score)
            results.append(
                RetrievalResult(
                    pid=candidate.pid,
                    text=candidate.text,
                    score=float(rerank_score),
                    rank=rank,
                    source="hybrid-rerank",
                    score_details=score_details,
                )
            )

        return results

"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from data_loader import MSMarcoDataLoader, Passage, QueryExample
from evaluator import Evaluator
from generators import BaseGenerator, build_generator
from retrievers import BM25Retriever, HybridRetriever, RetrievalResult, SentenceBERTRetriever
from settings import AppSettings
from subset_builder import SubsetBuilder
from utils import build_rag_prompt


class RAGPipeline:
    """End-to-end project pipeline for subset construction, indexing, retrieval, evaluation, and generation."""

    def __init__(self, settings: AppSettings) -> None:
        """
        Initializes the full RAG pipeline.

        @param settings: Application settings.
        @return: None.
        """

        self.settings = settings
        self.settings.ensure_directories()
        self.dense_retriever = SentenceBERTRetriever(
            embedding_model_name=settings.retrieval.embedding_model_name,
            faiss_index_path=settings.data.faiss_index_path,
            metadata_parquet_path=settings.data.metadata_parquet_path,
            dense_embeddings_path=settings.data.dense_embeddings_path,
        )
        self.passages: List[Passage] = []
        self.judged_queries: List[QueryExample] = []
        self.qrels: Dict[str, Dict[str, int]] = {}
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.generator_cache: Dict[str, BaseGenerator] = {}

    def build_subset(self) -> int:
        """
        Builds the judged MS MARCO subset used by the project.

        @return: Number of passages written to the subset collection.
        """

        builder = SubsetBuilder(self.settings)
        judged_queries, qrels, subset_count = builder.build()
        self.judged_queries = judged_queries
        self.qrels = qrels
        return subset_count

    def load_subset_data(self) -> None:
        """
        Loads subset passages, judged queries, qrels, and lexical retrievers.

        @return: None.
        """

        self.passages = MSMarcoDataLoader.load_passages(self.settings.data.subset_collection_tsv)
        self.judged_queries = MSMarcoDataLoader.load_queries(self.settings.data.subset_queries_tsv)
        self.qrels = MSMarcoDataLoader.load_qrels(self.settings.data.subset_qrels_tsv)
        self.bm25_retriever = BM25Retriever(self.passages)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            bm25_retriever=self.bm25_retriever,
            rrf_k=self.settings.retrieval.hybrid_rrf_k,
            candidate_multiplier=self.settings.retrieval.hybrid_candidate_multiplier,
            max_query_variants=self.settings.retrieval.hybrid_max_query_variants,
        )

    def build_dense_index(self) -> None:
        """
        Builds and saves the dense FAISS index from the loaded subset.

        @return: None.
        """

        if not self.passages:
            self.load_subset_data()
        self.dense_retriever.build_index(
            self.passages,
            batch_size=self.settings.retrieval.batch_size,
        )

    def load_dense_index(self) -> None:
        """
        Loads the saved dense index and makes sure subset data is available.

        @return: None.
        """

        self.dense_retriever.load_index()
        if not self.passages or self.bm25_retriever is None or self.hybrid_retriever is None:
            self.load_subset_data()

    def get_generator(self, backend: Optional[str] = None) -> BaseGenerator:
        """
        Returns a cached answer generator backend.

        @param backend: Optional backend override, such as local or openai.
        @return: Generator instance.
        """

        key = (backend or self.settings.generation.default_backend).lower()
        if key not in self.generator_cache:
            self.generator_cache[key] = build_generator(self.settings.generation, backend=key)
        return self.generator_cache[key]

    def get_rerank_retriever(self) -> CrossEncoderRerankRetriever:
        """
        Lazily initializes the optional hybrid-plus-cross-encoder reranker.

        The reranker is not loaded during normal startup because it is slower and
        may download a model the first time it is used.

        @return: Initialized CrossEncoderRerankRetriever instance.
        @raises ValueError: If the hybrid retriever has not been initialized.
        """

        if self.hybrid_retriever is None:
            raise ValueError("Hybrid retriever not initialized. Load subset data first.")

        if self.rerank_retriever is None:
            self.rerank_retriever = CrossEncoderRerankRetriever(
                base_retriever=self.hybrid_retriever,
                model_name=self.settings.retrieval.reranker_model_name,
                candidate_k=self.settings.retrieval.reranker_candidate_k,
            )

        return self.rerank_retriever

    def retrieve(self, query: str, retriever_type: str = "hybrid") -> List[RetrievalResult]:
        """
        Retrieves ranked passages using dense, BM25, hybrid, or hybrid-rerank retrieval.

        @param query: User query.
        @param retriever_type: Retrieval mode: dense, bm25, hybrid, or hybrid-rerank.
        @return: Ranked retrieval results.
        @raises ValueError: If the requested retriever is unsupported or unavailable.
        """

        retriever_name = retriever_type.lower().strip()
        top_k = self.settings.retrieval.top_k

        if retriever_name == "dense":
            return self.dense_retriever.retrieve(query, top_k)

        if retriever_name == "bm25":
            if self.bm25_retriever is None:
                raise ValueError("BM25 retriever not initialized. Load subset data first.")
            return self.bm25_retriever.retrieve(query, top_k)

        if retriever_name == "hybrid":
            if self.hybrid_retriever is None:
                raise ValueError("Hybrid retriever not initialized. Load subset data first.")
            return self.hybrid_retriever.retrieve(query, top_k)

        if retriever_name in {"hybrid-rerank", "rerank"}:
            return self.get_rerank_retriever().retrieve(query, top_k)

        raise ValueError(f"Unsupported retriever type: {retriever_type}")

    def search_query(self, query: str, retriever_type: str = "hybrid") -> Dict[str, object]:
        """
        Runs retrieval and builds the RAG prompt for a user query.

        @param query: User query.
        @param retriever_type: Retrieval mode.
        @return: Response dictionary containing query, results, prompt, and answer placeholder.
        """

        results = self.retrieve(query, retriever_type=retriever_type)
        prompt = build_rag_prompt(
            query=query,
            results=results,
            max_context_passages=self.settings.retrieval.max_context_passages,
        )
        return {
            "query": query,
            "retriever_type": retriever_type,
            "results": results,
            "prompt": prompt,
            "answer": None,
        }

    def answer_query(
        self,
        query: str,
        backend: Optional[str] = None,
        retriever_type: str = "hybrid",
    ) -> Dict[str, object]:
        """
        Runs retrieval and generates a grounded answer.

        @param query: User query.
        @param backend: Optional generation backend override.
        @param retriever_type: Retrieval mode.
        @return: Response dictionary containing retrieval results and generated answer.
        """

        response = self.search_query(query, retriever_type=retriever_type)
        generator = self.get_generator(backend)
        response["backend"] = backend or self.settings.generation.default_backend
        response["answer"] = generator.generate_answer(str(response["prompt"]))
        return response

    def run_evaluation(self, include_reranker: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Runs retrieval evaluation for the configured retrievers.

        The default evaluation compares dense, BM25, and hybrid retrieval. The
        optional reranker can be included when the user wants a stronger but slower
        method based on cross-encoder reranking.

        @param include_reranker: Whether to include hybrid-rerank evaluation.
        @return: Mapping containing detailed retriever tables and a comparison summary.
        """

        if self.bm25_retriever is None or self.hybrid_retriever is None:
            self.load_subset_data()

        evaluation_queries = Evaluator.build_project_query_set(
            self.judged_queries,
            self.settings.evaluation,
        )
        top_k = self.settings.retrieval.top_k
        tables: Dict[str, pd.DataFrame] = {
            "dense": Evaluator.evaluate_retriever(
                self.dense_retriever,
                evaluation_queries,
                self.qrels,
                top_k,
                "dense",
            ),
            "bm25": Evaluator.evaluate_retriever(
                self.bm25_retriever,
                evaluation_queries,
                self.qrels,
                top_k,
                "bm25",
            ),
            "hybrid": Evaluator.evaluate_retriever(
                self.hybrid_retriever,
                evaluation_queries,
                self.qrels,
                top_k,
                "hybrid",
            ),
        }

        if include_reranker:
            tables["hybrid_rerank"] = Evaluator.evaluate_retriever(
                self.get_rerank_retriever(),
                evaluation_queries,
                self.qrels,
                top_k,
                "hybrid-rerank",
            )

        tables["comparison_summary"] = Evaluator.build_comparison_summary(tables, top_k)
        return tables

    def subset_summary(self) -> Dict[str, object]:
        """
        Returns a concise summary of the loaded subset and limitations.

        @return: Summary dictionary for CLI or UI display.
        """

        return {
            "passages": len(self.passages),
            "judged_queries": len(self.judged_queries),
            "qrel_queries": len(self.qrels),
            "limitation": (
                "This project indexes a filtered judged MS MARCO subset, not the full collection. "
                "Manual queries may fail when the needed evidence is outside the subset."
            ),
        }

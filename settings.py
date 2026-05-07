"""Project settings for the MS MARCO RAG course project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DataPaths:
    """
    Holds dataset and artifact paths.

    @param collection_tsv: Path to the raw MS MARCO collection file.
    @param queries_dev_tsv: Path to the raw MS MARCO dev queries file.
    @param queries_eval_tsv: Path to the raw MS MARCO eval queries file.
    @param qrels_dev_tsv: Path to the raw MS MARCO dev qrels file.
    @param subset_collection_tsv: Path to the filtered subset collection file.
    @param subset_queries_tsv: Path to the filtered judged query file.
    @param subset_qrels_tsv: Path to the filtered judged qrels file.
    @param faiss_index_path: Path to the saved FAISS index.
    @param metadata_parquet_path: Path to the saved passage metadata file.
    @param dense_embeddings_path: Path to the saved dense embedding matrix.
    @param offload_dir: Path used for local model disk offloading.
    @param evaluations_dir: Directory for saved evaluation outputs.
    """

    collection_tsv: str = "data/raw/collection.tsv"
    queries_dev_tsv: str = "data/raw/queries.dev.tsv"
    queries_eval_tsv: str = "data/raw/queries.eval.tsv"
    qrels_dev_tsv: str = "data/raw/qrels.dev.tsv"

    subset_collection_tsv: str = "data/subsets/collection.subset.tsv"
    subset_queries_tsv: str = "data/subsets/queries.dev.judged.tsv"
    subset_qrels_tsv: str = "data/subsets/qrels.dev.judged.tsv"

    faiss_index_path: str = "artifacts/faiss/msmarco_subset.faiss"
    metadata_parquet_path: str = "artifacts/faiss/msmarco_subset_metadata.parquet"
    dense_embeddings_path: str = "artifacts/embeddings/msmarco_subset_embeddings.npy"
    offload_dir: str = "artifacts/offload"

    evaluations_dir: str = "outputs/evaluations"


@dataclass
class RetrievalSettings:
    """
    Holds retrieval configuration.

    @param embedding_model_name: SentenceTransformer model name.
    @param top_k: Number of passages to retrieve.
    @param max_context_passages: Number of passages to include in the RAG prompt.
    @param batch_size: Embedding batch size.
    @param hybrid_rrf_k: Reciprocal-rank-fusion smoothing constant for hybrid retrieval.
    @param hybrid_candidate_multiplier: Candidate multiplier used before hybrid top-k truncation.
    @param hybrid_max_query_variants: Maximum deterministic query variants used by hybrid retrieval.
    @param subset_max_judged_queries: Maximum judged dev queries to keep in the subset.
    @param subset_max_passages: Maximum judged passages to keep in the subset.
    @param reranker_model_name: Cross-encoder reranker model name for optional hybrid-rerank retrieval.
    @param reranker_candidate_k: Number of hybrid candidates to rerank.
    """

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    top_k: int = int(os.getenv("TOP_K", "10"))
    max_context_passages: int = int(os.getenv("MAX_CONTEXT_PASSAGES", "5"))
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))

    hybrid_rrf_k: int = int(os.getenv("HYBRID_RRF_K", "60"))
    hybrid_candidate_multiplier: int = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "5"))
    hybrid_max_query_variants: int = int(os.getenv("HYBRID_MAX_QUERY_VARIANTS", "4"))

    subset_max_judged_queries: int = int(os.getenv("SUBSET_MAX_JUDGED_QUERIES", "5000"))
    subset_max_passages: Optional[int] = (
        int(os.getenv("SUBSET_MAX_PASSAGES"))
        if os.getenv("SUBSET_MAX_PASSAGES")
        else 100000
    )
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_candidate_k: int = int(os.getenv("RERANKER_CANDIDATE_K", "50"))


@dataclass
class GenerationSettings:
    """
    Holds answer generation configuration.

    @param default_backend: Default generation backend.
    @param local_model_id: Local Hugging Face model id.
    @param local_device_map: Transformers device map.
    @param local_torch_dtype: Torch dtype string.
    @param openai_model: OpenAI model name.
    @param temperature: Sampling temperature.
    @param max_new_tokens: Maximum generated tokens.
    @param max_context_passages: Number of passages used to build the generation prompt.
    @param local_system_prompt: System prompt for the local generator.
    """

    default_backend: str = os.getenv("GENERATOR_BACKEND", "local")
    local_model_id: str = os.getenv("LOCAL_MODEL_ID", "openai/gpt-oss-20b")
    local_device_map: str = os.getenv("LOCAL_DEVICE_MAP", "auto")
    local_torch_dtype: str = os.getenv("LOCAL_TORCH_DTYPE", "auto")

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "256"))
    max_context_passages: int = int(os.getenv("MAX_CONTEXT_PASSAGES", "5"))

    local_system_prompt: str = os.getenv(
        "LOCAL_SYSTEM_PROMPT",
        (
            "You are a careful retrieval-augmented question answering assistant. "
            "Only answer using the provided passages. "
            "If the passages do not support the answer, clearly say the evidence is insufficient. "
            "Return only the final user-facing answer."
        ),
    )


@dataclass
class EvaluationSettings:
    """
    Holds evaluation configuration.

    @param random_seed: Seed for reproducible sampling.
    @param random_ms_marco_queries: Number of random judged MS MARCO queries.
    @param manual_queries: Manual student or instructor queries.
    """

    random_seed: int = int(os.getenv("EVAL_RANDOM_SEED", "42"))
    random_ms_marco_queries: int = int(os.getenv("EVAL_RANDOM_MS_MARCO_QUERIES", "5"))
    manual_queries: List[str] = field(
        default_factory=lambda: [
            "what is the role of phloem in plants",
            "what causes dark yellow urine",
            "what was the Manhattan Project",
            "what is restorative justice",
            "what causes pain under the left rib cage",
        ]
    )


@dataclass
class AppSettings:
    """
    Root application settings object.

    @param data: Dataset and artifact paths.
    @param retrieval: Retrieval settings.
    @param generation: Generation settings.
    @param evaluation: Evaluation settings.
    """

    data: DataPaths = field(default_factory=DataPaths)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    def ensure_directories(self) -> None:
        """
        Creates required output directories.

        @return: None.
        """

        paths = [
            self.data.subset_collection_tsv,
            self.data.subset_queries_tsv,
            self.data.subset_qrels_tsv,
            self.data.faiss_index_path,
            self.data.metadata_parquet_path,
            self.data.dense_embeddings_path,
            str(Path(self.data.offload_dir) / "dummy.txt"),
            str(Path(self.data.evaluations_dir) / "dummy.csv"),
        ]

        for path_str in paths:
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)

"""Command-line entry point for the MS MARCO RAG project."""

from __future__ import annotations

import argparse
import logging

from pipeline import RAGPipeline
from settings import AppSettings
from utils import format_evaluation_summary, format_results, save_evaluation_tables


def build_parser() -> argparse.ArgumentParser:
    """
    Builds the command-line argument parser.

    @return: Configured ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(description="MS MARCO passage RAG project")
    parser.add_argument("--build-subset", action="store_true", help="Build the judged subset files")
    parser.add_argument("--build-index", action="store_true", help="Build the dense FAISS index")
    parser.add_argument("--load-index", action="store_true", help="Load the saved dense FAISS index")
    parser.add_argument("--query", help="Run a search for a single query")
    parser.add_argument(
        "--retriever",
        choices=["dense", "bm25", "hybrid", "hybrid-rerank"],
        default="hybrid",
        help="Retriever type",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "openai"],
        default=None,
        help="Generation backend",
    )
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Generate a grounded RAG answer",
    )
    parser.add_argument("--evaluate", action="store_true", help="Run retrieval evaluation")
    parser.add_argument(
        "--include-reranker",
        action="store_true",
        help="Include slower hybrid-rerank evaluation with a cross-encoder reranker",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def _configure_logging(log_level: str) -> None:
    """
    Configures basic application logging.

    @param log_level: Requested log level name.
    @return: None.
    """

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """
    Runs the CLI workflow requested by the user.

    @return: None.
    """

    parser = build_parser()
    args = parser.parse_args()
    _configure_logging(args.log_level)

    settings = AppSettings()
    pipeline = RAGPipeline(settings)

    if args.build_subset:
        subset_count = pipeline.build_subset()
        print(f"Built judged subset with {subset_count} passages.")

    if args.build_index:
        pipeline.load_subset_data()
        pipeline.build_dense_index()
        print("Dense FAISS index built successfully.")

    if args.load_index or args.query or args.evaluate:
        pipeline.load_subset_data()
        pipeline.load_dense_index()
        summary = pipeline.subset_summary()
        print(
            "Subset summary: "
            f"{summary['passages']} passages, "
            f"{summary['judged_queries']} judged queries."
        )

    if args.query:
        response = pipeline.search_query(args.query, retriever_type=args.retriever)
        print(f"\n=== {args.retriever.upper()} Top-{settings.retrieval.top_k} ===")
        print(format_results(response["results"]))

        if args.generate_answer:
            answered = pipeline.answer_query(
                args.query,
                backend=args.backend,
                retriever_type=args.retriever,
            )
            print("\n=== Grounded Answer ===")
            print(answered["answer"])

    if args.evaluate:
        tables = pipeline.run_evaluation(include_reranker=args.include_reranker)
        save_evaluation_tables(tables, output_dir=settings.data.evaluations_dir)
        print("\n=== Retriever Comparison Summary ===")
        print(format_evaluation_summary(tables["comparison_summary"]))
        print("\nDetailed CSV files saved to outputs/evaluations.")
        print("Manual queries without official qrels are shown, but excluded from mean judged P@10.")


if __name__ == "__main__":
    main()

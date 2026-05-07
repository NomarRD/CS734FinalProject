"""General project utility helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd

from retrievers import RetrievalResult, TextPreprocessor


YEAR_PATTERN = re.compile(r"\b(18|19|20)\d{2}\b")


def build_rag_prompt(query: str, results: Sequence[RetrievalResult], max_context_passages: int) -> str:
    """
    Builds a grounded RAG prompt.

    @param query: User query.
    @param results: Retrieved passages.
    @param max_context_passages: Maximum number of passages to include in context.
    @return: Prompt string.
    """

    blocks = []
    for result in list(results)[:max_context_passages]:
        source_label = f"source={result.source}" if result.source else "source=unknown"
        blocks.append(
            f"[Passage {result.rank} | pid={result.pid} | {source_label} | score={result.score:.4f}]\n"
            f"{result.text}"
        )

    context = "\n\n".join(blocks) if blocks else "No retrieved passages were available."
    return (
        "Use only the retrieved passages to answer the question.\n"
        "If the evidence is insufficient, say so clearly.\n"
        "Do not add facts that are not supported by the passages.\n"
        "Keep the answer concise and grounded in the passages.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved Passages:\n{context}\n\n"
        "Output format:\n"
        "Answer: <short answer>\n"
        "Evidence: <brief supporting summary>"
    )


def annotate_result(query: str, result: RetrievalResult) -> Dict[str, object]:
    """
    Adds lightweight facet metadata to a result.

    @param query: User query.
    @param result: Retrieval result.
    @return: Annotated dictionary for display and filtering.
    """

    query_tokens = set(TextPreprocessor.tokenize(query))
    passage_tokens = TextPreprocessor.tokenize(result.text)
    overlap = len(query_tokens.intersection(passage_tokens))
    length = len(passage_tokens)

    if length < 40:
        length_bucket = "short"
    elif length < 90:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    if overlap <= 1:
        overlap_bucket = "low"
    elif overlap <= 3:
        overlap_bucket = "medium"
    else:
        overlap_bucket = "high"

    return {
        "pid": result.pid,
        "text": result.text,
        "score": result.score,
        "rank": result.rank,
        "source": result.source,
        "score_details": result.score_details,
        "length_bucket": length_bucket,
        "contains_digits": any(ch.isdigit() for ch in result.text),
        "contains_year": bool(YEAR_PATTERN.search(result.text)),
        "overlap_bucket": overlap_bucket,
    }


def annotate_results(query: str, results: Sequence[RetrievalResult]) -> list[Dict[str, object]]:
    """
    Adds display metadata to a ranked result list.

    @param query: User query.
    @param results: Retrieval results.
    @return: List of annotated result dictionaries.
    """

    return [annotate_result(query, result) for result in results]


def filter_annotated_results(
    annotated_results: Sequence[Dict[str, object]],
    allowed_length_buckets: Sequence[str],
    require_digits: bool,
    require_year: bool,
    allowed_overlap_buckets: Sequence[str],
) -> list[Dict[str, object]]:
    """
    Applies lightweight UI facet filters to annotated results.

    @param annotated_results: Annotated result dictionaries.
    @param allowed_length_buckets: Enabled length buckets.
    @param require_digits: Whether to require at least one digit.
    @param require_year: Whether to require a likely year.
    @param allowed_overlap_buckets: Enabled lexical-overlap buckets.
    @return: Filtered annotated result dictionaries.
    """

    filtered = []
    for item in annotated_results:
        if item["length_bucket"] not in allowed_length_buckets:
            continue
        if require_digits and not item["contains_digits"]:
            continue
        if require_year and not item["contains_year"]:
            continue
        if item["overlap_bucket"] not in allowed_overlap_buckets:
            continue
        filtered.append(item)
    return filtered


def format_results(results: Sequence[RetrievalResult]) -> str:
    """
    Formats retrieval results for command-line display.

    @param results: Retrieval result list.
    @return: Multi-line formatted result text.
    """

    lines = []
    for result in results:
        preview = result.text.replace("\n", " ").strip()
        if len(preview) > 180:
            preview = preview[:177] + "..."
        source = result.source or "unknown"
        lines.append(
            f"{result.rank:>2}. pid={result.pid} source={source} score={result.score:.4f} | {preview}"
        )
    return "\n".join(lines)


def save_evaluation_tables(tables: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Saves evaluation tables to CSV files.

    @param tables: Mapping from table name to DataFrame.
    @param output_dir: Directory where CSV files should be written.
    @return: None.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}_evaluation.csv", index=False)


def format_evaluation_summary(summary_table: pd.DataFrame) -> str:
    """
    Formats the compact evaluation comparison table for CLI output.

    @param summary_table: Comparison summary DataFrame.
    @return: Printable summary text.
    """

    if summary_table.empty:
        return "No evaluation summary was generated."
    return summary_table.to_string(index=False)

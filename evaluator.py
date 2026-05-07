"""Evaluation utilities for retrieval quality."""

from __future__ import annotations

import random
from typing import Dict, Optional, Sequence

import pandas as pd

from data_loader import QueryExample
from retrievers import RetrievalResult
from settings import EvaluationSettings


class Evaluator:
    """Computes retrieval metrics and builds the project evaluation set."""

    @staticmethod
    def _relevant_result_ranks(
        results: Sequence[RetrievalResult],
        relevant_pids: Optional[Dict[str, int]],
        k: int,
    ) -> list[int]:
        """
        Finds one-based ranks of relevant results in the top-k list.

        @param results: Ranked retrieval results.
        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: One-based ranks where relevant passages were retrieved.
        """

        if relevant_pids is None or k <= 0:
            return []

        ranks: list[int] = []
        for rank, result in enumerate(list(results)[:k], start=1):
            if relevant_pids.get(result.pid, 0) > 0:
                ranks.append(rank)
        return ranks

    @staticmethod
    def precision_at_k(
        results: Sequence[RetrievalResult],
        relevant_pids: Optional[Dict[str, int]],
        k: int = 10,
    ) -> float:
        """
        Computes Precision@k for judged queries.

        Note that many MS MARCO passage qrels have only one relevant passage for a
        query. In that common case, the maximum possible P@10 is 0.10. Therefore,
        Hit@10, Recall@10, MRR@10, and normalized P@10 are also reported to make
        the result easier to interpret.

        @param results: Ranked retrieval results.
        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: Precision@k value, or NaN for unjudged queries.
        """

        if relevant_pids is None:
            return float("nan")
        if k <= 0:
            return 0.0

        return len(Evaluator._relevant_result_ranks(results, relevant_pids, k)) / float(k)

    @staticmethod
    def max_precision_at_k(relevant_pids: Optional[Dict[str, int]], k: int = 10) -> float:
        """
        Computes the maximum possible Precision@k for a judged query.

        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: Maximum possible Precision@k, or NaN for unjudged queries.
        """

        if relevant_pids is None:
            return float("nan")
        if k <= 0:
            return 0.0

        positive_count = sum(1 for relevance in relevant_pids.values() if relevance > 0)
        return min(positive_count, k) / float(k)

    @staticmethod
    def normalized_precision_at_k(
        results: Sequence[RetrievalResult],
        relevant_pids: Optional[Dict[str, int]],
        k: int = 10,
    ) -> float:
        """
        Computes P@k divided by the query's maximum possible P@k.

        For single-answer MS MARCO qrels, this is easier to explain than raw P@10:
        retrieving the one relevant passage anywhere in the top 10 gives normalized
        P@10 = 1.0 for that query.

        @param results: Ranked retrieval results.
        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: Normalized Precision@k, or NaN for unjudged queries.
        """

        if relevant_pids is None:
            return float("nan")

        max_p = Evaluator.max_precision_at_k(relevant_pids, k)
        if max_p <= 0:
            return 0.0

        return Evaluator.precision_at_k(results, relevant_pids, k) / max_p

    @staticmethod
    def relevant_hits_at_k(
        results: Sequence[RetrievalResult],
        relevant_pids: Optional[Dict[str, int]],
        k: int = 10,
    ) -> Optional[int]:
        """
        Counts relevant hits in the top-k list.

        @param results: Ranked retrieval results.
        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: Relevant hit count, or None for unjudged queries.
        """

        if relevant_pids is None:
            return None

        return len(Evaluator._relevant_result_ranks(results, relevant_pids, k))

    @staticmethod
    def hit_at_k(
        results: Sequence[RetrievalResult],
        relevant_pids: Optional[Dict[str, int]],
        k: int = 10,
    ) -> float:
        """
        Computes Hit@k for judged queries.

        @param results: Ranked retrieval results.
        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: 1.0 if any relevant result appears in the top-k, 0.0 if not, or NaN if unjudged.
        """

        if relevant_pids is None:
            return float("nan")

        return 1.0 if Evaluator._relevant_result_ranks(results, relevant_pids, k) else 0.0

    @staticmethod
    def recall_at_k(
        results: Sequence[RetrievalResult],
        relevant_pids: Optional[Dict[str, int]],
        k: int = 10,
    ) -> float:
        """
        Computes Recall@k for judged queries.

        @param results: Ranked retrieval results.
        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: Recall@k, or NaN for unjudged queries.
        """

        if relevant_pids is None:
            return float("nan")

        positive_count = sum(1 for relevance in relevant_pids.values() if relevance > 0)
        if positive_count <= 0:
            return 0.0

        return len(Evaluator._relevant_result_ranks(results, relevant_pids, k)) / float(positive_count)

    @staticmethod
    def reciprocal_rank_at_k(
        results: Sequence[RetrievalResult],
        relevant_pids: Optional[Dict[str, int]],
        k: int = 10,
    ) -> float:
        """
        Computes reciprocal rank at k.

        @param results: Ranked retrieval results.
        @param relevant_pids: Relevance labels for the query, or None if unjudged.
        @param k: Rank cutoff.
        @return: Reciprocal rank for the first relevant result, 0.0 if missed, or NaN if unjudged.
        """

        if relevant_pids is None:
            return float("nan")

        ranks = Evaluator._relevant_result_ranks(results, relevant_pids, k)
        if not ranks:
            return 0.0

        return 1.0 / float(min(ranks))

    @staticmethod
    def build_project_query_set(
        judged_queries: Sequence[QueryExample],
        evaluation_settings: EvaluationSettings,
    ) -> list[QueryExample]:
        """
        Builds the project's ten-query evaluation set.

        The first group is sampled from judged MS MARCO dev queries. The second
        group is the manually supplied project queries, which are intentionally
        marked as unjudged unless matching qrels exist.

        @param judged_queries: Judged MS MARCO queries.
        @param evaluation_settings: Evaluation settings.
        @return: Combined evaluation query set.
        """

        rng = random.Random(evaluation_settings.random_seed)
        query_pool = list(judged_queries)
        random_count = min(evaluation_settings.random_ms_marco_queries, len(query_pool))
        random_queries = rng.sample(query_pool, random_count) if random_count else []
        manual_queries = [
            QueryExample(qid=f"manual_{index}", text=query_text)
            for index, query_text in enumerate(evaluation_settings.manual_queries, start=1)
        ]
        return list(random_queries) + manual_queries

    @staticmethod
    def evaluate_retriever(
        retriever,
        queries: Sequence[QueryExample],
        qrels: Dict[str, Dict[str, int]],
        top_k: int,
        retriever_name: str,
    ) -> pd.DataFrame:
        """
        Evaluates a retriever over multiple queries.

        Manual queries without qrels are shown in the table, but their judged
        metrics are NaN and do not affect the judged mean. This prevents unjudged
        manual examples from being incorrectly counted as zero-relevance failures.

        @param retriever: Retriever object with a retrieve method.
        @param queries: Evaluation queries.
        @param qrels: Query relevance judgments.
        @param top_k: Rank cutoff.
        @param retriever_name: Label for the retriever.
        @return: Evaluation table with a final MEAN_JUDGED row.
        @raises ValueError: If the retriever is None.
        """

        if retriever is None:
            raise ValueError(f"Retriever is not initialized: {retriever_name}")

        rows = []
        metric_col = f"P@{top_k}"
        max_metric_col = f"MaxP@{top_k}"
        normalized_metric_col = f"NormP@{top_k}"
        hit_col = f"Hit@{top_k}"
        recall_col = f"Recall@{top_k}"
        mrr_col = f"RR@{top_k}"

        for query in queries:
            results = retriever.retrieve(query.text, top_k)
            relevant_pids = qrels.get(query.qid)
            judged = relevant_pids is not None
            hit_count = Evaluator.relevant_hits_at_k(results, relevant_pids, top_k)
            query_type = "manual" if query.qid.startswith("manual_") else "ms_marco_dev"
            rows.append(
                {
                    "retriever": retriever_name,
                    "qid": query.qid,
                    "query_type": query_type,
                    "query": query.text,
                    "judged": judged,
                    "relevant_hits": hit_count,
                    metric_col: Evaluator.precision_at_k(results, relevant_pids, top_k),
                    max_metric_col: Evaluator.max_precision_at_k(relevant_pids, top_k),
                    normalized_metric_col: Evaluator.normalized_precision_at_k(results, relevant_pids, top_k),
                    hit_col: Evaluator.hit_at_k(results, relevant_pids, top_k),
                    recall_col: Evaluator.recall_at_k(results, relevant_pids, top_k),
                    mrr_col: Evaluator.reciprocal_rank_at_k(results, relevant_pids, top_k),
                    "result_count": len(results),
                    "retrieved_pids": [result.pid for result in results],
                    "note": "official qrels unavailable" if not judged else "judged by qrels",
                }
            )

        frame = pd.DataFrame(rows)
        judged_frame = frame.loc[frame["judged"]] if not frame.empty else frame

        summary = {
            "retriever": retriever_name,
            "qid": "MEAN_JUDGED",
            "query_type": "summary",
            "query": "",
            "judged": True,
            "relevant_hits": judged_frame["relevant_hits"].mean() if not judged_frame.empty else float("nan"),
            metric_col: judged_frame[metric_col].mean() if not judged_frame.empty else float("nan"),
            max_metric_col: judged_frame[max_metric_col].mean() if not judged_frame.empty else float("nan"),
            normalized_metric_col: judged_frame[normalized_metric_col].mean() if not judged_frame.empty else float("nan"),
            hit_col: judged_frame[hit_col].mean() if not judged_frame.empty else float("nan"),
            recall_col: judged_frame[recall_col].mean() if not judged_frame.empty else float("nan"),
            mrr_col: judged_frame[mrr_col].mean() if not judged_frame.empty else float("nan"),
            "result_count": "",
            "retrieved_pids": [],
            "note": "mean excludes unjudged manual queries",
        }
        return pd.concat([frame, pd.DataFrame([summary])], ignore_index=True)

    @staticmethod
    def build_comparison_summary(tables: Dict[str, pd.DataFrame], top_k: int) -> pd.DataFrame:
        """
        Builds a compact comparison summary across retrievers.

        @param tables: Mapping from retriever name to detailed evaluation table.
        @param top_k: Rank cutoff used during evaluation.
        @return: Summary DataFrame with one row per retriever.
        """

        metric_col = f"P@{top_k}"
        max_metric_col = f"MaxP@{top_k}"
        normalized_metric_col = f"NormP@{top_k}"
        hit_col = f"Hit@{top_k}"
        recall_col = f"Recall@{top_k}"
        mrr_col = f"RR@{top_k}"

        rows = []
        for retriever_name, table in tables.items():
            if table.empty:
                continue

            detail_rows = table[table["qid"] != "MEAN_JUDGED"]
            judged_rows = detail_rows[detail_rows["judged"] == True]
            unjudged_rows = detail_rows[detail_rows["judged"] == False]
            summary_rows = table[table["qid"] == "MEAN_JUDGED"]

            def summary_value(column_name: str) -> float:
                """
                Extracts one numeric summary value.

                @param column_name: Summary column name.
                @return: Numeric summary value or NaN.
                """

                if summary_rows.empty or column_name not in summary_rows:
                    return float("nan")
                value = summary_rows.iloc[0][column_name]
                return float(value) if pd.notna(value) else float("nan")

            rows.append(
                {
                    "retriever": retriever_name,
                    "judged_queries": len(judged_rows),
                    "unjudged_manual_queries": len(unjudged_rows),
                    f"mean_judged_{metric_col}": summary_value(metric_col),
                    f"mean_possible_{metric_col}": summary_value(max_metric_col),
                    f"mean_normalized_{metric_col}": summary_value(normalized_metric_col),
                    f"mean_{hit_col}": summary_value(hit_col),
                    f"mean_{recall_col}": summary_value(recall_col),
                    f"mean_MRR@{top_k}": summary_value(mrr_col),
                    "relevant_hits_total": int(judged_rows["relevant_hits"].fillna(0).sum()),
                    "best_judged_query": Evaluator._best_query_label(judged_rows, metric_col),
                    "interpretation_note": (
                        "Raw P@10 is capped by the number of relevant qrels. "
                        "For one-relevant-passage queries, the best possible P@10 is 0.10."
                    ),
                    "limitation_note": (
                        "Manual queries are displayed but excluded from judged metrics when qrels are unavailable."
                    ),
                }
            )

        summary = pd.DataFrame(rows)
        if summary.empty:
            return summary
        return summary.sort_values(
            by=[f"mean_normalized_{metric_col}", f"mean_MRR@{top_k}", f"mean_judged_{metric_col}"],
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    @staticmethod
    def _best_query_label(frame: pd.DataFrame, metric_col: str) -> str:
        """
        Gets a short label for the highest-scoring judged query.

        @param frame: Judged query rows.
        @param metric_col: Metric column name.
        @return: Query label or empty string.
        """

        if frame.empty or metric_col not in frame:
            return ""

        best_row = frame.sort_values(by=metric_col, ascending=False).iloc[0]
        return f"{best_row['qid']} ({best_row[metric_col]:.3f})"

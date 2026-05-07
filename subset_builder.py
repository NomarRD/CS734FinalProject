"""Subset construction utilities for filtered MS MARCO indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

from settings import AppSettings
from data_loader import MSMarcoDataLoader, QueryExample


class SubsetBuilder:
    """
    Builds a judged subset of the MS MARCO passage collection.

    @param settings: Application settings.
    """

    def __init__(self, settings: AppSettings) -> None:
        """
        Initializes the subset builder.

        @param settings: Application settings.
        @return: None.
        """

        self.settings = settings

    def build(self) -> Tuple[List[QueryExample], Dict[str, Dict[str, int]], int]:
        """
        Builds the judged query subset and collection subset files.

        @return: Tuple containing saved judged queries, saved qrels, and subset passage count.
        """

        data = self.settings.data
        retrieval = self.settings.retrieval

        all_queries = MSMarcoDataLoader.load_queries(data.queries_dev_tsv)
        qrels = MSMarcoDataLoader.load_qrels(data.qrels_dev_tsv)

        judged_queries = [query for query in all_queries if query.qid in qrels]
        judged_queries.sort(key=lambda item: int(item.qid) if item.qid.isdigit() else item.qid)

        if retrieval.subset_max_judged_queries is not None:
            judged_queries = judged_queries[: retrieval.subset_max_judged_queries]

        kept_query_ids = {query.qid for query in judged_queries}
        kept_qrels = {qid: qrels[qid] for qid in kept_query_ids}

        judged_passage_ids: List[str] = []
        seen: Set[str] = set()
        for query in judged_queries:
            for pid in kept_qrels[query.qid]:
                if pid not in seen:
                    judged_passage_ids.append(pid)
                    seen.add(pid)

        if retrieval.subset_max_passages is not None:
            judged_passage_ids = judged_passage_ids[: retrieval.subset_max_passages]
            allowed_pids = set(judged_passage_ids)
            trimmed_qrels: Dict[str, Dict[str, int]] = {}
            for qid, pid_map in kept_qrels.items():
                filtered = {pid: rel for pid, rel in pid_map.items() if pid in allowed_pids}
                if filtered:
                    trimmed_qrels[qid] = filtered
            kept_qrels = trimmed_qrels
            judged_queries = [query for query in judged_queries if query.qid in kept_qrels]
        else:
            allowed_pids = set(judged_passage_ids)

        MSMarcoDataLoader.save_queries(data.subset_queries_tsv, judged_queries)
        MSMarcoDataLoader.save_qrels(data.subset_qrels_tsv, kept_qrels)
        subset_count = self._write_collection_subset(data.collection_tsv, data.subset_collection_tsv, allowed_pids)
        return judged_queries, kept_qrels, subset_count

    @staticmethod
    def _write_collection_subset(source_collection_tsv: str, output_collection_tsv: str, allowed_pids: Set[str]) -> int:
        """
        Writes a filtered collection subset to disk.

        @param source_collection_tsv: Full collection file path.
        @param output_collection_tsv: Output subset file path.
        @param allowed_pids: Passage ids that should be retained.
        @return: Number of passages written.
        """

        output_path = Path(output_collection_tsv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for passage in MSMarcoDataLoader.iterate_collection(source_collection_tsv):
                if passage.pid in allowed_pids:
                    handle.write(f"{passage.pid}\t{passage.text}\n")
                    count += 1
        return count
